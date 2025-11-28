from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate
import torch.nn.functional as F


class OSTrackActor(BaseActor):
    """ Actor for training OSTrack models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg
        contrastive_cfg = getattr(cfg.TRAIN, 'CONTRASTIVE', None) if cfg is not None else None
        self.contrastive_enabled = getattr(contrastive_cfg, 'ENABLED', False) if contrastive_cfg is not None else False
        self.contrastive_weight = getattr(contrastive_cfg, 'LOSS_WEIGHT', 0.0) if contrastive_cfg is not None else 0.0
        self.contrastive_num_negatives = getattr(contrastive_cfg, 'NUM_NEGATIVES', 16) if contrastive_cfg is not None else 16
        self.contrastive_temperature = getattr(contrastive_cfg, 'TEMPERATURE', 0.07) if contrastive_cfg is not None else 0.07
        self.contrastive_pos_radius = getattr(contrastive_cfg, 'POS_RADIUS', 1) if contrastive_cfg is not None else 1
        self.contrastive_neg_margin = getattr(contrastive_cfg, 'NEG_MARGIN', 2) if contrastive_cfg is not None else 2

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(out_dict, data)

        return loss, status

    def forward_pass(self, data):
        # currently only support 1 template and 1 search region
        assert len(data['template_images']) == 1
        assert len(data['search_images']) == 1

        template_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1,
                                                             *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
            # template_att_i = data['template_att'][i].view(-1, *data['template_att'].shape[2:])  # (batch, 128, 128)
            template_list.append(template_img_i)

        search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 320, 320)
        # search_att = data['search_att'][0].view(-1, *data['search_att'].shape[2:])  # (batch, 320, 320)

        box_mask_z = None
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            box_mask_z = generate_mask_cond(self.cfg, template_list[0].shape[0], template_list[0].device,
                                            data['template_anno'][0])

            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
                                                total_epochs=ce_start_epoch + ce_warm_epoch,
                                                ITERS_PER_EPOCH=1,
                                                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])

        if len(template_list) == 1:
            template_list = template_list[0]

        out_dict = self.net(template=template_list,
                            search=search_img,
                            ce_template_mask=box_mask_z,
                            ce_keep_rate=ce_keep_rate,
                            return_last_attn=False)

        return out_dict

    def bbox_to_grid_indices(self, gt_bbox, feat_sz, search_size, stride):
        """Convert gt box to positive and negative grid indices on the feature map."""
        pos_indices = []
        neg_indices = []
        pos_margin = self.contrastive_pos_radius * stride
        neg_margin = self.contrastive_neg_margin * stride

        # heuristically convert to pixel if boxes are normalized
        bbox_max = gt_bbox.max().item()
        if bbox_max <= 2.0:
            bbox_px = gt_bbox * search_size
        else:
            bbox_px = gt_bbox

        for b in range(bbox_px.shape[0]):
            x1, y1, w, h = bbox_px[b]
            x2 = x1 + w
            y2 = y1 + h
            pos_list = []
            neg_list = []
            for iy in range(feat_sz):
                cy = (iy + 0.5) * stride
                for ix in range(feat_sz):
                    cx = (ix + 0.5) * stride
                    inside = (cx >= x1 - pos_margin) and (cx <= x2 + pos_margin) and (cy >= y1 - pos_margin) and (cy <= y2 + pos_margin)
                    outside = (cx < x1 - neg_margin) or (cx > x2 + neg_margin) or (cy < y1 - neg_margin) or (cy > y2 + neg_margin)
                    if inside:
                        pos_list.append((iy, ix))
                    elif outside:
                        neg_list.append((iy, ix))

            if len(pos_list) == 0:
                cx = x1 + 0.5 * w
                cy = y1 + 0.5 * h
                center_ix = min(max(int(cx / stride), 0), feat_sz - 1)
                center_iy = min(max(int(cy / stride), 0), feat_sz - 1)
                for iy in range(max(0, center_iy - self.contrastive_pos_radius), min(feat_sz, center_iy + self.contrastive_pos_radius + 1)):
                    for ix in range(max(0, center_ix - self.contrastive_pos_radius), min(feat_sz, center_ix + self.contrastive_pos_radius + 1)):
                        pos_list.append((iy, ix))

            if len(neg_list) == 0:
                for iy in range(feat_sz):
                    for ix in range(feat_sz):
                        if (iy, ix) not in pos_list:
                            neg_list.append((iy, ix))

            pos_indices.append(pos_list)
            neg_indices.append(neg_list)

        return pos_indices, neg_indices

    @staticmethod
    def l2_normalize(x, dim=-1, eps=1e-6):
        return x / (x.norm(dim=dim, keepdim=True) + eps)

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        # gt gaussian map
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # compute location loss
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)
        contrastive_loss = None
        if self.contrastive_enabled and pred_dict.get('memory_embedding', None) is not None:
            feat_all = pred_dict['backbone_feat']
            feat_all = feat_all[-1] if isinstance(feat_all, list) else feat_all
            backbone = self.net.backbone
            num_template = backbone.pos_embed_z.shape[1]
            num_search = backbone.pos_embed_x.shape[1]
            search_tokens = feat_all[:, num_template:, :]
            feat_sz = int(num_search ** 0.5)
            search_tokens_2d = search_tokens.view(search_tokens.shape[0], feat_sz, feat_sz, search_tokens.shape[-1])

            pos_idx, neg_idx = self.bbox_to_grid_indices(gt_bbox.clone(), feat_sz, self.cfg.DATA.SEARCH.SIZE,
                                                         self.cfg.MODEL.BACKBONE.STRIDE)
            pos_embeds = []
            neg_embeds = []
            for b in range(search_tokens_2d.shape[0]):
                pos_coords = torch.tensor(pos_idx[b], device=search_tokens_2d.device)
                pos_feats = search_tokens_2d[b, pos_coords[:, 0], pos_coords[:, 1], :]
                pos_embeds.append(pos_feats.mean(dim=0))

                neg_coords = torch.tensor(neg_idx[b], device=search_tokens_2d.device)
                if neg_coords.shape[0] >= self.contrastive_num_negatives:
                    perm = torch.randperm(neg_coords.shape[0], device=search_tokens_2d.device)[:self.contrastive_num_negatives]
                    neg_coords = neg_coords[perm]
                else:
                    repeat_idx = torch.randint(0, neg_coords.shape[0], (self.contrastive_num_negatives,),
                                               device=search_tokens_2d.device)
                    neg_coords = neg_coords[repeat_idx]
                neg_feats = search_tokens_2d[b, neg_coords[:, 0], neg_coords[:, 1], :]
                neg_embeds.append(neg_feats)

            pos_embed = torch.stack(pos_embeds, dim=0)
            neg_embed = torch.stack(neg_embeds, dim=0)
            mem_embed = pred_dict['memory_embedding']

            anchor = self.l2_normalize(mem_embed, dim=-1)
            pos_embed = self.l2_normalize(pos_embed, dim=-1)
            neg_embed = self.l2_normalize(neg_embed, dim=-1)

            sim_pos = (anchor * pos_embed).sum(dim=-1, keepdim=True)
            sim_neg = torch.matmul(anchor.unsqueeze(1), neg_embed.transpose(1, 2)).squeeze(1)
            logits = torch.cat([sim_pos, sim_neg], dim=1) / self.contrastive_temperature
            labels = torch.zeros(anchor.shape[0], dtype=torch.long, device=logits.device)
            contrastive_loss = F.cross_entropy(logits, labels)

        # weighted sum
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss
        if contrastive_loss is not None:
            loss = loss + self.contrastive_weight * contrastive_loss
        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      "IoU": mean_iou.item()}
            if contrastive_loss is not None:
                status["Loss/contrastive"] = contrastive_loss.item()
            return loss, status
        else:
            return loss
