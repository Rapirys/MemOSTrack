from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
import torch.nn.functional as F
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate


class OSTrackActor(BaseActor):
    """ Actor for training OSTrack models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg

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

        template_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1,
                                                             *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
            # template_att_i = data['template_att'][i].view(-1, *data['template_att'].shape[2:])  # (batch, 128, 128)
            template_list.append(template_img_i)

        search_images = data['search_images']
        num_search = search_images.shape[0]
        batch_size = search_images.shape[1]
        search_img_shape = search_images.shape[2:]
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

        mem_tokens = None
        mem_cfg = getattr(self.cfg.MODEL, "MEMORY", None)
        max_bptt_steps = int(getattr(mem_cfg, "BPTT_STEPS", -1)) if mem_cfg is not None else -1
        if hasattr(self.net, 'backbone') and hasattr(self.net.backbone, 'init_memory') and getattr(self.net.backbone, 'memory_tokens', 0) > 0:
            mem_tokens = self.net.backbone.init_memory(batch_size, device=search_images.device, dtype=search_images.dtype)

        out_dict = []
        for i in range(num_search):
            if max_bptt_steps > 0 and i > 0 and mem_tokens is not None:
                if i % (max_bptt_steps + 1) == 0:
                    mem_tokens = mem_tokens.detach()
            search_img = search_images[i].view(-1, *search_img_shape)  # (batch, 3, 320, 320)
            out_i = self.net(template=template_list,
                             search=search_img,
                             ce_template_mask=box_mask_z,
                             ce_keep_rate=ce_keep_rate,
                             return_last_attn=False,
                             mem_tokens=mem_tokens)
            if 'memory_tokens' in out_i:
                mem_tokens = out_i['memory_tokens']
            out_dict.append(out_i)

     #   print ("Debug: out_dict =", len(out_dict))

        return out_dict

    def compute_memory_filter_loss(self, pred_t, gt_gaussian_maps, device):
        # Memory filter loss (DiMP-style) for this frame
        if 'mem_filter_kernel' in pred_t and 'search_feat' in pred_t:
            mem_filter = pred_t['mem_filter_kernel']  # (B, C, k, k)
            search_feat = pred_t['search_feat']  # (B, C, H, W)

            Bf, Cf, Hf, Wf = search_feat.shape
            k = mem_filter.shape[-1]

            # Grouped conv: x * f for each sample in the batch
            x_merged = search_feat.view(1, Bf * Cf, Hf, Wf)  # (1, B*C, H, W)
            w_merged = mem_filter  # (B, C, k, k)
            mem_response = F.conv2d(x_merged, w_merged, groups=Bf, padding=k // 2)  # (1, B, H, W)
            mem_response = mem_response.view(Bf, 1, Hf, Wf)  # (B, 1, H, W)

            # ensure spatial sizes match (in case of minor mismatch)
            if mem_response.shape[-2:] != gt_gaussian_maps.shape[-2:]:
                gt_mem = F.interpolate(
                    gt_gaussian_maps,
                    size=mem_response.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
                print("warning, innvestigate gt_mem spatial size")
            else:
                gt_mem = gt_gaussian_maps

            # data term: mean squared residual over batch and spatial dims
            data_mem_loss = torch.mean((mem_response - gt_mem) ** 2)

            # regularization term on filter kernel
            memory_cfg = getattr(self.cfg.MODEL, "MEMORY", None)
            lambda_reg = getattr(memory_cfg, "FILTER_REG", 1e-4) if memory_cfg is not None else 1e-4
            reg_mem_loss = lambda_reg * torch.mean(mem_filter ** 2)

            mem_loss_t = data_mem_loss + reg_mem_loss #∥x∗f−c∥+λ∥f∥
        else:
            mem_loss_t = torch.tensor(0.0, device=device)

        return mem_loss_t

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        search_anno = gt_dict['search_anno']  # (Ns, B, 4)
        gt_gaussian_maps_all = generate_heatmap(search_anno, self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)

        num_frames = len(pred_dict)
        if num_frames == 0:
            raise ValueError("Empty prediction sequence in compute_losses.")

        device = pred_dict[0]['pred_boxes'].device
        sum_giou_loss = torch.tensor(0.0, device=device)
        sum_l1_loss = torch.tensor(0.0, device=device)
        sum_location_loss = torch.tensor(0.0, device=device)
        sum_mem_loss = torch.tensor(0.0, device=device)
        sum_iou = 0.0
        # TODO: maybe add check that num_frames matches GT length
        for t in range(num_frames):
            # gt for frame t
            gt_bbox = search_anno[t]  # (batch, 4)
            gt_gaussian_maps = gt_gaussian_maps_all[t].unsqueeze(1)  # (B, 1, Hf, Wf)

            # predictions for frame t
            pred_t = pred_dict[t]
            pred_boxes = pred_t['pred_boxes']
            if torch.isnan(pred_boxes).any():
                raise ValueError("Network outputs is NAN! Stop Training")
            num_queries = pred_boxes.size(1)
            pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
            gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                               max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)

            # compute giou and iou
            try:
                giou_loss_t, iou_t = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
            except:
                giou_loss_t, iou_t = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
            # compute l1 loss
            l1_loss_t = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
            # compute location loss for main head
            if 'score_map' in pred_t:
                location_loss_t = self.objective['focal'](pred_t['score_map'], gt_gaussian_maps)
            else:
                location_loss_t = torch.tensor(0.0, device=device)

            mem_loss_t = self.compute_memory_filter_loss(pred_t, gt_gaussian_maps, device)

            sum_giou_loss += giou_loss_t
            sum_l1_loss += l1_loss_t
            sum_location_loss += location_loss_t
            sum_mem_loss += mem_loss_t
            sum_iou += iou_t.detach().mean().item()

        giou_loss = sum_giou_loss / num_frames
        l1_loss = sum_l1_loss / num_frames
        location_loss = sum_location_loss / num_frames
        mem_loss = sum_mem_loss / num_frames
        mean_iou = torch.tensor(sum_iou / num_frames, device=device)

        # weighted sum
        memory_weight = self.loss_weight.get('memory', 0.001)
        loss = (self.loss_weight['giou'] * giou_loss
                + self.loss_weight['l1'] * l1_loss
                + self.loss_weight['focal'] * location_loss
                + memory_weight * mem_loss)

        if return_status:
            # status for log
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      "Loss/memory": mem_loss.item(),
                      "Loss/memory weighted": (memory_weight * mem_loss).item(),
                      "IoU": mean_iou.item()}
            return loss, status
        else:
            return loss
