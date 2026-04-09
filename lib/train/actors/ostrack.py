from . import BaseActor
from .memory_filter_loss import (
    compute_memory_filter_loss,
    compute_memory_filter_target_match_loss,
)
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
import os
import cv2
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
        # Temporary debug: save incoming search frames in strict sequence order.
        self.debug_save_seq = False
        self.debug_seq_idx = 0
        self.debug_seq_group_idx = 0
        self.debug_seq_dir = "/home/illia/PycharmProjects/MemOSTrack/output/videos"
        if self.debug_save_seq:
            os.makedirs(self.debug_seq_dir, exist_ok=True)

        self.memory_loss_weight = float(self.loss_weight.get('memory', 0.0))
        memory_loss_cfg = str(getattr(self.cfg.TRAIN, "MEMORY_LOSS_TYPE", "none")).strip().lower()
        self.memory_loss_fn, self.memory_loss_name = self._build_memory_loss(memory_loss_cfg)

        if self.memory_loss_fn is not None:
            memory_cfg = getattr(self.cfg.MODEL, "MEMORY", None)
            mem_enabled = bool(getattr(memory_cfg, "ENABLED", False)) if memory_cfg is not None else False
            mem_tokens = int(getattr(memory_cfg, "NUM_TOKENS", 0)) if memory_cfg is not None else 0
            if not (mem_enabled and mem_tokens > 0):
                raise ValueError(
                    "TRAIN.MEMORY_LOSS_TYPE='{}' requires MODEL.MEMORY.ENABLED=True "
                    "and MODEL.MEMORY.NUM_TOKENS>0.".format(self.memory_loss_name)
                )

    @staticmethod
    def _build_memory_loss(memory_loss_cfg):
        if memory_loss_cfg in {"", "none"}:
            return None, "none"

        if memory_loss_cfg in {"memoryfilterloss", "memory_filter_loss"}:
            return compute_memory_filter_loss, "memoryfilterloss"

        if memory_loss_cfg in {
            "dimpsteepestdescentsolver",
            "dimp_steepest_descent_solver"
        }:
            return compute_memory_filter_target_match_loss, "dimpsteepestdescentsolver"

        raise ValueError(
            "Unsupported TRAIN.MEMORY_LOSS_TYPE='{}'. Use one of: "
            "none, MemoryFilterLoss, DiMPSteepestDescentSolver.".format(memory_loss_cfg)
        )

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

        if self.debug_save_seq:
            seq_tag = f"seq_{self.debug_seq_group_idx:06d}"
            self.debug_seq_group_idx += 1
            for t_idx in range(data['template_images'].shape[0]):
                tmpl_img = data['template_images'][t_idx][0].detach().cpu().permute(1, 2, 0).clamp(0, 1)
                tmpl_img = (tmpl_img * 255).byte().numpy()
                tmpl_path = os.path.join(self.debug_seq_dir, f"{seq_tag}_template_{t_idx:02d}.jpg")
                cv2.imwrite(tmpl_path, cv2.cvtColor(tmpl_img, cv2.COLOR_RGB2BGR))

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
            if self.debug_save_seq:
                dbg_img = search_img[0].detach().cpu().permute(1, 2, 0).clamp(0, 1)
                dbg_img = (dbg_img * 255).byte().numpy()
                dbg_path = os.path.join(self.debug_seq_dir, f"{seq_tag}_search_{i:02d}_{self.debug_seq_idx:06d}.jpg")
                cv2.imwrite(dbg_path, cv2.cvtColor(dbg_img, cv2.COLOR_RGB2BGR))
                self.debug_seq_idx += 1
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

            sum_giou_loss += giou_loss_t
            sum_l1_loss += l1_loss_t
            sum_location_loss += location_loss_t
            sum_iou += iou_t.detach().mean().item()

        giou_loss = sum_giou_loss / num_frames
        l1_loss = sum_l1_loss / num_frames
        location_loss = sum_location_loss / num_frames
        memory_weight = self.memory_loss_weight
        if self.memory_loss_fn is not None and memory_weight > 0.0:
            mem_loss = self.memory_loss_fn(pred_dict, gt_gaussian_maps_all, self.cfg, device)
        else:
            mem_loss = torch.tensor(0.0, device=device)
        mean_iou = torch.tensor(sum_iou / num_frames, device=device)

        # weighted sum
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
