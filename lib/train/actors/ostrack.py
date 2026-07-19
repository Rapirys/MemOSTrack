from . import BaseActor
from .memory_filter_loss import (
    compute_memory_filter_loss,
    compute_memory_filter_target_match_loss,
)
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, box_xyxy_to_xywh
from lib.utils.template_corruption import MemoryTemplateCorruption
from lib.utils.inference_rollout_utils import (
    apply_crop_jitter_to_boxes,
    clip_xywh_boxes_to_image_bounds,
    compute_teacher_forcing_probability,
    map_search_boxes_back_to_image,
    project_box_to_crop_normalized,
    sample_shared_crop_jitter_params,
    sample_target_crops_from_bchw_tensor,
)
import torch
import os
import cv2
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate
from ...test.utils.hann import hann2d


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
        self.rollout_teacher_prob_start = float(getattr(self.cfg.TRAIN, "ROLLOUT_TEACHER_PROB_START", 1.0))
        self.rollout_teacher_prob_end = float(getattr(self.cfg.TRAIN, "ROLLOUT_TEACHER_PROB_END", 0.0))
        self.rollout_teacher_anneal_epochs = max(1, int(getattr(self.cfg.TRAIN, "ROLLOUT_TEACHER_ANNEAL_EPOCHS", 20)))
        self.rollout_crop_jitter = bool(getattr(self.cfg.TRAIN, "ROLLOUT_CROP_JITTER", False))
        self.search_per_frame_jitter = bool(getattr(self.cfg.TRAIN, "SEARCH_PER_FRAME_JITTER", False))
        self.template_center_jitter = float(getattr(self.cfg.DATA.TEMPLATE, "CENTER_JITTER", 0.0))
        self.template_scale_jitter = float(getattr(self.cfg.DATA.TEMPLATE, "SCALE_JITTER", 0.0))
        self.search_center_jitter = float(getattr(self.cfg.DATA.SEARCH, "CENTER_JITTER", 0.0))
        self.search_scale_jitter = float(getattr(self.cfg.DATA.SEARCH, "SCALE_JITTER", 0.0))
        self.pixel_mean = [float(x) for x in self.cfg.DATA.MEAN]
        self.pixel_std = [float(x) for x in self.cfg.DATA.STD]
        self.rollout_window = None
        memory_loss_cfg = str(getattr(self.cfg.TRAIN, "MEMORY_LOSS_TYPE", "none")).strip().lower()
        self.memory_loss_fn, self.memory_loss_name = self._build_memory_loss(memory_loss_cfg)
        train_cfg = getattr(self.cfg, "TRAIN", None)
        self.memory_template_corruption = MemoryTemplateCorruption.from_train_cfg(train_cfg)
        cfg_num_search = int(getattr(self.cfg.DATA.SEARCH, "NUMBER", 0))
        settings_num_search = int(getattr(self.settings, "num_search", cfg_num_search))
        effective_num_search = settings_num_search if settings_num_search > 0 else cfg_num_search
        self.memory_template_corruption.validate_for_num_search(effective_num_search)
        self.memory_blur_enabled = self.memory_template_corruption.enabled
        if self.memory_blur_enabled and self.memory_loss_fn is not None:
            raise ValueError(
                "TRAIN.MEMORY_BLUR_ENABLED=True is not supported together with "
                "TRAIN.MEMORY_LOSS_TYPE='{}'. Disable one of them.".format(self.memory_loss_name)
            )

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

    def _normalize_image_batch(self, images: torch.Tensor) -> torch.Tensor:
        mean = images.new_tensor(self.pixel_mean).view(1, -1, 1, 1)
        std = images.new_tensor(self.pixel_std).view(1, -1, 1, 1)
        return (images - mean) / std

    def _get_rollout_window(self, score_map: torch.Tensor) -> torch.Tensor:
        feat_sz = torch.tensor(score_map.shape[-2:], dtype=torch.long)
        if (self.rollout_window is None
                or self.rollout_window.shape[-2:] != score_map.shape[-2:]
                or self.rollout_window.device != score_map.device):
            self.rollout_window = hann2d(feat_sz, centered=True).to(device=score_map.device, dtype=score_map.dtype)
        return self.rollout_window

    def _select_rollout_boxes(self, out_i, batch_size: int) -> torch.Tensor:
        if all(k in out_i for k in ('score_map', 'size_map', 'offset_map')):
            response = self._get_rollout_window(out_i['score_map']) * out_i['score_map']
            pred_boxes = self.net.box_head.cal_bbox(response, out_i['size_map'], out_i['offset_map'])
            return pred_boxes.view(batch_size, -1, 4)
        return out_i['pred_boxes'].view(batch_size, -1, 4)

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
                                                             *data['template_images'].shape[2:])  # (batch, 3, Hf, Wf) i think Hf, Wf is (batch, 3, 128, 128)
            # template_att_i = data['template_att'][i].view(-1, *data['template_att'].shape[2:])  # (batch, Hf, Wf)
            template_list.append(template_img_i)

        search_images = data['search_images']
        search_gt_boxes = data['search_anno'].float()  # (Ns, B, 4), absolute xywh on frame canvas
        num_search = search_images.shape[0]
        batch_size = search_images.shape[1]
        search_img_shape = search_images.shape[2:]
        frame_h = int(search_images.shape[-2])
        frame_w = int(search_images.shape[-1])
        # search_att = data['search_att'][0].view(-1, *data['search_att'].shape[2:])  # (batch, Hf, Wf)

        if self.debug_save_seq:
            seq_tag = f"seq_{self.debug_seq_group_idx:06d}"
            self.debug_seq_group_idx += 1
            for t_idx in range(data['template_images'].shape[0]):
                tmpl_img = data['template_images'][t_idx][0].detach().cpu().permute(1, 2, 0).clamp(0, 1)
                tmpl_img = (tmpl_img * 255).byte().numpy()
                tmpl_path = os.path.join(self.debug_seq_dir, f"{seq_tag}_template_{t_idx:02d}.jpg")
                cv2.imwrite(tmpl_path, cv2.cvtColor(tmpl_img, cv2.COLOR_RGB2BGR))

        template_boxes = data['template_anno'][0].view(-1, 4).float()  # absolute xywh on frame canvas
        template_factor = float(self.cfg.DATA.TEMPLATE.FACTOR)
        template_size = int(self.cfg.DATA.TEMPLATE.SIZE)
        search_factor = float(self.cfg.DATA.SEARCH.FACTOR)
        search_size = int(self.cfg.DATA.SEARCH.SIZE)
        is_train = self.net.training
        use_crop_jitter = self.rollout_crop_jitter and is_train

        template_crop_boxes = template_boxes
        if use_crop_jitter:
            template_crop_boxes = apply_crop_jitter_to_boxes(
                template_boxes,
                center_jitter_factor=self.template_center_jitter,
                scale_jitter_factor=self.template_scale_jitter,
            )

        template_frames = template_list[0]
        template_list, template_resize_factors = sample_target_crops_from_bchw_tensor(
            template_frames, template_crop_boxes, template_factor, template_size
        )
        template_list = self._normalize_image_batch(template_list)

        box_mask_z = None
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_anno_crop = []
            for b in range(batch_size):
                gt_crop_b = project_box_to_crop_normalized(
                    gt_box_xywh=template_boxes[b],
                    crop_box_xywh=template_crop_boxes[b],
                    resize_factor=template_resize_factors[b],
                    crop_size=template_size,
                )
                template_anno_crop.append(gt_crop_b)
            template_anno_crop = torch.stack(template_anno_crop, dim=0)
            box_mask_z = generate_mask_cond(
                self.cfg,
                template_list.shape[0],
                template_list.device,
                template_anno_crop
            )

            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            ce_keep_rate = adjust_keep_rate(
                data['epoch'],
                warmup_epochs=ce_start_epoch,
                total_epochs=ce_start_epoch + ce_warm_epoch,
                ITERS_PER_EPOCH=1,
                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0]
            )

        mem_tokens = None
        is_first_frame = True
        mem_cfg = getattr(self.cfg.MODEL, "MEMORY", None)
        max_bptt_steps = int(getattr(mem_cfg, "BPTT_STEPS", -1)) if mem_cfg is not None else -1

        state_boxes = template_boxes.clone()
        if is_train:
            teacher_prob = compute_teacher_forcing_probability(
                epoch=data['epoch'],
                start_prob=self.rollout_teacher_prob_start,
                end_prob=self.rollout_teacher_prob_end,
                anneal_epochs=self.rollout_teacher_anneal_epochs,
            )
            template_corrupt_mask, full_corrupt_mask = self.memory_template_corruption.build_masks(
                num_search, batch_size, search_images.device)
        else:
            teacher_prob = 0.0
            template_corrupt_mask = torch.zeros((num_search, batch_size), dtype=torch.bool, device=search_images.device)
            full_corrupt_mask = torch.zeros((num_search, batch_size), dtype=torch.bool, device=search_images.device)
        out_dict = []
        shared_search_jitter = None
        if use_crop_jitter and (not self.search_per_frame_jitter):
            shared_search_jitter = sample_shared_crop_jitter_params(
                batch_size=batch_size,
                device=state_boxes.device,
                dtype=state_boxes.dtype,
            )

        for i in range(num_search):
            if max_bptt_steps > 0 and i > 0 and mem_tokens is not None:
                if i % (max_bptt_steps + 1) == 0:
                    mem_tokens = mem_tokens.detach()

            search_img = search_images[i].view(-1, *search_img_shape)  # (batch, 3, Hf, Wf)
            search_img_full = search_img
            frame_template_corrupt_mask = template_corrupt_mask[i]
            frame_full_corrupt_mask = full_corrupt_mask[i]
            template_in = template_list
            if frame_template_corrupt_mask.any():
                template_corrupt_selector = frame_template_corrupt_mask.view(-1, 1, 1, 1)
                corrupted_template = self.memory_template_corruption.apply(template_list)
                template_in = torch.where(template_corrupt_selector, corrupted_template, template_list)
            if frame_full_corrupt_mask.any():
                template_full_corrupt_selector = frame_full_corrupt_mask.view(-1, 1, 1, 1)
                template_in = torch.where(template_full_corrupt_selector, torch.zeros_like(template_in), template_in)

            if self.debug_save_seq:
                dbg_img = search_img_full[0].detach().cpu().permute(1, 2, 0).clamp(0, 1)
                dbg_img = (dbg_img * 255).byte().numpy()
                dbg_path = os.path.join(self.debug_seq_dir, f"{seq_tag}_search_{i:02d}_{self.debug_seq_idx:06d}.jpg")
                cv2.imwrite(dbg_path, cv2.cvtColor(dbg_img, cv2.COLOR_RGB2BGR))
                self.debug_seq_idx += 1

            state_before_i = state_boxes.detach().clone()
            crop_boxes_i = state_before_i
            if use_crop_jitter:
                crop_boxes_i = apply_crop_jitter_to_boxes(
                    state_before_i,
                    center_jitter_factor=self.search_center_jitter,
                    scale_jitter_factor=self.search_scale_jitter,
                    jitter_params=shared_search_jitter,
                )
            search_img, resize_factors_i_t = sample_target_crops_from_bchw_tensor(
                search_img_full, crop_boxes_i, search_factor, search_size
            )
            search_img = self._normalize_image_batch(search_img)
            if frame_full_corrupt_mask.any():
                search_corrupt_selector = frame_full_corrupt_mask.view(-1, 1, 1, 1)
                search_img = torch.where(search_corrupt_selector, torch.zeros_like(search_img), search_img)

            ce_keep_rate_i = ce_keep_rate
            if frame_template_corrupt_mask.any() and self.cfg.MODEL.BACKBONE.CE_LOC:
                ce_keep_rate_i = 1.0

            out_i = self.net(
                template=template_in,
                search=search_img,
                ce_template_mask=box_mask_z,
                ce_keep_rate=ce_keep_rate_i,
                return_last_attn=False,
                mem_tokens=mem_tokens,
                is_first_frame=is_first_frame,
            )
            is_first_frame = False

            mem_tokens_layers = out_i.get('memory_tokens_layers')
            if mem_tokens_layers is not None:
                mem_tokens = mem_tokens_layers

            pred_boxes = self._select_rollout_boxes(out_i, batch_size)
            pred_box_mean = pred_boxes.mean(dim=1) * (float(search_size) / resize_factors_i_t.view(-1, 1))
            pred_state = map_search_boxes_back_to_image(pred_box_mean, crop_boxes_i, search_size, resize_factors_i_t)
            pred_state = clip_xywh_boxes_to_image_bounds(
                pred_state,
                image_height=frame_h,
                image_width=frame_w,
                margin=10.0,
            )

            if i < (num_search - 1):
                teacher_mask = (torch.rand(batch_size, device=pred_state.device) < teacher_prob)
                next_teacher = search_gt_boxes[i].to(pred_state.device)
                state_boxes = torch.where(teacher_mask.view(-1, 1), next_teacher, pred_state.detach())
            else:
                state_boxes = pred_state.detach()

            gt_crop_i = []
            gt_valid_i = []
            for b in range(batch_size):
                gt_crop_b = project_box_to_crop_normalized(
                    gt_box_xywh=search_gt_boxes[i, b].to(search_img.device),
                    crop_box_xywh=crop_boxes_i[b].to(search_img.device),
                    resize_factor=resize_factors_i_t[b],
                    crop_size=search_size,
                )
                gt_crop_xyxy_b = box_xywh_to_xyxy(gt_crop_b.unsqueeze(0)).squeeze(0)
                gt_tl = gt_crop_xyxy_b[:2].clamp(min=0.0, max=1.0)
                gt_br = gt_crop_xyxy_b[2:].clamp(min=0.0, max=1.0)
                gt_br = torch.maximum(gt_br, gt_tl)
                gt_valid_i.append(((gt_br - gt_tl) > 1e-4).all())
                gt_crop_b = box_xyxy_to_xywh(torch.cat((gt_tl, gt_br), dim=0).unsqueeze(0)).squeeze(0)
                gt_crop_i.append(gt_crop_b)
            out_i['gt_box_in_crop'] = torch.stack(gt_crop_i, dim=0)
            out_i['gt_box_valid_in_crop'] = torch.stack(gt_valid_i, dim=0)
            out_i['template_corrupt_mask'] = frame_template_corrupt_mask.detach()
            out_i['full_corrupt_mask'] = frame_full_corrupt_mask.detach()
            out_i['is_blurred_frame'] = bool(frame_template_corrupt_mask.any().item())
            out_i['is_full_corrupt_frame'] = bool(frame_full_corrupt_mask.any().item())
            out_dict.append(out_i)

        #   print ("Debug: out_dict =", len(out_dict))
        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        if len(pred_dict) == 0 or ('gt_box_in_crop' not in pred_dict[0]):
            raise ValueError("Missing gt_box_in_crop for inference-like rollout loss.")
        search_anno = torch.stack([pred_t['gt_box_in_crop'] for pred_t in pred_dict], dim=0)  # (Ns, B, 4)
        if 'gt_box_valid_in_crop' in pred_dict[0]:
            valid_anno = torch.stack([pred_t['gt_box_valid_in_crop'] for pred_t in pred_dict], dim=0).to(torch.bool)
        else:
            valid_anno = torch.ones(search_anno.shape[:2], dtype=torch.bool, device=search_anno.device)
        gt_gaussian_maps_all = generate_heatmap(search_anno, self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        for t, gt_map in enumerate(gt_gaussian_maps_all):
            gt_map[~valid_anno[t].to(gt_map.device)] = 0.0

        num_frames = len(pred_dict)
        if num_frames == 0:
            raise ValueError("Empty prediction sequence in compute_losses.")

        device = pred_dict[0]['pred_boxes'].device
        frame_metrics = []
        full_corrupt_frame_metrics = []

        def _compute_frame_metrics_for_samples(pred_t, gt_bbox, gt_gaussian_maps, sample_mask):
            sample_mask = sample_mask.to(device=device, dtype=torch.bool)
            has_valid = bool(sample_mask.any().item())
            if not has_valid:
                return {
                    "giou": torch.tensor(0.0, device=device),
                    "l1": torch.tensor(0.0, device=device),
                    "location": torch.tensor(0.0, device=device),
                    "iou": 0.0,
                    "has_valid": False,
                    "valid_count": 0,
                }

            pred_boxes = pred_t['pred_boxes']
            pred_boxes_valid = pred_boxes[sample_mask]
            gt_bbox_valid = gt_bbox[sample_mask]
            num_queries = pred_boxes_valid.size(1)
            pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes_valid).view(-1, 4)
            gt_boxes_vec = box_xywh_to_xyxy(gt_bbox_valid)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4)
            gt_boxes_vec = gt_boxes_vec.clamp(min=0.0, max=1.0)

            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)
            if (not torch.isfinite(giou_loss)) or (not torch.isfinite(iou).all()):
                raise ValueError("Non-finite GIoU/IoU encountered during loss computation.")
            l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)
            if 'score_map' in pred_t:
                location_loss = self.objective['focal'](pred_t['score_map'][sample_mask], gt_gaussian_maps[sample_mask])
            else:
                location_loss = torch.tensor(0.0, device=device)

            return {
                "giou": giou_loss,
                "l1": l1_loss,
                "location": location_loss,
                "iou": float(iou.detach().mean().item()),
                "has_valid": True,
                "valid_count": int(sample_mask.sum().item()),
            }

        # TODO: maybe add check that num_frames matches GT length
        for t in range(num_frames):
            # gt for frame t
            gt_bbox = search_anno[t]  # (batch, 4)
            valid_t = valid_anno[t].to(device)
            gt_gaussian_maps = gt_gaussian_maps_all[t].unsqueeze(1)  # (B, 1, Hf, Wf)

            # predictions for frame t
            pred_t = pred_dict[t]
            pred_boxes = pred_t['pred_boxes']
            if torch.isnan(pred_boxes).any():
                raise ValueError("Network outputs is NAN! Stop Training")

            frame_metrics.append(_compute_frame_metrics_for_samples(pred_t, gt_bbox, gt_gaussian_maps, valid_t))

            full_corrupt_mask = pred_t.get("full_corrupt_mask", None)
            if full_corrupt_mask is None:
                full_corrupt_mask = torch.zeros_like(valid_t, dtype=torch.bool, device=device)
            else:
                full_corrupt_mask = full_corrupt_mask.to(device=device, dtype=torch.bool) & valid_t
            full_corrupt_frame_metrics.append(
                _compute_frame_metrics_for_samples(pred_t, gt_bbox, gt_gaussian_maps, full_corrupt_mask)
            )

        def _aggregate(indices, metrics_source=frame_metrics):
            indices = [idx for idx in indices if metrics_source[idx]["has_valid"]]
            if not indices:
                return {
                    "giou": torch.tensor(0.0, device=device),
                    "l1": torch.tensor(0.0, device=device),
                    "location": torch.tensor(0.0, device=device),
                    "iou": torch.tensor(0.0, device=device),
                }

            sum_giou = torch.tensor(0.0, device=device)
            sum_l1 = torch.tensor(0.0, device=device)
            sum_location = torch.tensor(0.0, device=device)
            sum_iou = 0.0
            for idx in indices:
                m = metrics_source[idx]
                sum_giou = sum_giou + m["giou"]
                sum_l1 = sum_l1 + m["l1"]
                sum_location = sum_location + m["location"]
                sum_iou += m["iou"]

            denom = float(len(indices))
            return {
                "giou": sum_giou / denom,
                "l1": sum_l1 / denom,
                "location": sum_location / denom,
                "iou": torch.tensor(sum_iou / denom, device=device),
            }

        all_indices = list(range(num_frames))
        blurred_indices = [i for i, pred_t in enumerate(pred_dict) if bool(pred_t.get("is_blurred_frame", False))]
        full_corrupt_indices = [
            i for i, pred_t in enumerate(pred_dict) if bool(pred_t.get("is_full_corrupt_frame", False))
        ]
        clean_indices = [i for i in all_indices if i not in blurred_indices]
        if not clean_indices:
            clean_indices = all_indices

        all_stats = _aggregate(all_indices)
        clean_stats = _aggregate(clean_indices)
        blur_stats = _aggregate(blurred_indices)
        full_corrupt_stats = _aggregate(full_corrupt_indices, full_corrupt_frame_metrics)

        memory_weight = self.memory_loss_weight
        if self.memory_loss_fn is not None and memory_weight > 0.0:
            mem_loss = self.memory_loss_fn(pred_dict, gt_gaussian_maps_all, self.cfg, device)
        else:
            mem_loss = torch.tensor(0.0, device=device)

        # Optimization loss uses all frames.
        loss = (self.loss_weight['giou'] * all_stats["giou"]
                + self.loss_weight['l1'] * all_stats["l1"]
                + self.loss_weight['focal'] * all_stats["location"]
                + memory_weight * mem_loss)

        if return_status:
            valid_samples = float(valid_anno.sum().item())
            total_samples = float(valid_anno.numel())
            clean_total = (self.loss_weight['giou'] * clean_stats["giou"]
                           + self.loss_weight['l1'] * clean_stats["l1"]
                           + self.loss_weight['focal'] * clean_stats["location"]
                           + memory_weight * mem_loss)
            blur_total = (self.loss_weight['giou'] * blur_stats["giou"]
                          + self.loss_weight['l1'] * blur_stats["l1"]
                          + self.loss_weight['focal'] * blur_stats["location"])
            full_corrupt_total = (self.loss_weight['giou'] * full_corrupt_stats["giou"]
                                  + self.loss_weight['l1'] * full_corrupt_stats["l1"]
                                  + self.loss_weight['focal'] * full_corrupt_stats["location"])

            # Status for logs. Blurred-frame metrics are reported separately.
            status = {
                "Loss/total": clean_total.item(),
                "Loss/total optim": loss.item(),
                "Loss/giou": clean_stats["giou"].item(),
                "Loss/l1": clean_stats["l1"].item(),
                "Loss/location": clean_stats["location"].item(),
                "Loss/memory": mem_loss.item(),
                "Loss/memory weighted": (memory_weight * mem_loss).item(),
                "IoU": clean_stats["iou"].item(),
                "Rollout/crop jitter": float(self.rollout_crop_jitter and self.net.training),
                "Rollout/valid gt samples": valid_samples,
                "Rollout/valid gt ratio": valid_samples / max(1.0, total_samples),
                "Blur/frames": float(len(blurred_indices)),
                "Blur/Loss/total": blur_total.item(),
                "Blur/Loss/giou": blur_stats["giou"].item(),
                "Blur/Loss/l1": blur_stats["l1"].item(),
                "Blur/Loss/location": blur_stats["location"].item(),
                "Blur/IoU": blur_stats["iou"].item(),
                "FullCorrupt/frames": float(len(full_corrupt_indices)),
                "FullCorrupt/Loss/total": full_corrupt_total.item(),
                "FullCorrupt/Loss/giou": full_corrupt_stats["giou"].item(),
                "FullCorrupt/Loss/l1": full_corrupt_stats["l1"].item(),
                "FullCorrupt/Loss/location": full_corrupt_stats["location"].item(),
                "FullCorrupt/IoU": full_corrupt_stats["iou"].item(),
            }
            return loss, status
        else:
            return loss
