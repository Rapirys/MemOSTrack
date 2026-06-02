from typing import Tuple

import torch
import torch.nn.functional as F

from lib.train.data.processing_utils import transform_image_to_crop


def compute_teacher_forcing_probability(epoch: int, start_prob: float, end_prob: float, anneal_epochs: int) -> float:
    clamped_epoch = max(1, int(epoch))
    clamped_anneal = max(1, int(anneal_epochs))
    progress = min(1.0, float(clamped_epoch - 1) / float(clamped_anneal))
    return float(start_prob + (end_prob - start_prob) * progress)


def clip_xywh_boxes_to_image_bounds(boxes_xywh: torch.Tensor, image_height: int, image_width: int,
                                    margin: float = 0.0) -> torch.Tensor:
    """Tensor equivalent of test-time clip_box(..., margin=margin)."""
    x1 = boxes_xywh[:, 0]
    y1 = boxes_xywh[:, 1]
    bw = boxes_xywh[:, 2]
    bh = boxes_xywh[:, 3]
    x2 = x1 + bw
    y2 = y1 + bh

    x1 = x1.clamp(min=0.0, max=max(0.0, image_width - margin))
    x2 = x2.clamp(min=margin, max=float(image_width))
    y1 = y1.clamp(min=0.0, max=max(0.0, image_height - margin))
    y2 = y2.clamp(min=margin, max=float(image_height))

    bw = (x2 - x1).clamp(min=margin)
    bh = (y2 - y1).clamp(min=margin)
    return torch.stack([x1, y1, bw, bh], dim=-1)


def map_search_boxes_back_to_image(pred_box_cxcywh: torch.Tensor, prev_state_xywh: torch.Tensor,
                                   search_size: int, resize_factor: torch.Tensor) -> torch.Tensor:
    cx_prev = prev_state_xywh[:, 0] + 0.5 * prev_state_xywh[:, 2]
    cy_prev = prev_state_xywh[:, 1] + 0.5 * prev_state_xywh[:, 3]
    cx = pred_box_cxcywh[:, 0]
    cy = pred_box_cxcywh[:, 1]
    bw = pred_box_cxcywh[:, 2]
    bh = pred_box_cxcywh[:, 3]
    half_side = 0.5 * float(search_size) / resize_factor
    cx_real = cx + (cx_prev - half_side)
    cy_real = cy + (cy_prev - half_side)
    return torch.stack([cx_real - 0.5 * bw, cy_real - 0.5 * bh, bw, bh], dim=-1)


def _validate_batched_crop_inputs(images_bchw: torch.Tensor, target_boxes_xywh: torch.Tensor):
    if images_bchw.dim() != 4:
        raise ValueError("images_bchw must have shape (B,C,H,W), got {}".format(tuple(images_bchw.shape)))
    if target_boxes_xywh.dim() != 2 or target_boxes_xywh.shape[-1] != 4:
        raise ValueError(
            "target_boxes_xywh must have shape (B,4), got {}".format(tuple(target_boxes_xywh.shape))
        )
    if images_bchw.shape[0] != target_boxes_xywh.shape[0]:
        raise ValueError(
            "batch size mismatch: images has {}, boxes has {}".format(
                images_bchw.shape[0], target_boxes_xywh.shape[0]
            )
        )


def _validate_finite_boxes(target_boxes_xywh: torch.Tensor):
    if not torch.isfinite(target_boxes_xywh).all():
        raise ValueError("target_boxes_xywh must contain only finite values.")


def _compute_sample_target_geometry(target_boxes_xywh: torch.Tensor,
                                    search_area_factor: float,
                                    output_sz: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    boxes = target_boxes_xywh.detach()
    _validate_finite_boxes(boxes)
    box_w = boxes[:, 2].clamp(min=1.0)
    box_h = boxes[:, 3].clamp(min=1.0)

    crop_sz = torch.ceil(torch.sqrt(box_w * box_h) * float(search_area_factor)).clamp(min=1.0)
    resize_factors = float(output_sz) / crop_sz

    crop_x1 = torch.round(boxes[:, 0] + 0.5 * box_w - 0.5 * crop_sz)
    crop_y1 = torch.round(boxes[:, 1] + 0.5 * box_h - 0.5 * crop_sz)
    return crop_x1, crop_y1, resize_factors


def _build_sample_target_grid(crop_x1: torch.Tensor,
                              crop_y1: torch.Tensor,
                              resize_factors: torch.Tensor,
                              image_height: int,
                              image_width: int,
                              output_sz: int) -> torch.Tensor:
    batch_size = crop_x1.shape[0]
    coords = torch.arange(int(output_sz), device=crop_x1.device, dtype=crop_x1.dtype)
    src_x = crop_x1[:, None] + (coords[None, :] + 0.5) / resize_factors[:, None] - 0.5
    src_y = crop_y1[:, None] + (coords[None, :] + 0.5) / resize_factors[:, None] - 0.5

    grid_x = src_x[:, None, :].expand(batch_size, int(output_sz), int(output_sz))
    grid_y = src_y[:, :, None].expand(batch_size, int(output_sz), int(output_sz))
    grid_x = 2.0 * (grid_x + 0.5) / float(image_width) - 1.0
    grid_y = 2.0 * (grid_y + 0.5) / float(image_height) - 1.0
    return torch.stack((grid_x, grid_y), dim=-1)


def sample_target_crops_from_bchw_tensor(images_bchw: torch.Tensor,
                                         target_boxes_xywh: torch.Tensor,
                                         search_area_factor: float,
                                         output_sz: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batched tensor crop using sample_target-compatible crop geometry."""
    _validate_batched_crop_inputs(images_bchw, target_boxes_xywh)

    _, _, image_height, image_width = images_bchw.shape
    boxes = target_boxes_xywh.to(device=images_bchw.device, dtype=images_bchw.dtype)
    crop_x1, crop_y1, resize_factors = _compute_sample_target_geometry(
        boxes,
        search_area_factor=search_area_factor,
        output_sz=output_sz,
    )
    grid = _build_sample_target_grid(
        crop_x1,
        crop_y1,
        resize_factors,
        image_height=image_height,
        image_width=image_width,
        output_sz=output_sz,
    )
    crops = F.grid_sample(
        images_bchw,
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False,
    )
    return crops, resize_factors


def project_box_to_crop_normalized(gt_box_xywh: torch.Tensor, crop_box_xywh: torch.Tensor,
                                   resize_factor: float, crop_size: int) -> torch.Tensor:
    crop_size_tensor = gt_box_xywh.new_tensor([crop_size, crop_size])
    return transform_image_to_crop(
        box_in=gt_box_xywh,
        box_extract=crop_box_xywh,
        resize_factor=resize_factor,
        crop_sz=crop_size_tensor,
        normalize=True,
    )


def sample_shared_crop_jitter_params(batch_size: int,
                                     device: torch.device,
                                     dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    scale_noise = torch.randn(batch_size, 2, device=device, dtype=dtype)
    center_noise = torch.rand(batch_size, 2, device=device, dtype=dtype) - 0.5
    return scale_noise, center_noise


def apply_crop_jitter_to_boxes(boxes_xywh: torch.Tensor,
                               center_jitter_factor: float,
                               scale_jitter_factor: float,
                               jitter_params: Tuple[torch.Tensor, torch.Tensor] = None) -> torch.Tensor:
    if boxes_xywh.numel() == 0:
        return boxes_xywh

    center_factor = float(center_jitter_factor)
    scale_factor = float(scale_jitter_factor)
    if center_factor == 0.0 and scale_factor == 0.0:
        return boxes_xywh

    batch = boxes_xywh.shape[0]
    device = boxes_xywh.device
    dtype = boxes_xywh.dtype

    if jitter_params is None:
        scale_noise = torch.randn(batch, 2, device=device, dtype=dtype)
        center_noise = torch.rand(batch, 2, device=device, dtype=dtype) - 0.5
    else:
        scale_noise, center_noise = jitter_params
        scale_noise = scale_noise.to(device=device, dtype=dtype)
        center_noise = center_noise.to(device=device, dtype=dtype)

        if scale_noise.dim() == 1:
            scale_noise = scale_noise.view(1, 2).expand(batch, 2)
        elif scale_noise.shape != (batch, 2):
            raise ValueError("scale_noise must have shape (2,) or (B,2), got {}".format(tuple(scale_noise.shape)))

        if center_noise.dim() == 1:
            center_noise = center_noise.view(1, 2).expand(batch, 2)
        elif center_noise.shape != (batch, 2):
            raise ValueError("center_noise must have shape (2,) or (B,2), got {}".format(tuple(center_noise.shape)))

    jittered_size = boxes_xywh[:, 2:4] * torch.exp(scale_noise * scale_factor)
    max_offset = jittered_size.prod(dim=1, keepdim=True).sqrt() * center_factor
    jittered_center = boxes_xywh[:, 0:2] + 0.5 * boxes_xywh[:, 2:4] + max_offset * center_noise

    return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=1)
