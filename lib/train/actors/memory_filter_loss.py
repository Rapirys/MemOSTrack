import torch
import torch.nn.functional as F


def compute_memory_filter_loss(pred_seq, gt_gaussian_maps_all, cfg, device):
    memory_cfg = getattr(cfg.MODEL, "MEMORY", None)
    num_frames = int(getattr(memory_cfg, "FILTER_LOSS_FRAMES", 1)) if memory_cfg is not None else 1
    if num_frames <= 0 or len(pred_seq) == 0:
        return torch.tensor(0.0, device=device)

    filter_frames = [
        i for i, pred_t in enumerate(pred_seq)
        if "mem_filter_kernel" in pred_t and "search_feat" in pred_t
    ]
    if not filter_frames:
        return torch.tensor(0.0, device=device)

    feature_frames = [
        i for i, pred_t in enumerate(pred_seq)
        if "search_feat" in pred_t
    ]
    if not feature_frames:
        return torch.tensor(0.0, device=device)

    data_mem_loss = torch.tensor(0.0, device=device)
    reg_mem_loss = torch.tensor(0.0, device=device)
    num_filters = 0

    for t in filter_frames:
        pred_t = pred_seq[t]
        mem_filter = pred_t["mem_filter_kernel"]  # (B, C, k, k)
        k = mem_filter.shape[-1]

        candidates = feature_frames
        if len(feature_frames) > 1 and t in feature_frames:
            candidates = [i for i in feature_frames if i != t]
        if not candidates:
            continue

        sample_count = min(num_frames, len(candidates))
        sample_idx = torch.randperm(len(candidates), device=device)[:sample_count].tolist()
        chosen = [candidates[i] for i in sample_idx]

        search_feats = []
        gt_maps = []
        for s in chosen:
            search_feat = pred_seq[s]["search_feat"].detach()
            search_feats.append(search_feat)
            gt_maps.append(gt_gaussian_maps_all[s].unsqueeze(1))

        # Stack features as batch for grouped conv; keep filter shape (B, C, k, k)
        search_feat = torch.stack(search_feats, dim=0)  # (F, B, C, H, W)
        gt_gaussian_maps = torch.stack(gt_maps, dim=0)  # (F, B, 1, H, W)

        ff, bf, cf, hf, wf = search_feat.shape
        x_merged = search_feat.reshape(ff, bf * cf, hf, wf)  # (F, B*C, H, W)
        w_merged = mem_filter  # (B, C, k, k)
        mem_response = F.conv2d(x_merged, w_merged, groups=bf, padding=k // 2)  # (F, B, H, W)
        mem_response = mem_response.view(ff, bf, 1, hf, wf)

        # Ensure spatial sizes match (in case of minor mismatch).
        if mem_response.shape[-2:] != gt_gaussian_maps.shape[-2:]:
            gt_mem = F.interpolate(
                gt_gaussian_maps.view(ff * bf, 1, *gt_gaussian_maps.shape[-2:]),
                size=mem_response.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).view(ff, bf, 1, hf, wf)
            print("warning, investigate gt_mem spatial size")
        else:
            gt_mem = gt_gaussian_maps

        data_mem_loss = data_mem_loss + torch.mean((mem_response - gt_mem) ** 2)
        reg_mem_loss = reg_mem_loss + torch.mean(mem_filter ** 2)
        num_filters += 1

    if num_filters == 0:
        return torch.tensor(0.0, device=device)

    data_mem_loss = data_mem_loss / num_filters
    reg_mem_loss = reg_mem_loss / num_filters

    # Regularization term on filter kernel.
    lambda_reg = getattr(memory_cfg, "FILTER_REG", 1e-4) if memory_cfg is not None else 1e-4

    return data_mem_loss + lambda_reg * reg_mem_loss  # ||x*f-c|| + lambda||f||
