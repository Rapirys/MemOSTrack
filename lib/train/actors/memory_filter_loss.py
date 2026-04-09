import torch
import torch.nn.functional as F


class MemoryFilterLoss:
    """
    Original sampled-frame memory filter loss.
    For each frame-level filter, samples `FILTER_LOSS_FRAMES` search features and applies DiMP-style loss.
    """

    def __call__(self, pred_seq, gt_gaussian_maps_all, cfg, device):
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


_DEFAULT_MEMORY_FILTER_LOSS = MemoryFilterLoss()


def compute_memory_filter_loss(pred_seq, gt_gaussian_maps_all, cfg, device):
    """
    Backward-compatible wrapper for the class-based sampled memory filter loss.
    """
    return _DEFAULT_MEMORY_FILTER_LOSS(
        pred_seq=pred_seq,
        gt_gaussian_maps_all=gt_gaussian_maps_all,
        cfg=cfg,
        device=device,
    )


class DiMPSteepestDescentSolver:
    """
    DiMP-style steepest descent with analytic step length (Gauss-Newton form).

    Objective:
      L(f) = mean(||A f - y||^2) + lambda_reg * mean(||f||^2)
    """

    def __init__(
        self,
        eps=1e-12,
        alpha_max=10.0,
        chunk_size=None,
        force_fp32=True,
    ):
        self.eps = eps
        self.alpha_max = alpha_max
        self.chunk_size = chunk_size
        self.force_fp32 = force_fp32
        self._has_conv2d_weight = hasattr(torch.nn, "grad") and hasattr(torch.nn.grad, "conv2d_weight")

    @torch.no_grad()
    def solve(
        self,
        x_merged,
        y_merged,
        init_filter,
        lambda_reg,
        iters,
        padding,
        groups,
    ):
        if iters <= 0:
            return init_filter.detach()

        if self.force_fp32 and init_filter.dtype != torch.float32:
            x = x_merged.float()
            y = y_merged.float()
            f = init_filter.float().clone()
        else:
            x = x_merged
            y = y_merged
            f = init_filter.clone()

        t_len, _, h, w = x.shape
        batch_size = groups
        _, channels, k, _ = f.shape

        n = float(t_len * batch_size * h * w)
        m = float(batch_size * channels * k * k)

        chunk = self.chunk_size
        stride = 1
        dilation = 1

        for _ in range(iters):
            g_data = torch.zeros_like(f)

            if chunk is None or chunk >= t_len:
                resp = F.conv2d(x, f, groups=batch_size, padding=padding)
                resid = resp - y
                grad_out = (2.0 / n) * resid

                if self._has_conv2d_weight:
                    g_data = torch.nn.grad.conv2d_weight(
                        input=x,
                        weight_size=f.shape,
                        grad_output=grad_out,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        groups=batch_size,
                    )
                else:
                    with torch.enable_grad():
                        f_req = f.detach().requires_grad_(True)
                        resp2 = F.conv2d(x, f_req, groups=batch_size, padding=padding)
                        data = (resp2 - y).pow(2).sum() / n
                        (g_data,) = torch.autograd.grad(data, f_req, create_graph=False)
                        g_data = g_data.detach()
            else:
                for t0 in range(0, t_len, chunk):
                    t1 = min(t_len, t0 + chunk)
                    x_t = x[t0:t1]
                    y_t = y[t0:t1]

                    resp = F.conv2d(x_t, f, groups=batch_size, padding=padding)
                    resid = resp - y_t
                    grad_out = (2.0 / n) * resid

                    if self._has_conv2d_weight:
                        g_data = g_data + torch.nn.grad.conv2d_weight(
                            input=x_t,
                            weight_size=f.shape,
                            grad_output=grad_out,
                            stride=stride,
                            padding=padding,
                            dilation=dilation,
                            groups=batch_size,
                        )
                    else:
                        with torch.enable_grad():
                            f_req = f.detach().requires_grad_(True)
                            resp2 = F.conv2d(x_t, f_req, groups=batch_size, padding=padding)
                            data = (resp2 - y_t).pow(2).sum() / n
                            (g_chunk,) = torch.autograd.grad(data, f_req, create_graph=False)
                            g_data = g_data + g_chunk.detach()

            g_reg = (2.0 * float(lambda_reg) / m) * f
            g = g_data + g_reg

            gg = (g * g).sum()

            ag2 = torch.zeros((), device=f.device, dtype=f.dtype)
            if chunk is None or chunk >= t_len:
                h_t = F.conv2d(x, g, groups=batch_size, padding=padding)
                ag2 = (h_t * h_t).sum()
            else:
                for t0 in range(0, t_len, chunk):
                    t1 = min(t_len, t0 + chunk)
                    h_t = F.conv2d(x[t0:t1], g, groups=batch_size, padding=padding)
                    ag2 = ag2 + (h_t * h_t).sum()

            denom = (ag2 / n) + (float(lambda_reg) * gg / m) + self.eps
            alpha = gg / denom
            if self.alpha_max is not None:
                alpha = torch.clamp(alpha, max=float(self.alpha_max))

            f = f - alpha * g

        if f.dtype != init_filter.dtype:
            f = f.to(dtype=init_filter.dtype)
        return f.detach()


def compute_memory_filter_target_match_loss_dimp_sd(
    pred_seq,
    gt_gaussian_maps_all,
    cfg,
    device,
    sd_solver=None,
):
    """
    Two-stage loss:
    1) Fit a sequence-level target filter on all frames using DiMP-style SD.
    2) Match each predicted per-frame filter to the trained target filter.
    """
    memory_cfg = getattr(cfg.MODEL, "MEMORY", None)
    if memory_cfg is None or len(pred_seq) == 0:
        return torch.tensor(0.0, device=device)

    lambda_reg = float(getattr(memory_cfg, "FILTER_REG", 1e-4))
    inner_steps = int(getattr(memory_cfg, "FILTER_TARGET_ITERS", 6))
    match_w = float(getattr(memory_cfg, "FILTER_MATCH_WEIGHT", 1.0))

    filter_frames = [i for i, pred_t in enumerate(pred_seq) if "mem_filter_kernel" in pred_t]
    feature_frames = [i for i, pred_t in enumerate(pred_seq) if "search_feat" in pred_t]
    if not filter_frames or not feature_frames:
        return torch.tensor(0.0, device=device)

    common_frames = [i for i in feature_frames if i < gt_gaussian_maps_all.shape[0]]
    if not common_frames:
        return torch.tensor(0.0, device=device)

    init_filters = torch.stack([pred_seq[i]["mem_filter_kernel"].detach() for i in filter_frames], dim=0)
    init_filter = init_filters.mean(dim=0)
    batch_size, channels, k, _ = init_filter.shape
    padding = k // 2

    search_feat_all = torch.stack([pred_seq[i]["search_feat"].detach() for i in common_frames], dim=0)
    gt_maps_all = torch.stack([gt_gaussian_maps_all[i].unsqueeze(1).detach() for i in common_frames], dim=0)

    t_len, batch_feat, channels_feat, h, w = search_feat_all.shape
    if batch_feat != batch_size or channels_feat != channels:
        raise ValueError(
            "Shape mismatch: "
            "init_filter is (B={},C={},k={}) but search_feat_all is (T={},B={},C={},H={},W={}).".format(
                batch_size, channels, k, t_len, batch_feat, channels_feat, h, w
            )
        )

    if gt_maps_all.shape[-2:] != (h, w):
        gt_maps_all = F.interpolate(
            gt_maps_all.view(t_len * batch_size, 1, *gt_maps_all.shape[-2:]),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).view(t_len, batch_size, 1, h, w)

    x_merged = search_feat_all.reshape(t_len, batch_size * channels, h, w).contiguous()
    y_merged = gt_maps_all.squeeze(2).contiguous()

    if sd_solver is None:
        sd_solver = DiMPSteepestDescentSolver(eps=1e-12, alpha_max=10.0, chunk_size=None, force_fp32=True)

    if inner_steps > 0:
        target_filter = sd_solver.solve(
            x_merged=x_merged,
            y_merged=y_merged,
            init_filter=init_filter,
            lambda_reg=lambda_reg,
            iters=inner_steps,
            padding=padding,
            groups=batch_size,
        )
    else:
        target_filter = init_filter.detach()

    target_f = target_filter.float()
    match_loss = torch.tensor(0.0, device=device)
    for i in filter_frames:
        f_i = pred_seq[i]["mem_filter_kernel"].float()
        match_loss = match_loss + (f_i - target_f).pow(2).mean()

    match_loss = match_loss / len(filter_frames)
    return match_w * match_loss


def compute_memory_filter_target_match_loss(pred_seq, gt_gaussian_maps_all, cfg, device, sd_solver=None):
    """
    Backward-compatible alias using DiMP-style steepest descent target fitting.
    """
    return compute_memory_filter_target_match_loss_dimp_sd(
        pred_seq=pred_seq,
        gt_gaussian_maps_all=gt_gaussian_maps_all,
        cfg=cfg,
        device=device,
        sd_solver=sd_solver,
    )
