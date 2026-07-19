import warnings
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class MemoryTemplateCorruption:
    enabled: bool
    mode: str
    num_frames: int
    full_corruption_num_frames: int
    kernel_size: int
    passes: int
    skip_first: int = 2

    @classmethod
    def from_train_cfg(cls, train_cfg):
        return cls(
            enabled=bool(getattr(train_cfg, "MEMORY_BLUR_ENABLED", False)),
            mode=str(getattr(train_cfg, "MEMORY_BLUR_MODE", "blur")).strip().lower(),
            num_frames=int(getattr(train_cfg, "MEMORY_BLUR_NUM_FRAMES", 2)),
            full_corruption_num_frames=int(getattr(train_cfg, "MEMORY_FULL_CORRUPTION_NUM_FRAMES", 0)),
            kernel_size=int(getattr(train_cfg, "MEMORY_BLUR_KERNEL_SIZE", 19)),
            passes=int(getattr(train_cfg, "MEMORY_BLUR_PASSES", 2)),
            skip_first=2,
        )

    @property
    def min_num_search(self):
        return self.num_frames * 2 + self.skip_first

    def validate_for_num_search(self, num_search: int):
        if not self.enabled:
            return

        if self.num_frames <= 0:
            raise ValueError("TRAIN.MEMORY_BLUR_NUM_FRAMES must be > 0 when memory blur is enabled.")
        if self.full_corruption_num_frames < 0:
            raise ValueError("TRAIN.MEMORY_FULL_CORRUPTION_NUM_FRAMES must be >= 0.")
        if self.full_corruption_num_frames > self.num_frames:
            raise ValueError(
                "TRAIN.MEMORY_FULL_CORRUPTION_NUM_FRAMES must be <= TRAIN.MEMORY_BLUR_NUM_FRAMES. "
                "Got {} > {}.".format(self.full_corruption_num_frames, self.num_frames)
            )
        if self.mode not in {"blur", "zero"}:
            raise ValueError(
                "TRAIN.MEMORY_BLUR_MODE must be one of: blur, zero. "
                "Got '{}'.".format(self.mode)
            )
        if self.mode == "blur":
            if self.kernel_size <= 1 or self.kernel_size % 2 == 0:
                raise ValueError(
                    "TRAIN.MEMORY_BLUR_KERNEL_SIZE must be an odd integer > 1. "
                    "Got {}.".format(self.kernel_size)
                )
            if self.passes <= 0:
                raise ValueError("TRAIN.MEMORY_BLUR_PASSES must be > 0 when memory blur is enabled.")

        if num_search <= self.min_num_search:
            raise ValueError(
                "Memory blur requires number of search frames > (2*k + skip_first). "
                "Got DATA.SEARCH.NUMBER={}, k={}, skip_first={}, threshold={}.".format(
                    num_search,
                    self.num_frames,
                    self.skip_first,
                    self.min_num_search,
                )
            )

    def select_frame_indices(self, num_search: int):
        if not self.enabled:
            return set()

        if num_search <= self.min_num_search:
            warnings.warn(
                "Received short sequence with {} search frames while memory blur needs > {} "
                "(k={}, skip_first={}, formula 2*k + skip_first). Skipping blur for this sample.".format(
                    num_search,
                    self.min_num_search,
                    self.num_frames,
                    self.skip_first,
                )
            )
            return set()

        candidates = torch.arange(self.skip_first, num_search, dtype=torch.long)
        if candidates.numel() < self.num_frames:
            warnings.warn(
                "Insufficient blur candidates ({} available, need {}). Skipping blur for this sample.".format(
                    candidates.numel(),
                    self.num_frames,
                )
            )
            return set()

        rand_idx = torch.randperm(candidates.numel())[:self.num_frames]
        selected = candidates[rand_idx].tolist()
        return set(int(i) for i in selected)

    def build_masks(self, num_search: int, batch_size: int, device):
        template_corrupt_mask = torch.zeros((num_search, batch_size), dtype=torch.bool, device=device)
        full_corrupt_mask = torch.zeros((num_search, batch_size), dtype=torch.bool, device=device)
        if not self.enabled:
            return template_corrupt_mask, full_corrupt_mask

        frame_indices = self.select_frame_indices(num_search)
        if not frame_indices:
            return template_corrupt_mask, full_corrupt_mask

        frame_indices = sorted(frame_indices)
        idx_tensor = torch.tensor(frame_indices, dtype=torch.long, device=device)
        template_corrupt_mask[idx_tensor, :] = True

        if self.full_corruption_num_frames > 0:
            full_count = min(self.full_corruption_num_frames, len(frame_indices))
            full_perm = torch.randperm(len(frame_indices))[:full_count]
            full_indices = [frame_indices[int(i)] for i in full_perm.tolist()]
            full_idx_tensor = torch.tensor(sorted(full_indices), dtype=torch.long, device=device)
            full_corrupt_mask[full_idx_tensor, :] = True

        return template_corrupt_mask, full_corrupt_mask

    def build_mask(self, num_search: int, batch_size: int, device):
        template_corrupt_mask, _ = self.build_masks(num_search, batch_size, device)
        return template_corrupt_mask

    def apply(self, template_tensor: torch.Tensor):
        if self.mode == "zero":
            return torch.zeros_like(template_tensor)

        blurred = template_tensor
        for _ in range(self.passes):
            blurred = F.avg_pool2d(
                blurred,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
                count_include_pad=False,
            )
        return blurred
