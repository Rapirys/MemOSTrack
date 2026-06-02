import unittest

import numpy as np
import torch

from lib.train.data.processing import InferenceLikeSequenceProcessing
from lib.train.data.processing_utils import sample_target
from lib.utils.box_ops import clip_box
from lib.utils.inference_rollout_utils import (
    _compute_sample_target_geometry,
    clip_xywh_boxes_to_image_bounds,
    sample_target_crops_from_bchw_tensor,
)


class InferenceRolloutUtilsTest(unittest.TestCase):
    def test_rollout_crop_matches_sample_target_geometry_without_resize_delta(self):
        image_hwc = np.arange(80 * 120 * 3, dtype=np.float32).reshape(80, 120, 3) / 255.0
        box_xywh = torch.tensor([20.0, 20.0, 20.0, 20.0], dtype=torch.float32)

        _, expected_resize, _ = sample_target(
            image_hwc,
            box_xywh,
            search_area_factor=2.0,
            output_sz=40,
        )

        crop_x1, crop_y1, resize_from_geometry = _compute_sample_target_geometry(
            box_xywh.view(1, 4),
            search_area_factor=2.0,
            output_sz=40,
        )
        image_bchw = torch.from_numpy(image_hwc).permute(2, 0, 1).unsqueeze(0)
        crops, resize_factors = sample_target_crops_from_bchw_tensor(
            image_bchw,
            box_xywh.view(1, 4),
            search_area_factor=2.0,
            output_sz=40,
        )

        self.assertAlmostEqual(float(crop_x1.item()), 10.0, places=6)
        self.assertAlmostEqual(float(crop_y1.item()), 10.0, places=6)
        self.assertAlmostEqual(float(resize_from_geometry.item()), float(expected_resize), places=6)
        self.assertEqual(crops.device, image_bchw.device)
        self.assertEqual(tuple(crops.shape), (1, 3, 40, 40))
        self.assertAlmostEqual(float(resize_factors.item()), float(expected_resize), places=6)

    def test_rollout_resize_factor_matches_sample_target(self):
        image_hwc = np.zeros((80, 120, 3), dtype=np.float32)
        box_xywh = torch.tensor([18.5, 14.0, 22.0, 16.0], dtype=torch.float32)

        _, expected_resize, _ = sample_target(
            image_hwc,
            box_xywh,
            search_area_factor=2.5,
            output_sz=64,
        )

        image_bchw = torch.from_numpy(image_hwc).permute(2, 0, 1).unsqueeze(0)
        crops, resize_factors = sample_target_crops_from_bchw_tensor(
            image_bchw,
            box_xywh.view(1, 4),
            search_area_factor=2.5,
            output_sz=64,
        )

        self.assertEqual(tuple(crops.shape), (1, 3, 64, 64))
        self.assertAlmostEqual(float(resize_factors.item()), float(expected_resize), places=6)

    def test_rollout_clipping_matches_test_clip_box(self):
        boxes = torch.tensor([
            [-5.0, 3.0, 8.0, 6.0],
            [115.0, 70.0, 20.0, 30.0],
            [10.0, 10.0, 2.0, 2.0],
        ])

        clipped = clip_xywh_boxes_to_image_bounds(boxes, image_height=80, image_width=120, margin=10.0)
        expected = torch.tensor([clip_box(box.tolist(), 80, 120, margin=10) for box in boxes])

        self.assertTrue(torch.equal(clipped, expected))

    def test_processing_letterbox_does_not_warp_bbox(self):
        processing = InferenceLikeSequenceProcessing(
            search_area_factor={'template': 2.0, 'search': 5.0},
            output_sz={'template': 50, 'search': 100},
            center_jitter_factor={},
            scale_jitter_factor={},
            mode='sequence',
        )
        image = np.full((50, 100, 3), 7, dtype=np.uint8)
        bbox = torch.tensor([10.0, 10.0, 20.0, 20.0])

        image_out, bbox_out, _ = processing._letterbox_frame_bbox_mask(image, bbox, None)

        self.assertEqual(image_out.shape, (100, 100, 3))
        self.assertTrue(torch.equal(bbox_out, torch.tensor([10.0, 35.0, 20.0, 20.0])))
        self.assertEqual(int(image_out[:25].max()), 0)
        self.assertEqual(int(image_out[25:75].min()), 7)


if __name__ == '__main__':
    unittest.main()
