import torch
import torchvision.transforms as transforms
from lib.utils import TensorDict
import torch.nn.functional as F
import numpy as np
import cv2 as cv


def stack_tensors(x):
    if isinstance(x, (list, tuple)) and isinstance(x[0], torch.Tensor):
        return torch.stack(x)
    return x


class BaseProcessing:
    """ Base class for Processing. Processing class is used to process the data returned by a dataset, before passing it
     through the network. For example, it can be used to crop a search region around the object, apply various data
     augmentations, etc."""
    def __init__(self, transform=transforms.ToTensor(), template_transform=None, search_transform=None, joint_transform=None):
        """
        args:
            transform       - The set of transformations to be applied on the images. Used only if template_transform or
                                search_transform is None.
            template_transform - The set of transformations to be applied on the template images. If None, the 'transform'
                                argument is used instead.
            search_transform  - The set of transformations to be applied on the search images. If None, the 'transform'
                                argument is used instead.
            joint_transform - The set of transformations to be applied 'jointly' on the template and search images.  For
                                example, it can be used to convert both template and search images to grayscale.
        """
        self.transform = {'template': transform if template_transform is None else template_transform,
                          'search':  transform if search_transform is None else search_transform,
                          'joint': joint_transform}

    def __call__(self, data: TensorDict):
        raise NotImplementedError


class InferenceLikeSequenceProcessing(BaseProcessing):
    """Inference-like processing path.

    Keeps full sequence frames, resizes them to a fixed canvas, applies augmentations, and keeps annotations
    in absolute image coordinates on that canvas. Search/template crops are deferred to actor rollout.
    """

    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor,
                 mode='pair', settings=None, *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.mode = mode
        self.settings = settings
        self.search_per_frame_jitter = bool(getattr(settings, "search_per_frame_jitter", False))

        self.canvas_size = int(output_sz['search'])

    def _resize_frame_bbox_mask(self, image, bbox, mask):
        if image is None or bbox is None:
            return None, bbox, mask

        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            raise ValueError("invalid frame size: {}x{}".format(h, w))

        out_h = self.canvas_size
        out_w = self.canvas_size
        resized = cv.resize(image, (out_w, out_h), interpolation=cv.INTER_LINEAR)

        if torch.is_tensor(bbox):
            bbox_out = bbox.clone().float()
        else:
            bbox_out = torch.tensor(bbox, dtype=torch.float32)

        scale_x = float(out_w) / float(w)
        scale_y = float(out_h) / float(h)
        bbox_out[0] = bbox_out[0] * scale_x
        bbox_out[1] = bbox_out[1] * scale_y
        bbox_out[2] = bbox_out[2] * scale_x
        bbox_out[3] = bbox_out[3] * scale_y

        if mask is None:
            mask_out = None
        elif torch.is_tensor(mask):
            mask_np = mask.detach().cpu().numpy().astype(np.uint8)
            mask_resized = cv.resize(mask_np, (out_w, out_h), interpolation=cv.INTER_NEAREST)
            mask_out = torch.from_numpy(mask_resized).to(mask.device, dtype=mask.dtype)
        else:
            mask_out = cv.resize(mask.astype(np.uint8), (out_w, out_h), interpolation=cv.INTER_NEAREST)

        return resized, bbox_out, mask_out

    def _resize_triplet(self, images, annos, masks):
        out_images, out_annos, out_masks = [], [], []
        for img, box, m in zip(images, annos, masks):
            img_out, box_out, mask_out = self._resize_frame_bbox_mask(img, box, m)
            out_images.append(img_out)
            out_annos.append(box_out)
            out_masks.append(mask_out)
        return out_images, out_annos, out_masks

    @staticmethod
    def _stack_masks(mask_list, size):
        out = []
        for m in mask_list:
            if m is None:
                out.append(torch.zeros((size, size), dtype=torch.float32))
            elif torch.is_tensor(m):
                out.append(m.float())
            else:
                out.append(torch.from_numpy(m).float())
        return out

    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'template_images', search_images', 'template_anno', 'search_anno'
        returns:
            TensorDict - output data block with following fields:
                'template_images', 'search_images', 'template_anno', 'search_anno', 'test_proposals', 'proposal_iou'
        """
        # Apply joint transforms
        if self.transform['joint'] is not None:
            data['template_images'], data['template_anno'], data['template_masks'] = self.transform['joint'](
                image=data['template_images'], bbox=data['template_anno'], mask=data['template_masks'])
            data['search_images'], data['search_anno'], data['search_masks'] = self.transform['joint'](
                image=data['search_images'], bbox=data['search_anno'], mask=data['search_masks'], new_roll=False)

        data['template_images'], data['template_anno'], data['template_masks'] = self._resize_triplet(
            data['template_images'], data['template_anno'], data['template_masks'])
        data['search_images'], data['search_anno'], data['search_masks'] = self._resize_triplet(
            data['search_images'], data['search_anno'], data['search_masks'])

        for s in ['template', 'search']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"

            # Apply transforms
            num_frames = len(data[s + '_images'])
            if num_frames > 1:
                new_roll = [True] + [False] * (num_frames - 1)
            else:
                new_roll = True

            data[s + '_images'], data[s + '_anno'], data[s + '_att'], data[s + '_masks'] = self.transform[s](
                image=data[s + '_images'],
                bbox=data[s + '_anno'],
                att=[np.zeros((self.canvas_size, self.canvas_size), dtype=np.bool_) for _ in range(num_frames)],
                mask=data[s + '_masks'],
                joint=False,
                new_roll=new_roll
            )

            # Check whether data is valid. Avoid too small bounding boxes
            boxes_t = torch.stack([a.float() for a in data[s + '_anno']], dim=0)
            if (boxes_t[:, 2:] <= 1.0).any():
                data['valid'] = False
                data['invalid_reason'] = "{} bbox too small after augmentation".format(s)
                return data

            # 2021.1.9 Check whether elements in data[s + '_att'] is all 1
            # Note that type of data[s + '_att'] is tuple, type of ele is torch.tensor
            for ele in data[s + '_att']:
                if (ele == 1).all():
                    data['valid'] = False
                    data['invalid_reason'] = "{} attention mask is all ones".format(s)
                    # print("Values of original attention mask are all one. Replace it with new data.")
                    return data
            # 2021.1.10 more strict conditions: require the donwsampled masks not to be all 1
            for ele in data[s + '_att']:
                feat_size = self.output_sz[s] // 16  # 16 is the backbone stride
                # (1,1,128,128) (1,1,256,256) --> (1,1,8,8) (1,1,16,16)
                mask_down = F.interpolate(ele[None, None].float(), size=feat_size).to(torch.bool)[0]
                if (mask_down == 1).all():
                    data['valid'] = False
                    data['invalid_reason'] = "{} downsampled attention mask is all ones".format(s)
                    # print("Values of down-sampled attention mask are all one. "
                    #       "Replace it with new data.")
                    return data

        data['valid'] = True
        data['invalid_reason'] = None
        # if we use copy-and-paste augmentation
        if data["template_masks"] is None:
            data["template_masks"] = [
                torch.zeros((self.canvas_size, self.canvas_size), dtype=torch.float32)
                for _ in range(len(data["template_images"]))
            ]
        else:
            data["template_masks"] = self._stack_masks(data["template_masks"], self.canvas_size)

        if data["search_masks"] is None:
            data["search_masks"] = [
                torch.zeros((self.canvas_size, self.canvas_size), dtype=torch.float32)
                for _ in range(len(data["search_images"]))
            ]
        else:
            data["search_masks"] = self._stack_masks(data["search_masks"], self.canvas_size)

        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        return data
