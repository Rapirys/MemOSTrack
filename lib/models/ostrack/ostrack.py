"""
Basic OSTrack model.
"""
import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones
import torch.nn.functional as F

from lib.models.layers.head import build_box_head, build_memory_filter_head
from lib.models.ostrack.vit import vit_base_patch16_224
from lib.models.ostrack.vit_ce import vit_large_patch16_224_ce, vit_base_patch16_224_ce
from lib.utils.box_ops import box_xyxy_to_cxcywh


class OSTrack(nn.Module):
    """ This is the base class for OSTrack """

    def __init__(self, transformer, box_head, mem_filter_head=None, aux_loss=False, head_type="CORNER",
                 compute_memory_head=False):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head
        self.mem_filter_head = mem_filter_head
        self.compute_memory_head = bool(compute_memory_head)

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                mem_tokens=None,
                ):
        x, aux_dict = self.backbone(z=template, x=search,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn,
                                    mem_tokens=mem_tokens, )

        # Forward head
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]

        # Extract memory tokens from aux_dict for heads
        memory_tokens = aux_dict.get('memory_tokens', None)
        out = self.forward_head(feat_last, None, memory_tokens=memory_tokens)

        # ------------------------------------------------------------------
        # Memory filter head: predict DiMP-style filter from memory tokens.
        # No convolution here; trainer will use mem_filter_kernel + features.
        # ------------------------------------------------------------------
        if (self.compute_memory_head
                and hasattr(self, 'mem_filter_head')
                and self.mem_filter_head is not None
                and memory_tokens is not None):
            # memory_tokens: (B, K, C)
            filter_kernel = self.mem_filter_head(memory_tokens)  # (B, C, k, k)
            out['mem_filter_kernel'] = filter_kernel

            # Extract search features (B, C, H, W) from concatenated template+search sequence
            # feat_last: (B, L_z + L_x, C); take last feat_len_s tokens as search
            enc_opt_mem = feat_last[:, -self.feat_len_s:]  # (B, HW, C)
            opt_mem = (enc_opt_mem.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()  # (B, 1, C, HW)
            bs_mem, Nq_mem, C_feat, HW_feat = opt_mem.size()
            search_feat = opt_mem.view(-1, C_feat, self.feat_sz_s, self.feat_sz_s)  # (B, C, H, W)

            out['search_feat'] = search_feat

        out.update(aux_dict)
        return out

    def forward_head(self, cat_feature, gt_score_map=None, memory_tokens=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        memory_tokens: memory tokens from backbone (B, num_mem_tokens, C) or None
                       (not used directly in the main prediction head here).
        """
        enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError


def build_ostrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('OSTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
        backbone = vit_base_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                           )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224_ce':
        backbone = vit_large_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                            )

        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    else:
        raise NotImplementedError

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    # Build memory filter head if memory is enabled in config
    mem_filter_head = None
    compute_memory_head = False
    memory_cfg = getattr(cfg.MODEL, "MEMORY", None)
    if memory_cfg is not None:
        mem_enabled = getattr(memory_cfg, "ENABLED", False)
        mem_k = int(getattr(memory_cfg, "NUM_TOKENS", 0))
        if mem_enabled and mem_k > 0:
            mem_filter_head = build_memory_filter_head(cfg, hidden_dim)
            memory_loss_type = str(getattr(cfg.TRAIN, "MEMORY_LOSS_TYPE", "none")).strip().lower()
            memory_weight = float(getattr(cfg.TRAIN, "MEMORY_WEIGHT", 0.0))
            memory_loss_disabled = memory_loss_type in {"", "none", "off", "disabled", "no"}
            compute_memory_head = training and (not memory_loss_disabled) and memory_weight > 0.0

    model = OSTrack(
        backbone,
        box_head,
        mem_filter_head=mem_filter_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
        compute_memory_head=compute_memory_head,
    )

    if 'OSTrack' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model
