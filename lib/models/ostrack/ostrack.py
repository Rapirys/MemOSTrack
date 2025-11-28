"""
Basic OSTrack model.
"""
import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
from lib.models.layers.memory_block import MemoryTransformer
from lib.models.ostrack.vit import vit_base_patch16_224
from lib.models.ostrack.vit_ce import vit_large_patch16_224_ce, vit_base_patch16_224_ce
from lib.utils.box_ops import box_xyxy_to_cxcywh
from timm.models.layers import trunc_normal_


class OSTrack(nn.Module):
    """ This is the base class for OSTrack """

    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER",
                 memory_enabled=False, memory_num_tokens=8, memory_num_blocks=1,
                 memory_num_heads=8, memory_mlp_ratio=4.0, memory_dropout=0.0):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

        # memory tokens configuration
        self.use_memory = memory_enabled
        self.num_memory_tokens = memory_num_tokens
        self.memory_dim = self.backbone.embed_dim

        self.mem_init = nn.Parameter(torch.zeros(1, self.num_memory_tokens, self.memory_dim))
        trunc_normal_(self.mem_init, std=0.02)

        self.memory_transformer = MemoryTransformer(
            dim=self.memory_dim,
            num_heads=memory_num_heads,
            mlp_ratio=memory_mlp_ratio,
            depth=memory_num_blocks,
            drop=memory_dropout,
            attn_drop=memory_dropout,
            drop_path=0.0,
        )

        # persistent state for inference
        self.mem_state = None

    def reset_memory(self):
        self.mem_state = None

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                use_memory=True,
                reset_memory=False,
                ):
        x, aux_dict = self.backbone(z=template, x=search,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn, )

        # Forward head
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]

        if self.use_memory and use_memory:
            B, _, _ = feat_last.shape
            if self.training:
                mem_in = self.mem_init.expand(B, -1, -1)
            else:
                if reset_memory or self.mem_state is None or self.mem_state.shape[0] != B:
                    self.mem_state = self.mem_init.expand(B, -1, -1).clone()
                mem_in = self.mem_state

            concat = torch.cat([mem_in, feat_last], dim=1)
            concat = self.memory_transformer(concat)
            mem_out = concat[:, :self.num_memory_tokens, :]
            feat_last = concat[:, self.num_memory_tokens:, :]

            if not self.training:
                self.mem_state = mem_out.detach()

            aux_dict["memory_tokens"] = mem_out
            aux_dict["memory_embedding"] = mem_out.mean(dim=1)
        else:
            aux_dict["memory_tokens"] = None
            aux_dict["memory_embedding"] = None
        out = self.forward_head(feat_last, None)

        out.update(aux_dict)
        out['backbone_feat'] = x
        out['feat_after_memory'] = feat_last
        return out

    def forward_head(self, cat_feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
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

    memory_cfg = getattr(cfg.MODEL, 'MEMORY', None)
    memory_enabled = getattr(memory_cfg, 'ENABLED', False) if memory_cfg is not None else False
    memory_num_tokens = getattr(memory_cfg, 'NUM_TOKENS', 8) if memory_cfg is not None else 8
    memory_num_blocks = getattr(memory_cfg, 'NUM_BLOCKS', 1) if memory_cfg is not None else 1
    backbone_heads = getattr(backbone, 'num_heads', 8)
    memory_num_heads = getattr(memory_cfg, 'NUM_HEADS', backbone_heads) if memory_cfg is not None else backbone_heads
    memory_mlp_ratio = getattr(memory_cfg, 'MLP_RATIO', 4.0) if memory_cfg is not None else 4.0
    memory_dropout = getattr(memory_cfg, 'DROPOUT', 0.0) if memory_cfg is not None else 0.0

    model = OSTrack(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
        memory_enabled=memory_enabled,
        memory_num_tokens=memory_num_tokens,
        memory_num_blocks=memory_num_blocks,
        memory_num_heads=memory_num_heads,
        memory_mlp_ratio=memory_mlp_ratio,
        memory_dropout=memory_dropout,
    )

    if 'OSTrack' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model
