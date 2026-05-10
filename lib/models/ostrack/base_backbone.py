from functools import partial
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import resize_pos_embed
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from lib.models.layers.patch_embed import PatchEmbed
from lib.models.ostrack.utils import combine_tokens, recover_tokens


class BaseBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        # for original ViT
        self.pos_embed = None
        self.img_size = [224, 224]
        self.patch_size = 16
        self.embed_dim = 384

        self.cat_mode = 'direct'

        self.pos_embed_z = None
        self.pos_embed_x = None

        self.template_segment_pos_embed = None
        self.search_segment_pos_embed = None

        self.return_inter = False
        self.return_stage = [2, 5, 8, 11]

        self.add_cls_token = False
        self.add_sep_seg = False

        self.memory_tokens = 0
        self.mem_token_embed = None
        self.read_mem_embed = None
        self.mem_grus = None
        self.mem_norms = None

    def finetune_track(self, cfg, patch_start_index=1):

        search_size = to_2tuple(cfg.DATA.SEARCH.SIZE)
        template_size = to_2tuple(cfg.DATA.TEMPLATE.SIZE)
        new_patch_size = cfg.MODEL.BACKBONE.STRIDE

        self.cat_mode = cfg.MODEL.BACKBONE.CAT_MODE
        self.return_inter = cfg.MODEL.RETURN_INTER
        self.return_stage = cfg.MODEL.RETURN_STAGES
        self.add_sep_seg = cfg.MODEL.BACKBONE.SEP_SEG

        # resize patch embedding
        if new_patch_size != self.patch_size:
            print('Inconsistent Patch Size With The Pretrained Weights, Interpolate The Weight!')
            old_patch_embed = {}
            for name, param in self.patch_embed.named_parameters():
                if 'weight' in name:
                    param = nn.functional.interpolate(param, size=(new_patch_size, new_patch_size),
                                                      mode='bicubic', align_corners=False)
                    param = nn.Parameter(param)
                old_patch_embed[name] = param
            self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=new_patch_size, in_chans=3,
                                          embed_dim=self.embed_dim)
            self.patch_embed.proj.bias = old_patch_embed['proj.bias']
            self.patch_embed.proj.weight = old_patch_embed['proj.weight']

        # for patch embedding
        patch_pos_embed = self.pos_embed[:, patch_start_index:, :]
        patch_pos_embed = patch_pos_embed.transpose(1, 2)
        B, E, Q = patch_pos_embed.shape
        P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)

        # for search region
        H, W = search_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        search_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                           align_corners=False)
        search_patch_pos_embed = search_patch_pos_embed.flatten(2).transpose(1, 2)

        # for template region
        H, W = template_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        template_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                             align_corners=False)
        template_patch_pos_embed = template_patch_pos_embed.flatten(2).transpose(1, 2)

        self.pos_embed_z = nn.Parameter(template_patch_pos_embed)
        self.pos_embed_x = nn.Parameter(search_patch_pos_embed)

        # for cls token (keep it but not used)
        if self.add_cls_token and patch_start_index > 0:
            cls_pos_embed = self.pos_embed[:, 0:1, :]
            self.cls_pos_embed = nn.Parameter(cls_pos_embed)

        # separate token and segment token
        if self.add_sep_seg:
            self.template_segment_pos_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.template_segment_pos_embed = trunc_normal_(self.template_segment_pos_embed, std=.02)
            self.search_segment_pos_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.search_segment_pos_embed = trunc_normal_(self.search_segment_pos_embed, std=.02)

        # self.cls_token = None
        # self.pos_embed = None

        if self.return_inter:
            for i_layer in self.return_stage:
                if i_layer != 11:
                    norm_layer = partial(nn.LayerNorm, eps=1e-6)
                    layer = norm_layer(self.embed_dim)
                    layer_name = f'norm{i_layer}'
                    self.add_module(layer_name, layer)

        memory_cfg = getattr(cfg.MODEL, "MEMORY", None)
        mem_enabled = getattr(memory_cfg, "ENABLED", False) if memory_cfg is not None else False
        mem_k = int(getattr(memory_cfg, "NUM_TOKENS", 0)) if memory_cfg is not None else 0
        self.memory_tokens = mem_k if mem_enabled and mem_k > 0 else 0
        if self.memory_tokens > 0:
            self.mem_token_embed = nn.Parameter(torch.zeros(1, self.memory_tokens, self.embed_dim))
            trunc_normal_(self.mem_token_embed, std=.02)
            self.read_mem_embed = nn.Parameter(torch.zeros(1, self.memory_tokens, self.embed_dim))
            trunc_normal_(self.read_mem_embed, std=.02)
            num_layers = self._num_backbone_blocks()
            print('[DEBUG], num_layers', num_layers)
            self.mem_grus = nn.ModuleList([
                nn.GRUCell(self.embed_dim, self.embed_dim) for _ in range(num_layers)
            ])
            self.mem_norms = nn.ModuleList([
                nn.LayerNorm(self.embed_dim) for _ in range(num_layers)
            ])
        else:
            self.mem_token_embed = None
            self.read_mem_embed = None
            self.mem_grus = None
            self.mem_norms = None

    def _init_memory(self, batch_size, device=None, dtype=None):
        mem = self.mem_token_embed
        if device is not None:
            mem = mem.to(device=device)
        if dtype is not None:
            mem = mem.to(dtype=dtype)
        return mem.expand(batch_size, -1, -1)

    def _num_backbone_blocks(self):
        blocks = getattr(self, "blocks", None)
        if blocks is None:
            return 0
        if isinstance(blocks, nn.Sequential):
            return len(blocks)
        raise TypeError(f"Expected self.blocks to be nn.Sequential, got {type(blocks).__name__}")

    def _prepare_layer_memory(self, mem_tokens, batch_size, device, dtype):
        num_layers = self._num_backbone_blocks()
        if self.memory_tokens <= 0 or num_layers <= 0:
            return None

        if mem_tokens is None:
            base_mem = self._init_memory(batch_size, device=device, dtype=dtype)
            # expand() creates a shared-storage view, so clone() makes real per-layer storage.
            return base_mem.unsqueeze(0).expand(num_layers, -1, -1, -1).clone()

        mem_tokens = mem_tokens.to(device=device, dtype=dtype)
        expected_shape = (
            num_layers,
            batch_size,
            self.memory_tokens,
            self.embed_dim,
        )

        if tuple(mem_tokens.shape) != expected_shape:
            raise ValueError(
            f"Expected mem_tokens shape {expected_shape}, got {tuple(mem_tokens.shape)}"
            )
        return mem_tokens

    def _update_memory_with_gru(self, layer_idx, M_prev_time, M_hat):
        if self.mem_grus is None or layer_idx >= len(self.mem_grus):
            return M_hat

        if M_prev_time.shape != M_hat.shape:
            raise ValueError(
                f"M_prev_time and M_hat must have same shape, "
                f"got {tuple(M_prev_time.shape)} and {tuple(M_hat.shape)}"
            )

        B, K, C = M_prev_time.shape
        h_prev = M_prev_time.reshape(B * K, C)  # M_{t-1,l}
        x_in = M_hat.reshape(B * K, C) # M_hat_{t,l}
        h_new = self.mem_grus[layer_idx](x_in, h_prev)
        M_current = h_new.reshape(B, K, C)      # M_{t,l}
        if self.mem_norms is not None and layer_idx < len(self.mem_norms):
            M_current = self.mem_norms[layer_idx](M_current)

        return M_current

    def forward_features(self, z, x, mem_tokens=None, is_first_frame=False):
        B, H, W = x.shape[0], x.shape[2], x.shape[3]

        x = self.patch_embed(x)
        z = self.patch_embed(z)

        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.cls_pos_embed

        z += self.pos_embed_z
        x += self.pos_embed_x

        if self.add_sep_seg:
            x += self.search_segment_pos_embed
            z += self.template_segment_pos_embed

        x = combine_tokens(z, x, mode=self.cat_mode)
        if self.add_cls_token:
            x = torch.cat([cls_tokens, x], dim=1)

        x = self.pos_drop(x)

        M_out = None
        M_layers_out = None
        if self.memory_tokens > 0:
            use_gru = (mem_tokens is not None) and (not is_first_frame)
            M_prev_layers = self._prepare_layer_memory(mem_tokens, B, x.device, x.dtype)

            self.debug_print_memtokens_shape(M_prev_layers, mem_tokens, x)

            M_prev_layer = M_prev_layers[0]
            if self.read_mem_embed is not None:
                M_prev_layer = M_prev_layer + self.read_mem_embed.to(device=x.device, dtype=x.dtype)
            visual_tokens = x
            M_current_layers = []

            for l, blk in enumerate(self.blocks):
                M_in = M_prev_layer
                x = torch.cat([M_in, visual_tokens], dim=1)

                x = blk(x)
                M_hat = x[:, :self.memory_tokens, :]
                visual_tokens = x[:, self.memory_tokens:, :]

                if use_gru:
                    M_prev_time = M_prev_layers[l]
                    M_current = self._update_memory_with_gru(l, M_prev_time, M_hat)
                else:
                    M_current = M_hat
                    if self.mem_norms is not None and l < len(self.mem_norms):
                        M_current = self.mem_norms[l](M_current)
                M_current_layers.append(M_current)

                M_prev_layer = M_current

            x = torch.cat([M_current, visual_tokens], dim=1)
            M_layers_out = torch.stack(M_current_layers, dim=0)
        else:
            for l, blk in enumerate(self.blocks):
                x = blk(x)

        if self.memory_tokens > 0:
            M_out = x[:, :self.memory_tokens, :]
            x = x[:, self.memory_tokens:, :]

        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]
        x = recover_tokens(x, lens_z, lens_x, mode=self.cat_mode)

        aux_dict = {"attn": None, "memory_tokens": M_out, "memory_tokens_layers": M_layers_out}
        return self.norm(x), aux_dict

    def debug_print_memtokens_shape(self, mem_state: Optional[Any], mem_tokens, x):
        if self.training and getattr(self, "_dbg_once", False) is False:
            if mem_tokens is None:
                print("DBG: before layer-wise mem:", x.shape, "mem_k=", self.memory_tokens,
                      "mem_in_shape=None", "mem_state_shape=", mem_state.shape)
                self._dbg_once = True
            else:
                print("DBG: before layer-wise mem:", x.shape, "mem_k=", self.memory_tokens,
                      "mem_in_shape=", mem_tokens.shape, "mem_state_shape=", mem_state.shape)
                self._dbg_once = True

    def forward(self, z, x, mem_tokens=None, is_first_frame=False, **kwargs):
        """
        Joint feature extraction and relation modeling for the basic ViT backbone.
        Args:
            z (torch.Tensor): template feature, [B, C, H_z, W_z]
            x (torch.Tensor): search region feature, [B, C, H_x, W_x]

        Returns:
            x (torch.Tensor): merged template and search region feature, [B, L_z+L_x, C]
            attn : None
        """
        x, aux_dict = self.forward_features(z, x, mem_tokens=mem_tokens, is_first_frame=is_first_frame)

        return x, aux_dict
