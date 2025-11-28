import torch
from torch import nn
from timm.models.layers import DropPath


class MemoryBlock(nn.Module):
    """Transformer encoder block for memory interaction."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=attn_drop, batch_first=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention block
        shortcut = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x)
        x = shortcut + self.drop_path(attn_out)

        # MLP block
        shortcut = x
        x = self.norm2(x)
        x = shortcut + self.drop_path(self.mlp(x))
        return x


class MemoryTransformer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, depth, drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            MemoryBlock(dim=dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        drop=drop,
                        attn_drop=attn_drop,
                        drop_path=drop_path)
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x
