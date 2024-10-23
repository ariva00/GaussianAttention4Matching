import torch
import torch.nn as nn
from gaussian_x_transformers import Encoder
from point_gaussian import GaussianAttention


class EncoderPointTransfomer(nn.Module):
    def __init__(
            self,
            dim=512,
            heads=8,
            gaussian_heads=0,
            sigma=[],
            dim_head=64,
            custom_layers=None,
            depth=6
            ) -> None:
        super(EncoderPointTransfomer, self).__init__()

        self.gaussian_heads = gaussian_heads

        self.encoder = Encoder(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head_custom = dim_head,
            attn_dim_head = dim_head,
            pre_norm=False,
            residual_attn=True,
            rotary_pos_emb=True,
            rotary_emb_dim = dim_head,
            custom_layers=custom_layers,
            gauss_gaussian_heads=gaussian_heads,
        )

        self.gauss_attn = GaussianAttention(sigma)

        self.linear_in = nn.Sequential(
            nn.Linear(3, 16),
            nn.Tanh(),
            nn.Linear(16, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, dim)
        )

        self.linear_out = nn.Sequential(
            nn.Linear(dim, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 3)
        )
    
    def forward(self, x:torch.Tensor, sep_idx=None, mask_head=[], return_hiddens=False):
        if sep_idx is None:
            sep_idx = x.shape[1] // 2
        dim1 = sep_idx
        dim2 = sep_idx + 1

        if self.gaussian_heads:
            shape1_gaussian_attn = self.gauss_attn(x[:, :dim1])
            shape2_gaussian_attn = self.gauss_attn(x[:, dim2:])

        x = self.linear_in(x)
        attn_mask = torch.ones((8, x.shape[1], x.shape[1]), device=x.device) if mask_head else None
        fixed_attn = torch.zeros((x.shape[0], self.gaussian_heads, x.shape[1], x.shape[1]), device=x.device) if self.gaussian_heads else None
        if (self.gaussian_heads):
            if self.gaussian_heads:
                fixed_attn[:, :, :dim1, :dim1] = shape1_gaussian_attn
                fixed_attn[:, :, dim2:, dim2:] = shape2_gaussian_attn

        if mask_head:
            attn_mask[mask_head, :, :] = 0

        if attn_mask is not None:
            attn_mask = attn_mask.type(torch.bool)

        if return_hiddens:
            x, hiddens = self.encoder(x, gaussian_attn=fixed_attn, shape_sep_idx=dim1, attn_mask=attn_mask, return_hiddens=True)
        else:
            x = self.encoder(x, gaussian_attn=fixed_attn, shape_sep_idx=dim1, attn_mask=attn_mask)
        x = self.linear_out(x)

        if return_hiddens:
            return x, hiddens
        return x
