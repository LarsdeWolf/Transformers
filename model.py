import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Union
from utils import get_2d_sincos_pos_embed


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        hidden_activation: type[nn.Module] = nn.ReLU,
        output_activation: type[nn.Module] = None,
        flatten: bool = False,
        p_dropout: float = 0.0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.flatten = flatten
        self.p_dropout = p_dropout
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.dtype = dtype

        layers: list[nn.Module] = [] if not self.flatten else [nn.Flatten(-3)]
        d_in = self.input_dim
        for d_out in self.hidden_dims:
            layers.append(nn.Linear(d_in, d_out, dtype=dtype))
            layers.append(self.hidden_activation())
            layers.append(nn.Dropout(self.p_dropout))
            d_in = d_out
        layers.append(nn.Linear(d_in, self.output_dim, dtype=dtype))
        if self.output_activation is not None:
            layers.append(self.output_activation())
        self.layers = nn.Sequential(*layers)

        self.apply(self._init_weights)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

class MoE(nn.Module):
    """

    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        out_dim: int,
        num_experts: int,
        hidden_act: type[nn.Module] = nn.ReLU,
        top_k: int = 2,
        p_dropout: float = 0.,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        self.experts = nn.ModuleList([
            MLP(input_dim, hidden_dims, out_dim, hidden_act, p_dropout=p_dropout)
            for _ in range(num_experts)
        ])
        self.gate = nn.Sequential(
            nn.Linear(input_dim, num_experts),
            nn.Softmax(-1)
        )


    def forward(self, x):
        gate_weights = self.gate(x)
        gate_idx = torch.topk(gate_weights, self.top_k, -1)[1]

        out = torch.zeros_like(x)
        for exp in range(self.num_experts):
            # Boolean for each token if processed by exp
            mask = (gate_idx == exp).any(-1) # [B, S]
            if mask.any():
                tokens = x[mask] # [N, D]
                out_exp = self.experts[exp](tokens)
                out[mask] = (gate_weights[mask][:, exp].unsqueeze(-1) * out_exp)
        return out


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention layer."""

    def __init__(
            self,
            d_emb: int,
            n_heads: int,
            return_scores: bool = False
    ) -> None:
        super().__init__()
        assert d_emb % n_heads == 0, "d_emb must be divisible by n_heads"
        self.qkv_dim = d_emb // n_heads
        self.d_emb = d_emb
        self.n_heads = n_heads
        self.return_scores = return_scores

        self.to_qkv = nn.Linear(self.d_emb, 3 * self.d_emb)
        self.output = nn.Linear(self.d_emb, self.d_emb)

        self.apply(self._init_weights)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> tuple[Tensor, Optional[Tensor]]:
        B, S, D = x.shape
        # project to qkv embedding and split into separate q, k, v
        x = self.to_qkv(x).view(B, S, self.n_heads, 3 * self.qkv_dim).permute(0, 2, 1, 3)
        q, k, v = torch.chunk(x, 3, dim=-1)
        # dot product between q & k and apply scaling
        x = (q @ k.permute(0, 1, 3, 2)) / (self.qkv_dim**0.5)
        if mask is not None:
            x = x.masked_fill(mask == 0, float("-inf"))
        # attention scores
        x = x.softmax(-1)
        scores = x if self.return_scores else None
        x = (x @ v).permute(0, 2, 1, 3).reshape(B, S, D)
        return self.output(x), scores

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class TransformerBlock(nn.Module):
    """Transformer block with MHSA + MLP."""

    def __init__(
        self,
        d_emb: int,
        n_heads: int,
        mlp_factor: int = 4,
        act_fn: type[nn.Module] = nn.ReLU,
        att_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        pre_ln: bool = True,
        return_scores: bool = False,
        num_exp: int = 1,
    ) -> None:
        super().__init__()
        self.pre_ln = pre_ln
        self.return_scores = return_scores

        self.mhsa = MultiHeadSelfAttention(d_emb, n_heads, self.return_scores)
        if num_exp > 1:
            self.mlp = MoE(d_emb, [(mlp_factor * d_emb) // num_exp], d_emb, num_exp)
        else:
            self.mlp = MLP(d_emb, [mlp_factor * d_emb], d_emb, act_fn, p_dropout=mlp_dropout)
        self.ln1 = nn.LayerNorm(d_emb)
        self.ln2 = nn.LayerNorm(d_emb)
        self.dropout = nn.Dropout(att_dropout)

    def forward(self, x: Tensor) -> tuple[Tensor, Optional[Tensor]]:
        res = x
        if self.pre_ln:
            x = self.ln1(x)
        x, scores = self.mhsa(x)
        x = res + self.dropout(x)
        if not self.pre_ln:
            x = self.ln1(x)

        res = x
        if self.pre_ln:
            x = self.ln2(x)
        x = self.mlp(x)
        x = res + self.dropout(x)
        if not self.pre_ln:
            x = self.ln2(x)

        return x, scores


class ViT(nn.Module):
    """Vision Transformer (ViT)."""

    def __init__(
        self,
        x_dim: tuple[int, int, int, int],
        patch_dim: int,
        d_emb: int,
        n_heads: int,
        n_blocks: int,
        n_class: int,
        class_token: bool = True,
        mlp_factor: int = 4,
        act_fn: type[nn.Module] = nn.GELU,
        att_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        pre_ln: bool = True,
        learned_encodings: bool = True,
        disable_head: bool = False,
        return_scores: bool = False,
        num_exp: int = 1,
    ) -> None:
        super().__init__()
        self.input_size = x_dim
        self.d_emb = d_emb
        self.return_scores = return_scores
        self.disable_head = disable_head
        self.learned_encodings = learned_encodings

        _, self.x_c, self.x_h, self.x_w = x_dim
        self.p_dim = patch_dim
        self.class_token = class_token
        self.n_patch = self.x_h * self.x_w // (self.p_dim**2)
        assert (self.x_h % self.p_dim == 0) and (self.x_w % self.p_dim == 0), (
            "Input height/width should be divisible by patch_dim"
        )

        self.embedding = nn.Linear((self.p_dim**2) * self.x_c, d_emb)
        if class_token:
            self.cls = nn.Parameter(torch.zeros(1, 1, d_emb))
            self.n_patch += 1
        if learned_encodings:
            self.pos_emb = nn.Parameter(torch.zeros(1, self.n_patch, d_emb))
        else:
            self.pos_emb = nn.Parameter(get_2d_sincos_pos_embed(d_emb, self.x_h // self.p_dim, self.class_token),
                                        requires_grad=False)
        self.blocks = nn.ModuleList(
            TransformerBlock(
                d_emb=d_emb,
                n_heads=n_heads,
                mlp_factor=mlp_factor,
                act_fn=act_fn,
                att_dropout=att_dropout,
                mlp_dropout=mlp_dropout,
                pre_ln=pre_ln,
                return_scores=return_scores,
                num_exp=num_exp,
            )
            for _ in range(n_blocks)
        )
        self.output = MLP(d_emb, [d_emb * 2], n_class) if not disable_head else None

        self.apply(self._init_weights)

    def forward(self, x: Tensor) -> Union[Tensor, tuple[Tensor, Tensor]]:
        B = x.shape[0]
        x = self.patchify(x)
        x = self.embedding(x)
        if self.class_token:
            x = torch.cat([self.cls.expand(B, -1, -1), x], dim=1)
        x = x + self.pos_emb

        att_scores = []
        for block in self.blocks:
            x, scores = block(x)
            if scores is not None:
                att_scores.append(scores)

        x = x[:, 0] if self.class_token else x.mean(dim=1)
        x = self.output(x) if not self.disable_head else x
        if self.return_scores:
            att_scores = torch.stack(att_scores, dim=1)
            return x, att_scores
        return x

    def patchify(self, x: Tensor) -> Tensor:
        n_patch = self.n_patch - 1 if self.class_token else self.n_patch
        return (x.unfold(2, self.p_dim, self.p_dim)
                .unfold(3, self.p_dim, self.p_dim)
                .contiguous()
                .view(-1, n_patch, (self.p_dim**2) * self.x_c)
        )

    def depatchify(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs the original image from patches.

        Args:
            patches: Tensor of shape (batch, n_patches, patch_size^2 * channels)
            h: original image height
            w: original image width

        Returns:
            Reconstructed image of shape (batch, channels, h, w)
        """
        b, _, _ = patches.shape
        p = self.p_dim
        c = self.x_c

        # number of patches per row/col
        n_h = self.x_h // p
        n_w = self.x_w // p

        # reshape patches into grid
        patches = patches.view(b, n_h, n_w, p, p, c)
        patches = patches.permute(0, 5, 1, 3, 2, 4).contiguous()
        img = patches.view(b, c, self.x_h, self.x_w)
        return img

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Parameter):
            nn.init.trunc_normal_(m, std=0.02)


class DiT(nn.Module):
    def __init__(self):
        super(DiT, self).__init__()

if __name__=='__main__':
    inp = torch.randn(64, 1, 28, 28)
    model = ViT([64, 1, 28, 28],
                7,
                64,
                4,
                2,
                10,
                return_scores=False,
                disable_head=False,
                class_token=True,
                learned_encodings=False,
                num_exp=5)
    model(inp)
    print()