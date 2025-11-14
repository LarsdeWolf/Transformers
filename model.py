import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Union, Any
from utils import get_2d_sincos_pos_embed


class MLP(nn.Module):
    """
    Builds a sequence of Linear -> Activation -> Dropout layers
    (for all but the final layer). The final layer optionally applies a
    separate output activation.

    Parameters
    ----------
    input_dim:
        Size of the input feature dimension.
    hidden_dims:
        Sequence of hidden layer widths.
    out_dim:
        Output feature dimension.
    hidden_act:
        Activation class to instantiate between layers
    out_act:
        Optional activation class applied after the final projection.
    flatten:
        If True, first flattens the last 3 dims.
    p_dropout:
        Dropout probability applied after hidden activations.
    dtype:
        Torch dtype used for linear layers.

    Returns
    -------
    Tensor
        Output of shape (B, out_dim) for input of shape (B, input_dim)
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dims: list[int],
            out_dim: int,
            hidden_act: type[nn.Module] = nn.ReLU,
            out_act: type[nn.Module] = None,
            flatten: bool = False,
            p_dropout: float = 0.0,
            dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.out_dim = out_dim
        self.flatten = flatten
        self.p_dropout = p_dropout
        self.hidden_act = hidden_act
        self.out_act = out_act
        self.dtype = dtype

        dims = [input_dim] + hidden_dims + [out_dim]
        layers = []
        if self.flatten: layers.append(nn.Flatten(-3))
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1], dtype=dtype))
            layers.append(hidden_act())
            if p_dropout > 0:
                layers.append(nn.Dropout(p_dropout))
        layers.append(nn.Linear(dims[-2], dims[-1], dtype=dtype))
        if out_act is not None: layers.append(out_act())

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
    Contains `num_experts` independent MLPs. A linear gating network
    computes scores of shape (tokens, num_experts); these scores are
    then masked (except top-k) and rescaled into a probability distribution,
    used to combine the expert outputs.

    Parameters
    ----------
    input_dim, hidden_dims, out_dim:
        Dimensions of expert MLPs.
    num_experts:
        Number of expert MLPs.
    hidden_act:
        Activation used inside each expert.
    top_k:
        Number of experts to select per token.
    p_dropout:
        Dropout probability of experts.
    return_weights:
        If True, the forward output contains `.gate_weights`
        and `.gate_idx` attributes which are the gating scores and
        expert indices respectively.

    Returns
    -------
    Tensor
        Output of shape (B, out_dim) for input of shape (B, input_dim)
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
            return_weights: bool = False,
            dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.return_weights = return_weights

        self.experts = nn.ModuleList([
            MLP(input_dim, hidden_dims, out_dim, hidden_act, p_dropout=p_dropout)
            for _ in range(num_experts)
        ])

        self.gate = nn.Linear(input_dim, num_experts)
        self.softmax = nn.Softmax(-1)

    def forward(self, x):
        gate_weights = self.gate(x)
        gate_idx = torch.topk(gate_weights, self.top_k, -1)[1]
        mask = torch.zeros_like(gate_weights, dtype=torch.bool).scatter(-1, gate_idx, True)
        gate_weights = self.softmax(gate_weights.masked_fill(~mask, float('-inf')))

        out_tokens = torch.zeros_like(x)
        for exp in range(self.num_experts):
            mask = (gate_idx == exp).any(-1)  # [B, S]
            if mask.any():
                tokens = x[mask]  # [N, D]
                tokens = self.experts[exp](tokens)
                out_tokens[mask] += (gate_weights[mask][:, exp].unsqueeze(-1) * tokens)

        if self.return_weights:
            out_tokens.__setattr__('gate_weights', gate_weights)
            out_tokens.__setattr__('gate_idx', gate_idx)

        return out_tokens

    @staticmethod
    def load_balancing_loss(gate_weights: torch.Tensor, gate_idx: torch.Tensor, num_experts: int):
        """
        Computes an "importance" term (sum of gate probabilities per expert) and a
        "load" term (number of tokens routed to each expert) and returns
        a scalar loss based on their elementwise product.

        Parameters
        ----------
        gate_weights:
            Tensor of shape [B, S, num_experts] containing per-token
            gating probabilities (after softmax and masking).
        gate_idx:
            Integer tensor with selected expert indices for the top-k
            gating choices (used to compute load).
        num_experts:
            Number of experts (size of the expert axis).

        Returns
        -------
        Tensor
            Scalar loss tensor (single-element) encouraging balanced
            expert utilization.
        """
        # Importance: sum of probabilities per expert
        importance = gate_weights.sum(dim=[0, 1])  # [num_experts]
        importance = importance / importance.sum()

        # Load: number of tokens routed to each expert
        load = torch.zeros(num_experts, device=gate_weights.device)
        for e in range(num_experts):
            load[e] = (gate_idx == e).any(dim=-1).float().sum()
        load = load / (load.sum() + 1e-10)

        # Coefficient of variation loss
        loss = num_experts * (importance * load).sum()
        return loss


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention (MHSA).

    A straightforward implementation of transformer-style scaled
    dot-product attention. The module computes concatenated qkv via a
    single Linear projection, splits heads, applies attention with an
    optional mask, and projects back to the original embedding
    dimension.

    Parameters
    ----------
    d_emb:
        Total embedding dimension (will be split across heads).
    n_heads:
        Number of attention heads. `d_emb` must be divisible by
        `n_heads`.
    return_scores:
        If True, attn score tensors will be attached to the returned
        tensor as the attribute `.scores`.
    """

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
        """Compute self-attention for input tokens.

        Parameters
        ----------
        x:
            Tensor of shape [B, S, D] containing token embeddings.
        mask:
            Optional boolean or byte mask broadcastable to attention
            weights; positions with mask==0 will be ignored (set to
            -inf before softmax).

        Returns
        -------
        Tensor
            Output tensor of shape [B, S, D] after attention and output
            projection. If `return_scores` is True, the attention
            weights tensor of shape [B, n_heads, S, S] is attached to
            the returned tensor as `.scores`.
        """
        B, S, D = x.shape
        # project to qkv embedding and split into separate q, k, v
        x = self.to_qkv(x).view(B, S, self.n_heads, 3 * self.qkv_dim).permute(0, 2, 1, 3)
        q, k, v = torch.chunk(x, 3, dim=-1)
        # dot product between q & k and apply scaling
        x = (q @ k.permute(0, 1, 3, 2)) / (self.qkv_dim ** 0.5)
        if mask is not None:
            x = x.masked_fill(mask == 0, float("-inf"))
        # attention scores
        scores = x.softmax(-1)
        x = (scores @ v).permute(0, 2, 1, 3).reshape(B, S, D)
        x = self.output(x)

        if self.return_scores:
            x.__setattr__('scores', scores)
        return x

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class TransformerBlock(nn.Module):
    """Transformer block combining MHSA, residuals and MLP.

    This block implements a standard transformer block with optional
    pre-layernorm (`pre_ln`). It supports replacing the dense MLP with
    a Mixture-of-Experts by setting `num_exp > 1`. When `lb_weights` is
    True the block will expose expert gating weights on the block's
    output for later aggregation or loss computation.
    """

    def __init__(
            self,
            d_emb: int,
            n_heads: int,
            mlp_factor: int = 4,
            act_fn: type[nn.Module] = nn.ReLU,
            att_dropout: float = 0.0,
            mlp_dropout: float = 0.0,
            pre_ln: bool = True,
            return_attscores: bool = False,
            lb_weights: bool = False,
            num_exp: int = 1,
    ) -> None:
        super().__init__()
        assert (not lb_weights) or (num_exp > 1), 'use num_exp > 1 to return weights'
        self.pre_ln = pre_ln
        self.return_attscores = return_attscores
        self.lb_weights = lb_weights

        self.mhsa = MultiHeadSelfAttention(d_emb, n_heads, self.return_attscores)
        if num_exp > 1:
            self.mlp = MoE(d_emb, [(mlp_factor * d_emb) // num_exp], d_emb, num_exp, return_weights=self.lb_weights)
        else:
            self.mlp = MLP(d_emb, [mlp_factor * d_emb], d_emb, act_fn, p_dropout=mlp_dropout)
        self.ln1 = nn.LayerNorm(d_emb)
        self.ln2 = nn.LayerNorm(d_emb)
        self.dropout = nn.Dropout(att_dropout)

    def forward(self, x: Tensor) -> tuple[Union[Tensor, Any], Any, list[Any]]:
        outputs = {}
        res = x
        if self.pre_ln:
            x = self.ln1(x)
        x = self.mhsa(x)
        if self.return_attscores:
            outputs['scores'] = x.scores
        x = res + self.dropout(x)
        if not self.pre_ln:
            x = self.ln1(x)

        res = x
        if self.pre_ln:
            x = self.ln2(x)
        x = self.mlp(x)
        if self.lb_weights:
            outputs['gate_weights'], outputs['gate_idx'] = x.gate_weights, x.gate_idx
        x = res + self.dropout(x)
        if not self.pre_ln:
            x = self.ln2(x)

        for k, v in outputs.items():
            x.__setattr__(k, v)
        return x


class ViT(nn.Module):
    """Minimal Vision Transformer (ViT) implementation.

    The ViT accepts images of shape [B, C, H, W], splits them into
    non-overlapping square patches of size `patch_dim`, projects each
    patch to `d_emb` dimensions, optionally prepends a learnable class
    token, adds positional encodings, and processes the sequence with
    `n_blocks` TransformerBlock modules. The final classification head
    is an MLP mapping the pooled token(s) to `n_class` logits.

    Parameters
    ----------
    x_dim:
        Tuple describing input batch shape template (unused batch dim
        first), e.g. `[B, C, H, W]` where B may be arbitrary. Only C,
        H, W are used to infer patch counts.
    patch_dim:
        Width/height of each square patch.
    d_emb, n_heads, n_blocks, n_class:
        Standard transformer sizes.
    class_token:
        If True, a learnable CLS token is prepended and used for
        classification.
    learned_encodings:
        If True, learnable positional embeddings are used; otherwise a
        fixed 2D sin/cos embedding is used (not learned).
    disable_head:
        If True, the final MLP head is disabled and the pooled token is
        returned directly (useful for feature extraction).

    Returns
    -------
    Tensor
        The forward pass returns either logits (if head enabled) or the
        feature tensor. Depending on flags the returned tensor may be
        annotated with attributes `.att_scores`, `.lb_loss`, or
        `.moe_scores`.
    """

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
            return_attscores: bool = False,
            return_moescores: bool = False,
            num_exp: int = 1,
            lb_loss: bool = False,
    ) -> None:
        super().__init__()
        assert (not lb_loss) or num_exp > 1, 'use num_exp > 1 to enable lb_loss'
        self.input_size = x_dim
        self.d_emb = d_emb
        self.return_attscores = return_attscores
        self.return_moescores = return_moescores
        self.disable_head = disable_head
        self.learned_encodings = learned_encodings
        self.lb_loss = lb_loss
        self.num_exp = num_exp

        _, self.x_c, self.x_h, self.x_w = x_dim
        self.p_dim = patch_dim
        self.class_token = class_token
        self.n_patch = self.x_h * self.x_w // (self.p_dim ** 2)
        assert (self.x_h % self.p_dim == 0) and (self.x_w % self.p_dim == 0), (
            "Input height/width should be divisible by patch_dim"
        )

        self.embedding = nn.Linear((self.p_dim ** 2) * self.x_c, d_emb)
        if class_token:
            self.cls = nn.Parameter(torch.zeros(1, 1, d_emb))
            nn.init.trunc_normal_(self.cls, std=0.02)
            self.n_patch += 1
        if learned_encodings:
            self.pos_emb = nn.Parameter(torch.zeros(1, self.n_patch, d_emb))
            nn.init.trunc_normal_(self.pos_emb, std=0.02)
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
                return_attscores=return_attscores,
                num_exp=num_exp,
                lb_weights=self.lb_loss
            )
            for _ in range(n_blocks)
        )
        self.output = MLP(d_emb, [d_emb * 2], n_class) if not disable_head else None

        self.apply(self._init_weights)

    def forward(self, x: Tensor) -> Union[
        Union[tuple[Union[Tensor, Any], Tensor], tuple[Union[Tensor, Any], float], Tensor], Any]:
        B = x.shape[0]
        x = self.patchify(x)
        x = self.embedding(x)
        if self.class_token:
            x = torch.cat([self.cls.expand(B, -1, -1), x], dim=1)
        x = x + self.pos_emb

        att_scores = []
        lb_losses = []
        moe_scores = []
        for block in self.blocks:
            x = block(x)

            if self.return_attscores:
                att_scores.append(x.scores)
            if self.lb_loss:
                lb_losses.append(block.mlp.load_balancing_loss(x.gate_weights, x.gate_idx, block.mlp.num_experts))
            if self.return_moescores:
                moe_scores.append(x.gate_idx)

        x = x[:, 0] if self.class_token else x.mean(dim=1)
        x = self.output(x) if not self.disable_head else x

        if self.return_attscores: x.__setattr__('att_scores', torch.stack(att_scores, dim=1))
        if self.lb_loss: x.__setattr__('lb_loss', (sum(lb_losses) / len(lb_losses)))
        if self.return_moescores: x.__setattr__('moe_scores', torch.stack(moe_scores, dim=1))
        return x

    def patchify(self, imgs: Tensor) -> Tensor:
        """
        Convert a batch of images to flattened patches.

        Args:
            imgs: Tensor of shape [B, C, H, W]

        Returns:
            patches: Tensor of shape [B, n_patches, patch_size^2 * C]
                     where n_patches = (H//p) * (W//p) and p = self.p_dim
        """
        # basic shape checks
        B, C, H, W = imgs.shape
        p = self.p_dim

        gh = H // p
        gw = W // p

        patches = imgs.reshape(B, C, gh, p, gw, p)
        patches = patches.permute(0, 2, 4, 3, 5, 1)
        patches = patches.reshape(B, gh * gw, p * p * C)
        return patches.contiguous()

    def depatchify(self, patches: Tensor) -> Tensor:
        """
        Inverse of patchify: reconstruct images from flattened patches.

        Args:
            patches: Tensor of shape [B, n_patches, patch_size^2 * C]

        Returns:
            imgs: Tensor of shape [B, C, H, W]
        """
        B, S, D = patches.shape
        p = self.p_dim
        C = self.x_c
        gh = self.x_h // p
        gw = self.x_w // p

        patches = patches.reshape(B, gh, gw, p, p, C)
        patches = patches.permute(0, 5, 1, 3, 2, 4)
        imgs = patches.reshape(B, C, gh * p, gw * p)
        return imgs.contiguous()

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Parameter):
            nn.init.trunc_normal_(m, std=0.02)


class DiT(nn.Module):
    """Placeholder for a diffusion transformer (DiT) variant.

    The class is currently empty and provided as a minimal placeholder
    for future extension. It intentionally does not implement forward
    logic yet.
    """

    def __init__(self):
        super(DiT, self).__init__()


if __name__ == '__main__':
    inp = torch.randn(64, 1, 28, 28)
    model = ViT([64, 1, 28, 28],
                7,
                64,
                4,
                2,
                10,
                return_attscores=False,
                disable_head=False,
                class_token=True,
                learned_encodings=False,
                num_exp=5,
                lb_loss=True)
    model(inp)
    print()