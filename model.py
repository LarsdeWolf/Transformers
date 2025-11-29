import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Union, Any, Tuple
from utils import get_2d_sincos_pos_embed, patchify, depatchify


class MLP(nn.Module):
    """MLP Module

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
    """Mixture-of-Experts Module

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

        out = torch.zeros(x.shape[:-1] + (self.experts[0].out_dim,), device=x.device, dtype=x.dtype)
        for exp in range(self.num_experts):
            mask = (gate_idx == exp).any(dim=-1)
            if mask.any():
                tokens_for_expert = x[mask]
                expert_output = self.experts[exp](tokens_for_expert)
                weights_for_expert = gate_weights[mask][:, exp].unsqueeze(-1)
                out[mask] += weights_for_expert * expert_output

        if self.return_weights:
            out.__setattr__('gate_weights', gate_weights)
            out.__setattr__('gate_idx', gate_idx)

        return out

    @staticmethod
    def load_balancing_loss(gate_weights: torch.Tensor, gate_idx: torch.Tensor, num_experts: int):
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
    """Multi-head self-attention (MHSA)

    This module computes concatenated qkv via a single Linear projection,
    splits heads, applies attention with an optional mask,
    and projects back to the original embedding dimension.

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

    Returns
    -------
    Tensor
        Output of shape (B, S, D) for input of shape (B, S, D)
    """

    def __init__(
            self,
            hidden_dim: int,
            n_heads: int,
            return_scores: bool = False
    ) -> None:
        super().__init__()
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"
        self.qkv_dim = hidden_dim // n_heads
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.return_scores = return_scores

        self.to_qkv = nn.Linear(self.hidden_dim, 3 * self.hidden_dim)
        self.output = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.apply(self._init_weights)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> tuple[Tensor, Optional[Tensor]]:
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
    """Transformer block combining MHSA, residuals and MLP

    
    Implements a standard transformer block with optional
    pre-layernorm (`pre_ln`). Supports replacing the dense MLP with
    a Mixture-of-Experts by setting `num_exp > 1`. When `lb_weights` is
    True the block will expose expert gating weights on the block's
    output for later aggregation or loss computation.

    Parameters
    ----------
    hidden_dim:
        Total embedding dimension.
    n_heads:
        Number of attention heads.
    mlp_factor:
        Factor to determine the hidden dimension size of the MLP.
    act_fn:
        Activation function to use in the MLP.
    att_dropout:
        Dropout rate for the attention mechanism.
    mlp_dropout:
        Dropout rate for the MLP.
    pre_ln:
        If True, applies LayerNorm before the attention and MLP blocks.
    return_attscores:
        If True, attention scores are attached to the output.
    lb_weights:
        If True, enables load balancing loss calculation for MoE.
    num_exp:
        Number of experts to use in the MoE layer. If 1, a standard MLP is used.
 
    Returns
    -------
    Tensor
        The output tensor of shape (B, S, D). Depending on the configuration,
        this tensor may have attached attributes:
        - `.scores`: Attention scores from the MHSA module.
        - `.gate_weights`: Gating weights from the MoE layer.
        - `.gate_idx`: Expert indices chosen by the MoE gating network.
    """

    def __init__(
            self,
            hidden_dim: int,
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
        self.hidden_dim = hidden_dim
        self.pre_ln = pre_ln
        self.return_attscores = return_attscores
        self.lb_weights = lb_weights

        self.mhsa = MultiHeadSelfAttention(hidden_dim, n_heads, self.return_attscores)
        if num_exp > 1:
            self.mlp = MoE(hidden_dim, [(mlp_factor * hidden_dim) // num_exp], hidden_dim, num_exp, return_weights=self.lb_weights)
        else:
            self.mlp = MLP(hidden_dim, [mlp_factor * hidden_dim], hidden_dim, act_fn, p_dropout=mlp_dropout)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
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
    

class DiTBlock(TransformerBlock):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_dim, n_heads, mlp_factor=4, **block_kwargs):
        super().__init__(hidden_dim, n_heads, mlp_factor, **block_kwargs)
        self.adamlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim, bias=True)
        )
        self.ln1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, bias=False)
        self.ln2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, bias=False)

        nn.init.constant_(self.adamlp[-1].weight, 0)
        nn.init.constant_(self.adamlp[-1].bias, 0)

    def forward(self, x, cond):
        """
        x: [B, S, D]
        cond: [S, D]
        """
        outputs = {}
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.adamlp(cond).chunk(6, dim=-1)
        res = x
        x = self.ln1(x) * (1 + gamma1.unsqueeze(1)) + beta1.unsqueeze(1)
        x = self.mhsa(x)
        if self.return_attscores:
            outputs['scores'] = x.scores
        x = self.dropout(x) * (alpha1.unsqueeze(1)) + res
        res = x 
        x = self.ln2(x) * (1 + gamma2.unsqueeze(1)) + beta2.unsqueeze(1)
        x = self.mlp(x) 
        if self.lb_weights:
            outputs['gate_weights'], outputs['gate_idx'] = x.gate_weights, x.gate_idx
        x = self.dropout(x) * (alpha2.unsqueeze(1)) + res
        for k, v in outputs.items():
            x.__setattr__(k, v)
        return x


class ViT(nn.Module):
    """Vision Transformer (ViT) Module

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
        first), e.g., `[B, C, H, W]` where B may be arbitrary. Only C,
        H, and W are used to infer patch counts.
    patch_dim:
        Width/height of each square patch.
    hidden_dim:
        Total embedding dimension.
    n_heads:
        Number of attention heads.
    n_blocks:
        Number of transformer blocks.
    n_class:
        Number of output classes for the classification head.
    mlp_factor:
        Factor to determine the hidden dimension size of the MLP in transformer blocks.
    act_fn:
        Activation function to use in the MLP.
    att_dropout:
        Dropout rate for the attention mechanism.
    mlp_dropout:
        Dropout rate for the MLP.
    class_token:
        If True, a learnable CLS token is prepended and used for
        classification.
    learned_encodings:
        If True, learnable positional embeddings are used; otherwise a
        fixed 2D sin/cos embedding is used (not learned).
    disable_head:
        If True, the final MLP head is disabled, and the pooled token is
        returned directly (useful for feature extraction).
    return_attscores:
        If True, attention scores are attached to the output.
    return_moescores:
        If True, expert indices from MoE layers are attached to the output.
    num_exp:
        Number of experts to use in MoE layers. If 1, a standard MLP is used.
    lb_loss:
        If True, enables load balancing loss calculation for MoE layers.

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
            hidden_dim: int,
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
            DiT: bool = False,
    ) -> None:
        super().__init__()
        assert (not lb_loss) or num_exp > 1, 'use num_exp > 1 to enable lb_loss'
        self.input_size = x_dim
        self.hidden_dim = hidden_dim
        self.return_attscores = return_attscores
        self.return_moescores = return_moescores
        self.disable_head = disable_head
        self.learned_encodings = learned_encodings
        self.lb_loss = lb_loss
        self.num_exp = num_exp
        self.DiT = DiT

        _, self.x_c, self.x_h, self.x_w = x_dim
        self.p_dim = patch_dim
        self.class_token = class_token
        self.n_patch = self.x_h * self.x_w // (self.p_dim ** 2)
        assert (self.x_h % self.p_dim == 0) and (self.x_w % self.p_dim == 0), (
            "Input height/width should be divisible by patch_dim"
        )

        self.embedding = nn.Linear((self.p_dim ** 2) * self.x_c, hidden_dim)
        if class_token:
            self.cls = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            nn.init.trunc_normal_(self.cls, std=0.02)
            self.n_patch += 1
        if learned_encodings:
            self.pos_emb = nn.Parameter(torch.zeros(1, self.n_patch, hidden_dim))
            nn.init.trunc_normal_(self.pos_emb, std=0.02)
        else:
            self.pos_emb = nn.Parameter(get_2d_sincos_pos_embed(hidden_dim, self.x_h // self.p_dim, self.class_token),
                                        requires_grad=False)
        mod = TransformerBlock if not DiT else DiTBlock
        self.blocks = nn.ModuleList(
            mod(
                hidden_dim=hidden_dim,
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
        self.output = MLP(hidden_dim, [hidden_dim * 2], n_class) if not disable_head else None

        self.apply(self._init_weights)

    def forward(self, x: Tensor) -> Union[
        Union[tuple[Union[Tensor, Any], Tensor], tuple[Union[Tensor, Any], float], Tensor], Any]:
        B = x.shape[0]
        x = patchify(x, self.p_dim)
        x = self.embedding(x)
        if self.class_token:
            x = torch.cat([self.cls.expand(B, -1, -1), x], dim=1)
        x = x + self.pos_emb

        outputs = {'att_scores': [], 'lb_losses': [], 'moe_scores': []}
        for block in self.blocks:
            x = block(x) if not self.DiT else block(x, x[0, 0, :])

            if self.return_attscores:
                outputs['att_scores'].append(x.scores)
            if self.lb_loss:
                loss = block.mlp.load_balancing_loss(x.gate_weights, x.gate_idx, block.mlp.num_experts)
                outputs['lb_losses'].append(loss)
            if self.return_moescores:
                outputs['moe_scores'].append(x.gate_idx)

        x = x[:, 0] if self.class_token else x.mean(dim=1)
        x = self.output(x) if not self.disable_head else x

        if self.return_attscores: x.__setattr__('att_scores', torch.stack(outputs['att_scores'], dim=1))
        if self.lb_loss: x.__setattr__('lb_loss', (sum(outputs['lb_losses']) / len(outputs['lb_losses'])))
        if self.return_moescores: x.__setattr__('moe_scores', torch.stack(outputs['moe_scores'], dim=1))
        return x

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Parameter):
            nn.init.trunc_normal_(m, std=0.02)

class DiT(nn.Module):
    """
    Full DiT model with multiple DiTBlocks
    """
    def __init__(self, hidden_dim, n_heads, n_blocks, mlp_factor=4, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, n_heads, mlp_factor, **kwargs)
            for _ in range(n_blocks)
        ])
    
    def forward(self, x, cond):
        """
        x: [B, S, D] - sequence of patch embeddings
        cond: [B, D] - conditioning vector (time + class)
        """
        for block in self.blocks:
            x = block(x, cond)
        return x


                 


if __name__ == '__main__':
    # inp = torch.randn(64, 1, 28, 28)
    # model = ViT([64, 1, 28, 28],
    #             7,
    #             64,
    #             4,
    #             2,
    #             10,
    #             return_attscores=False,
    #             disable_head=False,
    #             class_token=True,
    #             learned_encodings=False,
    #             num_exp=5,
    #             lb_loss=True,
    #             DiT=True)
    inp = torch.randn(13, 64, 256)
    cond = torch.randn(64, 256)
    model = DiTBlock(256, 4)
    out = model(inp, cond)
    print(out.shape)
