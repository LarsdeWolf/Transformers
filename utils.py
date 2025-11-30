import torch
import torch.nn.functional as F
import math


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    embed_dim: output dimension for each position (must be even)
    grid_size: int of the grid height and width
    cls_token: whether to add a [CLS] token embedding
    return:
        pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim]
    """
    # generate grid of positions
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")  # (2, grid_size, grid_size)
    grid = torch.stack(grid, dim=0)  # (2, grid_size, grid_size)
    grid = grid.reshape([2, 1, grid_size, grid_size])

    # get positional embeddings
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

    if cls_token:
        cls_pos_embed = torch.zeros([1, embed_dim])  # cls token has a fixed zero embedding
        pos_embed = torch.cat([cls_pos_embed, pos_embed], dim=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions for x, half for y
    emb_h = get_1d_sincos_pos_embed(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed(embed_dim // 2, grid[1])  # (H*W, D/2)
    emb = torch.cat([emb_h, emb_w], dim=1)  # (H*W, D)
    return emb

def get_1d_sincos_pos_embed(embed_dim, pos):
    """
    embed_dim: dimension of the positional embedding
    pos: torch.Tensor of shape (H, W)
    return: (H*W, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega = 1. / (10000 ** (omega / (embed_dim // 2)))

    pos = pos.reshape(-1)  # flatten
    out = torch.einsum('p,d->pd', pos, omega)  # (M, D/2)
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb


def patchify(imgs: torch.Tensor, p_dim: int) -> torch.Tensor:
    """
    Convert a batch of images to flattened patches.

    Args:
        imgs: Tensor of shape [B, C, H, W]
        p_dim: Patch dimension.

    Returns:
        patches: Tensor of shape [B, n_patches, patch_size^2 * C]
                 where n_patches = (H//p) * (W//p)
    """
    B, C, H, W = imgs.shape
    gh = H // p_dim
    gw = W // p_dim

    patches = imgs.reshape(B, C, gh, p_dim, gw, p_dim)
    patches = patches.permute(0, 2, 4, 3, 5, 1)
    patches = patches.reshape(B, gh * gw, p_dim * p_dim * C)
    return patches.contiguous()


def depatchify(patches: torch.Tensor, p_dim: int, x_c: int, x_h: int, x_w: int) -> torch.Tensor:
    """
    Inverse of patchify: reconstruct images from flattened patches.

    Args:
        patches: Tensor of shape [B, n_patches, patch_size^2 * C]
        p_dim: Patch dimension.
        x_c: Number of channels in the image.
        x_h: Height of the image.
        x_w: Width of the image.

    Returns:
        imgs: Tensor of shape [B, C, H, W]
    """
    B, S, D = patches.shape
    gh = x_h // p_dim
    gw = x_w // p_dim

    patches = patches.reshape(B, gh, gw, p_dim, p_dim, x_c)
    patches = patches.permute(0, 5, 1, 3, 2, 4)
    imgs = patches.reshape(B, x_c, gh * p_dim, gw * p_dim)
    return imgs.contiguous()

def time_embed(t, dim):
    half = dim // 2
    freqs = torch.exp(
        torch.arange(half, dtype=torch.float32, device=t.device) * (-math.log(10000) / (half - 1))
    )
    args = t[:, None].float() * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return emb

def to_img(x):
    x = x.detach().cpu().clamp(0, 1)
    if x.shape[0] == 1:
        return x[0].float().numpy() 
    else:
        return x.permute(1, 2, 0).float().numpy()
    
import torch
import math
from dataclasses import dataclass

@dataclass
class NoiseSchedulerConfig:
    T: int = 1000                # number of diffusion steps
    schedule: str = "cosine"     # "linear" or "cosine"
    beta_start: float = 1e-4     # for linear schedule
    beta_end: float = 2e-2       # for linear schedule


class NoiseScheduler(torch.nn.Module):
    """
    Provides betas, alphas, and alpha_bar for the diffusion process.
    Supports:
      - 'linear'  (DDPM)
      - 'cosine'  (Improved DDPM, Nichol & Dhariwal)
    """

    def __init__(self, config: NoiseSchedulerConfig):
        super().__init__()
        self.T = config.T

        if config.schedule == "linear":
            betas = torch.linspace(config.beta_start, config.beta_end, config.T)
        elif config.schedule == "cosine":
            betas = self._cosine_schedule(config.T)
        else:
            raise ValueError(f"Unknown schedule: {config.schedule}")

        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas", alphas.float())
        self.register_buffer("alpha_bar", alpha_bar.float())

    @staticmethod
    def _cosine_schedule(T):
        """
        Cosine schedule from:
        Improved Denoising Diffusion Probabilistic Models (Nichol & Dhariwal, 2021)
        https://arxiv.org/abs/2102.09672
        """

        steps = T + 1
        s = 0.008

        t = torch.linspace(0, T, steps) / T
        f_t = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2

        alpha_bar = f_t / f_t[0]      # normalize to 1 at step 0
        betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
        betas = betas.clamp(1e-6, 0.999)

        return betas

    def get_alpha_bar(self, t):
        return self.alpha_bar[t]
