import torch


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

