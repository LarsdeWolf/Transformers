import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import *
from lit_model import *
import numpy as np
import torch.nn as nn
import seaborn as sns
from matplotlib.patches import Rectangle
import torch


def visualize_attention_patterns(attention_scores, sample_idx=0, save_path=None):
    """
    Comprehensive attention visualization for ViT
    attention_scores: (batch, blocks, heads, seq, seq)
    """
    batch, blocks, heads, seq_len, _ = attention_scores.shape

    # Convert to numpy if tensor
    if torch.is_tensor(attention_scores):
        attn = attention_scores[sample_idx].detach().cpu().numpy()
    else:
        attn = attention_scores[sample_idx]

    # 1. HEATMAP GRID: All blocks and heads
    fig, axes = plt.subplots(blocks, heads, figsize=(3 * heads, 3 * blocks))
    if blocks == 1:
        axes = axes.reshape(1, -1)
    if heads == 1:
        axes = axes.reshape(-1, 1)

    fig.suptitle(f'Attention Patterns - Sample {sample_idx}', fontsize=16)

    for block in range(blocks):
        for head in range(heads):
            ax = axes[block, head] if blocks > 1 else axes[head]

            # Plot heatmap
            im = ax.imshow(attn[block, head], cmap='Blues', aspect='auto')
            ax.set_title(f'Block {block}, Head {head}')
            ax.set_xlabel('Key Position')
            ax.set_ylabel('Query Position')

            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046)

            # Highlight CLS token interactions (assuming position 0 is CLS)
            if seq_len > 1:
                # CLS row and column
                ax.add_patch(Rectangle((0, 0), seq_len, 1, fill=False, edgecolor='red', lw=2))
                ax.add_patch(Rectangle((0, 0), 1, seq_len, fill=False, edgecolor='red', lw=2))

    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}_grid.png", dpi=150, bbox_inches='tight')
    plt.show()


def attention_summary_plots(attention_scores, sample_idx=0, save_path=None):
    """
    Summary statistics and patterns
    """
    if torch.is_tensor(attention_scores):
        attn = attention_scores[sample_idx].detach().cpu().numpy()
    else:
        attn = attention_scores[sample_idx]

    batch, blocks, heads, seq_len, _ = attention_scores.shape

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Attention entropy per head/block
    entropy = -np.sum(attn * np.log(attn + 1e-12), axis=-1)  # (blocks, heads, seq)
    mean_entropy = np.mean(entropy, axis=-1)  # (blocks, heads)

    im1 = axes[0, 0].imshow(mean_entropy, cmap='viridis', aspect='auto')
    axes[0, 0].set_title('Mean Attention Entropy')
    axes[0, 0].set_xlabel('Head')
    axes[0, 0].set_ylabel('Block')
    plt.colorbar(im1, ax=axes[0, 0])

    # 2. CLS token attention (how much other tokens attend to CLS)
    cls_attention = attn[:, :, :, 0]  # (blocks, heads, seq)
    mean_cls_attn = np.mean(cls_attention, axis=-1)  # (blocks, heads)

    im2 = axes[0, 1].imshow(mean_cls_attn, cmap='Reds', aspect='auto')
    axes[0, 1].set_title('Mean Attention to CLS Token')
    axes[0, 1].set_xlabel('Head')
    axes[0, 1].set_ylabel('Block')
    plt.colorbar(im2, ax=axes[0, 1])

    # 3. Attention from CLS token (how CLS attends to patches)
    cls_from_attention = attn[:, :, 0, :]  # (blocks, heads, seq)
    mean_cls_from = np.mean(cls_from_attention, axis=-1)  # (blocks, heads)

    im3 = axes[0, 2].imshow(mean_cls_from, cmap='Greens', aspect='auto')
    axes[0, 2].set_title('Mean Attention from CLS Token')
    axes[0, 2].set_xlabel('Head')
    axes[0, 2].set_ylabel('Block')
    plt.colorbar(im3, ax=axes[0, 2])

    # 4. Attention distance analysis
    # Create distance matrix
    patch_size = int(np.sqrt(seq_len - 1))  # Assuming square patches + CLS
    distances = []

    for i in range(1, seq_len):  # Skip CLS token
        for j in range(1, seq_len):
            pos_i = [(i - 1) // patch_size, (i - 1) % patch_size]
            pos_j = [(j - 1) // patch_size, (j - 1) % patch_size]
            dist = abs(pos_i[0] - pos_j[0]) + abs(pos_i[1] - pos_j[1])  # Manhattan distance
            distances.append(dist)

    max_dist = max(distances) if distances else 1
    dist_bins = range(max_dist + 2)

    # Average attention by distance
    attn_by_distance = [[] for _ in range(max_dist + 1)]

    idx = 0
    for i in range(1, seq_len):
        for j in range(1, seq_len):
            if i != j:  # Skip self-attention
                dist = distances[idx] if idx < len(distances) else 0
                # Average across all blocks and heads
                avg_attn = np.mean(attn[:, :, i, j])
                attn_by_distance[dist].append(avg_attn)
                idx += 1

    mean_attn_by_dist = [np.mean(attn_list) if attn_list else 0 for attn_list in attn_by_distance]

    axes[1, 0].bar(range(len(mean_attn_by_dist)), mean_attn_by_dist)
    axes[1, 0].set_title('Attention vs Spatial Distance')
    axes[1, 0].set_xlabel('Manhattan Distance')
    axes[1, 0].set_ylabel('Mean Attention Score')

    # 5. Head specialization (attention pattern diversity)
    # Compute pairwise correlation between heads
    head_correlations = []
    for block in range(blocks):
        block_corrs = []
        for h1 in range(heads):
            for h2 in range(h1 + 1, heads):
                # Flatten attention matrices and compute correlation
                attn1 = attn[block, h1].flatten()
                attn2 = attn[block, h2].flatten()
                corr = np.corrcoef(attn1, attn2)[0, 1]
                block_corrs.append(corr)
        head_correlations.append(np.mean(block_corrs) if block_corrs else 0)

    axes[1, 1].bar(range(blocks), head_correlations)
    axes[1, 1].set_title('Head Similarity by Block')
    axes[1, 1].set_xlabel('Block')
    axes[1, 1].set_ylabel('Mean Pairwise Correlation')
    axes[1, 1].set_ylim(0, 1)

    # 6. Attention sparsity (how concentrated is attention?)
    # Gini coefficient as sparsity measure
    def gini_coefficient(x):
        x = np.sort(x.flatten())
        n = len(x)
        cumsum = np.cumsum(x)
        return (2 * np.sum((np.arange(1, n + 1) * x))) / (n * cumsum[-1]) - (n + 1) / n

    sparsity_scores = []
    for block in range(blocks):
        block_sparsity = []
        for head in range(heads):
            gini = gini_coefficient(attn[block, head])
            block_sparsity.append(gini)
        sparsity_scores.append(np.mean(block_sparsity))

    axes[1, 2].bar(range(blocks), sparsity_scores)
    axes[1, 2].set_title('Attention Sparsity by Block (Gini)')
    axes[1, 2].set_xlabel('Block')
    axes[1, 2].set_ylabel('Gini Coefficient')

    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}_summary.png", dpi=150, bbox_inches='tight')
    plt.show()


def plot_attention_rollout(attention_scores, sample_idx=0, save_path=None):
    """
    Attention rollout - shows effective attention from CLS to patches
    """
    if torch.is_tensor(attention_scores):
        attn = attention_scores[sample_idx].detach().cpu().numpy()
    else:
        attn = attention_scores[sample_idx]

    blocks, heads, seq_len, _ = attn.shape

    # Average across heads for each block
    attn_avg = np.mean(attn, axis=1)  # (blocks, seq, seq)

    # Compute rollout
    rollout = attn_avg[0]
    for i in range(1, blocks):
        rollout = np.matmul(rollout, attn_avg[i])

    # Extract CLS attention to patches (excluding CLS->CLS)
    cls_to_patches = rollout[0, 1:]  # Shape: (seq_len-1,)

    # Reshape to spatial grid (assuming square patches)
    patch_size = int(np.sqrt(len(cls_to_patches)))
    if patch_size * patch_size == len(cls_to_patches):
        attention_map = cls_to_patches.reshape(patch_size, patch_size)

        plt.figure(figsize=(8, 6))
        plt.imshow(attention_map, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title('CLS Token Attention Rollout')
        plt.xlabel('Patch X')
        plt.ylabel('Patch Y')
        if save_path:
            plt.savefig(f"{save_path}_att.png", dpi=150, bbox_inches='tight')
        plt.show()
    else:
        # If not square, plot as 1D
        plt.figure(figsize=(12, 4))
        plt.bar(range(len(cls_to_patches)), cls_to_patches)
        plt.title('CLS Token Attention Rollout')
        plt.xlabel('Patch Index')
        plt.ylabel('Attention Score')
        if save_path:
            plt.savefig(f"{save_path}_att.png", dpi=150, bbox_inches='tight')
        plt.show()


# Example usage:
if __name__ == "__main__":
    model = LitClassification.load_from_checkpoint('lightning_logs/mnist/ViT/version_0/checkpoints/epoch=19-step=4700.ckpt',
                                                   model=ViT([64, 1, 28, 28], 7, 64, 4,
                                                             2, 10, return_scores=True),
                                                   loss_fn=torch.nn.CrossEntropyLoss)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Resize(28)
    ])
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=False,
        transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=False,
        transform=transform
    )

    img, label = next(iter(test_loader))
    i = np.random.randint(0, 63)
    blk, h = 0, 0  # first sample, first block, first head
    pred, scores = model.model(img)
    # pred, scores = pred[i].argmax(), scores[i, blk, h].detach().cpu().numpy()
    # plt.figure(figsize=(6, 6))
    plt.imshow(img[i, 0])
    plt.show()
    # plt.imshow(scores, cmap="viridis")
    # plt.colorbar(label="Attention weight")
    # plt.title(f"Attention Map (Block {blk}, Head {h})")
    # plt.xlabel("Key sequence position")
    # plt.ylabel("Query sequence position")
    # plt.show()

    # Simulate attention scores (replace with your actual data)
    attention_scores = scores

    # Normalize to make proper attention (sum to 1 along last dim)

    # Run visualizations
    visualize_attention_patterns(attention_scores, sample_idx=i, save_path='./')
    attention_summary_plots(attention_scores, sample_idx=i, save_path='./')
    plot_attention_rollout(attention_scores, sample_idx=i, save_path='./')