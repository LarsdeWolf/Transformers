import torch
import torch.nn as nn
import lightning as L
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from model import ViT, MLP, MoE
from lit_model import *


# Dataset registry with transforms and metadata
DATASETS = {
    "flowers102": {
        "cls": datasets.Flowers102,
        "num_classes": 102,
        "resize": (128, 128),
        "mean": [0.4330, 0.3819, 0.2964],
        "std": [0.2923, 0.2439, 0.2712],
        "splits": ["train", "val", "test"],
    },
    "mnist": {
        "cls": datasets.MNIST,
        "num_classes": 10,
        "resize": (28, 28),
        "mean": [0.1307],
        "std": [0.3081],
        "splits": ["train", "test"],
    },
    "cifar10": {
        "cls": datasets.CIFAR10,
        "num_classes": 10,
        "resize": (32, 32),
        "mean": [0.4914, 0.4822, 0.4465],
        "std": [0.2023, 0.1994, 0.2010],
        "splits": ["train", "test"],
    },
}


def get_datasets(name: str, root: str = "./data"):
    """Return train/val/test datasets for a given dataset name."""
    cfg = DATASETS[name.lower()]

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(cfg["resize"]),
            transforms.Normalize(cfg["mean"], cfg["std"]),
        ]
    )

    datasets_out = {}
    if name.lower() == "flowers102":
        datasets_out["train"] = cfg["cls"](root=root, split="train", download=True, transform=transform)
        datasets_out["val"] = cfg["cls"](root=root, split="val", download=True, transform=transform)
        datasets_out["test"] = cfg["cls"](root=root, split="test", download=True, transform=transform)
    elif name.lower() in ["mnist", "cifar10"]:
        datasets_out["train"] = cfg["cls"](root=root, train=True, download=True, transform=transform)
        datasets_out["val"] = None  # no separate val, can split train
        datasets_out["test"] = cfg["cls"](root=root, train=False, download=True, transform=transform)

    return datasets_out, cfg


def main(dataset_name: str = "flowers102", log_graph=False):
    # Load datasets
    datasets_out, cfg = get_datasets(dataset_name)

    train_loader = DataLoader(datasets_out["train"], batch_size=128, shuffle=True, num_workers=6, persistent_workers=True, pin_memory=True)
    val_loader = (
        DataLoader(datasets_out["val"], batch_size=128, shuffle=False, num_workers=3, persistent_workers=True, pin_memory=True)
        if datasets_out["val"] is not None
        else None
    )
    test_loader = DataLoader(datasets_out["test"], batch_size=128, shuffle=False, num_workers=6, persistent_workers=True, pin_memory=True)

    # Determine input channels from dataset
    sample, _ = datasets_out["train"][0]
    c, h, w = list(sample.shape)


    lit_model = LitClassification(model= ViT(x_dim=[1, c, h, w],
                                             patch_dim=h // 7,
                                             d_emb=256,
                                             n_heads=1,
                                             n_blocks=1,
                                             n_class=cfg['num_classes'],
                                             class_token=True,
                                             num_exp=4
                                             ),
                                  loss_fn=nn.CrossEntropyLoss(),
                                  wd=.1)
    # lit_model = LitMaskedAutoEncoder(
    #     ViT([64, 1, 128, 128], 4, 500, 6,4, cfg['num_classes'], return_scores=False,
    #         disable_head=True, class_token=False, learned_encodings=False, num_exp=8),
    #     ViT([64, 3, 28, 28], 4, 256, 4, 4, 10, return_scores=False,
    #         disable_head=True, class_token=False, learned_encodings=False, num_exp=500),
    #     save_train=10
    # )

    # Trainer with early stopping
    early_stop = L.pytorch.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")
    logger = TensorBoardLogger("lightning_logs", name=dataset_name + '/' + lit_model.__class__.__name__, log_graph=log_graph)
    trainer = L.Trainer(
        logger=logger,
       #precision="bf16-mixed", #"16-mixed",
        enable_model_summary=True,
        max_epochs=100,
        enable_progress_bar=True,
        callbacks=[early_stop] if val_loader is not None else [],
    )

    trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(lit_model, dataloaders=test_loader)


if __name__ == "__main__":
    # Switch dataset here: "flowers102", "mnist", "cifar10"
    main("mnist")
