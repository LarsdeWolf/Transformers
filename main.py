import torch
import torch.nn as nn
import lightning as L
from torchvision import datasets
from torchvision.transforms import v2

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
    transform = v2.Compose([
        v2.ToTensor(),
        v2.Resize(cfg["resize"]),
        *AUGMENTS,
        v2.Normalize(cfg["mean"], cfg["std"]),
    ])

    datasets_out = {
        "train": cfg["cls"](root=root, split="train", download=True, transform=transform),
        "test": cfg["cls"](root=root, split="test", download=True, transform=transform)
    }
    if 'val' in cfg['splits']:
        datasets_out["val"] = cfg["cls"](root=root, split="val", download=True, transform=transform)

    return datasets_out, cfg


def main(model, config, dataset_name: str = "flowers102", log_graph=False):
    datasets_out, cfg = get_datasets(dataset_name)
    train_loader = DataLoader(datasets_out["train"], batch_size=N_BATCH, shuffle=True, num_workers=N_WORKERS, persistent_workers=True, pin_memory=False)
    val_loader = (
        DataLoader(datasets_out["val"], batch_size=N_BATCH, shuffle=False, num_workers=N_WORKERS, persistent_workers=True, pin_memory=False)
        if datasets_out["val"] is not None
        else None
    )
    test_loader = DataLoader(datasets_out["test"], batch_size=N_BATCH, shuffle=False, num_workers=N_WORKERS, persistent_workers=True, pin_memory=False)

    sample, _ = datasets_out["train"][0]
    c, h, w = list(sample.shape)
    if model == LitClassification:
        encoder, config = config['Encoder']
        model = model(
            model=encoder(x_dim=[1, c, h, w], n_class=cfg['num_classes'], **config),
            lr=LR,
            wd=WD,
            epochs=N_EPOCH,
            scheduler=SCHEDULER
        )
    else:
        model = model(
            encoder=config['Encoder'][0](x_dim=[1, c, h, w], n_class=cfg['num_classes'], **config['Encoder'][1]),
            decoder=config['Decoder'][0](x_dim=[1, c, h, w], n_class=cfg['num_classes'], **config['Decoder'][1]),
            lr=LR,
            wd=WD,
            epochs=N_EPOCH,
            scheduler=SCHEDULER,
            **config['MISC']
        )


    logger = TensorBoardLogger("lightning_logs", name=dataset_name + '/' + model.__class__.__name__, log_graph=log_graph)
    monitor = 'val_loss' if val_loader else 'train_loss'
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        dirpath=f"{logger.log_dir}/checkpoints/",          
        filename="epoch{epoch:03d}",    
        save_top_k=3,                 
        every_n_epochs=10,            
        monitor=monitor
    )
    early_stop = L.pytorch.callbacks.EarlyStopping(monitor=monitor, patience=15, mode="min")
    trainer = L.Trainer(
        logger=logger,
       #precision="bf16-mixed", #"16-mixed",
        enable_model_summary=True,
        max_epochs=N_EPOCH,
        enable_progress_bar=True,
        callbacks=[checkpoint_callback, early_stop] if val_loader is not None else [checkpoint_callback],
    )

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(model, dataloaders=test_loader)


if __name__ == "__main__":
    # TRAIN PARAMS
    N_BATCH = 32
    N_EPOCH = 5
    N_WORKERS = 3

    # DATA PARAMS
    N_TRAIN = 0
    AUGMENTS = []

    # LEARN PARAMS
    LR = 3e-4
    WD = .05
    SCHEDULER = None 

    # model, config = LitClassification, {
    #     'Encoder': (ViT, {
    #         'patch_dim': 7,
    #         'd_emb': 80,
    #         'n_heads': 2,
    #         'n_blocks': 2,
    #         'class_token': False,
    #         'num_exp': 5,
    #         'lb_loss': True,
    #         'return_attscores': True,
    #         'return_moescores': True,
    #         'att_dropout': .1,
    #         'mlp_dropout': .1,
    #         })
    # }

    model, config = LitMaskedAutoEncoder, {
        'Encoder': (ViT, {
            'd_emb': 140,
            'patch_dim': 4,
            'n_heads': 2,
            'n_blocks': 2,
            'class_token': False,
            'disable_head': True,
            'num_exp': 4,
            'lb_loss': True,
            'return_moescores': True,
            'att_dropout': .1,
            'mlp_dropout': .1,
        }),
        'Decoder': (ViT, {
            'd_emb': 140,
            'patch_dim': 4,
            'n_heads': 2,
            'n_blocks': 2,
            'class_token': False,
            'disable_head': True,
            'num_exp': 4,
            'lb_loss': True,
            'return_moescores': True,
            'att_dropout': .1,
            'mlp_dropout': .1,
        }),
        'MISC': {'save_train': 10}
    }
    
    main(model, config, "mnist")
