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
        "resize": (224, 224),
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
    import inspect

    cfg = DATASETS[name.lower()]
    transform = v2.Compose([
        v2.ToTensor(),
        v2.Resize(cfg["resize"]),
        *AUGMENTS,
        v2.Normalize(cfg["mean"], cfg["std"]),
    ])

    def make_split(split_name: str):
        """Instantiate the dataset for the given split name.

        Some torchvision datasets (e.g. Flowers102) expect `split=` while
        others (MNIST, CIFAR10) expect `train=` boolean. Detect the
        constructor signature and call with the appropriate kwarg.
        """
        cls = cfg["cls"]
        sig = inspect.signature(cls.__init__)
        params = sig.parameters
        if 'split' in params:
            return cls(root=root, split=split_name, download=True, transform=transform)
        elif 'train' in params:
            train_flag = True if split_name == 'train' else False
            return cls(root=root, train=train_flag, download=True, transform=transform)

    datasets_out = {
        "train": make_split("train"),
        "test": make_split("test"),
    }

    # Ensure 'val' key exists so callers don't KeyError; set to None when not available
    datasets_out["val"] = make_split("val") if 'val' in cfg.get('splits', []) else None

    return datasets_out, cfg


def main(model, config, dataset_name: str = "flowers102", log_graph=False):
    datasets_out, cfg = get_datasets(dataset_name)
    if N_TRAIN != 0:
        datasets_out['train'] = torch.utils.data.Subset(datasets_out['train'],
                                                        torch.randperm(len(datasets_out["train"]))[:N_TRAIN])
    train_loader = DataLoader(datasets_out["train"], batch_size=N_BATCH, shuffle=True, num_workers=N_WORKERS,
                              persistent_workers=True, pin_memory=False)
    val_loader = (
        DataLoader(datasets_out["val"], batch_size=N_BATCH, shuffle=False, num_workers=N_WORKERS,
                   persistent_workers=True, pin_memory=False)
        if datasets_out["val"] is not None
        else None
    )
    test_loader = DataLoader(datasets_out["test"], batch_size=N_BATCH, shuffle=False, num_workers=N_WORKERS,
                             persistent_workers=True, pin_memory=False)

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

    logger = TensorBoardLogger("lightning_logs", name=dataset_name + '/' + model.__class__.__name__,
                               log_graph=log_graph)
    monitor = 'Loss/val' if val_loader else 'Loss/train'
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
        precision="bf16-mixed",  # "16-mixed",
        enable_model_summary=True,
        max_epochs=N_EPOCH,
        enable_progress_bar=True,
        callbacks=[checkpoint_callback, early_stop] if val_loader is not None else [checkpoint_callback],
    )
    torch.set_float32_matmul_precision('medium')

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(model, dataloaders=test_loader)


if __name__ == "__main__":
    # TRAIN PARAMS
    N_BATCH = 100
    N_EPOCH = 500
    N_WORKERS = 5

    # DATA PARAMS
    DNAME = 'cifar10'
    N_TRAIN = 500
    AUGMENTS = []

    # LEARN PARAMS
    LR = 3e-4
    WD = .05
    SCHEDULER = None

    # model, config = LitClassification, {
    #     'Encoder': (ViT, {
    #         'patch_dim': 4,
    #         'd_emb': 480,
    #         'n_heads': 6,
    #         'n_blocks': 4,
    #         'class_token': False,
    #         'num_exp': 1,
    #         'lb_loss': False,
    #         'return_attscores': True,
    #         'return_moescores': False,
    #         'att_dropout': .1,
    #         'mlp_dropout': .1,
    #         })
    # }

    model, config = LitMaskedAutoEncoder, {
        'Encoder': (ViT, {
            'd_emb': 384,
            'patch_dim': 4,
            'n_heads': 6,
            'n_blocks': 8,
            'class_token': False,
            'disable_head': True,
            'num_exp': 1,
            'lb_loss': False,
            'return_moescores': False,
            'att_dropout': .05,
            'mlp_dropout': .05,
        }),
        'Decoder': (ViT, {
            'd_emb': 192,
            'patch_dim': 4,
            'n_heads': 4,
            'n_blocks': 4,
            'class_token': False,
            'disable_head': True,
            'num_exp': 1,
            'lb_loss': False,
            'return_moescores': False,
            'att_dropout': .05,
            'mlp_dropout': .05,
        }),
        'MISC': {'save_train': 10, 'mean': DATASETS[DNAME]['mean'], 'std': DATASETS[DNAME]['std']}
    }

    main(model, config, DNAME)
