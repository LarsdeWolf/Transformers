import torch
import torch.nn as nn
import lightning as L
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.cli import LightningCLI
from lit_model import LitClassification, LitClassCondDiffusion, LitMaskedAutoEncoder
from model import *
from data import DataModule
from utils import *


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
        "resize": (32, 32),
        "mean": [.5],
        "std": [.5],
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
        *AUGMENTS, # Ensure AUGMENTS is empty for MNIST
        v2.Normalize(cfg["mean"], cfg["std"])
        # v2.Normalize([0.5] * len(cfg["mean"]), [0.5] * len(cfg["std"]))
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

def main_():
    cli = LightningCLI()

def main(model, config, dataset_name: str, n_batch: int, n_workers: int, n_train: int, log_graph=False):
    datasets_out, cfg = get_datasets(dataset_name)
    if n_train != 0:
        datasets_out['train'] = torch.utils.data.Subset(datasets_out['train'],
                                                        torch.randperm(len(datasets_out["train"]))[:n_train])
    train_loader = DataLoader(datasets_out["train"], batch_size=n_batch, shuffle=True, num_workers=n_workers,
                              persistent_workers=True, pin_memory=False)
    val_loader = (
        DataLoader(datasets_out["val"], batch_size=n_batch, shuffle=False, num_workers=n_workers,
                   persistent_workers=True, pin_memory=False)
        if datasets_out["val"] is not None
        else None
    )
    test_loader = DataLoader(datasets_out["test"], batch_size=n_batch, shuffle=False, num_workers=n_workers,
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
    elif model == LitMaskedAutoEncoder:
        model = model(
            encoder=config['Encoder'][0](x_dim=[1, c, h, w], n_class=cfg['num_classes'], **config['Encoder'][1]),
            decoder=config['Decoder'][0](x_dim=[1, c, h, w], n_class=cfg['num_classes'], **config['Decoder'][1]),
            lr=LR,
            wd=WD,
            epochs=N_EPOCH,
            scheduler=SCHEDULER,
            **config['MISC']
        )
    else:
        model = model(
            input_shape = [1, c, h, w],
            scheduler=SCHEDULER,
            **config
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
    N_BATCH = 32
    N_EPOCH = 120
    N_WORKERS = 3

    # DATA PARAMS
    DNAME = 'mnist'
    N_TRAIN = 200
    AUGMENTS = []

    # LEARN PARAMS
    LR = 3e-4
    WD = .0
    SCHEDULER = torch.optim.lr_scheduler.CosineAnnealingLR

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

    model, config = LitClassCondDiffusion, {
        'p_dim': 8,
        'vae': None,
        'model': DiT(
            hidden_dim=80,
            n_heads=2,
            n_blocks=2,
            mlp_factor=2,
            act_fn=nn.GELU,
            att_dropout=0.0,
            mlp_dropout=0.0,
        ), # Ensure hidden_dim is divisible by n_heads
        'n_class': 10,
        'noise_scheduler': NoiseScheduler(NoiseSchedulerConfig()),
        'save_train': 10, 
    }

    # model, config = LitMaskedAutoEncoder, {
    #     'Encoder': (ViT, {
    #         'd_emb': 384,
    #         'patch_dim': 4,
    #         'n_heads': 2,
    #         'n_blocks': 2,
    #         'class_token': False,
    #         'disable_head': True,
    #         'num_exp': 1,
    #         'lb_loss': False,
    #         'return_moescores': False,
    #         'att_dropout': .05,
    #         'mlp_dropout': .05,
    #     }),
    #     'Decoder': (ViT, {
    #         'd_emb': 192,
    #         'patch_dim': 4,
    #         'n_heads': 2,
    #         'n_blocks': 2,
    #         'class_token': False,
    #         'disable_head': True,
    #         'num_exp': 1,
    #         'lb_loss': False,
    #         'return_moescores': False,
    #         'att_dropout': .05,
    #         'mlp_dropout': .05,
    #     }),
    #     'MISC': {'save_train': 10, 'mean': DATASETS[DNAME]['mean'], 'std': DATASETS[DNAME]['std']}
    # }
    # main_()

    main(
        model=model,
        config=config,
        dataset_name=DNAME,
        n_batch=N_BATCH,
        n_workers=N_WORKERS,
        n_train=N_TRAIN
    )
