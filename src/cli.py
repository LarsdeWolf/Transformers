import torch
import os
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor, EMAWeightAveraging
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.cli import LightningCLI
from lit_model import LitClassification, LitClassCondDiffusion, LitMaskedAutoEncoder
from data import DataModule


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # Early stopping + LR Monitor + model checkpoints
        parser.add_lightning_class_args(EarlyStopping, "early_stopping")
        parser.set_defaults({"early_stopping.monitor": "Loss/val", "early_stopping.patience": 50})
        parser.add_lightning_class_args(LearningRateMonitor, "learning_rate")
        parser.add_lightning_class_args(ModelCheckpoint, "model_checkpoint")
        parser.add_lightning_class_args(EMAWeightAveraging, "ema_averaging")
        parser.set_defaults({"ema_averaging.update_starting_at_epoch": 10})
        parser.set_defaults({
            "model_checkpoint.monitor": "Loss/val",
            "model_checkpoint.filename": "epoch{epoch:03d}",
            "model_checkpoint.save_top_k": 3,
            "model_checkpoint.every_n_epochs": 5,
        })

    def before_fit(self):
        # Ensures consistent logging dir
        data_name = self.datamodule.hparams.name
        model_name = self.model.__class__.__name__
        save_dir = os.path.join("logs", data_name, model_name)
        logger = TensorBoardLogger(save_dir=save_dir, name="", version=None)
        self.trainer.logger = logger
        for cb in self.trainer.callbacks:
            if isinstance(cb, ModelCheckpoint):
                cb.dirpath = os.path.join(self.trainer.logger.log_dir, "checkpoints")
                os.makedirs(cb.dirpath, exist_ok=True)
            
def main_():
    CLI(trainer_defaults={
        'precision': '16-mixed',
        'logger': None #{'class_path': 'lightning.pytorch.loggers.TensorBoardLogger', 'init_args': {'save_dir': ''}}
    })
    torch.set_float32_matmul_precision('medium')


if __name__ == "__main__":
    main_()
