import torch
import torch.nn as nn
import lightning as L
from torch import Tensor
from typing import Any
import matplotlib.pyplot as plt
from model import ViT


class LitClassification(L.LightningModule):
    """Lightning module for training classification models."""

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module = nn.CrossEntropyLoss(),
        lr: float = 3e-4,
        wd: float = 0.3,
        epochs: int = 100,
        input_shape: tuple[int, int, int, int] = None,
        scheduler: nn.Module = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model", "loss_fn", "scheduler"])  
        self.model = model
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        
        if input_shape is not None:
            self.example_input_array = torch.randn(input_shape)
        self.lb = isinstance(self.model, ViT) and getattr(self.model, "lb_loss", False)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = batch
        x = self.forward(x)
        loss = self.loss_fn(x, y)
        acc = (x.argmax(dim=1) == y).float().mean()
        self.log("Loss/train", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("Accuracy/train", acc, on_step=True, on_epoch=True, prog_bar=True)

        if self.lb:
            loss = loss + x.lb_loss
            self.log('LB/train', x.lb_loss, on_step=True, on_epoch=True, prog_bar=False)
            if self.model.return_moescores:
                tb = self.logger.experiment
                for i in range(self.model.num_exp):
                    tb.add_scalar(f'Expert_Load/train-exp{i}',
                                  (x.moe_scores == i).sum().item(),
                                  self.global_step)
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        x, y = batch
        x = self.forward(x)
        loss = self.loss_fn(x, y)
        acc = (x.argmax(dim=1) == y).float().mean()
        self.log("Loss/val", loss, on_epoch=True, prog_bar=True)
        self.log("Accuracy/val", acc, on_epoch=True, prog_bar=True)

        if self.lb:
            self.log('LB/val', x.lb_loss, on_step=True, on_epoch=True, prog_bar=False)
            if self.model.return_moescores:
                tb = self.logger.experiment
                for i in range(self.model.num_exp):
                    tb.add_scalar(f'Expert_Load/val-exp{i}',
                                  (x.moe_scores == i).sum().item(),
                                  self.global_step)

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        x, y = batch
        x = self.forward(x)
        loss = self.loss_fn(x, y)
        acc = (x.argmax(dim=1) == y).float().mean()
        self.log("Loss/test", loss)
        self.log("Accuracy/test", acc)

        if self.lb:
            self.log('LB/test', x.lb_loss, on_step=True, on_epoch=True, prog_bar=False)
            if self.model.return_moescores:
                tb = self.logger.experiment
                for i in range(self.model.num_exp):
                    tb.add_scalar(f'Expert_Load/test-exp{i}',
                                  (x.moe_scores == i).sum().item(),
                                  self.global_step)

    def on_before_optimizer_step(self, optimizer):
        """Compute and log global gradient norm across all parameters."""
        # Stack all gradients into one vector and compute its L2 norm
        grads = [p.grad.detach().flatten() for p in self.model.parameters() if p.grad is not None]
        if grads:  # avoid empty list
            all_grads = torch.cat(grads)
            total_norm = all_grads.norm(2)  # L2 norm
            self.log("grad_norm/global", total_norm, on_step=True, on_epoch=True, prog_bar=False)

    def on_fit_start(self) -> None:
        if self.model.return_moescores:
            tb = self.logger.experiment
            tb.add_custom_scalars({
                "Expert_Load": {
                    "train": ["Multiline", [f'train-exp{x}' for x in range(self.model.num_exp)]],
                    "val": ["Multiline", [f'val-exp{x}' for x in range(self.model.num_exp)]],
                    "test": ["Multiline", [f'test-exp{x}' for x in range(self.model.num_exp)]],
                },
            })

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd
        )

        if self.scheduler is None:
            return {"optimizer": optimizer} 
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }




class LitMaskedAutoEncoder(L.LightningModule):
    def __init__(
            self,
            encoder: nn.Module,
            decoder: nn.Module,
            lr: float = 3e-3,
            wd: float = 0.05,
            scheduler: nn.Module = None,
            epochs: int = 100,
            mask_ratio: float = .6,
            save_train: int = 0,
            save_val: int = 0,
            save_test: int = 10,
    ) -> None:
        super(LitMaskedAutoEncoder, self).__init__()
        self.save_hyperparameters(ignore=["encoder", "decoder"])
        self.encoder = encoder
        self.decoder = decoder
        self.mask_ratio = mask_ratio
        self.scheduler = scheduler
        self.enc_lb = isinstance(self.encoder, ViT) and getattr(self.encoder, "lb_loss", False)
        self.dec_lb = isinstance(self.encoder, ViT) and getattr(self.encoder, "lb_loss", False)


        self.loss_fn = nn.MSELoss()
        self.enc_todec = nn.Linear(self.encoder.d_emb, self.decoder.d_emb)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder.d_emb))
        self.to_pix = nn.Linear(self.decoder.d_emb, (self.encoder.p_dim**2) * self.encoder.x_c)

        self.train_epoch_outputs = []
        self.val_epoch_outputs = []
        self.test_epoch_outputs = []

    def forward(self, x: Tensor) -> tuple[Any, Tensor, Tensor, int]:
        # ENCODER
        x = self.encoder.patchify(x)
        x = self.encoder.embedding(x) + self.encoder.pos_emb

        B, N, D = x.shape
        n_mask = int(N * self.mask_ratio)
        idx_shuffle = torch.rand(B, N, device=x.device).argsort(dim=1)
        idx_restore = idx_shuffle.argsort(dim=1)

        x = torch.gather(x, dim=1, index=idx_shuffle.unsqueeze(-1).expand(-1, -1, D))[:, n_mask:, :]
        enc_lb_losses = []
        enc_moe_scores = []
        for bl in self.encoder.blocks:
            x = bl(x)
            if self.enc_lb:
                enc_lb_losses.append(bl.mlp.load_balancing_loss(x.gate_weights, x.gate_idx, bl.mlp.num_experts))
                enc_moe_scores.append(x.gate_idx)

        # DECODER
        x = self.enc_todec(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], n_mask, 1)
        mask_tokens = torch.cat([mask_tokens, x], dim=1)
        x = torch.gather(mask_tokens, dim=1, index=idx_restore.unsqueeze(-1).expand(-1, -1, self.decoder.d_emb))
        x = x + self.decoder.pos_emb
        dec_lb_losses = []
        dec_moe_scores = []
        for bl in self.decoder.blocks:
            x = bl(x)
            if self.enc_lb:
                dec_lb_losses.append(bl.mlp.load_balancing_loss(x.gate_weights, x.gate_idx, bl.mlp.num_experts))
                dec_moe_scores.append(x.gate_idx)
        x = self.to_pix(x)

        if self.enc_lb: x.__setattr__('enc_lb_loss', (sum(enc_lb_losses) / len(enc_lb_losses)))
        if self.dec_lb: x.__setattr__('dec_lb_loss', (sum(dec_lb_losses) / len(dec_lb_losses)))
        if self.encoder.return_moescores: x.__setattr__('enc_moe_scores', torch.stack(enc_moe_scores, dim=1))
        if self.decoder.return_moescores: x.__setattr__('dec_moe_scores', torch.stack(dec_moe_scores, dim=1))

        return x, idx_shuffle, idx_restore, n_mask

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, _ = batch
        preds, idx_shuffle, idx_restore, n_mask = self.forward(x)
        # Select masked patches
        masked = torch.gather(preds,
                              dim=1,
                              index=idx_shuffle.unsqueeze(-1).expand(-1, -1, preds.shape[-1]))[:, :n_mask, :]

        gt = self.encoder.patchify(x)
        gt = torch.gather(gt,
                          dim=1,
                          index=idx_shuffle.unsqueeze(-1).expand(-1, -1, preds.shape[-1]))[:, :n_mask, :]
        loss = self.loss_fn(masked, gt)
        self.log("Loss/train", loss.item(), on_step=True, on_epoch=True, prog_bar=True)

        if self.enc_lb:
            self.log('LB/train-encoder', preds.enc_lb_loss, on_step=True, on_epoch=True, prog_bar=False)
            loss = loss + preds.enc_lb_loss
        if self.dec_lb:
            self.log('LB/train-decoder', preds.dec_lb_loss, on_step=True, on_epoch=True, prog_bar=False)
            loss = loss + preds.dec_lb_loss
        tb = self.logger.experiment
        if self.encoder.return_moescores:
            for i in range(self.encoder.num_exp):
                tb.add_scalar(f'Expert_Load/train-enc-exp{i}',
                              (preds.enc_moe_scores == i).sum().item(),
                              self.global_step)
                tb.add_scalar(f'Expert_Load/train-dec-exp{i}',
                              (preds.dec_moe_scores == i).sum().item(),
                              self.global_step)

        if len(self.train_epoch_outputs) < 1 and self.hparams.save_train > 0:
            preds = self.encoder.depatchify(preds)
            self.train_epoch_outputs.append([preds[:self.hparams.save_train].detach().cpu(),
                                                x[:self.hparams.save_train].detach().cpu()])
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        x, _ = batch
        preds, idx_shuffle, idx_restore, n_mask = self.forward(x)
        # Select masked patches
        masked = torch.gather(preds,
                              dim=1,
                              index=idx_shuffle.unsqueeze(-1).expand(-1, -1, preds.shape[-1]))[:, :n_mask, :]

        gt = self.encoder.patchify(x)
        gt = torch.gather(gt,
                          dim=1,
                          index=idx_shuffle.unsqueeze(-1).expand(-1, -1, preds.shape[-1]))[:, :n_mask, :]
        loss = self.loss_fn(masked, gt)
        self.log("Loss/val", loss.item(), on_epoch=True, prog_bar=True)


        if self.enc_lb: self.log('LB/val-encoder', preds.enc_lb_loss, on_step=True, on_epoch=True, prog_bar=False)
        if self.dec_lb: self.log('LB/val-decoder', preds.dec_lb_loss, on_step=True, on_epoch=True, prog_bar=False)
        tb = self.logger.experiment
        if self.encoder.return_moescores:
            for i in range(self.encoder.num_exp):
                tb.add_scalar(f'Expert_Load/val-enc-exp{i}',
                              (preds.enc_moe_scores == i).sum().item(),
                              self.global_step)
                tb.add_scalar(f'Expert_Load/val-dec-exp{i}',
                              (preds.dec_moe_scores == i).sum().item(),
                              self.global_step)

        if len(self.val_epoch_outputs) < 1 and self.hparams.save_val > 0:
            preds = self.encoder.depatchify(preds)
            self.val_epoch_outputs.append([preds[:self.hparams.save_val].detach().cpu(),
                                                x[:self.hparams.save_val].detach().cpu()])
        return loss

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        x, _ = batch
        preds, idx_shuffle, idx_restore, n_mask = self.forward(x)
        # Select masked patches
        masked = torch.gather(preds,
                              dim=1,
                              index=idx_shuffle.unsqueeze(-1).expand(-1, -1, preds.shape[-1]))[:, :n_mask, :]

        gt = self.encoder.patchify(x)
        gt = torch.gather(gt,
                          dim=1,
                          index=idx_shuffle.unsqueeze(-1).expand(-1, -1, preds.shape[-1]))[:, :n_mask, :]
        loss = self.loss_fn(masked, gt)
        self.log("Loss/test", loss.item())


        if self.enc_lb: self.log('LB/test-encoder', preds.enc_lb_loss, on_step=True, on_epoch=True, prog_bar=False)
        if self.dec_lb: self.log('LB/test-decoder', preds.dec_lb_loss, on_step=True, on_epoch=True, prog_bar=False)
        tb = self.logger.experiment
        if self.encoder.return_moescores:
            for i in range(self.encoder.num_exp):
                tb.add_scalar(f'Expert_Load/test-enc-exp{i}',
                              (preds.enc_moe_scores == i).sum().item(),
                              self.global_step)
                tb.add_scalar(f'Expert_Load/test-dec-exp{i}',
                              (preds.dec_moe_scores == i).sum().item(),
                              self.global_step)

        if len(self.test_epoch_outputs) < 1 and self.hparams.save_test > 0:
            preds = self.encoder.depatchify(preds)
            self.test_epoch_outputs.append([preds[:self.hparams.save_test].detach().cpu(),
                                                x[:self.hparams.save_test].detach().cpu()])

        return loss

    def on_train_epoch_end(self):
        if self.hparams.save_train > 0 and self.current_epoch % 5 == 0:
            all_preds = torch.cat([x[0] for x in self.train_epoch_outputs], dim=0)
            all_gt = torch.cat([x[1] for x in self.train_epoch_outputs], dim=0)
            tensorboard = self.logger.experiment
            fig, axs = plt.subplots(2, all_preds.shape[0])
            for i in range(self.hparams.save_train):
                axs[0, i].imshow(all_preds[i, 0].numpy())
                axs[1, i].imshow(all_gt[i, 0].numpy())
            tensorboard.add_figure(f'Train Epoch: {self.current_epoch}', plt.gcf())
            self.train_epoch_outputs.clear()  # free memory

    def on_validation_epoch_end(self):
        if self.hparams.save_val > 0 and self.current_epoch % 5 == 0:
            all_preds = torch.cat([x[0] for x in self.val_epoch_outputs], dim=0)
            all_gt = torch.cat([x[1] for x in self.val_epoch_outputs], dim=0)
            tensorboard = self.logger.experiment
            fig, axs = plt.subplots(2, all_preds.shape[0] + 1)
            print(f'{all_preds.shape[0]}')
            print(range(self.hparams.save_val))
            for i in range(self.hparams.save_val):
                axs[0, i].imshow(all_preds[i, 0].numpy())
                axs[1, i].imshow(all_gt[i, 0].numpy())
            tensorboard.add_figure(f'Val Epoch: {self.current_epoch}', plt.gcf())
            self.val_epoch_outputs.clear()  # free memory

    def on_test_end(self):
        if self.hparams.save_test > 0:
            all_preds = torch.cat([x[0] for x in self.test_epoch_outputs], dim=0)
            all_gt = torch.cat([x[1] for x in self.test_epoch_outputs], dim=0)
            tensorboard = self.logger.experiment
            fig, axs = plt.subplots(2, all_preds.shape[0] + 1)
            print(f'{all_preds.shape[0]}')
            print(self.hparams.save_test)
            for i in range(self.hparams.save_test):
                axs[0, i].imshow(all_preds[i, 0].numpy())
                axs[1, i].imshow(all_gt[i, 0].numpy())
            tensorboard.add_figure(f'Test Epoch: {self.current_epoch}', plt.gcf())
            self.test_epoch_outputs.clear()  # free memory

    def on_before_optimizer_step(self, optimizer):
        """Compute and log global gradient norm across all parameters."""
        # Stack all gradients into one vector and compute its L2 norm
        grads = [p.grad.detach().flatten() for p in self.parameters() if p.grad is not None]
        if grads:  # avoid empty list
            all_grads = torch.cat(grads)
            total_norm = all_grads.norm(2)  # L2 norm
            self.log("grad_norm/global", total_norm, on_step=True, on_epoch=True, prog_bar=False)

    def on_fit_start(self) -> None:
        if self.encoder.return_moescores:
            tb = self.logger.experiment
            tb.add_custom_scalars({
                "Expert_Load": {
                    "train_encoder": ["Multiline", [f'Expert_Load/train-enc-exp{x}' for x in range(self.encoder.num_exp)]],
                    "val_encoder": ["Multiline", [f'Expert_Load/val-enc-exp{x}' for x in range(self.encoder.num_exp)]],
                    "test_encoder": ["Multiline", [f'Expert_Load/test-enc-exp{x}' for x in range(self.encoder.num_exp)]],
                    "train_decoder": ["Multiline", [f'Expert_Load/train-dec-exp{x}' for x in range(self.decoder.num_exp)]],
                    "val_decoder": ["Multiline", [f'Expert_Load/val-dec-exp{x}' for x in range(self.decoder.num_exp)]],
                    "test_decoder": ["Multiline", [f'Expert_Load/test-dec-exp{x}' for x in range(self.decoder.num_exp)]],
                },
            })

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd
        )
        if self.scheduler is None:
            return {"optimizer": optimizer} 
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }


if __name__=='__main__':
    from model import *
    model = LitMaskedAutoEncoder(
        ViT([64, 1, 28, 28], 7, 64, 4,2, 10, return_scores=False,
            disable_head=True, class_token=False),
        ViT([64, 1, 28, 28], 7, 64, 4, 2, 10, return_scores=False,
            disable_head=True, class_token=False)
    )
    inp = torch.randn(64, 1, 28, 28)
    inp = torch.arange(28 * 28, dtype=torch.float32).reshape(1, 1, 28, 28).repeat(64, 1, 1, 1)
    out = model(inp)
    print()

