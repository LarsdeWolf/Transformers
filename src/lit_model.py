import torch
import torch.nn as nn
import lightning as L
from torch import Tensor
from typing import Any
import matplotlib.pyplot as plt
from model import ViT
from utils import *

class LitClassification(L.LightningModule):
    """Lightning module for training classification models."""

    def __init__(
            self,
            encoder: nn.Module,
            loss_fn: nn.Module = nn.CrossEntropyLoss(),
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["encoder", "loss_fn"])
        self.model = encoder
        self.loss_fn = loss_fn
        self.lb = isinstance(self.model, ViT) and getattr(self.model, "lb_loss", False)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def _forward_step(self, batch, stage: str):
        x, y = batch
        x = self.forward(x)
        loss = self.loss_fn(x, y)
        acc = (x.argmax(dim=1) == y).float().mean()
        if self.lb:
            loss.lb = x.lb_loss
            tb = self.logger.experiment
            for i in range(self.model.num_exp):
                tb.add_scalar(f'Expert_Load/{stage}-exp{i}',
                              (x.moe_scores == i).sum().item(),
                              self.global_step)
        return loss, acc

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        loss, acc = self._forward_step(batch, 'train')
        self.log("Loss/train", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("Accuracy/train", acc, on_step=True, on_epoch=True, prog_bar=True)

        if self.lb:
            self.log('LB/train', loss.lb, on_step=True, on_epoch=True, prog_bar=False)
            loss = loss + loss.lb
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        loss, acc = self._forward_step(batch, 'val')
        self.log("Loss/val", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("Accuracy/val", acc, on_step=True, on_epoch=True, prog_bar=True)

        if self.lb:
            self.log('LB/val', loss.lb, on_step=True, on_epoch=True, prog_bar=False)

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        loss, acc = self._forward_step(batch, 'test')
        self.log("Loss/test", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("Accuracy/test", acc, on_step=True, on_epoch=True, prog_bar=True)

        if self.lb:
            self.log('LB/test', loss.lb, on_step=True, on_epoch=True, prog_bar=False)

    def on_before_optimizer_step(self, optimizer):
        """Compute and log global gradient norm across all parameters."""
        grads = [p.grad.detach().flatten() for p in self.model.parameters() if p.grad is not None]
        if grads: 
            all_grads = torch.cat(grads)
            total_norm = all_grads.norm(2)  
            self.log("grad_norm/global", total_norm, on_step=True, on_epoch=True, prog_bar=False)

    def on_fit_start(self) -> None:
        if self.lb:
            tb = self.logger.experiment
            tb.add_custom_scalars({
                "Expert_Load": {
                    "train": ["Multiline", [f'train-exp{x}' for x in range(self.model.num_exp)]],
                    "val": ["Multiline", [f'val-exp{x}' for x in range(self.model.num_exp)]],
                    "test": ["Multiline", [f'test-exp{x}' for x in range(self.model.num_exp)]],
                },
            })

    def configure_optimizers(self) -> dict[str, Any]:
        return super().configure_optimizers()


class LitMaskedAutoEncoder(L.LightningModule):
    def __init__(
            self,
            encoder: nn.Module,
            decoder: nn.Module,
            lr: float = 3e-3,
            wd: float = 0.05,
            scheduler: nn.Module = None,
            epochs: int = 100,
            mask_ratio: float = .5,
            save_train: int = 0,
            save_val: int = 0,
            save_test: int = 10,
            mean: list[float, float, float] = None,
            std: list[float, float, float] = None
    ) -> None:
        super(LitMaskedAutoEncoder, self).__init__()
        self.save_hyperparameters(ignore=["encoder", "decoder"])
        self.encoder = encoder
        self.decoder = decoder
        self.scheduler = scheduler
        self.mean = mean
        self.std = std  

        self.loss_fn = nn.MSELoss()
        self.enc_todec = nn.Linear(self.encoder.hidden_dim, self.decoder.hidden_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder.hidden_dim))
        self.to_pix = nn.Linear(self.decoder.hidden_dim, (self.encoder.p_dim ** 2) * self.encoder.x_c)

        self.train_epoch_outputs = [] if save_train > 0 else None
        self.val_epoch_outputs = [] if save_val > 0 else None
        self.test_epoch_outputs = [] if save_test > 0 else None

        self.enc_lb = isinstance(self.encoder, ViT) and getattr(self.encoder, "lb_loss", False)
        self.dec_lb = isinstance(self.decoder, ViT) and getattr(self.encoder, "lb_loss", False)

        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor):
        """
        Forward pass of a masked autoencoder.
        Returns:
            preds        -- [B, N, patch_dim]
            idx_shuffle  -- permutation indices used for masking
            idx_restore  -- inverse permutation
            n_mask       -- number of masked patches
        """

        x = patchify(x, self.encoder.p_dim)
        x = self.encoder.embedding(x)  
        B, N, D = x.shape

        if self.encoder.class_token:
            x = torch.cat([self.encoder.cls.expand(B, -1, -1), x], dim=1)  
        x = x + self.encoder.pos_emb

        n_mask = int(N * self.mask_ratio)
        idx_shuffle = torch.rand(B, N, device=x.device).argsort(dim=1)
        idx_restore = idx_shuffle.argsort(dim=1)

        # keep only visible patches
        if self.encoder.class_token:
            # Exclude class token from selection
            x = torch.cat([x[:, :1], torch.gather(x[:, 1:], dim=1, index=idx_shuffle[:, n_mask:].unsqueeze(-1).expand(-1, -1, D)
        )], dim=1)
        
        x = torch.gather(x, dim=1, index=idx_shuffle[:, n_mask:].unsqueeze(-1).expand(-1, -1, D)
            )  
        if self.enc_lb:
            enc_lb_losses, enc_gate_scores = [], []

        for block in self.encoder.blocks:
            x = block(x)
            if self.enc_lb:
                enc_lb_losses.append(block.mlp.load_balancing_loss(
                    x.gate_weights, x.gate_idx, block.mlp.num_experts
                ))
                enc_gate_scores.append(x.gate_idx)

        if self.encoder.class_token:
            x = x[:, 1:]  # Discard the class token

        x = self.enc_todec(x)  
        mask_tokens = self.mask_token.repeat(B, n_mask, 1)  
        x = torch.cat([mask_tokens, x], dim=1) 
        x = torch.gather(
            x,
            dim=1,
            index=idx_restore.unsqueeze(-1).expand(-1, -1, self.decoder.d_emb)
        )

        if self.decoder.class_token:
            x = torch.cat([self.decoder.cls.expand(B, -1, -1), x], dim=1)  
        x = x + self.decoder.pos_emb
        if self.dec_lb:
            dec_lb_losses, dec_gate_scores = [], []

        for block in self.decoder.blocks:
            x = block(x)
            if self.dec_lb:
                dec_lb_losses.append(block.mlp.load_balancing_loss(
                    x.gate_weights, x.gate_idx, block.mlp.num_experts
                ))
                dec_gate_scores.append(x.gate_idx)

        if self.decoder.class_token:
            x = x[:, 1:]  # Discard the decoder's class token

        x = self.to_pix(x)  

        if self.enc_lb:
            x.enc_lb_loss = sum(enc_lb_losses) / len(enc_lb_losses)
        if self.dec_lb:
            x.dec_lb_loss = sum(dec_lb_losses) / len(dec_lb_losses)
        if self.encoder.return_moescores:
            x.enc_moe_scores = torch.stack(enc_gate_scores, dim=1)
        if self.decoder.return_moescores:
            x.dec_moe_scores = torch.stack(dec_gate_scores, dim=1)

        return x, idx_shuffle, idx_restore, n_mask

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, _ = batch
        preds, idx_shuffle, idx_restore, n_mask = self.forward(x)
               # Select masked patches
        masked = torch.gather(preds,
                              dim=1,
                              index=idx_shuffle.unsqueeze(-1).expand(-1, -1, preds.shape[-1]))[:, :n_mask, :]

        gt = patchify(x, self.encoder.p_dim)
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
            preds = depatchify(preds, self.encoder.p_dim, self.encoder.x_c, self.encoder.x_h,
                               self.encoder.x_w)
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

        gt = patchify(x, self.encoder.p_dim)
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
            preds = depatchify(preds, self.encoder.p_dim, self.encoder.x_c, self.encoder.x_h,
                               self.encoder.x_w)
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

        gt = patchify(x, self.encoder.p_dim)
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
            preds = depatchify(preds, self.encoder.p_dim, self.encoder.x_c, self.encoder.x_h,
                               self.encoder.x_w)
            self.test_epoch_outputs.append([preds[:self.hparams.save_test].detach().cpu(),
                                            x[:self.hparams.save_test].detach().cpu()])

        return loss

    def on_train_epoch_end(self):
        if self.hparams.save_train > 0 and self.current_epoch % 5 == 0:
            all_preds = torch.cat([x[0] for x in self.train_epoch_outputs], dim=0)
            all_gt = torch.cat([x[1] for x in self.train_epoch_outputs], dim=0)

            if self.mean and self.std:
                all_preds = self.denormalize(all_preds, self.mean, self.std)
                all_gt = self.denormalize(all_gt, self.mean, self.std)

            tensorboard = self.logger.experiment
            fig, axs = plt.subplots(2, all_preds.shape[0])

            for i in range(min(self.hparams.save_train, all_preds.shape[0])):
                axs[0, i].imshow(self.to_img(all_preds[i]))
                axs[1, i].imshow(self.to_img(all_gt[i]))
                axs[0, i].axis("off")
                axs[1, i].axis("off")

            tensorboard.add_figure(f'Train Epoch: {self.current_epoch}', plt.gcf())
            self.train_epoch_outputs.clear() 

    def on_validation_epoch_end(self):
        if self.hparams.save_val > 0 and self.current_epoch % 5 == 0:
            all_preds = torch.cat([x[0] for x in self.val_epoch_outputs], dim=0)
            all_gt = torch.cat([x[1] for x in self.val_epoch_outputs], dim=0)

            if self.mean and self.std:
                all_preds = self.denormalize(all_preds, self.mean, self.std)
                all_gt = self.denormalize(all_gt, self.mean, self.std)

            tensorboard = self.logger.experiment
            fig, axs = plt.subplots(2, all_preds.shape[0] + 1)
            for i in range(min(self.hparams.save_val, all_preds.shape[0])):
                axs[0, i].imshow(self.to_img(all_preds[i]))
                axs[1, i].imshow(self.to_img(all_gt[i]))
                axs[0, i].axis("off")
                axs[1, i].axis("off")
            tensorboard.add_figure(f'Val Epoch: {self.current_epoch}', plt.gcf())
            self.val_epoch_outputs.clear()  

    def on_test_end(self):
        if self.hparams.save_test > 0:
            all_preds = torch.cat([x[0] for x in self.test_epoch_outputs], dim=0)
            all_gt = torch.cat([x[1] for x in self.test_epoch_outputs], dim=0)

            if self.mean and self.std:
                all_preds = self.denormalize(all_preds, self.mean, self.std)
                all_gt = self.denormalize(all_gt, self.mean, self.std)

            tensorboard = self.logger.experiment
            fig, axs = plt.subplots(2, all_preds.shape[0] + 1)
            for i in range(min(self.hparams.save_test, all_preds.shape[0])):
                axs[0, i].imshow(self.to_img(all_preds[i]))
                axs[1, i].imshow(self.to_img(all_gt[i]))
                axs[0, i].axis("off")
                axs[1, i].axis("off")
            tensorboard.add_figure(f'Test Epoch: {self.current_epoch}', plt.gcf())
            self.test_epoch_outputs.clear()  

    def on_before_optimizer_step(self, optimizer):
        grads = [p.grad.detach().flatten() for p in self.parameters() if p.grad is not None]
        if grads:  
            all_grads = torch.cat(grads)
            total_norm = all_grads.norm(2)  
            self.log("grad_norm/global", total_norm, on_step=True, on_epoch=True, prog_bar=False)

    def on_fit_start(self) -> None:
        if self.encoder.return_moescores:
            tb = self.logger.experiment
            tb.add_custom_scalars({
                "Expert_Load": {
                    "train_encoder": ["Multiline",
                                      [f'Expert_Load/train-enc-exp{x}' for x in range(self.encoder.num_exp)]],
                    "val_encoder": ["Multiline", [f'Expert_Load/val-enc-exp{x}' for x in range(self.encoder.num_exp)]],
                    "test_encoder": ["Multiline",
                                     [f'Expert_Load/test-enc-exp{x}' for x in range(self.encoder.num_exp)]],
                    "train_decoder": ["Multiline",
                                      [f'Expert_Load/train-dec-exp{x}' for x in range(self.decoder.num_exp)]],
                    "val_decoder": ["Multiline", [f'Expert_Load/val-dec-exp{x}' for x in range(self.decoder.num_exp)]],
                    "test_decoder": ["Multiline",
                                     [f'Expert_Load/test-dec-exp{x}' for x in range(self.decoder.num_exp)]],
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

    @staticmethod
    def denormalize(img_tensor, mean, std):
        """Denormalizes a tensor of images."""
        mean = torch.tensor(mean, device=img_tensor.device)
        std = torch.tensor(std, device=img_tensor.device)

        mean = mean.view(1, -1, 1, 1)
        std = std.view(1, -1, 1, 1)

        img_denorm = (img_tensor * std) + mean
        return img_denorm
    

class LitClassCondDiffusion(L.LightningModule):
    def __init__(self,
                 input_shape: tuple[int, int, int, int],
                 p_dim: int,
                 model: nn.Module,
                 n_class: int,
                 vae: nn.Module = None,
                 noise_scheduler: str = 'cosine',
                 T: int = 1000,           
                 save_train: int = 10,
                 save_val: int = 0,
                 save_test: int = 10,
                 **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["vae", "model", "noise_scheduler"])
        self.vae = vae
        self.model = model   
        noise_scheduler = NoiseScheduler(T, noise_scheduler, device=self.device)
        self.hidden_dim = model.hidden_dim

        _, self.C, self.H, self.W = input_shape
        
        self.label_emb = nn.Embedding(n_class, self.hidden_dim)
        self.patch_emb = nn.Linear((self.hparams.p_dim ** 2) * self.C, self.hidden_dim) 
        self.pos_enc = nn.Parameter(get_2d_sincos_pos_embed(self.hidden_dim, self.H // self.hparams.p_dim, False).unsqueeze(0),
                                    requires_grad=False)
        
        self.final_ln = nn.LayerNorm(self.hidden_dim, elementwise_affine=False)
        self.final_linear = nn.Linear(self.hidden_dim, (p_dim ** 2) * self.C)
        
        # Zero-init final layer (DiT paper recommendation)
        nn.init.constant_(self.final_linear.weight, 0)
        nn.init.constant_(self.final_linear.bias, 0)

        # Noise schedule
        self.alphas = noise_scheduler.alphas
        self.betas = noise_scheduler.betas
        self.alpha_bar = noise_scheduler.alpha_bar

        self.train_epoch_outputs = [] if save_train > 0 else None
        self.val_epoch_outputs = [] if save_val > 0 else None
        self.test_epoch_outputs = [] if save_test > 0 else None
        

    
    def forward(self, x_t, t, cond):
        """
        Predict noise from noisy image
        Args:
            x_t: noisy image [B, C, H, W]
            t: timestep [B]
            cond: class labels [B]
        Returns:
            predicted noise [B, C, H, W]
        """        
        # Patchify and embed
        x = patchify(x_t, self.hparams.p_dim)  # [B, N, patch_dim^2 * C]
        x = self.patch_emb(x)           # [B, N, D]
        x = x + self.pos_enc            # Add pos encoding
        
        # Time + class conditioning
        time_emb = time_embed(t, self.hidden_dim)  # [B, D]
        label_emb = self.label_emb(cond)            # [B, D]
        cond_emb = time_emb + 2.0 * label_emb     # [B, D]
        
        # Pass through DiT blocks
        x = self.model(x, cond_emb)  # [B, N, D]
        
        # Output: predict noise
        x = self.final_ln(x)          # [B, N, D]
        x = self.final_linear(x)      # [B, N, patch_dim^2 * C]
        
        # Depatchify
        eps_pred = depatchify(x, self.hparams.p_dim, self.C, self.H, self.W)  # [B, C, H, W]
        
        return eps_pred
    
    def apply_noise(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        
        abar = self.alpha_bar[t].view(-1, 1, 1, 1)
        x_t = torch.sqrt(abar) * x0 + torch.sqrt(1 - abar) * noise
        return x_t, noise

    @torch.no_grad()
    def sample(self, cond, steps=1000):
        """DDPM sampling"""
        B = cond.shape[0]
        device = cond.device
        
        # pure noise
        x_t = torch.randn(B, self.C, self.H, self.W, device=device)        
        for t in range(steps-1, -1, -1):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            eps_pred = self.forward(x_t, t_batch, cond)
            
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_bar[t]
            beta_t = self.betas[t]
            
            if t > 0:
                alpha_bar_prev = self.alpha_bar[t-1]
                mu = (1 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * eps_pred)
                beta_tilde = ((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * beta_t
                noise = torch.randn_like(x_t)
                x_t = mu + torch.sqrt(beta_tilde) * noise
            else:
                x_t = (1 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * eps_pred)
        
        x_t = self.vae.decode(x_t) if self.vae else x_t
        # Denormalize
        x_t = (x_t + 1) / 2  
        x_t = x_t.clamp(0, 1)
        return x_t

    def training_step(self, batch, batch_idx):
        x, cond = batch
        B = x.shape[0]
        x = self.vae.encode(x) if self.vae else x
        t = torch.randint(0, self.hparams.T, (B,), device=x.device, dtype=torch.long)
        x_t, noise = self.apply_noise(x, t)
        eps_pred = self.forward(x_t, t, cond)
        loss = F.mse_loss(eps_pred, noise)
        self.log("Loss/train", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Generate samples 
        if self.hparams.save_train > 0 and self.current_epoch % 25 == 0 and len(self.train_epoch_outputs) < 1:
            with torch.no_grad():
                preds = self.sample(cond[:self.hparams.save_train])
                gt_denorm = (batch[0][:self.hparams.save_train] * 0.5) + 0.5  # Denormalize GT
                self.train_epoch_outputs.append([
                    preds.detach().cpu(),
                    gt_denorm.detach().cpu()
                ])
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, cond = batch
        B = x.shape[0]

        # Encode to latent if using VAE
        x = self.vae.encode(x) if self.vae else x
        
        # Sample random timesteps
        t = torch.randint(0, self.hparams.T, (B,), device=x.device, dtype=torch.long)
        
        # Add noise
        x_t, noise = self.apply_noise(x, t)
        
        # Predict noise
        eps_pred = self.forward(x_t, t, cond)
        
        # Simple MSE loss on noise prediction
        loss = F.mse_loss(eps_pred, noise)
        
        self.log("Loss/val", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Generate samples periodically
        if self.hparams.save_val > 0 and self.current_epoch % 25 == 0 and len(self.val_epoch_outputs) < 1:
            with torch.no_grad():
                preds = self.sample(cond[:self.hparams.save_train])
                gt_denorm = (batch[0][:self.hparams.save_train] * 0.5) + 0.5  # Denormalize GT
                self.val_epoch_outputs.append([
                    preds.detach().cpu(),
                    gt_denorm.detach().cpu()
                ])
    
    def on_train_epoch_end(self):
        if self.hparams.save_train > 0 and self.current_epoch % 10 == 0 and self.train_epoch_outputs:
            all_preds = torch.cat([x[0] for x in self.train_epoch_outputs], dim=0)
            all_gt = torch.cat([x[1] for x in self.train_epoch_outputs], dim=0)

            tensorboard = self.logger.experiment
            n_imgs = min(self.hparams.save_train, all_preds.shape[0])
            fig, axs = plt.subplots(2, n_imgs, figsize=(2*n_imgs, 4))
            
            if n_imgs == 1:
                axs = axs.reshape(2, 1)

            for i in range(n_imgs):
                pred_img = to_img(all_preds[i])
                gt_img = to_img(all_gt[i])
                
                if self.C == 1:
                    axs[0, i].imshow(pred_img, cmap='gray', vmin=0, vmax=1)
                    axs[1, i].imshow(gt_img, cmap='gray', vmin=0, vmax=1)
                else:
                    axs[0, i].imshow(pred_img)
                    axs[1, i].imshow(gt_img)
                    
                axs[0, i].axis("off")
                axs[1, i].axis("off")
                if i == 0:
                    axs[0, i].set_title("Generated", fontsize=10)
                    axs[1, i].set_title("Ground Truth", fontsize=10)

            plt.tight_layout()
            tensorboard.add_figure(f'Samples/Epoch_{self.current_epoch}', fig, self.current_epoch)
            plt.close()
            self.train_epoch_outputs.clear() 
    

    def configure_optimizers(self) -> dict[str, Any]:
        return super().configure_optimizers()



if __name__ == '__main__':
    from model import *

    # model = LitMaskedAutoEncoder(
    #     ViT([64, 1, 28, 28], 7, 64, 4, 2, 10, return_scores=False,
    #         disable_head=True, class_token=False),
    #     ViT([64, 1, 28, 28], 7, 64, 4, 2, 10, return_scores=False,
    #         disable_head=True, class_token=False)
    # )
    shape = (64, 1, 32, 32)
    inp = torch.randn(shape)
    schedule = NoiseScheduler(NoiseSchedulerConfig())
    model = LitClassCondDiffusion(
        input_shape=shape,
        p_dim=4,
        vae=None,
        model=DiT(
            hidden_dim=192,
            n_heads=6,
            n_blocks=6,
            mlp_factor=4,
            act_fn=nn.GELU,
            att_dropout=0.0,
            mlp_dropout=0.0,
        ),
        n_class=10,
        noise_scheduler=schedule,

    )
    out = model.training_step((inp, torch.tensor(0)), 0)
    print()
