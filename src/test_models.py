import unittest
import torch
import torch.nn as nn
from model import MLP, MoE, MultiHeadSelfAttention, ViT, TransformerBlock
from lit_model import LitClassification, LitMaskedAutoEncoder
from utils import patchify, depatchify

class TestModels(unittest.TestCase):

    def setUp(self):
        self.batch_size = 2
        self.input_dim = 32
        self.hidden_dims = [64, 64]
        self.out_dim = 10
        self.num_experts = 4
        self.d_emb = 32
        self.n_heads = 4
        self.patch_dim = 4
        self.img_size = (1, 3, 32, 32) # B, C, H, W
        self.n_blocks = 2
        self.n_class = 10

    def test_mlp(self):
        model = MLP(self.input_dim, self.hidden_dims, self.out_dim)
        x = torch.randn(self.batch_size, self.input_dim)
        out = model(x)
        self.assertEqual(out.shape, (self.batch_size, self.out_dim))

    def test_moe(self):
        model = MoE(self.input_dim, self.hidden_dims, self.out_dim, self.num_experts, return_weights=True)
        x = torch.randn(self.batch_size, 32, self.input_dim)
        out = model(x)
        self.assertEqual(out.shape, (self.batch_size, 32, self.out_dim))
        self.assertTrue(hasattr(out, 'gate_weights'))
        self.assertTrue(hasattr(out, 'gate_idx'))
        
        # Test load balancing loss
        loss = MoE.load_balancing_loss(out.gate_weights, out.gate_idx, self.num_experts)
        self.assertTrue(isinstance(loss, torch.Tensor))

    def test_mhsa(self):
        model = MultiHeadSelfAttention(self.d_emb, self.n_heads, return_scores=True)
        x = torch.randn(self.batch_size, 10, self.d_emb) # B, S, D
        out = model(x)
        self.assertEqual(out.shape, (self.batch_size, 10, self.d_emb))
        self.assertTrue(hasattr(out, 'scores'))

    def test_vit(self):
        model = ViT(self.img_size, self.patch_dim, self.d_emb, self.n_heads, self.n_blocks, self.n_class, return_attscores=True)
        x = torch.randn(self.img_size)
        out = model(x)
        self.assertEqual(out.shape, (1, self.n_class)) # Batch size is 1 in img_size
        self.assertTrue(hasattr(out, 'att_scores'))

    def test_vit_patchify_depatchify(self):
        model = ViT(self.img_size, self.patch_dim, self.d_emb, self.n_heads, self.n_blocks, self.n_class)
        x = torch.randn(self.img_size)
        patches = patchify(x, model.p_dim)
        reconstructed = depatchify(patches, model.p_dim, model.x_c, model.x_h, model.x_w)
        self.assertEqual(x.shape, reconstructed.shape)
        # Note: reconstruction might not be exact due to potential lossy operations if any, 
        # but patchify/depatchify should be inverse for shape and content if no processing is done.
        # Here we just check shape and close values.
        self.assertTrue(torch.allclose(x, reconstructed, atol=1e-5))

    def test_lit_classification(self):
        vit = ViT(self.img_size, self.patch_dim, self.d_emb, self.n_heads, self.n_blocks, self.n_class)
        model = LitClassification(vit)
        x = torch.randn(self.img_size)
        out = model(x)
        self.assertEqual(out.shape, (1, self.n_class))

    def test_lit_masked_autoencoder(self):
        encoder = ViT(self.img_size, self.patch_dim, self.d_emb, self.n_heads, self.n_blocks, self.n_class, disable_head=True)
        decoder = ViT(self.img_size, self.patch_dim, self.d_emb, self.n_heads, self.n_blocks, self.n_class, disable_head=True)
        model = LitMaskedAutoEncoder(encoder, decoder)
        x = torch.randn(self.img_size)
        preds, idx_shuffle, idx_restore, n_mask = model(x)
        # Preds shape: [B, N, patch_dim] -> [B, N, p*p*C] actually? 
        # Let's check model.py forward: 
        # patches = self.encoder.patchify(x) -> [B, N, D_patch]
        # ...
        # x = self.to_pix(x) -> [B, N, D_patch]
        # So preds should be [B, N, D_patch]
        
        expected_patches = (self.img_size[2] // self.patch_dim) * (self.img_size[3] // self.patch_dim)
        patch_content_dim = (self.patch_dim ** 2) * self.img_size[1]
        
        self.assertEqual(preds.shape, (1, expected_patches, patch_content_dim))

    def test_gradients(self):
        # Test gradients for ViT
        model = ViT(self.img_size, self.patch_dim, self.d_emb, self.n_heads, self.n_blocks, self.n_class)
        x = torch.randn(self.img_size)
        y = torch.randint(0, self.n_class, (1,))
        
        out = model(x)
        loss = nn.CrossEntropyLoss()(out, y)
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Gradient for {name} is None")
                self.assertFalse(torch.isnan(param.grad).any(), f"Gradient for {name} contains NaN")
                # It's possible for some grads to be zero (e.g. ReLU dead zone), but unlikely for all.
                # We won't strictly assert non-zero for every single param, but maybe for the input embedding.
        
        self.assertNotEqual(model.embedding.weight.grad.abs().sum(), 0, "Embedding gradients are zero")

if __name__ == '__main__':
    unittest.main()
