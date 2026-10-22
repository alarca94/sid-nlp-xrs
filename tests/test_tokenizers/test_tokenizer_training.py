"""
Unit tests for tokenizer training components: Encoder, Decoder,
ResidualVectorQuantizer, and RQVAEModel.
"""

import os
import tempfile
import unittest
from types import SimpleNamespace

import torch

from nlp_xrs.semantic_ids.quantizer import ResidualVectorQuantizer
from nlp_xrs.semantic_ids.rqvae_model import MLP, RQVAEModel

# Disable tokenizers parallelism to avoid warnings in tests
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TestEncoder(unittest.TestCase):
    def setUp(self):
        self.encoder_dims = [32, 16, 8]
        self.batch_size = 4

    def test_encoder_forward_shape(self):
        encoder = MLP(self.encoder_dims)
        x = torch.randn(self.batch_size, self.encoder_dims[0])
        z = encoder(x)
        self.assertEqual(z.shape, (self.batch_size, self.encoder_dims[-1]))

    def test_encoder_gradient_flow(self):
        encoder = MLP(self.encoder_dims)
        x = torch.randn(self.batch_size, self.encoder_dims[0], requires_grad=True)
        z = encoder(x)
        loss = z.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)


class TestDecoder(unittest.TestCase):
    def setUp(self):
        self.decoder_dims = [8, 16, 32]
        self.batch_size = 4

    def test_decoder_forward_shape(self):
        decoder = MLP(self.decoder_dims)
        z = torch.randn(self.batch_size, self.decoder_dims[0])
        x_recon = decoder(z)
        self.assertEqual(x_recon.shape, (self.batch_size, self.decoder_dims[-1]))

    def test_decoder_gradient_flow(self):
        decoder = MLP(self.decoder_dims)
        z = torch.randn(self.batch_size, self.decoder_dims[0], requires_grad=True)
        x_recon = decoder(z)
        loss = x_recon.sum()
        loss.backward()
        self.assertIsNotNone(z.grad)


class TestResidualVectorQuantizer(unittest.TestCase):
    def setUp(self):
        self.n_codebooks = 3
        self.codebook_size = 32
        self.embedding_dim = 8
        self.commitment_cost = 0.25
        self.batch_size = 4

    def test_rvq_forward_shape(self):
        rvq = ResidualVectorQuantizer(
            n_codebooks=self.n_codebooks,
            codebook_size=self.codebook_size,
            embedding_dim=self.embedding_dim,
            commitment_cost=self.commitment_cost,
        )
        x = torch.randn(self.batch_size, self.embedding_dim)
        z_q_st, z_q, indices, vq_loss = rvq(x)
        self.assertEqual(z_q_st.shape, (self.batch_size, self.embedding_dim))
        self.assertEqual(z_q.shape, (self.batch_size, self.embedding_dim))
        self.assertEqual(indices.shape, (self.batch_size, self.n_codebooks))
        self.assertTrue(vq_loss.dim() == 0 or vq_loss.numel() == 1)
        self.assertTrue(vq_loss.item() >= 0)

    def test_rvq_quantize_dequantize_shape(self):
        rvq = ResidualVectorQuantizer(
            n_codebooks=self.n_codebooks,
            codebook_size=self.codebook_size,
            embedding_dim=self.embedding_dim,
        )
        x = torch.randn(self.batch_size, self.embedding_dim)
        z_q, indices = rvq.quantize(x)
        self.assertEqual(z_q.shape, (self.batch_size, self.embedding_dim))
        self.assertEqual(indices.shape, (self.batch_size, self.n_codebooks))
        recon = rvq.dequantize(indices)
        self.assertEqual(recon.shape, (self.batch_size, self.embedding_dim))

    def test_rvq_dequantize_equals_quantize_output(self):
        rvq = ResidualVectorQuantizer(
            n_codebooks=self.n_codebooks,
            codebook_size=self.codebook_size,
            embedding_dim=self.embedding_dim,
        )
        x = torch.randn(self.batch_size, self.embedding_dim)
        z_q, indices = rvq.quantize(x)
        recon = rvq.dequantize(indices)
        torch.testing.assert_close(z_q, recon)

    def test_rvq_vocab_size(self):
        rvq = ResidualVectorQuantizer(
            n_codebooks=self.n_codebooks,
            codebook_size=self.codebook_size,
            embedding_dim=self.embedding_dim,
        )
        expected = self.codebook_size**self.n_codebooks
        self.assertEqual(rvq.vocab_size, expected)

    def test_rvq_cosine_metric(self):
        rvq = ResidualVectorQuantizer(
            n_codebooks=self.n_codebooks,
            codebook_size=self.codebook_size,
            embedding_dim=self.embedding_dim,
            distance_metric="cosine",
        )
        x = torch.randn(self.batch_size, self.embedding_dim)
        _z_q_st, _z_q, indices, vq_loss = rvq(x)
        self.assertEqual(indices.shape, (self.batch_size, self.n_codebooks))
        self.assertTrue(vq_loss.item() >= 0)

    def test_rvq_euclidean_metric(self):
        rvq = ResidualVectorQuantizer(
            n_codebooks=self.n_codebooks,
            codebook_size=self.codebook_size,
            embedding_dim=self.embedding_dim,
            distance_metric="euclidean",
        )
        x = torch.randn(self.batch_size, self.embedding_dim)
        _z_q_st, _z_q, indices, _vq_loss = rvq(x)
        self.assertEqual(indices.shape, (self.batch_size, self.n_codebooks))

    def test_rvq_invalid_distance_metric(self):
        with self.assertRaises(ValueError) as ctx:
            ResidualVectorQuantizer(
                n_codebooks=self.n_codebooks,
                codebook_size=self.codebook_size,
                embedding_dim=self.embedding_dim,
                distance_metric="invalid",
            )
        self.assertIn("Unknown similarity metric", str(ctx.exception))

    def test_rvq_init_codebooks_uniform(self):
        rvq = ResidualVectorQuantizer(
            n_codebooks=self.n_codebooks,
            codebook_size=self.codebook_size,
            embedding_dim=self.embedding_dim,
        )
        # Latent-dim embeddings (as would be produced by encoder)
        num_items = 100
        latent_embeddings = torch.randn(num_items, self.embedding_dim)
        rvq.init_codebooks(latent_embeddings, init_kwargs=SimpleNamespace(init_method="uniform"))
        self.assertTrue(torch.all(rvq.codebooks[0].weight.data.abs() <= 1.0 / self.codebook_size).item())
        self.assertTrue(torch.all(rvq.codebooks[1].weight.data.abs() <= 1.0 / self.codebook_size).item())
        self.assertTrue(torch.all(rvq.codebooks[2].weight.data.abs() <= 1.0 / self.codebook_size).item())

    def test_rvq_init_codebooks_kmeans(self):
        rvq = ResidualVectorQuantizer(
            n_codebooks=self.n_codebooks,
            codebook_size=min(16, self.codebook_size),
            embedding_dim=self.embedding_dim,
        )
        num_items = 200
        latent_embeddings = torch.randn(num_items, self.embedding_dim)
        rvq.init_codebooks(latent_embeddings, init_kwargs=SimpleNamespace(init_method="kmeans", init_batch_size=100))
        z_q, indices = rvq.quantize(latent_embeddings[: self.batch_size])
        self.assertEqual(indices.shape, (self.batch_size, self.n_codebooks))


class TestRQVAEModel(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace(
            encoder_dims=[32, 16, 8],
            decoder_dims=[8, 16, 32],
            n_codebooks=2,
            codebook_size=32,
            commitment_cost=0.25,
            distance_metric="l2",
            emb_config=SimpleNamespace(embedding_model=None, sem_embedding_dim=None),
        )
        self.batch_size = 4
        self.num_items = 50

    def test_rqvae_forward_shape_and_losses(self):
        model = RQVAEModel(self.config)
        x = torch.randn(self.batch_size, self.config.encoder_dims[0])
        x_recon, indices, losses = model(x)
        self.assertEqual(x_recon.shape, (self.batch_size, self.config.decoder_dims[-1]))
        self.assertEqual(indices.shape, (self.batch_size, self.config.n_codebooks))
        self.assertIn("total", losses)
        self.assertIn("reconstruction", losses)
        self.assertIn("quantization", losses)
        self.assertTrue(losses["total"].item() >= 0)
        self.assertTrue(losses["reconstruction"].item() >= 0)
        self.assertTrue(losses["quantization"].item() >= 0)

    def test_rqvae_encode_decode_shape(self):
        model = RQVAEModel(self.config)
        x = torch.randn(self.batch_size, self.config.encoder_dims[0])
        indices = model.tokenize(x)
        self.assertEqual(indices.shape, (self.batch_size, self.config.n_codebooks))
        x_recon = model.detokenize(indices)
        self.assertEqual(x_recon.shape, (self.batch_size, self.config.decoder_dims[-1]))

    def test_rqvae_train_model_smoke(self):
        model = RQVAEModel(self.config)
        item_embeddings = torch.randn(self.num_items, self.config.encoder_dims[0])
        model.train_model(
            item_embeddings,
            num_epochs=2,
            batch_size=16,
            learning_rate=1e-3,
        )
        indices = model.tokenize(item_embeddings[: self.batch_size])
        self.assertEqual(indices.shape, (self.batch_size, self.config.n_codebooks))

    def test_rqvae_train_model_cuda(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")
        model = RQVAEModel(self.config)
        model.to("cuda")
        item_embeddings = torch.randn(self.num_items, self.config.encoder_dims[0])
        model.train_model(
            item_embeddings,
            num_epochs=2,
            batch_size=16,
            learning_rate=1e-3,
        )
        indices = model.tokenize(item_embeddings[: self.batch_size].to("cuda"))
        self.assertEqual(indices.shape, (self.batch_size, self.config.n_codebooks))

    def test_rqvae_train_model_with_scheduler(self):
        config = SimpleNamespace(
            **{
                k: getattr(self.config, k)
                for k in [
                    "encoder_dims",
                    "decoder_dims",
                    "n_codebooks",
                    "codebook_size",
                    "commitment_cost",
                    "distance_metric",
                    "emb_config",
                ]
            },
            scheduler=SimpleNamespace(name="warmup_linear", warmup_steps=2, total_steps=10, min_ratio=0.1),
        )
        model = RQVAEModel(config)
        item_embeddings = torch.randn(self.num_items, self.config.encoder_dims[0])
        model.train_model(item_embeddings, num_epochs=2, batch_size=16, learning_rate=1e-3)
        indices = model.tokenize(item_embeddings[: self.batch_size])
        self.assertEqual(indices.shape, (self.batch_size, self.config.n_codebooks))

    def test_rqvae_save_load(self):
        model = RQVAEModel(self.config)
        x = torch.randn(2, self.config.encoder_dims[0])
        _, indices_before, _ = model(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(os.path.join(tmpdir, "rqvae_model.pt"))
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "rqvae_model.pt")))

            model2 = RQVAEModel(self.config)
            model2.load(os.path.join(tmpdir, "rqvae_model.pt"))
            _, indices_after, _ = model2(x)
            torch.testing.assert_close(indices_before, indices_after)

    def test_rqvae_compute_codebook_metrics(self):
        model = RQVAEModel(self.config)
        item_embeddings = torch.randn(self.num_items, self.config.encoder_dims[0])
        with torch.no_grad():
            _, z = model.encode(item_embeddings)
        model.init_codebooks(item_embeddings)
        metrics = model.compute_codebook_metrics(item_embeddings)
        self.assertIn("total_utilization_rate", metrics)
        self.assertIn("total_gini_index", metrics)
        self.assertIn("total_entropy", metrics)
        self.assertIn("total_perplexity", metrics)
        self.assertGreaterEqual(metrics["total_utilization_rate"], 0)
        self.assertLessEqual(metrics["total_utilization_rate"], 1)


if __name__ == "__main__":
    unittest.main()
