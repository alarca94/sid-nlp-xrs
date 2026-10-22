"""
RQ-VAE model for learning semantic item IDs.

Optionally applies PCA (with optional whitening) before the encoder, following
the same approach as :class:`~nlp_xrs.semantic_ids.opq_model.OPQModel`.
"""

import logging
import os
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from tqdm import tqdm

from nlp_xrs.semantic_ids.base_quantizer import BaseQuantizerModel
from nlp_xrs.semantic_ids.quantizer import ResidualVectorQuantizer, WhitenL2Norm
from nlp_xrs.semantic_ids.utils import get_hidden_dim, get_scheduler
from nlp_xrs.utils.constants import PACKAGE_NAME


class MLP(nn.Module):
    """Encoder network that maps item embeddings to latent space."""

    def __init__(self, layer_dims: List[int], dropout: float = 0.0):
        super().__init__()

        layers = []
        for i in range(len(layer_dims) - 2):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(layer_dims[-2], layer_dims[-1]))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class RQVAEModel(BaseQuantizerModel):
    """
    Residual Quantized Variational AutoEncoder for semantic item IDs.

    This model learns to compress item representations into discrete codes
    that preserve semantic similarity.

    Metric computation, pretty-printing, and checkpoint I/O are inherited
    from :class:`~nlp_xrs.semantic_ids.base_quantizer.BaseQuantizerModel`.
    """

    def __init__(self, config):
        """
        Initialize RQ-VAE model.

        Args:
            config: Configuration with parameters:
                - encoder_dims: Dimensions of encoder layers
                - decoder_dims: Dimensions of decoder layers
                - n_codebooks: Number of quantization layers
                - codebook_size: Size of each codebook
                - commitment_cost: Weight for commitment loss
                - distance_metric: Distance metric ('l2' or 'cosine')
                - normalize_input: Whether to normalize input embeddings.
                                   Forced to ``None`` (Identity) when ``pca_whiten=True``.
                - normalize_residual: Whether to normalize residual
                - init_kwargs: Initialization kwargs
                - pca_dim (int): If > 0, reduce to this many dims via PCA before
                                 the encoder.
                - pca_whiten (bool): Apply ZCA whitening in the PCA step.
                                     Disables ``normalize_input`` automatically.
        """
        super().__init__()

        self.config = config
        self.encoder_dims = getattr(config, "encoder_dims", [768, 256, 128])
        self.decoder_dims = getattr(config, "decoder_dims", [128, 256, 768])
        self.n_codebooks = getattr(config, "n_codebooks", 3)
        self.codebook_size = getattr(config, "codebook_size", 256)
        self.dropout = getattr(config, "dropout", 0.0)
        self.commitment_cost = getattr(config, "commitment_cost", 0.25)
        self.distance_metric = getattr(config, "distance_metric", "l2")
        self.normalize_input = getattr(config, "normalize_input", "layer_norm")
        self.normalize_residual = getattr(config, "normalize_residual", False)
        self.init_kwargs = getattr(config, "init_kwargs", SimpleNamespace())
        self.use_ema = getattr(config, "use_ema", False)
        self.ema_decay = getattr(config, "ema_decay", 0.99)
        self.ema_eps = getattr(config, "ema_eps", 1e-5)
        self.low_usage_threshold = getattr(config, "low_usage_threshold", -1)
        self.pca_dim = getattr(config, "pca_dim", 0)
        self.pca_whiten = getattr(config, "pca_whiten", False)

        # Resolve raw embedding dimensionality
        sem_embedding_dim = None
        emb_config = config.emb_config
        if getattr(emb_config, "embedding_model", None) is not None:
            sem_embedding_dim = get_hidden_dim(emb_config.embedding_model)

        if sem_embedding_dim is None:
            sem_embedding_dim = getattr(emb_config, "sem_embedding_dim", None)

        # When PCA is configured the encoder sees pca_dim-dimensional inputs, not
        # the raw embedding dimension.
        effective_input_dim = self.pca_dim if self.pca_dim > 0 else sem_embedding_dim

        if effective_input_dim is not None and effective_input_dim != self.encoder_dims[0]:
            self.encoder_dims = [effective_input_dim] + self.encoder_dims
            self.decoder_dims = self.decoder_dims + [effective_input_dim]

        self.logger = logging.getLogger(PACKAGE_NAME)

        if self.pca_whiten and self.normalize_input not in (False, None, "none"):
            self.logger.warning(
                "[RQVAE] pca_whiten=True: normalize_input forced to None/Identity "
                "(whitening already decorrelates and unit-normalises the space)."
            )
            self.normalize_input = None

        # KL concept supervision (Phase 3)
        self.sae_codebook_size = getattr(config, "sae_codebook_size", None)
        self.sae_preset = getattr(config, "sae_preset", None)
        if self.sae_codebook_size is None and self.sae_preset is not None:
            sae_cfg_path = os.path.join("config", "sae", f"{self.sae_preset}.yaml")
            if os.path.isfile(sae_cfg_path):
                with open(sae_cfg_path, "r") as f:
                    preset = yaml.safe_load(f)
                self.sae_codebook_size = preset["tokenizer"]["codebook_size"]
            else:
                self.sae_codebook_size = 0
        else:
            self.sae_codebook_size = 0
        self.kl_weight = getattr(config, "kl_weight", 0.0)

        if self.normalize_input == "layer_norm":
            self.input_norm_layer = nn.LayerNorm(self.encoder_dims[0])
        elif self.normalize_input == "whiten_l2":
            self.input_norm_layer = WhitenL2Norm(self.encoder_dims[0])
        else:
            self.input_norm_layer = nn.Identity()

        # Build encoder
        self.encoder = MLP(layer_dims=self.encoder_dims, dropout=self.dropout)

        # Concept-distribution projection head (disabled when sae_codebook_size=0)
        if self.sae_codebook_size > 0 and self.kl_weight > 0:
            self.concept_head = nn.Linear(self.encoder_dims[-1], self.sae_codebook_size)
        else:
            self.concept_head = None

        # Build quantizer
        self.quantizer = ResidualVectorQuantizer(
            n_codebooks=self.n_codebooks,
            codebook_size=self.codebook_size,
            embedding_dim=self.encoder_dims[-1],
            commitment_cost=self.commitment_cost,
            distance_metric=self.distance_metric,
            normalize_residual=self.normalize_residual,
            use_ema=self.use_ema,
            ema_decay=self.ema_decay,
            ema_eps=self.ema_eps,
            low_usage_threshold=self.low_usage_threshold,
        )

        # Build decoder
        self.decoder = MLP(layer_dims=self.decoder_dims, dropout=self.dropout)

        self._pca = None  # fitted sklearn PCA (or None)

    def forward(
        self,
        x: torch.Tensor,
        concept_targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Forward pass through the model.

        Args:
            x: Input item embeddings (batch_size, input_dim)
            concept_targets: Optional concept-distribution tensor of shape
                (batch_size, sae_codebook_size), indexed by item ID.  When provided
                and ``config.kl_weight > 0``, an auxiliary KL divergence loss is
                added during quantizer training to align learned representations
                with the SAE concept space.
        Returns:
            reconstructed: Reconstructed embeddings
            indices: Quantization indices
            losses: Dictionary of losses
        """
        norm_x, z = self.encode(x)
        z_q_detached, z_q, indices, vq_loss = self.quantizer(z)
        x_recon = self.decoder(z_q_detached)

        recon_loss = F.mse_loss(x_recon, norm_x)
        total_loss = recon_loss + vq_loss

        losses = {"total": total_loss, "reconstruction": recon_loss, "quantization": vq_loss}

        if concept_targets is not None and self.concept_head is not None and self.kl_weight > 0:
            # KL-flow on the encoder
            log_pred = F.log_softmax(self.concept_head(z), dim=-1)
            kl = F.kl_div(log_pred, concept_targets.to(z.device), reduction="batchmean")
            # KL-flow on the quantizer codebooks
            log_pred = F.log_softmax(self.concept_head(z_q), dim=-1)
            kl += F.kl_div(log_pred, concept_targets.to(z_q.device), reduction="batchmean")
            losses["kl"] = kl
            losses["total"] = losses["total"] + self.kl_weight * kl

        return x_recon, indices, losses

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode item embeddings to latent space.

        Args:
            x: Input item embeddings

        Returns:
            (norm_x, z): Normalized input and latent embeddings
        """
        norm_x = self.input_norm_layer(x)
        return norm_x, self.encoder(norm_x)

    def _apply_pca(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project embeddings through the fitted PCA transform.

        Args:
            x: Raw embeddings tensor (N, input_dim) on any device.

        Returns:
            PCA-projected tensor (N, pca_dim) on the same device.
        """
        device = x.device
        x_np = x.cpu().float().numpy()
        x_np = self._pca.transform(x_np).astype(np.float32)
        return torch.from_numpy(x_np).to(device)

    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tokenize items to discrete codes.

        Args:
            x: Input item embeddings (raw, before any PCA).

        Returns:
            Quantization indices of shape (batch_size, n_codebooks)
        """
        self.eval()
        with torch.no_grad():
            if self._pca is not None:
                x = self._apply_pca(x)
            _, z = self.encode(x)
            _, indices = self.quantizer.quantize(z)
        return indices

    def detokenize(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Detokenize discrete codes to item embeddings.

        Args:
            indices: Quantization indices

        Returns:
            Reconstructed embeddings
        """
        self.eval()
        with torch.no_grad():
            z_q = self.quantizer.dequantize(indices)
            x_recon = self.decoder(z_q)
        return x_recon

    def init_codebooks(self, item_embeddings: torch.Tensor, verbose: bool = True):
        """
        Initialize codebooks.

        Args:
            item_embeddings: Tensor of item embeddings (num_items, embedding_dim)
            verbose: Whether to print verbose logging.
        """
        device = next(self.parameters()).device

        with torch.no_grad():
            latent_embeddings = None
            if getattr(self.init_kwargs, "init_method", "uniform") in ["kmeans", "minibatch_kmeans"]:
                init_batch_size = getattr(self.init_kwargs, "init_batch_size", 1000)
                if init_batch_size == "all":
                    init_batch_size = item_embeddings.size(0)
                perm = torch.randperm(item_embeddings.size(0))
                batch_indices = perm[:init_batch_size]
                batch_embeddings = item_embeddings[batch_indices].to(device)
                _, latent_embeddings = self.encode(batch_embeddings)

            self.quantizer.init_codebooks(latent_embeddings, init_kwargs=self.init_kwargs)

    def _log_to_tensorboard(
        self,
        tb_writer,
        epoch: int,
        epoch_losses: Dict[str, float],
        scheduler,
        codebook_metrics: Dict,
    ):
        """Write training scalars to TensorBoard at the given epoch."""
        for key, value in epoch_losses.items():
            tb_writer.add_scalar(f"Loss/{key}", value, epoch)
        if scheduler is not None:
            tb_writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)
        for name, value in codebook_metrics.items():
            if isinstance(value, list):
                for q in range(len(value)):
                    tb_writer.add_scalar(f"Codebook/{name}_cb{q}", value[q], epoch)
            else:
                tb_writer.add_scalar(f"Codebook/{name}", value, epoch)

    def train_model(
        self,
        item_embeddings: torch.Tensor,
        batch_size: int,
        learning_rate: float,
        num_epochs: int,
        num_steps: Optional[int] = None,
        report_interval: int = 10,
        tb_writer=None,
        verbose: bool = True,
        **kwargs,
    ) -> None:
        """
        Train the RQ-VAE model. The training will stop after either the number of epochs or steps is reached.

        Args:
            item_embeddings: Tensor of training item embeddings (num_items, embedding_dim)
            num_epochs: Number of training epochs
            num_steps: Number of training steps
            batch_size: Batch size for training
            learning_rate: Learning rate
            report_interval: Log loss every this many epochs.
            tb_writer: Optional TensorBoard SummaryWriter.
            verbose: Whether to print verbose logging.
        """
        device = next(self.parameters()).device

        # Optional PCA dimensionality reduction — project all embeddings once
        # for training efficiency (avoids repeated numpy transforms in the loop).
        # tokenize() re-applies PCA at inference time for external callers.
        if self.pca_dim > 0:
            from sklearn.decomposition import PCA

            self.logger.info(
                f"[RQVAE] Applying PCA: {item_embeddings.shape[1]}d → {self.pca_dim}d (whiten={self.pca_whiten})"
            )
            self._pca = PCA(n_components=self.pca_dim, whiten=self.pca_whiten)
            proj_np = self._pca.fit_transform(item_embeddings.cpu().float().numpy()).astype(np.float32)
            proj = torch.from_numpy(proj_np)
        else:
            proj = item_embeddings

        num_items = proj.shape[0]
        num_batches = (num_items + batch_size - 1) // batch_size

        optimizer = torch.optim.Adagrad(self.parameters(), lr=learning_rate, weight_decay=0.0)
        if hasattr(self.config, "scheduler"):
            total_steps = num_steps if num_steps is not None else num_epochs * num_batches
            self.config.scheduler.total_steps = getattr(self.config.scheduler, "total_steps", total_steps)
            scheduler = get_scheduler(self.config.scheduler, optimizer)
        else:
            scheduler = None

        self.init_codebooks(proj, verbose=verbose)

        self.codebook_metrics = self.compute_codebook_metrics(item_embeddings, verbose=False)
        if verbose:
            self.print_codebook_metrics(
                self.codebook_metrics,
                title="Codebook Quality Metrics (Initialized Codebooks)",
            )

        aux_targets = kwargs.get("aux_targets", None)
        use_kl = aux_targets is not None and self.concept_head is not None and self.kl_weight > 0
        if use_kl:
            self.logger.info(
                f"[RQVAE] KL supervision enabled (kl_weight={self.kl_weight}, concept_dim={aux_targets.shape[1]})"
            )

        num_steps = num_steps or 10**10
        if num_epochs > num_steps / num_batches:
            self.logger.info(
                f"Training RQ-VAE on {num_items} items for {num_steps} steps ({num_steps / num_batches:.2f} epochs)..."
            )
            num_epochs = int(np.ceil(num_steps / num_batches))
        else:
            self.logger.info(f"Training RQ-VAE on {num_items} items for {num_epochs} epochs...")

        pbar = tqdm(range(num_epochs), desc="Training RQ-VAE")
        for epoch in pbar:
            pbar.set_description(f"Epoch {epoch + 1}/{num_epochs}")
            self.train()
            epoch_losses: Dict[str, float] = {"total": 0.0, "reconstruction": 0.0, "quantization": 0.0}
            if use_kl:
                epoch_losses["kl"] = 0.0

            perm = torch.randperm(num_items)

            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, num_items)
                batch_indices = perm[start_idx:end_idx]

                batch_embeddings = proj[batch_indices].to(device)

                optimizer.zero_grad()

                if use_kl:
                    batch_targets = aux_targets[batch_indices].to(device)
                    _, _, losses = self(batch_embeddings, batch_targets)
                else:
                    _, _, losses = self(batch_embeddings)
                losses["total"].backward()

                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                for key in epoch_losses:
                    if key in losses:
                        epoch_losses[key] += losses[key].item()

                curr_step = (batch_idx + 1) + num_batches * epoch
                if curr_step >= num_steps:
                    return

            for key in epoch_losses:
                epoch_losses[key] /= num_batches

            if (epoch + 1) % report_interval == 0 or epoch == 0:
                log_metrics = {
                    "Total Loss": f"{epoch_losses['total']:.4f}",
                    "Recon Loss": f"{epoch_losses['reconstruction']:.4f}",
                    "Quant Loss": f"{epoch_losses['quantization']:.4f}",
                }
                if use_kl:
                    log_metrics["KL Loss"] = f"{epoch_losses['kl']:.4f}"
                if scheduler is not None:
                    log_metrics["LR"] = f"{scheduler.get_last_lr()[0]:.2e}"
                self.logger.info(
                    f"Epoch {epoch + 1}/{num_epochs} (Steps: {curr_step}/{num_steps}) - {', '.join([f'{k}: {v}' for k, v in log_metrics.items()])}"
                )
                # Pass raw item_embeddings so tokenize() can apply PCA consistently
                codebook_metrics = self.compute_codebook_metrics(item_embeddings, verbose=False)
                if verbose:
                    self.print_codebook_metrics(
                        codebook_metrics,
                        title=f"Codebook Quality Metrics (Epoch {epoch + 1})",
                    )

                if tb_writer is not None:
                    self._log_to_tensorboard(tb_writer, epoch + 1, epoch_losses, scheduler, codebook_metrics)

        if (epoch + 1) % report_interval != 0:
            codebook_metrics = self.compute_codebook_metrics(item_embeddings, verbose=False)
            if verbose:
                self.print_codebook_metrics(
                    codebook_metrics,
                    title="Codebook Quality Metrics (Final)",
                )
            if tb_writer is not None:
                self._log_to_tensorboard(tb_writer, epoch + 1, epoch_losses, scheduler, codebook_metrics)

        self.logger.info("Training complete!")

    # ------------------------------------------------------------------
    # Checkpoint I/O (override base to persist fitted PCA)
    # ------------------------------------------------------------------

    def save(
        self,
        save_path: Union[str, Path],
        item_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Save model state dict, optional PCA, and config to a checkpoint.

        Args:
            save_path:       Checkpoint file path (.pt).
            item_embeddings: If provided, codebook metrics are computed and
                             stored in the checkpoint.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint: Dict = {
            "model_state_dict": self.state_dict(),
            "config": self.config,
            "pca_bytes": pickle.dumps(self._pca) if self._pca is not None else None,
        }

        if item_embeddings is not None:
            checkpoint["codebook_metrics"] = self.compute_codebook_metrics(item_embeddings, verbose=False)

        torch.save(checkpoint, save_path)
        self.logger.info(f"[RQVAE] Model saved to {save_path}")

    def load(self, load_path: Union[str, Path]):
        """
        Load model state dict and optional PCA from a checkpoint.

        Args:
            load_path: Checkpoint file path (.pt).
        """
        load_path = Path(load_path)
        checkpoint = torch.load(load_path, weights_only=False, map_location="cpu")

        self.load_state_dict(checkpoint["model_state_dict"])
        self.config = checkpoint["config"]

        if checkpoint.get("pca_bytes") is not None:
            self._pca = pickle.loads(checkpoint["pca_bytes"])

        if "codebook_metrics" in checkpoint:
            self.codebook_metrics = checkpoint["codebook_metrics"]
            self.logger.info("[RQVAE] Loaded checkpoint with pre-computed codebook metrics.")
