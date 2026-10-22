"""
Residual Vector Quantizer (RQ-VAE quantization component).
"""

import logging
from types import SimpleNamespace
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans, kmeans_plusplus
from torch import nn


class WhitenL2Norm(nn.Module):
    """
    Input normalization layer used in GRID.
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.norm = torch.nn.BatchNorm1d(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.norm(x), p=2, dim=1)


class ResidualVectorQuantizer(nn.Module):
    """
    Residual Vector Quantizer for RQ-VAE.

    This module quantizes continuous embeddings into discrete codes using
    multiple codebooks in a residual manner.

    Codebooks can be updated either via gradient descent (default) or via
    Exponential Moving Average (EMA). EMA updates are generally more stable
    and prevent codebook collapse. When EMA is enabled, the commitment loss
    only trains the encoder (one-sided), since codebook weights are no longer
    differentiable parameters. Dead-code reassignment is also available
    exclusively when EMA is active, as it relies on the EMA cluster-size stats.
    """

    def __init__(
        self,
        n_codebooks: int = 3,
        codebook_size: int = 256,
        embedding_dim: int = 64,
        commitment_cost: float = 0.25,
        distance_metric: str = "l2",
        normalize_residual: bool = False,
        use_ema: bool = False,
        ema_decay: float = 0.99,
        ema_eps: float = 1e-5,
        low_usage_threshold: int = -1,
    ):
        """
        Initialize the Residual Vector Quantizer.

        Args:
            n_codebooks: Number of quantization layers (depth)
            codebook_size: Number of codes in each codebook
            embedding_dim: Dimension of embeddings
            commitment_cost: Weight for commitment loss
            distance_metric: Distance metric ('l2' or 'cosine')
            normalize_residual: Whether to normalize residual between codebook levels
            use_ema: Use EMA updates for codebooks instead of gradient descent.
                When True, codebook weights have requires_grad=False and are updated
                via exponential moving average of cluster assignments.
            ema_decay: EMA decay factor (higher = slower updates). Only used when
                use_ema=True.
            ema_eps: Small constant for Laplace smoothing of cluster sizes, preventing
                division by zero. Only used when use_ema=True.
            low_usage_threshold: Reassign dead codebook entries (those with EMA cluster
                size below 1.0) when the number of dead entries exceeds this threshold.
                Set to -1 to disable. Requires use_ema=True.
        """
        super().__init__()
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.distance_metric = distance_metric
        self.normalize_residual = normalize_residual
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema_eps = ema_eps
        self.low_usage_threshold = low_usage_threshold

        if low_usage_threshold != -1 and not use_ema:
            raise ValueError("low_usage_threshold requires use_ema=True (cluster sizes are only tracked via EMA)")

        # Create multiple codebooks
        self.codebooks = nn.ModuleList([nn.Embedding(codebook_size, embedding_dim) for _ in range(n_codebooks)])

        # Initialize codebook weights; freeze them when using EMA (updated out-of-graph)
        for codebook in self.codebooks:
            codebook.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)
            if use_ema:
                codebook.weight.requires_grad_(False)

        # Per-codebook EMA buffers (only allocated when needed)
        if use_ema:
            for i, codebook in enumerate(self.codebooks):
                self.register_buffer(f"cluster_size_{i}", torch.zeros(codebook_size))
                self.register_buffer(f"embed_avg_{i}", codebook.weight.data.clone())

        # Distance function
        if distance_metric == "cosine":
            self.dist_fn = lambda x, y: 1 - torch.matmul(F.normalize(x, p=2, dim=1), F.normalize(y, p=2, dim=1).t())
        elif distance_metric in ["l2", "euclidean"]:
            self.dist_fn = lambda x, y: (
                torch.sum(x**2, dim=1, keepdim=True) + torch.sum(y**2, dim=1) - 2 * torch.matmul(x, y.t())
            )
        else:
            raise ValueError(f"Unknown similarity metric: {distance_metric}")

    # ------------------------------------------------------------------
    # Private EMA helpers
    # ------------------------------------------------------------------

    def _ema_update(self, cb_idx: int, residual: torch.Tensor, indices: torch.Tensor) -> None:
        """
        Update EMA statistics for one codebook level.

        Accumulates per-cluster usage counts (cluster_size) and per-cluster
        embedding sums (embed_avg) with exponential decay. The actual codebook
        weight update is deferred to :meth:`_reassign_low_usage_clusters` so
        that dead-code resets are reflected before normalization.

        Args:
            cb_idx: Index of the codebook being updated.
            residual: Pre-quantization residual of shape (batch, embedding_dim).
            indices: Nearest-code indices of shape (batch,).
        """
        cluster_size: torch.Tensor = getattr(self, f"cluster_size_{cb_idx}")
        embed_avg: torch.Tensor = getattr(self, f"embed_avg_{cb_idx}")

        one_hot = F.one_hot(indices, self.codebook_size).to(residual.dtype)  # (B, K)

        cluster_size.data.mul_(self.ema_decay).add_(one_hot.sum(0), alpha=1 - self.ema_decay)
        embed_avg.data.mul_(self.ema_decay).add_(one_hot.t() @ residual, alpha=1 - self.ema_decay)

    def _reassign_low_usage_clusters(self, cb_idx: int, residual: torch.Tensor) -> int:
        """
        Reassign dead codebook entries and update codebook weights.

        Must be called *after* :meth:`_ema_update` so that cluster_size
        reflects the current batch. Dead entries (EMA cluster size < 1.0)
        are replaced with random data points from the current batch when
        their count exceeds ``low_usage_threshold``. The codebook weights
        are then refreshed via Laplace-smoothed normalization of embed_avg.

        Args:
            cb_idx: Index of the codebook being updated.
            residual: Pre-quantization residual of shape (batch, embedding_dim),
                used as the pool of replacement vectors.

        Returns:
            Number of entries that were reassigned (0 when reassignment is
            disabled via ``low_usage_threshold == -1``).
        """
        cluster_size: torch.Tensor = getattr(self, f"cluster_size_{cb_idx}")
        embed_avg: torch.Tensor = getattr(self, f"embed_avg_{cb_idx}")
        codebook = self.codebooks[cb_idx]

        n_reassigned = 0
        if self.low_usage_threshold != -1:
            dead = cluster_size < 1.0
            n_dead = int(dead.sum().item())
            if n_dead > self.low_usage_threshold:
                sampled_idx = torch.randint(0, residual.size(0), (n_dead,), device=residual.device)
                sampled = residual[sampled_idx].clone().detach()
                embed_avg.data[dead] = sampled
                cluster_size.data[dead] = 0.99
                n_reassigned = n_dead

        # Laplace-smoothed normalization: update codebook weights from embed_avg
        n = cluster_size.sum()
        smoothed = (cluster_size + self.ema_eps) / (n + self.codebook_size * self.ema_eps) * n
        codebook.weight.data.copy_(embed_avg / smoothed.unsqueeze(1))

        return n_reassigned

    # ------------------------------------------------------------------

    def init_codebooks(
        self, batch_embeddings: Optional[torch.Tensor] = None, init_kwargs: SimpleNamespace = SimpleNamespace()
    ):
        """
        Initialize codebooks.

        Args:
            batch_embeddings: Tensor of batch embeddings (batch_size, embedding_dim)
            init_kwargs: Initialization configuration
                init_method: Method to initialize codebooks ("uniform" or "kmeans")
                init_batch_size: Number of items to sample for kmeans initialization
        """
        init_method = getattr(init_kwargs, "init_method", "uniform")
        if init_method == "uniform":
            logging.info("Initializing codebooks with uniform distribution...")
            for codebook in self.codebooks:
                codebook.weight.data.uniform_(-1.0 / self.codebook_size, 1.0 / self.codebook_size)
        elif init_method == "kmeans":
            logging.info("Initializing codebooks with kmeans...")
            assert batch_embeddings is not None, "batch_embeddings must be provided for kmeans initialization"
            # Normalize to the same space used in forward() so centroids live
            # on the unit sphere where training distances are computed
            residuals = batch_embeddings.detach().cpu().numpy()
            # Run kmeans_plusplus for each codebook
            for codebook in self.codebooks:
                if self.normalize_residual:
                    residuals = residuals / np.linalg.norm(residuals, axis=1, keepdims=True)

                # We only initialize the weights--do not overwrite the nn.Parameter object itself
                centroids, _ = kmeans_plusplus(residuals, n_clusters=self.codebook_size)
                codebook.weight.data = torch.from_numpy(centroids).to(
                    codebook.weight.data.device, dtype=codebook.weight.data.dtype
                )
                # Map embeddings to centroids and update to residuals
                distances = self.dist_fn(torch.tensor(residuals), torch.tensor(centroids))
                indices = torch.argmin(distances, dim=1)
                residuals = residuals - centroids[indices.cpu().numpy()]
        elif init_method == "minibatch_kmeans":
            logging.info("Initializing codebooks with minibatch kmeans...")
            assert batch_embeddings is not None, "batch_embeddings must be provided for minibatch kmeans initialization"
            # Normalize to the same space used in forward() so centroids live
            # on the unit sphere where training distances are computed
            residuals = batch_embeddings.detach().cpu().numpy()
            # Run kmeans_plusplus for each codebook
            max_iter = getattr(init_kwargs, "max_iter", 100)
            for codebook in self.codebooks:
                if self.normalize_residual:
                    residuals = residuals / np.linalg.norm(residuals, axis=1, keepdims=True)

                mbkmeans = MiniBatchKMeans(
                    n_clusters=self.codebook_size, init_size=residuals.shape[0], max_iter=max_iter
                )
                mbkmeans.fit(residuals)
                codebook.weight.data = torch.from_numpy(mbkmeans.cluster_centers_).to(
                    codebook.weight.data.device, dtype=codebook.weight.data.dtype
                )
                # Map embeddings to centroids and update to residuals
                distances = self.dist_fn(torch.tensor(residuals), torch.tensor(mbkmeans.cluster_centers_))
                indices = torch.argmin(distances, dim=1)
                residuals = residuals - mbkmeans.cluster_centers_[indices.cpu().numpy()]
        else:
            raise ValueError(f"Unknown initialization method: {init_method}")

        # Keep embed_avg in sync with the freshly-initialized weights so that
        # the first EMA update starts from a meaningful baseline rather than
        # the uniform-random values registered at construction time.
        if self.use_ema:
            for i, codebook in enumerate(self.codebooks):
                embed_avg: torch.Tensor = getattr(self, f"embed_avg_{i}")
                embed_avg.data.copy_(codebook.weight.data)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize input embeddings using residual quantization.

        Args:
            x: Input tensor of shape (batch_size, embedding_dim)

        Returns:
            quantized_out_detached: Quantized embeddings with detached residual
            quantized_out: Quantized embeddings
            indices: Quantization indices (batch_size, n_codebooks)
            loss: Quantization loss
        """
        # Storage for outputs (initialized in the normalized space)
        quantized_out = torch.zeros_like(x)
        indices_out = []

        residual = x
        total_loss = torch.tensor(0.0, device=x.device)

        # Apply each quantizer sequentially
        for cb_idx, codebook in enumerate(self.codebooks):
            # Normalize residual
            if self.normalize_residual:
                residual = F.normalize(residual, p=2, dim=1)

            # Compute distances to all codes
            distances = self.dist_fn(residual, codebook.weight)

            # Get nearest codes
            indices = torch.argmin(distances, dim=1)
            indices_out.append(indices)

            # Get quantized values
            quantized = codebook(indices)

            # Accumulate quantized output
            quantized_out = quantized_out + quantized

            if self.use_ema:
                # EMA path: codebook weights updated out-of-graph; only train encoder
                if self.training:
                    self._ema_update(cb_idx, residual, indices)
                    self._reassign_low_usage_clusters(cb_idx, residual)
                loss = self.commitment_cost * F.mse_loss(quantized.detach(), residual)
            else:
                # Gradient path: two-sided VQ loss trains both encoder and codebook
                e_latent_loss = F.mse_loss(quantized.detach(), residual)
                q_latent_loss = F.mse_loss(quantized, residual.detach())
                loss = q_latent_loss + self.commitment_cost * e_latent_loss
            total_loss = total_loss + loss

            # Update residual
            residual = residual - quantized.detach()

        # Stack indices
        indices_out = torch.stack(indices_out, dim=1)

        # Straight-through estimator: forward value is quantized_out (sum of
        # codebook entries), backward pass routes dL/d(quantized_out) directly
        # to x so reconstruction gradients reach the encoder.
        quantized_out_detached = x + (quantized_out - x).detach()

        return quantized_out_detached, quantized_out, indices_out, total_loss

    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize without computing loss (for inference).

        Args:
            x: Input tensor of shape (batch_size, embedding_dim)

        Returns:
            quantized_out_detached: Quantized embeddings with detached residual
            quantized: Quantized embeddings
            indices: Quantization indices
        """
        quantized_out = torch.zeros_like(x)
        indices_out = []

        residual = x

        for codebook in self.codebooks:
            # Normalize residual
            if self.normalize_residual:
                residual = F.normalize(residual, p=2, dim=1)

            distances = self.dist_fn(residual, codebook.weight)

            indices = torch.argmin(distances, dim=1)
            indices_out.append(indices)

            quantized = codebook(indices)
            quantized_out = quantized_out + quantized
            residual = residual - quantized

        indices_out = torch.stack(indices_out, dim=1)

        return quantized_out, indices_out

    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct embeddings from quantization indices.

        Args:
            indices: Quantization indices of shape (batch_size, n_codebooks)

        Returns:
            Reconstructed embeddings
        """
        batch_size = indices.shape[0]
        quantized = torch.zeros(batch_size, self.embedding_dim, device=indices.device)

        for i, codebook in enumerate(self.codebooks):
            quantized = quantized + codebook(indices[:, i])

        return quantized

    @property
    def vocab_size(self) -> int:
        """Get total vocabulary size (codebook_size ^ n_codebooks)."""
        return self.codebook_size**self.n_codebooks
