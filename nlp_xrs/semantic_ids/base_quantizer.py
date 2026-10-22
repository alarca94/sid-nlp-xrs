"""
Shared base class for residual quantizer models (RQ-VAE, R-KMeans, ...).

Provides common codebook metric computation, pretty-printing, and checkpoint
I/O so that concrete subclasses only need to implement model-specific logic
(architecture, training, tokenize).
"""

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import tabulate
import torch
from torch import nn

from nlp_xrs.utils.constants import PACKAGE_NAME


class BaseQuantizerModel(nn.Module, ABC):
    """
    Abstract base class for residual quantizer tokenizer models.

    Subclasses **must**:
    - Implement :meth:`tokenize`.
    - Set ``self.n_codebooks``, ``self.codebook_size``, and ``self.config``
      as instance attributes in ``__init__``.
    - Set ``self.logger = logging.getLogger(PACKAGE_NAME)`` in ``__init__``.

    Inherited for free:
    - :meth:`_compute_codebook_metrics` (static utility)
    - :meth:`compute_codebook_metrics` (full pipeline over item embeddings)
    - :meth:`print_codebook_metrics` (tabular summary)
    - :meth:`save` / :meth:`load` (checkpoint I/O)
    """

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(PACKAGE_NAME)
        self.n_codebooks: int
        self.codebook_size: int
        self.config = None

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map input embeddings to discrete code indices.

        Args:
            x: Input embeddings (N, input_dim).

        Returns:
            Index tensor of shape (N, n_codebooks).
        """

    @abstractmethod
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
        **kwargs: Any,
    ) -> None:
        """
        Train the quantizer on item embeddings.

        Args:
            item_embeddings: Pre-computed item embedding tensor (N, input_dim).
            batch_size:      Mini-batch size.
            learning_rate:   Learning rate (ignored by faiss-backed backends).
            num_epochs:      Training epochs (ignored by faiss-backed backends).
            num_steps:       Hard cap on gradient steps; overrides epochs when set.
            report_interval: Log frequency (epochs).
            tb_writer:       Optional TensorBoard SummaryWriter.
            verbose:         Whether to emit progress logs.
            **kwargs:        Additional keyword arguments.
        """

    @property
    def codebook_offsets(self) -> List[int]:
        """
        Return the per-codebook offsets for embedding lookup.

        By default, this assumes contiguous codebook ranges of size
        ``codebook_size``.  Subclasses can override if they use a different
        scheme (e.g., interleaved codes).
        """
        return [q * self.codebook_size for q in range(self.n_codebooks)]

    @property
    def codebook_sizes(self) -> List[int]:
        """
        Return the size of each codebook.
        """
        return [self.codebook_size] * self.n_codebooks

    @property
    def vocab_size(self) -> int:
        """
        Return the total vocabulary size across all codebooks.

        By default, this is ``n_codebooks * codebook_size``.  Subclasses can
        override if they use a different scheme (e.g., interleaved codes).
        """
        return self.n_codebooks * self.codebook_size

    @property
    def device(self) -> torch.device:
        """
        Return the device hosting this model's tensors.
        """
        return self._get_device()

    # ------------------------------------------------------------------
    # Codebook quality metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_codebook_metrics(
        indices: torch.Tensor, space_size: int, num_items: int, skip_metrics: List[str] = []
    ) -> Dict:
        """
        Compute quality metrics for a single quantization index tensor.

        Works for both a flat per-layer index (N,) and a joint multi-layer
        index (N, D): ``torch.unique`` treats each row as a code tuple in
        the latter case.

        Args:
            indices:    Index tensor — shape (N,) or (N, D).
            space_size: Total number of distinct codes in this space
                        (``codebook_size`` for single layer,
                        ``codebook_size ** n_codebooks`` for joint).
            num_items:  Number of items (N).
            skip_metrics: List of metrics to skip.

        Returns:
            Dictionary with keys:
            ``utilization_rate``, ``actual_utilization_rate``,
            ``max_utilization_rate``, ``gini_index``, ``entropy``,
            ``perplexity``.
        """
        metrics: Dict = {}
        unique_codes, counts = torch.unique(indices, return_counts=True, dim=0)
        num_used = len(unique_codes)

        metrics["utilization_rate"] = num_used / space_size
        metrics["actual_utilization_rate"] = num_used / min(num_items, space_size)
        metrics["max_utilization_rate"] = min(num_items / space_size, 1.0)

        probs = counts.float() / num_items
        if "gini_index" not in skip_metrics:
            sorted_probs, _ = torch.sort(probs)
            n = min(num_items, space_size)
            offset = n - num_used
            gini = (
                2.0
                * torch.sum(torch.arange(offset + 1, n + 1, device=sorted_probs.device).float() * sorted_probs)
                / (n * torch.sum(sorted_probs))
                - (n + 1) / n
            )
            metrics["gini_index"] = gini.item()

        entropy = -torch.sum(probs * torch.log(probs + 1e-10))
        if "entropy" not in skip_metrics:
            norm_entropy = entropy / np.log(min(num_items, space_size))
            metrics["entropy"] = norm_entropy.item()

        if "perplexity" not in skip_metrics:
            perplexity = torch.exp(entropy).item()
            metrics["perplexity"] = perplexity / min(num_items, space_size)

        return metrics

    @staticmethod
    def metrics_from_tokens(indices: torch.Tensor, codebook_size: int) -> Dict:
        """
        Compute codebook metrics from precomputed token indices.

        This is the checkpoint/offline counterpart to
        :meth:`compute_codebook_metrics`: it applies the same single-layer and
        joint-code calculations without running a tokenizer model.

        Args:
            indices: Integer token matrix with shape ``(num_items, num_codebooks)``.
            codebook_size: Number of entries in each codebook.

        Returns:
            Dictionary with per-codebook metric lists and ``total_*`` metrics
            for the joint token tuples.
        """
        if indices.ndim != 2:
            raise ValueError(f"indices must be (N, C), got shape {tuple(indices.shape)}")
        if indices.shape[0] == 0:
            raise ValueError("indices is empty")
        if indices.shape[1] == 0:
            raise ValueError("indices must contain at least one codebook")

        num_items, n_codebooks = indices.shape
        metrics: Dict = defaultdict(list)

        for q in range(n_codebooks):
            codebook_metrics = BaseQuantizerModel._compute_codebook_metrics(
                indices[:, q],
                space_size=codebook_size,
                num_items=num_items,
            )
            for name, value in codebook_metrics.items():
                metrics[name].append(value)

        total = BaseQuantizerModel._compute_codebook_metrics(
            indices,
            space_size=codebook_size**n_codebooks,
            num_items=num_items,
        )
        for name, value in total.items():
            metrics[f"total_{name}"] = value

        return dict(metrics)

    def _get_device(self) -> torch.device:
        """
        Return the device hosting this model's tensors.

        Falls back to buffers (for parameter-free models such as OPQModel),
        then to CPU if neither parameters nor buffers are present.
        """
        try:
            return next(self.parameters()).device
        except StopIteration:
            pass
        try:
            return next(self.buffers()).device
        except StopIteration:
            pass
        return torch.device("cpu")

    def compute_codebook_metrics(self, item_embeddings: torch.Tensor, verbose: bool = True) -> Dict:
        """
        Compute per-layer and aggregate codebook quality metrics.

        Tokenizes all item embeddings (using chunked inference when
        ``config.chunk_size`` is set) and reports utilization, Gini index,
        entropy, and perplexity for each codebook layer and jointly.

        Args:
            item_embeddings: All item embeddings (num_items, input_dim).
            verbose: Whether to print a summary table via :meth:`print_codebook_metrics`.

        Returns:
            Dictionary with per-layer metric lists (``utilization_rate``,
            ``gini_index``, ``entropy``, ``perplexity``) and aggregate
            scalars prefixed with ``total_``.
        """
        self.eval()
        device = self._get_device()
        chunk_size = getattr(self.config, "chunk_size", None)

        with torch.no_grad():
            if chunk_size is not None:
                parts = []
                for i in range(0, item_embeddings.shape[0], chunk_size):
                    parts.append(self.tokenize(item_embeddings[i : i + chunk_size].to(device)))
                indices = torch.cat(parts, dim=0)
            else:
                indices = self.tokenize(item_embeddings.to(device))

            metrics = self.metrics_from_tokens(indices, self.codebook_size)

            if verbose:
                self.print_codebook_metrics(metrics)
            return metrics

    def print_codebook_metrics(self, metrics: Dict, title: str = "Codebook Quality Metrics"):
        """
        Pretty-print a metrics table produced by :meth:`compute_codebook_metrics`.

        Args:
            metrics: Dictionary returned by :meth:`compute_codebook_metrics`.
            title:   Header string for the table.
        """
        self.logger.info(f"\n=== {title} ===")

        rows = []
        for q in range(self.n_codebooks):
            rows.append(
                {
                    "Codebook": q,
                    "Utilization": f"{metrics['utilization_rate'][q]:.2%}",
                    "Gini Index": f"{metrics['gini_index'][q]:.4f}",
                    "Entropy": f"{metrics['entropy'][q]:.4f}",
                    "Perplexity": f"{metrics['perplexity'][q]:.2f}",
                }
            )
        rows.append(
            {
                "Codebook": "Total",
                "Utilization": f"{metrics['total_utilization_rate']:.2%}",
                "Gini Index": f"{metrics['total_gini_index']:.4f}",
                "Entropy": f"{metrics['total_entropy']:.4f}",
                "Perplexity": f"{metrics['total_perplexity']:.2f}",
            }
        )

        df = pd.DataFrame(rows)
        table_str = tabulate.tabulate(
            df,
            headers="keys",
            tablefmt="github",
            showindex=False,
            colalign=("center", "center", "center", "center", "center"),
        )
        self.logger.info("\n\n" + table_str)

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save(
        self,
        save_path: Union[str, Path],
        item_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Save model state dict, config, and optionally codebook metrics.

        Args:
            save_path:       File path for the checkpoint (.pt).
            item_embeddings: If provided, codebook metrics are computed and
                             stored inside the checkpoint.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": self.config,
        }

        if item_embeddings is not None:
            checkpoint["codebook_metrics"] = self.compute_codebook_metrics(item_embeddings, verbose=False)

        torch.save(checkpoint, save_path)

    def load(self, load_path: Union[str, Path]):
        """
        Load model state dict and config from a checkpoint.

        Args:
            load_path: File path of the checkpoint (.pt).
        """
        load_path = Path(load_path)
        checkpoint = torch.load(load_path, weights_only=False, map_location="cpu")
        self.load_state_dict(checkpoint["model_state_dict"])
        self.config = checkpoint["config"]

        if "codebook_metrics" in checkpoint:
            self.codebook_metrics = checkpoint["codebook_metrics"]
            self.logger.info("Loaded checkpoint with codebook metrics.")
