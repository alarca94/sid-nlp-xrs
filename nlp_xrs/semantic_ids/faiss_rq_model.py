"""
Faiss Residual Quantization (RQ) model for learning semantic item IDs.

Wraps ``faiss.IndexResidualQuantizer`` which applies successive residual
quantization using learned codebooks.  Each codebook has ``codebook_size``
entries (must be a power of 2); the number of bits per codebook is
``log2(codebook_size)``.

Optionally applies PCA (with optional whitening) before quantization, following
the same approach as :class:`~nlp_xrs.semantic_ids.opq_model.OPQModel`.

Reference:
    Babenko, A., & Lempitsky, V. (2014).
    The inverted multi-index.
    IEEE TPAMI.
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from nlp_xrs.semantic_ids.base_quantizer import BaseQuantizerModel
from nlp_xrs.semantic_ids.utils import get_hidden_dim
from nlp_xrs.utils.constants import PACKAGE_NAME


class FaissRQModel(BaseQuantizerModel):
    """
    Residual Quantization tokenizer backed by ``faiss.IndexResidualQuantizer``.

    Each codebook entry uses ``n_bits = log2(codebook_size)`` bits.  The faiss
    index is serialised into the checkpoint, so faiss is required at both
    training and inference time.

    Pipeline:

    1. (Optional) L2-normalise embeddings.
    2. (Optional) Apply PCA for dimensionality reduction.
    3. Train the faiss ``IndexResidualQuantizer`` on training embeddings.
    4. At inference: preprocess + ``rq.compute_codes`` + bitstring decode.

    Args:
        config: Configuration object with attributes:

            - n_codebooks (int):           Number of residual quantization layers.
            - codebook_size (int):         Codes per layer; must be a power of 2
                                           (e.g. 256 → 8 bits).
            - normalize_input (bool):      L2-normalise embeddings before quantization.
                                           Forced to ``False`` when ``pca_whiten=True``.
            - pca_dim (int):               If > 0, reduce to this many dims via PCA
                                           before quantization.
            - pca_whiten (bool):           Apply ZCA whitening in the PCA step.
                                           Disables ``normalize_input`` automatically.
            - faiss_omp_num_threads (int): CPU OpenMP threads for faiss.
            - emb_config:                  Sub-config providing ``embedding_model``
                                           or ``sem_embedding_dim``.
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.n_codebooks = getattr(config, "n_codebooks", 3)
        self.codebook_size = getattr(config, "codebook_size", 256)
        self.normalize_input = getattr(config, "normalize_input", True)
        self.pca_dim = getattr(config, "pca_dim", 0)
        self.pca_whiten = getattr(config, "pca_whiten", False)
        self.faiss_omp_num_threads = getattr(config, "faiss_omp_num_threads", 16)

        self.n_bits = int(np.log2(self.codebook_size))
        if 2**self.n_bits != self.codebook_size:
            raise ValueError(
                f"codebook_size must be a power of 2, got {self.codebook_size}. "
                "Use 64 (6 bits), 128 (7 bits), 256 (8 bits), etc."
            )

        self.input_dim: Optional[int] = None
        emb_config = config.emb_config
        if getattr(emb_config, "embedding_model", None) is not None:
            self.input_dim = get_hidden_dim(emb_config.embedding_model)
        if self.input_dim is None:
            self.input_dim = getattr(emb_config, "sem_embedding_dim", None)
        if self.input_dim is None:
            raise ValueError(
                "input_dim could not be inferred.  Set emb_config.embedding_model "
                "to a known model or provide emb_config.sem_embedding_dim."
            )

        # Effective dimension fed into faiss (after PCA, if configured)
        self.effective_dim: int = self.pca_dim if self.pca_dim > 0 else self.input_dim

        self._faiss_index = None
        self._pca = None  # fitted sklearn PCA (or None)

        self.logger = logging.getLogger(PACKAGE_NAME)

        if self.pca_whiten and self.normalize_input:
            self.logger.warning(
                "[FaissRQ] pca_whiten=True: normalize_input forced to False "
                "(whitening already decorrelates and unit-normalises the space)."
            )
            self.normalize_input = False

    # ------------------------------------------------------------------
    # BaseQuantizerModel overrides for parameter-free model
    # ------------------------------------------------------------------

    def _get_device(self) -> torch.device:
        return torch.device("cpu")

    # ------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------

    def _preprocess(self, embs: np.ndarray) -> np.ndarray:
        """
        Apply L2 normalisation and optional PCA to a numpy embedding matrix.

        Args:
            embs: Float32 array (N, d) — modified in-place for L2 norm.

        Returns:
            Processed float32 array (N, effective_dim).
        """
        import faiss

        embs = np.ascontiguousarray(embs, dtype=np.float32)

        if self.normalize_input:
            faiss.normalize_L2(embs)

        if self._pca is not None:
            embs = self._pca.transform(embs)

        return embs

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_model(
        self,
        item_embeddings: torch.Tensor,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        num_epochs: Optional[int] = None,
        num_steps: Optional[int] = None,
        report_interval: int = 1,
        tb_writer=None,
        verbose: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Train the faiss ``IndexResidualQuantizer`` on item embeddings.

        ``batch_size``, ``learning_rate``, ``num_epochs``, and ``num_steps``
        are accepted for API compatibility but unused — optimisation is
        handled entirely by faiss.

        Args:
            item_embeddings: Training embeddings (N, input_dim).
            batch_size:      Ignored.
            learning_rate:   Ignored.
            num_epochs:      Ignored.
            num_steps:       Ignored.
            report_interval: Ignored.
            tb_writer:       Ignored.
            verbose:         Ignored (faiss RQ does not expose a verbose flag).
        """
        try:
            import faiss
        except ImportError as exc:
            raise ImportError(
                "faiss is required for FaissRQModel.  "
                "Install with:  pip install faiss-cpu  (or faiss-gpu for GPU support)."
            ) from exc

        embs = np.ascontiguousarray(item_embeddings.cpu().float().numpy())

        if self.normalize_input:
            faiss.normalize_L2(embs)

        # Optional PCA dimensionality reduction
        if self.pca_dim > 0:
            from sklearn.decomposition import PCA

            self.logger.info(f"[FaissRQ] Applying PCA: {embs.shape[1]}d → {self.pca_dim}d (whiten={self.pca_whiten})")
            self._pca = PCA(n_components=self.pca_dim, whiten=self.pca_whiten)
            embs = self._pca.fit_transform(embs).astype(np.float32)
            self.effective_dim = self.pca_dim

        faiss.omp_set_num_threads(self.faiss_omp_num_threads)

        self.logger.info(
            f"[FaissRQ] Training IndexResidualQuantizer on {embs.shape[0]} items, "
            f"d={embs.shape[1]}, n_codebooks={self.n_codebooks}, n_bits={self.n_bits}"
        )

        index = faiss.IndexResidualQuantizer(embs.shape[1], self.n_codebooks, self.n_bits, faiss.METRIC_INNER_PRODUCT)
        index.train(embs)
        index.add(embs)

        self._faiss_index = index
        self.logger.info("[FaissRQ] Training complete!")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode item embeddings into residual quantization code indices.

        Args:
            x: Input embeddings (N, input_dim).

        Returns:
            Code index tensor (N, n_codebooks), dtype ``long``.
        """
        if self._faiss_index is None:
            raise RuntimeError(
                "FaissRQModel must be trained before tokenizing.  Call train_model() first or load a checkpoint."
            )

        import faiss

        embs = self._preprocess(x.cpu().float().numpy())

        uint8_codes = self._faiss_index.rq.compute_codes(embs)
        n_bytes = uint8_codes.shape[1]
        codes = []
        for u8_code in uint8_codes:
            bs = faiss.BitstringReader(faiss.swig_ptr(u8_code), n_bytes)
            codes.append([bs.read(self.n_bits) for _ in range(self.n_codebooks)])

        return torch.tensor(codes, dtype=torch.long)

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save(
        self,
        save_path: Union[str, Path],
        item_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Save the faiss index, optional PCA, and config to a checkpoint.

        Args:
            save_path:       Checkpoint file path (.pt).
            item_embeddings: If provided, codebook metrics are computed and
                             stored in the checkpoint.
        """
        import faiss

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint: Dict = {
            "config": self.config,
            "faiss_index_bytes": (
                faiss.serialize_index(self._faiss_index).tobytes() if self._faiss_index is not None else None
            ),
            "pca_bytes": pickle.dumps(self._pca) if self._pca is not None else None,
            "effective_dim": self.effective_dim,
        }

        if item_embeddings is not None:
            checkpoint["codebook_metrics"] = self.compute_codebook_metrics(item_embeddings, verbose=False)

        torch.save(checkpoint, save_path)
        self.logger.info(f"[FaissRQ] Model saved to {save_path}")

    def load(self, load_path: Union[str, Path]):
        """
        Load the faiss index and optional PCA from a checkpoint.

        Args:
            load_path: Checkpoint file path (.pt).
        """
        import faiss

        load_path = Path(load_path)
        checkpoint = torch.load(load_path, weights_only=False, map_location="cpu")

        self.config = checkpoint["config"]
        self.effective_dim = checkpoint.get("effective_dim", self.effective_dim)

        if checkpoint.get("faiss_index_bytes") is not None:
            buf = np.frombuffer(checkpoint["faiss_index_bytes"], dtype=np.uint8)
            self._faiss_index = faiss.deserialize_index(buf)

        if checkpoint.get("pca_bytes") is not None:
            self._pca = pickle.loads(checkpoint["pca_bytes"])

        if "codebook_metrics" in checkpoint:
            self.codebook_metrics = checkpoint["codebook_metrics"]
            self.logger.info("[FaissRQ] Loaded checkpoint with pre-computed codebook metrics.")
