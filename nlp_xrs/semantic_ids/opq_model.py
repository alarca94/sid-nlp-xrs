"""
Optimized Product Quantization (OPQ) model for learning semantic item IDs.

OPQ jointly learns an orthogonal rotation of the embedding space and a
Product Quantizer that minimises reconstruction error, outperforming vanilla PQ.

Training uses the faiss ``index_factory`` pipeline
``OPQ{M},IVF1,PQ{M}x{nbits}`` (GPU-accelerated when configured).
The trained faiss index is serialised into the checkpoint, so faiss is also
required at inference time.

Optionally applies PCA (with optional whitening) before quantization, following
the approach in the reference codebase.

Reference:
    Ge, T., He, K., Ke, Q., & Sun, J. (2013).
    Optimized product quantization for approximate nearest neighbor search.
    CVPR 2013.
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


class OPQModel(BaseQuantizerModel):
    """
    Optimized Product Quantization tokenizer backed by faiss.

    Uses the faiss ``index_factory`` string
    ``"OPQ{M},IVF1,PQ{M}x{nbits}"`` to jointly train the OPQ rotation and
    the Product Quantizer in one pass.  GPU resources are used during
    training when ``use_gpu=True``.  The serialised faiss index is stored in
    the checkpoint, so faiss is required at both training and inference time.

    Pipeline:

    1. (Optional) L2-normalise embeddings.
    2. (Optional) Apply PCA for dimensionality reduction.
    3. Train the faiss ``OPQ + IVF1 + PQ`` index on training embeddings.
    4. At inference: preprocess + ``index.sa_encode`` + bitstring decode.

    Metric computation, pretty-printing, and checkpoint I/O use the
    faiss-serialised index rather than PyTorch buffers/parameters.

    Args:
        config: Configuration object with attributes:

            - n_codebooks (int):           Number of PQ sub-quantizers (M).
            - codebook_size (int):         Codes per sub-quantizer; must be a
                                           power of 2 (64, 128, 256, …).
            - normalize_input (bool):      L2-normalise before quantization.
            - opq_niter (int):             OPQ alternating-minimisation iters.
            - pca_dim (int):               If > 0, reduce to this many dims
                                           via PCA before quantization.
            - pca_whiten (bool):           Apply ZCA whitening in PCA step.
            - use_gpu (bool):              Use GPU faiss resources for training.
            - gpu_id (int):                GPU device index (when use_gpu=True).
            - faiss_omp_num_threads (int): CPU OpenMP threads for faiss.
            - emb_config:                  Sub-config providing
                                           ``embedding_model`` or
                                           ``sem_embedding_dim``.
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.n_codebooks = getattr(config, "n_codebooks", 4)
        self.codebook_size = getattr(config, "codebook_size", 256)
        self.normalize_input = getattr(config, "normalize_input", True)
        self.opq_niter = getattr(config, "opq_niter", 50)
        self.pca_dim = getattr(config, "pca_dim", 0)
        self.pca_whiten = getattr(config, "pca_whiten", True)
        self.use_gpu = getattr(config, "use_gpu", False)
        self.gpu_id = getattr(config, "gpu_id", 0)
        self.faiss_omp_num_threads = getattr(config, "faiss_omp_num_threads", 16)

        # faiss PQ requires codebook_size to be a power of 2
        self.n_bits = int(np.log2(self.codebook_size))
        if 2**self.n_bits != self.codebook_size:
            raise ValueError(
                f"codebook_size must be a power of 2, got {self.codebook_size}. "
                "Use 64 (6 bits), 128 (7 bits), 256 (8 bits), etc."
            )

        # Resolve raw embedding dimensionality
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

        # Set after training
        self._faiss_index = None  # serialised faiss Index
        self._pca = None  # fitted sklearn PCA (or None)

        self.logger = logging.getLogger(PACKAGE_NAME)

        if self.pca_whiten and self.normalize_input:
            self.logger.warning(
                "[OPQ] pca_whiten=True: normalize_input forced to False "
                "(whitening already decorrelates and unit-normalises the space)."
            )
            self.normalize_input = False

    # ------------------------------------------------------------------
    # Index factory string (mirrors the reference implementation exactly)
    # ------------------------------------------------------------------

    @property
    def _index_factory_str(self) -> str:
        return f"OPQ{self.n_codebooks},IVF1,PQ{self.n_codebooks}x{self.n_bits}"

    def _get_device(self) -> torch.device:
        if self.use_gpu and torch.cuda.is_available():
            return torch.device(f"cuda:{min(self.gpu_id, torch.cuda.device_count() - 1)}")
        else:
            return torch.device("cpu")

    # ------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------

    def _preprocess(self, embs: np.ndarray) -> np.ndarray:
        """
        Apply L2 normalisation and optional PCA to a numpy embedding matrix.

        Args:
            embs: Float32 array (N, d) — will be copied before modification.

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
        Train the OPQ faiss index on item embeddings.

        ``batch_size``, ``learning_rate``, ``num_epochs``, and ``num_steps``
        are accepted for API compatibility but are not used — optimisation is
        controlled entirely by faiss.  Adjust ``opq_niter`` in the config to
        control the number of OPQ alternating-minimisation iterations.

        Args:
            item_embeddings: Training embeddings (N, input_dim).
            batch_size:      Ignored.
            learning_rate:   Ignored.
            num_epochs:      Ignored.
            num_steps:       Ignored.
            report_interval: Ignored.
            tb_writer:       Ignored.
            verbose:         Passed to faiss index as ``verbose`` flag.
        """
        try:
            import faiss
        except ImportError as exc:
            raise ImportError(
                "faiss is required for OPQModel.  Install with:  pip install faiss-cpu  (or faiss-gpu for GPU support)."
            ) from exc

        embs = item_embeddings.cpu().float().numpy()
        if not embs.flags["C_CONTIGUOUS"]:
            embs = np.ascontiguousarray(embs)

        if self.normalize_input:
            faiss.normalize_L2(embs)

        # Optional PCA dimensionality reduction
        if self.pca_dim > 0:
            from sklearn.decomposition import PCA

            self.logger.info(f"[OPQ] Applying PCA: {embs.shape[1]}d → {self.pca_dim}d (whiten={self.pca_whiten})")
            self._pca = PCA(n_components=self.pca_dim, whiten=self.pca_whiten)
            embs = self._pca.fit_transform(embs)
            self.effective_dim = self.pca_dim

        faiss.omp_set_num_threads(self.faiss_omp_num_threads)

        d = embs.shape[1]
        if d % self.n_codebooks != 0:
            raise ValueError(
                f"Embedding dimension ({d}) must be divisible by "
                f"n_codebooks ({self.n_codebooks}).  "
                f"Adjust n_codebooks or set pca_dim to a suitable value."
            )

        self.logger.info(f"[OPQ] Building index '{self._index_factory_str}' on {embs.shape[0]} items, d={d}")

        index = faiss.index_factory(d, self._index_factory_str, faiss.METRIC_INNER_PRODUCT)

        # Set OPQ alternating-minimisation iterations before training
        opq_transform = faiss.downcast_VectorTransform(index.chain.at(0))
        opq_transform.niter = self.opq_niter

        index.verbose = verbose

        if self.use_gpu:
            res = faiss.StandardGpuResources()
            res.setTempMemory(1024 * 1024 * 512)
            co = faiss.GpuClonerOptions()
            # Use float16 for very wide codes (matches reference heuristic)
            co.useFloat16 = (self.n_codebooks * self.n_bits) >= 56
            index = faiss.index_cpu_to_gpu(res, self.gpu_id, index, co)

        self.logger.info("[OPQ] Training index…")
        index.train(embs)

        if self.use_gpu:
            index = faiss.index_gpu_to_cpu(index)

        self._faiss_index = index
        self.logger.info("[OPQ] Training complete!")

    # ------------------------------------------------------------------
    # Bitstring coding helpers
    # ------------------------------------------------------------------

    def _sa_encode_to_codes(self, embs: np.ndarray) -> np.ndarray:
        """
        Encode preprocessed embeddings to per-subquantizer indices.

        Uses ``index.sa_encode`` to get packed bytes, then reads each
        sub-quantizer's ``n_bits``-wide field with faiss ``BitstringReader``
        — exactly the decoding scheme from the reference implementation.

        Args:
            embs: Preprocessed float32 array (N, effective_dim).

        Returns:
            Integer code array of shape (N, n_codebooks).
        """
        import faiss

        # faiss ≥1.7 numpy wrapper: sa_encode(x) → uint8 (N, code_size)
        codes_packed = self._faiss_index.sa_encode(embs)
        code_size = codes_packed.shape[1]

        codes = []
        for u8code in codes_packed:
            bs = faiss.BitstringReader(faiss.swig_ptr(u8code), code_size)
            codes.append([bs.read(self.n_bits) for _ in range(self.n_codebooks)])

        return np.array(codes, dtype=np.int64)

    def _codes_to_sa_bytes(self, codes: np.ndarray) -> np.ndarray:
        """
        Pack per-subquantizer indices back into the ``sa_encode`` byte format.

        Args:
            codes: Integer array (N, n_codebooks).

        Returns:
            Packed uint8 array (N, code_size).
        """
        import faiss

        n = codes.shape[0]
        code_size = self._faiss_index.sa_code_size()
        codes_packed = np.zeros((n, code_size), dtype=np.uint8)

        for i, code in enumerate(codes):
            bw = faiss.BitstringWriter(faiss.swig_ptr(codes_packed[i]), code_size)
            for c in code:
                bw.write(int(c), self.n_bits)

        return codes_packed

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode item embeddings into OPQ code indices.

        Args:
            x: Input embeddings (N, input_dim).

        Returns:
            Code index tensor (N, n_codebooks), dtype ``long``.
        """
        if self._faiss_index is None:
            raise RuntimeError(
                "OPQModel must be trained before tokenizing.  Call train_model() first or load a checkpoint."
            )

        embs = self._preprocess(x.cpu().float().numpy())
        codes = self._sa_encode_to_codes(embs)
        return torch.from_numpy(codes)

    def detokenize(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct approximate embeddings from OPQ code indices.

        Uses ``index.sa_decode``, which applies centroid lookup (PQ) and the
        inverse OPQ rotation.

        Args:
            indices: Code indices (N, n_codebooks).

        Returns:
            Reconstructed embeddings (N, effective_dim).
        """

        if self._faiss_index is None:
            raise RuntimeError(
                "OPQModel must be trained before detokenizing.  Call train_model() first or load a checkpoint."
            )

        codes_np = indices.cpu().numpy().astype(np.int64)
        codes_packed = self._codes_to_sa_bytes(codes_np)

        # faiss ≥1.7 numpy wrapper: sa_decode(codes) → float32 (N, d)
        decoded = self._faiss_index.sa_decode(codes_packed)

        return torch.from_numpy(decoded)

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
        self.logger.info(f"[OPQ] Model saved to {save_path}")

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
            self.logger.info("[OPQ] Loaded checkpoint with pre-computed codebook metrics.")
