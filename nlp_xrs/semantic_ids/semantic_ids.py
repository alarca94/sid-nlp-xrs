"""
Semantic ID tokenizer using RQ-VAE or other methods.
"""

import logging
import random
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch

from nlp_xrs.semantic_ids.base import BaseItemTokenizer
from nlp_xrs.semantic_ids.faiss_rq_model import FaissRQModel
from nlp_xrs.semantic_ids.opq_model import OPQModel
from nlp_xrs.semantic_ids.rqvae_model import RQVAEModel
from nlp_xrs.utils.constants import PACKAGE_NAME, Key


class SemanticIDTokenizer(BaseItemTokenizer):
    """
    Tokenizer that uses RQ-VAE or similar models to create semantic item IDs.

    This tokenizer learns to map items to sequences of discrete tokens that
    capture semantic similarity between items.
    """

    def __init__(
        self,
        config: SimpleNamespace,
        special_tokens: Optional[Dict[str, int]] = None,
        device: str = "cpu",
        dedup_items_on_load: bool = True,
    ):
        """
        Initialize the Semantic ID tokenizer.
        """
        super().__init__(config)
        self.tokenizer_type = getattr(config, "tokenizer_type", "rqvae")
        self.codebook_size = getattr(config, "codebook_size", 256)
        self.dedup_items_on_creation = getattr(config, "dedup_items_on_creation", False)
        self.dedup_items_on_load = dedup_items_on_load
        if self.dedup_items_on_creation and self.dedup_items_on_load:
            raise ValueError(
                "dedup_items_on_creation and dedup_items_on_load are mutually exclusive. "
                "Set at most one of them to True."
            )
        self.logger = logging.getLogger(PACKAGE_NAME)

        self.item_to_tokens: pd.Series

        # Initialize the underlying quantization model
        if self.tokenizer_type == "rqvae":
            self.model = RQVAEModel(config)
        elif self.tokenizer_type == "opq":
            if not hasattr(config, "n_codebooks"):
                raise ValueError("n_codebooks must be set for OPQModel")
            self.model = OPQModel(config)
        elif self.tokenizer_type == "faiss_rq":
            if not hasattr(config, "n_codebooks"):
                raise ValueError("n_codebooks must be set for FaissRQModel")
            self.model = FaissRQModel(config)
        else:
            raise NotImplementedError(f"Tokenizer type {self.tokenizer_type} not implemented")

    @property
    def add_dedup_tokens(self) -> bool:
        """True when a deduplication codebook is present (either stored or added on load)."""
        return self.dedup_items_on_creation or self.dedup_items_on_load

    @property
    def codebook_offsets(self) -> List[int]:
        if self.add_dedup_tokens:
            return self.model.codebook_offsets + [self.model.codebook_offsets[-1] + self.model.codebook_sizes[-1]]
        return self.model.codebook_offsets

    @property
    def codebook_sizes(self) -> List[int]:
        if self.add_dedup_tokens:
            max_dedup_ix = np.array(self.item_to_tokens.values.tolist())[:, -1].max()
            return self.model.codebook_sizes + [max_dedup_ix + 1]
        return self.model.codebook_sizes

    @property
    def n_codebooks(self) -> int:
        if self.add_dedup_tokens:
            return self.model.n_codebooks + 1
        return self.model.n_codebooks

    @property
    def vocab_size(self) -> int:
        return sum(self.codebook_sizes)

    @property
    def is_orderless(self) -> bool:
        return getattr(self.config, "is_orderless", False)

    def to(self, device: str):
        """Move model to specified device."""
        if hasattr(self, "model"):
            self.model.to(device)

    def exists(self, load_path: Union[str, Path]) -> bool:
        """Check if the tokenizer exists."""
        load_path = Path(load_path)
        if load_path.suffix == ".pt":
            return load_path.is_file()
        return (load_path / self.model_fname).exists() and (load_path / self.mapping_fname).exists()

    def train(
        self,
        data: pd.DataFrame,
        item_features: Optional[Dict] = None,
        tb_writer=None,
        verbose: bool = True,
        key: str = Key.ITEM,
    ):
        """
        Train the semantic ID tokenizer.

        Args:
            data: DataFrame containing the interaction training data
            item_features: Dictionary with item features:
                - 'embeddings': Item embeddings for all items(collaborative or content-based)
                - 'metadata': Item metadata (optional)
                - 'concept_targets': Optional concept-distribution tensor of shape
                  (N_items, sae_codebook_size), indexed by item ID.  When provided
                  and ``config.kl_weight > 0``, an auxiliary KL divergence loss is
                  added during quantizer training to align learned representations
                  with the SAE concept space.
            tb_writer: Optional TensorBoard SummaryWriter for logging losses and metrics.
            verbose: Whether to print verbose logging.
            key: Key of the item column to use for training
        """
        self.logger.info(f"Training {self.tokenizer_type} tokenizer...")

        # Prepare item representations
        if item_features is None or "embeddings" not in item_features:
            raise ValueError("item_features must contain 'embeddings' for training")

        item_embeddings = item_features["embeddings"]
        aux_targets = item_features.get("concept_targets", None)

        if isinstance(data[key].iloc[0], list):
            train_items = data[key].str[-1].unique().tolist()
        else:
            train_items = data[key].unique().tolist()

        # Slice aux_targets to match training items (same indexing as item_embeddings)
        train_aux = aux_targets[train_items] if aux_targets is not None else None

        # Train the model (e.g., RQ-VAE)
        self.model.train_model(
            item_embeddings=item_embeddings[train_items],
            num_epochs=getattr(self.config, "num_epochs", 20000),
            num_steps=getattr(self.config, "num_steps", None),
            batch_size=getattr(self.config, "batch_size", 256),
            learning_rate=getattr(self.config, "learning_rate", 0.5),
            report_interval=getattr(self.config, "report_interval", 10),
            tb_writer=tb_writer,
            verbose=verbose,
            aux_targets=train_aux,
        )

        # Compute codebook quality metrics
        self.codebook_metrics = self.model.compute_codebook_metrics(item_embeddings, verbose=verbose)

        # Generate token mappings for all items (includes collision analysis).
        # Dedup tokens are embedded in the file only when dedup_items_on_creation is set;
        # when dedup_items_on_load is set they are added lazily at load time instead.
        self._generate_token_mappings(item_embeddings, add_dedup_tokens=self.dedup_items_on_creation)

        self.is_trained = True
        self.logger.info(f"\nTokenizer training complete. Vocab size: {self.vocab_size}")

    def _add_dedup_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Add deduplication tokens to the token sequences.
        """
        new_tokens = torch.zeros(tokens.shape[0], dtype=tokens.dtype, device=tokens.device)

        if self.is_orderless:
            tokens = tokens.sort(dim=1)

        _, inverse_ixs, counts = torch.unique(tokens, return_inverse=True, return_counts=True, dim=0)
        order = torch.argsort(inverse_ixs, stable=True)

        group_occurrence = torch.cat([torch.arange(count, device=tokens.device) for count in counts])
        new_tokens[order] = group_occurrence
        logging.info(f"Maximum deduplication token: {new_tokens.max()}")

        tokens = torch.cat([tokens, new_tokens[:, None]], dim=1)
        return tokens

    def _generate_token_mappings(self, item_embeddings: torch.Tensor, add_dedup_tokens: bool = False):
        """
        Generate token mappings for all items using the trained model.
        Analyzes collisions but does not resolve them (keeps fixed-length tokens).

        Args:
            item_embeddings: Tensor of shape (num_items, embedding_dim)
        """
        self.model.eval()
        device = self.model.device
        chunk_size = getattr(self.config, "chunk_size", None)
        with torch.no_grad():
            # Encode items to get token sequences
            if chunk_size is not None:
                parts = []
                for i in range(0, item_embeddings.shape[0], chunk_size):
                    chunk = item_embeddings[i : i + chunk_size].to(device)
                    parts.append(self.model.tokenize(chunk))
                tokens = torch.cat(parts, dim=0)
            else:
                tokens = self.model.tokenize(item_embeddings.to(device))

            # Detect collisions for analysis
            code_to_items = {}
            for item_id in range(tokens.shape[0]):
                code = tuple(tokens[item_id].cpu().tolist())
                if code not in code_to_items:
                    code_to_items[code] = []
                code_to_items[code].append(item_id)

            # Analyze collisions
            collision_groups = {code: items for code, items in code_to_items.items() if len(items) > 1}
            num_unique_codes = len(code_to_items)
            num_collision_codes = len(collision_groups)
            max_collision_size = max((len(items) for items in collision_groups.values()), default=0)
            num_items_with_collision = sum(len(items) for items in collision_groups.values())
            num_items_without_collision = tokens.shape[0] - num_items_with_collision
            ratio_without_collision = num_items_without_collision / tokens.shape[0] if tokens.shape[0] > 0 else 0

            # Log collision statistics
            self.logger.info("\n=== Collision Analysis ===")
            self.logger.info(f"Total items: {tokens.shape[0]}")
            self.logger.info(f"Unique codes: {num_unique_codes}")
            self.logger.info(f"Codes with collisions: {num_collision_codes}")
            self.logger.info(f"Maximum collision size: {max_collision_size}")
            self.logger.info(f"Items without collision: {num_items_without_collision} ({ratio_without_collision:.2%})")
            self.logger.info(f"Items with collision: {num_items_with_collision} ({1 - ratio_without_collision:.2%})")

            # Store collision statistics
            self.collision_stats = {
                "num_unique_codes": num_unique_codes,
                "num_collision_codes": num_collision_codes,
                "max_collision_size": max_collision_size,
                "num_items_without_collision": num_items_without_collision,
                "ratio_without_collision": ratio_without_collision,
            }

            if add_dedup_tokens:
                tokens = self._add_dedup_tokens(tokens)

            # Store mappings (fixed-length tokens)
            self.item_to_tokens = pd.Series(tokens.cpu().tolist())

            # Create reverse mapping (maps to first item in collision group)
            self.tokens_to_item = self._get_reverse_mapping(self.item_to_tokens)

    def _get_reverse_mapping(self, item_to_tokens: pd.Series) -> pd.Series:
        return item_to_tokens.map(tuple).reset_index(name="tokens").groupby("tokens")["index"].agg(list)

    def tokenize(self, item_ids: Union[int, List[int]], n_reserved: int = 0) -> Union[List[int], List[List[int]]]:
        """
        Convert item ID(s) to token sequence(s).

        Args:
            item_ids: Single item ID or list of item IDs

        Returns:
            Token sequence(s) for the item(s)
        """
        if not self.is_trained:
            raise ValueError("Tokenizer must be trained before tokenizing")

        single_input = isinstance(item_ids, int)
        if single_input:
            item_ids = [item_ids]

        codebook_offsets = self.codebook_offsets
        tokens = (np.array(self.item_to_tokens.loc[item_ids].tolist()) + codebook_offsets + n_reserved).tolist()

        return tokens[0] if single_input else tokens

    def detokenize(self, tokens: Union[List[int], List[List[int]]], n_reserved: int = 0) -> Union[int, List[int]]:
        """
        Convert token sequence(s) back to item ID(s).

        Note: This may not always give a perfect reconstruction depending on the method.

        Args:
            tokens: Token sequence or list of token sequences

        Returns:
            Item ID(s) corresponding to the token sequence(s)
        """
        if not self.is_trained:
            raise ValueError("Tokenizer must be trained before detokenizing")

        # Check if single sequence
        single_input = isinstance(tokens[0], int)
        if single_input:
            tokens = [tokens]

        codebook_offsets = self.codebook_offsets

        # Create reverse mapping if not exists
        if not hasattr(self, "tokens_to_item"):
            self.tokens_to_item = self._get_reverse_mapping(self.item_to_tokens)

        item_ids = []
        for token_seq in tokens:
            token_seq = [token - codebook_offsets[i] - n_reserved for i, token in enumerate(token_seq)]
            token_tuple = tuple(token_seq)
            item_id = self.tokens_to_item.get(token_tuple)
            if item_id is None:
                item_id = self._find_closest_item(token_seq)
            elif isinstance(item_id, list):
                item_id = random.choice(item_id)
            item_ids.append(item_id)

        return item_ids[0] if single_input else item_ids

    def _find_closest_item(self, token_seq: List[int]) -> int:
        """
        Find the closest item using hierarchical token matching.

        Args:
            token_seq: Token sequence

        Returns:
            Closest item ID
        """
        if not hasattr(self, "token_tree"):
            self._build_token_tree()

        # Try to find exact match or closest prefix
        current = self.token_tree
        candidates = self.item_to_tokens.index.tolist()

        for depth, token in enumerate(token_seq):
            if token in current:
                current = current[token]
                # Filter candidates that match up to this depth
                candidates = [
                    item_id
                    for item_id in candidates
                    if len(self.item_to_tokens.loc[item_id]) > depth
                    and self.item_to_tokens.loc[item_id][depth] == token
                ]
            else:
                # No exact match, find items with longest matching prefix
                break

        if not candidates:
            # Fallback: return any item
            return self.item_to_tokens.index.tolist()[0]

        # If multiple candidates, find closest by Hamming distance
        min_dist = float("inf")
        closest_item = candidates[0]

        for item_id in candidates:
            item_tokens = self.item_to_tokens.loc[item_id]
            dist = sum(t1 != t2 for t1, t2 in zip(token_seq, item_tokens))
            if dist < min_dist:
                min_dist = dist
                closest_item = item_id

        return closest_item

    def _build_token_tree(self):
        """Build a hierarchical tree for efficient token lookup."""
        self.token_tree = {}
        for item_id, token_seq in self.item_to_tokens.items():
            current = self.token_tree
            for token in token_seq[:-1]:
                if token not in current:
                    current[token] = {}
                current = current[token]
            # Store item_id at the leaf
            last_token = token_seq[-1]
            if last_token not in current:
                current[last_token] = []
            current[last_token].append(item_id)

    def save(self, save_path: Union[str, Path]):
        """
        Save the tokenizer and underlying model.

        A path ending in ``.pt`` is one bundle (mapping + model). Any other
        path is a directory with separate ``mapping_*.pt`` and ``model_*.pt`` files.

        Args:
            save_path: Directory, or a single ``.pt`` bundle file.
        """
        save_path = Path(save_path)
        if save_path.suffix == ".pt":
            self._save_bundle(save_path)
            return

        save_path.mkdir(parents=True, exist_ok=True)

        # Save base tokenizer state (includes collision_stats)
        super().save(save_path)

        # Save model with metrics
        self.model.save(save_path / self.model_fname, item_embeddings=None)

    def _save_bundle(self, save_path: Path) -> None:
        tmp = Path(tempfile.mkdtemp())
        try:
            super().save(tmp)
            self.model.save(tmp / self.model_fname, item_embeddings=None)
            bundle = {
                "mapping": torch.load(tmp / self.mapping_fname, weights_only=False),
                "model": torch.load(tmp / self.model_fname, weights_only=False),
                "model_fname": self.model_fname,
            }
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(bundle, save_path)
            self.logger.info("Saved tokenizer bundle to %s", save_path)
        finally:
            shutil.rmtree(tmp)

    def _load_bundle(self, load_path: Path) -> None:
        bundle = torch.load(load_path, weights_only=False, map_location="cpu")
        tmp = Path(tempfile.mkdtemp())
        try:
            torch.save(bundle["mapping"], tmp / self.mapping_fname)
            torch.save(bundle["model"], tmp / bundle.get("model_fname", self.model_fname))
            self._load_directory(tmp)
        finally:
            shutil.rmtree(tmp)

    def load(self, load_path: Union[str, Path]):
        """
        Load the tokenizer and underlying model.

        Args:
            load_path: Directory, or a single ``.pt`` bundle file.
        """
        load_path = Path(load_path)
        if load_path.suffix == ".pt":
            if not load_path.is_file():
                self.is_trained = False
                return
            self._load_bundle(load_path)
            return
        self._load_directory(load_path)

    def _load_directory(self, load_path: Path):
        """Load a directory checkpoint written by :meth:`save`."""
        # Load base tokenizer state
        super().load(load_path)

        if not self.is_trained:
            return

        if len(self.item_to_tokens.values[0]) == (self.model.n_codebooks + 1):
            if not self.dedup_items_on_creation:
                logging.warning(
                    "Deduplication codebook is present in the saved file, but dedup_items_on_creation is not set. "
                    "Setting dedup_items_on_creation to True and dedup_items_on_load to False."
                )
            self.dedup_items_on_creation = True
            self.dedup_items_on_load = False

        if self.dedup_items_on_creation:
            assert self.item_to_tokens.apply(tuple).nunique() == len(self.item_to_tokens), (
                "Deduplication codebook is not unique"
            )

        # When dedup_items_on_load is active the saved file does NOT contain the
        # deduplication codebook column — we inject it here instead.
        if self.dedup_items_on_load and self.is_trained:
            tokens = torch.tensor(self.item_to_tokens.tolist())
            tokens = self._add_dedup_tokens(tokens)
            self.item_to_tokens = pd.Series(tokens.cpu().tolist())
            self.tokens_to_item = self._get_reverse_mapping(self.item_to_tokens)

        # Load model
        if not (load_path / self.model_fname).exists():
            logging.warning(f"Model file {load_path / self.model_fname} does not exist. Training new tokenizer.")
            return
        self.model.load(load_path / self.model_fname)
