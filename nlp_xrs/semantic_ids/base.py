"""
Base class for item tokenizers.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch


class BaseItemTokenizer(ABC):
    """
    Base class for item tokenizers that convert items to token sequences.

    This can be used for:
    - Semantic IDs (e.g., RQ-VAE based)
    - Collaborative filtering based IDs
    - Content-based IDs
    """

    def __init__(self, config):
        """
        Initialize the tokenizer.

        Args:
            config: Configuration object with tokenizer parameters
        """
        self.config = config
        self.is_trained = False
        self.item_to_tokens: pd.Series = None  # Mapping from item_id to token sequence
        self.model_fname = f"model_{self.config.tokenizer_type}.pt"
        self.mapping_fname = f"mapping_{self.config.tokenizer_type}.pt"
        self.codebook_metrics = None

    @abstractmethod
    def train(self, data, item_features: Optional[Dict] = None):
        """
        Train the tokenizer on the dataset.

        Args:
            data: Training data (e.g., DataFrame with user-item interactions)
            item_features: Optional dictionary of item features (embeddings, metadata, etc.)
        """

    @abstractmethod
    def tokenize(self, item_ids: Union[int, List[int]]) -> Union[List[int], List[List[int]]]:
        """
        Convert item ID(s) to token sequence(s).

        Args:
            item_ids: Single item ID or list of item IDs

        Returns:
            Token sequence(s) for the item(s)
        """

    @abstractmethod
    def detokenize(self, tokens: Union[List[int], List[List[int]]]) -> Union[int, List[int]]:
        """
        Convert token sequence(s) back to item ID(s).

        Args:
            tokens: Token sequence or list of token sequences

        Returns:
            Item ID(s) corresponding to the token sequence(s)
        """

    def save(self, save_path: Union[str, Path]):
        """
        Save the trained tokenizer to disk.

        Args:
            save_path: Path to save the tokenizer
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save mapping
        checkpoint = {
            "item_to_tokens": self.item_to_tokens.to_dict(),
            "config": self.config,
            "is_trained": self.is_trained,
        }
        if self.codebook_metrics is not None:
            checkpoint["codebook_metrics"] = self.codebook_metrics

        torch.save(
            checkpoint,
            save_path / self.mapping_fname,
        )

    def load(self, load_path: Union[str, Path]):
        """
        Load a trained tokenizer from disk.

        Args:
            load_path: Path to load the tokenizer from
        """
        load_path = Path(load_path)

        if not (load_path / self.mapping_fname).exists():
            logging.warning(f"Mapping file {load_path / self.mapping_fname} does not exist. Skipping loading.")
            self.is_trained = False
            return

        # Load mapping
        checkpoint = torch.load(load_path / self.mapping_fname, weights_only=False, map_location="cpu")
        self.item_to_tokens = pd.Series(checkpoint["item_to_tokens"])
        self.config = checkpoint["config"]
        self.is_trained = checkpoint["is_trained"]
        self.codebook_metrics = checkpoint.get("codebook_metrics", None)

    @property
    def vocab_size(self) -> int:
        """
        Return the per-codebook vocabulary size.

        For RQ-VAE / SID tokenizers each codebook position has its own
        alphabet of size ``codebook_size``.  The total number of *distinct
        tokens across all positions* is therefore ``n_codebooks *
        codebook_size``, not the number of unique values in the flat mapping
        (which under-counts because each position reuses the same integer
        range).

        If the config exposes ``codebook_size`` we use that; otherwise we
        fall back to the empirical count per-codebook-position.
        """
        if not self.is_trained:
            raise ValueError("Tokenizer must be trained before accessing vocab_size")
        codebook_size = getattr(self.config, "codebook_size", None)
        n_codebooks = getattr(self.config, "n_codebooks", None)
        if codebook_size is not None and n_codebooks is not None:
            return int(codebook_size) * n_codebooks

        logging.warning("Vocab size not found in config. Using empirical fallback which may not be correct.")
        # Empirical fallback: count max value per position and multiply.
        tokens_list = np.array(self.item_to_tokens.values.tolist())
        return np.prod(tokens_list.max(axis=0))

    @property
    def num_items(self) -> int:
        """
        Get the number of items in the tokenizer.

        Returns:
            Number of items
        """
        return len(self.item_to_tokens)

    def get_token_length(self) -> int:
        """
        Get the length of token sequences.

        Returns:
            Length of token sequences (assumes all sequences have same length)
        """
        if not self.is_trained or self.item_to_tokens is None:
            raise ValueError("Tokenizer must be trained before accessing token length")

        # NOTE: This would need to change in case of variable-length tokens
        return len(self.item_to_tokens.iloc[0])
