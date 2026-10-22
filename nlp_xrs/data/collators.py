# nlp_xrs/data/collators.py
"""Collator functions for batching dataset samples."""

import time
from collections import defaultdict
from typing import Dict, List, Optional, Union

import torch
from transformers import PreTrainedTokenizerFast

from nlp_xrs.data.utils import FieldManager, Mapper, pad2len, pad2longest
from nlp_xrs.utils.constants import PAD_TOK, Key


class BaseCollator:
    """Base collator with common batching logic."""

    def __init__(
        self,
        tok: PreTrainedTokenizerFast,
        field_handler: FieldManager,
        mappers: Optional[Dict[str, Mapper]] = None,
        bench: Optional[Dict[str, float]] = None,
    ):
        self.tok = tok
        self.field_handler = field_handler
        self.mappers = mappers or {}
        self._bench = bench  # shared dict with BaseDataset; None means benchmarking disabled

    def pad_encoded_fields(self, batch) -> Dict[str, torch.Tensor]:
        """Pad encoded text fields using tokenizer."""
        for col, _ in self.field_handler.enc.items():
            if col in batch:
                batch[col] = self.tok.pad({"input_ids": batch[col]}, return_tensors="pt", padding="longest")[
                    "input_ids"
                ]
        return batch

    def pad_masks(self, batch) -> Dict[str, torch.Tensor]:
        """Create padding masks for specified columns."""
        for k in self.field_handler.pad_mask:
            mask_key = k.replace("_mask", "")
            if mask_key in batch or k in batch:
                # Handle both "col" and "col_mask" naming
                source_key = k if k in batch else mask_key
                batch[f"{mask_key}_mask"] = pad2longest(
                    batch.get(f"{mask_key}_mask", batch.get(source_key, [])), dim=0, side="right", value=0
                )
        return batch

    def stack_or_cat_tensors(self, batch) -> Dict[str, torch.Tensor]:
        """Stack or concatenate tensor lists, handling variable lengths."""
        aux = {}
        for k in list(batch.keys()):
            if isinstance(batch[k], list):
                try:
                    batch[k] = torch.stack(batch[k])
                except RuntimeError:
                    # Handle variable-length sequences
                    lens = list(map(len, batch[k]))
                    if len(set(lens)) > 1 and not k.endswith("_bix") and f"{k}_bix" not in batch:
                        aux[f"{k}_bix"] = torch.repeat_interleave(torch.arange(len(batch[k])), torch.tensor(lens))
                    batch[k] = torch.cat(batch[k])

        if aux:
            batch.update(aux)
        return batch

    def __call__(self, batch_list) -> Dict[str, torch.Tensor]:
        """Main collate function."""
        if self._bench is None:
            batch = batch_list2dict(batch_list, self.field_handler.pad_mask)
            batch = self.pad_encoded_fields(batch)
            batch = self.pad_masks(batch)
            batch = self.stack_or_cat_tensors(batch)
            return batch

        t = time.perf_counter()
        batch = batch_list2dict(batch_list, self.field_handler.pad_mask)
        self._bench["collate_batch_list2dict"] += time.perf_counter() - t

        t = time.perf_counter()
        batch = self.pad_encoded_fields(batch)
        self._bench["collate_pad_encoded"] += time.perf_counter() - t

        t = time.perf_counter()
        batch = self.pad_masks(batch)
        self._bench["collate_pad_masks"] += time.perf_counter() - t

        t = time.perf_counter()
        batch = self.stack_or_cat_tensors(batch)
        self._bench["collate_stack_or_cat"] += time.perf_counter() - t

        return batch


class SEQUERCollator(BaseCollator):
    """Collator for SEQUER dataset with fixed-length padding for sequences."""

    def __init__(self, tok, field_handler, mappers, item_seq_len, bench=None):
        super().__init__(tok, field_handler, mappers, bench=bench)
        self.item_seq_len = item_seq_len

    def pad_sequences(self, batch) -> Dict[str, torch.Tensor]:
        """Pad item and rating sequences to fixed length."""
        i_pad_ix = self.mappers[Key.ITEM].get_idx(PAD_TOK)

        if Key.ITEM in batch:
            batch[Key.ITEM] = pad2len(batch[Key.ITEM], self.item_seq_len, side="right", value=i_pad_ix)

        if Key.NEG_OPT in batch:
            batch[Key.NEG_OPT] = pad2len(batch[Key.NEG_OPT], self.item_seq_len, side="right", value=i_pad_ix)

        if Key.RATING in batch:
            batch[Key.RATING] = pad2len(batch[Key.RATING], self.item_seq_len, side="right")

        return batch

    def __call__(self, batch_list) -> Dict[str, torch.Tensor]:
        """SEQUER-specific collate with sequence padding."""
        if self._bench is None:
            batch = batch_list2dict(batch_list, self.field_handler.pad_mask)
            batch = self.pad_encoded_fields(batch)
            batch = self.pad_sequences(batch)
            batch = self.pad_masks(batch)
            batch = self.stack_or_cat_tensors(batch)
            return batch

        t = time.perf_counter()
        batch = batch_list2dict(batch_list, self.field_handler.pad_mask)
        self._bench["collate_batch_list2dict"] += time.perf_counter() - t

        t = time.perf_counter()
        batch = self.pad_encoded_fields(batch)
        self._bench["collate_pad_encoded"] += time.perf_counter() - t

        t = time.perf_counter()
        batch = self.pad_sequences(batch)
        self._bench["collate_pad_sequences"] += time.perf_counter() - t

        t = time.perf_counter()
        batch = self.pad_masks(batch)
        self._bench["collate_pad_masks"] += time.perf_counter() - t

        t = time.perf_counter()
        batch = self.stack_or_cat_tensors(batch)
        self._bench["collate_stack_or_cat"] += time.perf_counter() - t

        return batch


class SEQUERTextFusionCollator(SEQUERCollator):
    """SEQUER collator that preserves raw item IDs alongside SID codes."""

    def __init__(self, tok, field_handler, mappers, item_seq_len, n_codebooks, raw_item_pad_idx, bench=None):
        super().__init__(tok, field_handler, mappers, item_seq_len, bench=bench)
        self.n_codebooks = n_codebooks
        self.raw_item_pad_idx = raw_item_pad_idx

    def pad_sequences(self, batch) -> Dict[str, torch.Tensor]:
        batch = super().pad_sequences(batch)
        if Key.ITEM_RAW in batch:
            raw_item_seq_len = self.item_seq_len // self.n_codebooks
            batch[Key.ITEM_RAW] = pad2len(
                batch[Key.ITEM_RAW], raw_item_seq_len, side="right", value=self.raw_item_pad_idx
            )
        return batch


def batch_list2dict(
    batch: List[Dict[str, Union[List, torch.Tensor]]], pad_mask_cols: Optional[List[str]] = None
) -> Dict[str, Union[List, torch.Tensor, int]]:
    batch_size = len(batch)
    new_batch = {k: [] for k in batch[0]}

    aux = defaultdict(list)
    for bix, sample in enumerate(batch):
        for k in new_batch:
            if pad_mask_cols and not k.endswith("_mask") and k in pad_mask_cols:
                if isinstance(sample[k], (list, tuple)):
                    aux[f"{k}_mask"].extend([torch.BoolTensor([1] * len(l)) for l in sample[k]])
                else:
                    aux[f"{k}_mask"].append(torch.BoolTensor([1] * len(sample[k])))
            if isinstance(sample[k], (list, tuple)):
                aux[f"{k}_bix"].append(torch.ones(len(sample[k]), dtype=torch.long) * bix)
                new_batch[k].extend(sample[k])
            else:
                new_batch[k].append(sample[k])

    new_batch.update(aux)
    new_batch["size"] = batch_size
    return new_batch
