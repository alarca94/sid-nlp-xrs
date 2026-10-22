"""
Unit tests for nlp_xrs.data.collators.

Covers: batch_list2dict, BaseCollator (pad_encoded_fields, pad_masks,
        stack_or_cat_tensors, __call__), SEQUERCollator.pad_sequences.
"""

import unittest
from collections import defaultdict
from unittest.mock import MagicMock

import pandas as pd
import torch

from nlp_xrs.data.collators import BaseCollator, SEQUERCollator, batch_list2dict
from nlp_xrs.data.utils import FieldManager, Mapper
from nlp_xrs.utils.constants import PAD_TOK, Key

# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------


def _make_mapper(items):
    return Mapper(pd.Series(items, index=range(len(items))))


def _make_field_manager(**kwargs):
    """Return a FieldManager, optionally seeded with enc/pad_mask entries."""
    fm = FieldManager()
    for col, mapper_key in kwargs.get("enc", {}).items():
        fm.add_encoding(col, mapper_key)
    for col in kwargs.get("pad_mask", []):
        fm.add_pad_mask(col)
    return fm


def _mock_tok():
    """Minimal tokenizer stub for pad_encoded_fields."""
    tok = MagicMock()

    # tok.pad({"input_ids": [...]}, ...) should return {"input_ids": tensor}
    def fake_pad(inputs, return_tensors=None, padding=None):
        ids = inputs["input_ids"]
        max_len = max(len(row) for row in ids)
        padded = [row + [0] * (max_len - len(row)) for row in ids]
        return {"input_ids": torch.tensor(padded, dtype=torch.long)}

    tok.pad.side_effect = fake_pad
    return tok


# ---------------------------------------------------------------------------
# batch_list2dict
# ---------------------------------------------------------------------------


class TestBatchList2Dict(unittest.TestCase):
    def test_scalar_tensor_values_collected_into_lists(self):
        batch = [
            {Key.USER: torch.tensor(0), Key.ITEM: torch.tensor(10)},
            {Key.USER: torch.tensor(1), Key.ITEM: torch.tensor(20)},
        ]
        result = batch_list2dict(batch)
        self.assertIsInstance(result[Key.USER], list)
        self.assertEqual(len(result[Key.USER]), 2)

    def test_size_key_equals_batch_size(self):
        batch = [{Key.USER: torch.tensor(i)} for i in range(5)]
        result = batch_list2dict(batch)
        self.assertEqual(result["size"], 5)

    def test_list_values_flattened_and_bix_created(self):
        batch = [
            {Key.ITEM: [torch.tensor(1), torch.tensor(2)]},
            {Key.ITEM: [torch.tensor(3)]},
        ]
        result = batch_list2dict(batch)
        self.assertEqual(result[Key.ITEM], [torch.tensor(1), torch.tensor(2), torch.tensor(3)])
        self.assertIn(f"{Key.ITEM}_bix", result)

    def test_bix_has_correct_batch_indices(self):
        """bix[i] == batch index of the i-th flattened element."""
        batch = [
            {Key.ITEM: [torch.tensor(10), torch.tensor(11)]},  # batch 0
            {Key.ITEM: [torch.tensor(20)]},  # batch 1
        ]
        result = batch_list2dict(batch)
        bix = torch.cat(result[f"{Key.ITEM}_bix"])
        self.assertEqual(bix.tolist(), [0, 0, 1])

    def test_pad_mask_cols_creates_bool_masks(self):
        batch = [
            {Key.ITEM: torch.tensor([1, 2, 3])},
            {Key.ITEM: torch.tensor([4, 5])},
        ]
        result = batch_list2dict(batch, pad_mask_cols=[Key.ITEM])
        self.assertIn(f"{Key.ITEM}_mask", result)
        masks = result[f"{Key.ITEM}_mask"]
        self.assertEqual(len(masks[0]), 3)
        self.assertEqual(len(masks[1]), 2)
        self.assertTrue(all(masks[0].tolist()))

    def test_pad_mask_cols_with_list_of_lists_creates_per_element_masks(self):
        batch = [
            {Key.ITEM: [torch.tensor([1, 2]), torch.tensor([3])]},
            {Key.ITEM: [torch.tensor([4, 5, 6])]},
        ]
        result = batch_list2dict(batch, pad_mask_cols=[Key.ITEM])
        masks = result[f"{Key.ITEM}_mask"]
        # First sample contributes 2 masks, second sample 1 mask
        self.assertEqual(len(masks), 3)

    def test_scalar_tensor_values_not_extended_in_bix(self):
        """Scalar tensors should NOT produce a _bix entry."""
        batch = [
            {Key.USER: torch.tensor(0)},
            {Key.USER: torch.tensor(1)},
        ]
        result = batch_list2dict(batch)
        self.assertNotIn(f"{Key.USER}_bix", result)

    def test_single_sample_batch(self):
        batch = [{Key.USER: torch.tensor(7), Key.ITEM: torch.tensor(42)}]
        result = batch_list2dict(batch)
        self.assertEqual(result["size"], 1)
        self.assertEqual(result[Key.USER][0].item(), 7)

    def test_no_pad_mask_cols_no_mask_entries(self):
        batch = [{Key.ITEM: torch.tensor([1, 2])}]
        result = batch_list2dict(batch, pad_mask_cols=None)
        self.assertNotIn(f"{Key.ITEM}_mask", result)

    def test_multiple_keys_all_present_in_output(self):
        batch = [
            {Key.USER: torch.tensor(0), Key.ITEM: torch.tensor(1), Key.RATING: torch.tensor(5)},
            {Key.USER: torch.tensor(2), Key.ITEM: torch.tensor(3), Key.RATING: torch.tensor(3)},
        ]
        result = batch_list2dict(batch)
        self.assertIn(Key.USER, result)
        self.assertIn(Key.ITEM, result)
        self.assertIn(Key.RATING, result)


# ---------------------------------------------------------------------------
# BaseCollator.stack_or_cat_tensors
# ---------------------------------------------------------------------------


class TestStackOrCatTensors(unittest.TestCase):
    def _collator(self):
        fm = _make_field_manager()
        return BaseCollator(tok=_mock_tok(), field_handler=fm)

    def test_equal_shape_tensors_stacked(self):
        col = self._collator()
        batch = {Key.USER: [torch.tensor([1, 2]), torch.tensor([3, 4])]}
        result = col.stack_or_cat_tensors(batch)
        self.assertEqual(result[Key.USER].shape, (2, 2))
        self.assertIsInstance(result[Key.USER], torch.Tensor)

    def test_variable_length_tensors_cat_not_stacked(self):
        col = self._collator()
        batch = {Key.ITEM: [torch.tensor([1, 2, 3]), torch.tensor([4, 5])]}
        result = col.stack_or_cat_tensors(batch)
        # Should be concatenated, not stacked
        self.assertEqual(result[Key.ITEM].shape, (5,))

    def test_variable_length_creates_bix_when_not_present(self):
        col = self._collator()
        batch = {Key.ITEM: [torch.tensor([1, 2, 3]), torch.tensor([4, 5])]}
        result = col.stack_or_cat_tensors(batch)
        self.assertIn(f"{Key.ITEM}_bix", result)
        bix = result[f"{Key.ITEM}_bix"]
        self.assertEqual(bix.tolist(), [0, 0, 0, 1, 1])

    def test_bix_not_duplicated_when_already_present(self):
        col = self._collator()
        existing_bix = torch.tensor([0, 0, 1])
        batch = {
            Key.ITEM: [torch.tensor([1, 2]), torch.tensor([3])],
            f"{Key.ITEM}_bix": existing_bix,
        }
        result = col.stack_or_cat_tensors(batch)
        # _bix key should still be a tensor (from cat), not doubled
        self.assertIn(f"{Key.ITEM}_bix", result)

    def test_non_list_values_left_unchanged(self):
        col = self._collator()
        t = torch.tensor(42)
        batch = {"size": 2, "scalar": t}
        result = col.stack_or_cat_tensors(batch)
        self.assertEqual(result["size"], 2)

    def test_single_tensor_in_list_stacked(self):
        col = self._collator()
        batch = {Key.USER: [torch.tensor([7, 8])]}
        result = col.stack_or_cat_tensors(batch)
        self.assertEqual(result[Key.USER].shape, (1, 2))

    def test_variable_length_bix_key_not_given_bix(self):
        """A key ending in _bix must not generate a nested _bix_bix entry."""
        col = self._collator()
        batch = {f"{Key.ITEM}_bix": [torch.tensor([0, 0]), torch.tensor([1])]}
        result = col.stack_or_cat_tensors(batch)
        self.assertNotIn(f"{Key.ITEM}_bix_bix", result)


# ---------------------------------------------------------------------------
# BaseCollator.pad_masks
# ---------------------------------------------------------------------------


class TestPadMasks(unittest.TestCase):
    def test_masks_padded_to_longest(self):
        fm = _make_field_manager(pad_mask=["seq"])
        col = BaseCollator(tok=_mock_tok(), field_handler=fm)
        batch = {
            "seq": [torch.tensor([1, 2, 3]), torch.tensor([4, 5])],
            "seq_mask": [torch.BoolTensor([1, 1, 1]), torch.BoolTensor([1, 1])],
        }
        result = col.pad_masks(batch)
        self.assertIn("seq_mask", result)
        self.assertEqual(result["seq_mask"].shape[1], 3)
        # Shorter mask padded with 0
        self.assertEqual(result["seq_mask"][1, 2].item(), False)

    def test_no_pad_mask_cols_batch_unchanged(self):
        fm = FieldManager()  # no pad_mask columns
        col = BaseCollator(tok=_mock_tok(), field_handler=fm)
        batch = {Key.USER: [torch.tensor(0), torch.tensor(1)]}
        result = col.pad_masks(dict(batch))
        self.assertEqual(list(result.keys()), [Key.USER])


# ---------------------------------------------------------------------------
# BaseCollator.pad_encoded_fields
# ---------------------------------------------------------------------------


class TestPadEncodedFields(unittest.TestCase):
    def test_encoded_field_padded_to_longest(self):
        fm = _make_field_manager(enc={Key.EXP: Key.EXP})
        col = BaseCollator(tok=_mock_tok(), field_handler=fm)
        batch = {
            Key.EXP: [[101, 202, 303], [104, 205]],
        }
        result = col.pad_encoded_fields(batch)
        self.assertIsInstance(result[Key.EXP], torch.Tensor)
        self.assertEqual(result[Key.EXP].shape, (2, 3))
        # Shorter sequence padded with 0
        self.assertEqual(result[Key.EXP][1, 2].item(), 0)

    def test_non_enc_fields_not_touched(self):
        fm = _make_field_manager()  # no enc columns
        col = BaseCollator(tok=_mock_tok(), field_handler=fm)
        batch = {Key.USER: [0, 1, 2]}
        result = col.pad_encoded_fields(dict(batch))
        self.assertEqual(result[Key.USER], [0, 1, 2])


# ---------------------------------------------------------------------------
# BaseCollator.__call__ (full pipeline)
# ---------------------------------------------------------------------------


class TestBaseCollatorCall(unittest.TestCase):
    def test_full_pipeline_returns_tensor_dict(self):
        fm = FieldManager()
        col = BaseCollator(tok=_mock_tok(), field_handler=fm)
        batch_list = [
            {Key.USER: torch.tensor(0), Key.ITEM: torch.tensor(10)},
            {Key.USER: torch.tensor(1), Key.ITEM: torch.tensor(20)},
        ]
        result = col(batch_list)
        self.assertIn(Key.USER, result)
        self.assertIn(Key.ITEM, result)
        self.assertIsInstance(result[Key.USER], torch.Tensor)

    def test_bench_dict_populated_when_provided(self):
        fm = FieldManager()
        bench = defaultdict(float)
        col = BaseCollator(tok=_mock_tok(), field_handler=fm, bench=bench)
        batch_list = [
            {Key.USER: torch.tensor(0)},
            {Key.USER: torch.tensor(1)},
        ]
        col(batch_list)
        self.assertIn("collate_batch_list2dict", bench)
        self.assertGreaterEqual(bench["collate_batch_list2dict"], 0.0)

    def test_bench_none_skips_timing(self):
        fm = FieldManager()
        col = BaseCollator(tok=_mock_tok(), field_handler=fm, bench=None)
        batch_list = [{Key.USER: torch.tensor(i)} for i in range(3)]
        result = col(batch_list)  # must not raise
        self.assertIn(Key.USER, result)

    def test_output_size_key_equals_batch_length(self):
        fm = FieldManager()
        col = BaseCollator(tok=_mock_tok(), field_handler=fm)
        n = 4
        batch_list = [{Key.USER: torch.tensor(i)} for i in range(n)]
        result = col(batch_list)
        self.assertEqual(result["size"], n)


# ---------------------------------------------------------------------------
# SEQUERCollator.pad_sequences
# ---------------------------------------------------------------------------


class _StubMapperWithPad:
    """Minimal mapper stub providing get_idx for PAD_TOK."""

    def __init__(self, pad_idx=0):
        self._pad_idx = pad_idx

    def get_idx(self, token):
        if token == PAD_TOK:
            return self._pad_idx
        raise KeyError(token)


class TestSEQUERCollatorPadSequences(unittest.TestCase):
    SEQ_LEN = 5

    def _collator(self, pad_idx=0):
        fm = FieldManager()
        mappers = {Key.ITEM: _StubMapperWithPad(pad_idx)}
        return SEQUERCollator(
            tok=_mock_tok(),
            field_handler=fm,
            mappers=mappers,
            item_seq_len=self.SEQ_LEN,
        )

    def test_item_sequences_padded_to_seq_len(self):
        col = self._collator(pad_idx=0)
        batch = {
            Key.ITEM: [torch.tensor([1, 2]), torch.tensor([3, 4, 5])],
        }
        result = col.pad_sequences(batch)
        self.assertEqual(result[Key.ITEM].shape, (2, self.SEQ_LEN))

    def test_item_padding_uses_pad_idx(self):
        pad_idx = 99
        col = self._collator(pad_idx=pad_idx)
        batch = {Key.ITEM: [torch.tensor([7])]}
        result = col.pad_sequences(batch)
        # Positions 1..4 should be pad_idx
        self.assertEqual(result[Key.ITEM][0, 1:].tolist(), [pad_idx] * (self.SEQ_LEN - 1))

    def test_rating_sequences_padded_to_seq_len(self):
        col = self._collator()
        batch = {
            Key.ITEM: [torch.tensor([1, 2, 3])],
            Key.RATING: [torch.tensor([4.0, 5.0])],
        }
        result = col.pad_sequences(batch)
        self.assertEqual(result[Key.RATING].shape, (1, self.SEQ_LEN))

    def test_neg_opt_padded_to_seq_len(self):
        col = self._collator()
        batch = {
            Key.ITEM: [torch.tensor([1])],
            Key.NEG_OPT: [torch.tensor([2, 3])],
        }
        result = col.pad_sequences(batch)
        self.assertEqual(result[Key.NEG_OPT].shape, (1, self.SEQ_LEN))

    def test_missing_optional_keys_ignored(self):
        """Rating and neg_opt keys are optional; absence must not raise."""
        col = self._collator()
        batch = {Key.ITEM: [torch.tensor([1, 2])]}
        result = col.pad_sequences(batch)  # should not raise
        self.assertIn(Key.ITEM, result)
        self.assertNotIn(Key.RATING, result)

    def test_already_full_length_sequence_unchanged(self):
        col = self._collator(pad_idx=0)
        full = list(range(1, self.SEQ_LEN + 1))
        batch = {Key.ITEM: [torch.tensor(full)]}
        result = col.pad_sequences(batch)
        self.assertEqual(result[Key.ITEM][0].tolist(), full)

    def test_sequer_call_pipeline_returns_tensors(self):
        """Full SEQUERCollator.__call__ smoke test."""
        col = self._collator()
        batch_list = [
            {
                Key.USER: torch.tensor(0),
                Key.ITEM: torch.tensor([1, 2]),
            },
            {
                Key.USER: torch.tensor(1),
                Key.ITEM: torch.tensor([3]),
            },
        ]
        result = col(batch_list)
        self.assertIn(Key.USER, result)
        self.assertIn(Key.ITEM, result)
        self.assertEqual(result[Key.ITEM].shape[-1], self.SEQ_LEN)


# ---------------------------------------------------------------------------
# Edge-case: batch_list2dict with pad_mask_cols=[] (empty list, not None)
# ---------------------------------------------------------------------------


class TestBatchList2DictEdgeCases(unittest.TestCase):
    def test_empty_pad_mask_list_treated_same_as_none(self):
        batch = [{Key.ITEM: torch.tensor([1, 2])}]
        result = batch_list2dict(batch, pad_mask_cols=[])
        self.assertNotIn(f"{Key.ITEM}_mask", result)

    def test_batch_with_empty_list_value(self):
        """A sample whose list-valued field is empty should contribute no elements."""
        batch = [
            {Key.ITEM: [torch.tensor(1), torch.tensor(2)]},
            {Key.ITEM: []},
        ]
        result = batch_list2dict(batch)
        # Only 2 elements from first sample
        self.assertEqual(len(result[Key.ITEM]), 2)

    def test_all_samples_have_empty_list_values(self):
        batch = [{Key.ITEM: []}, {Key.ITEM: []}]
        result = batch_list2dict(batch)
        self.assertEqual(result[Key.ITEM], [])

    def test_keys_from_first_sample_define_schema(self):
        """All output keys are determined by the first sample's keys."""
        batch = [
            {Key.USER: torch.tensor(0), Key.ITEM: torch.tensor(1)},
            {Key.USER: torch.tensor(2), Key.ITEM: torch.tensor(3)},
        ]
        result = batch_list2dict(batch)
        self.assertIn(Key.USER, result)
        self.assertIn(Key.ITEM, result)


if __name__ == "__main__":
    unittest.main()
