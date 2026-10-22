"""
Unit tests for the per-entity SID (Semantic ID) loading logic in BaseDataset.

Tests cover:
  1. _iter_sid_entities – nested and legacy-flat schema detection.
  2. apply_sid_replace  – both Key.ITEM and Key.USER replacement.
  3. _finish_init SID registration – lengths are multiplied and the correct
     partials are appended to data_transform_fns.

All tests use lightweight stubs; no real data or checkpoints are required.
"""

import unittest
from functools import partial
from types import SimpleNamespace

import pandas as pd
import torch

from nlp_xrs.data.collators import SEQUERTextFusionCollator
from nlp_xrs.data.dataset import SEQUERTextFusionDataset
from nlp_xrs.utils.constants import Key

# ---------------------------------------------------------------------------
# Minimal stubs
# ---------------------------------------------------------------------------


def _make_stub_tokenizer(n_codebooks: int, codebook_size: int, entity_ids: list):
    """Return a minimal object that mimics SemanticIDTokenizer.

    item_to_tokens maps entity_id -> [c0, c1, …] (raw, offset-free codes).
    tokenize(ids) returns the offset-applied codes (each code += cb_offset).
    """

    class _StubTokenizer:
        def __init__(self):
            self.n_codebooks = n_codebooks
            self.codebook_size = codebook_size
            self.is_trained = True
            # Assign a unique, repeatable code sequence per entity id.
            self.item_to_tokens = pd.Series(
                [[(eid + cb) % codebook_size for cb in range(n_codebooks)] for eid in entity_ids],
                index=entity_ids,
            )
            # vocab_size = n_codebooks * codebook_size (no PAD yet)
            self.vocab_size = n_codebooks * codebook_size

            # Pre-compute codebook offsets: [0, C, 2C, …]
            self._codebook_offsets = [cb * codebook_size for cb in range(n_codebooks)]

        def tokenize(self, ids, n_reserved=0):
            results = []
            for eid in ids:
                raw = self.item_to_tokens.loc[eid]
                results.append([r + self._codebook_offsets[i] + n_reserved for i, r in enumerate(raw)])
            return results

    return _StubTokenizer()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestIterSidEntities(unittest.TestCase):
    """_iter_sid_entities: nested vs. legacy-flat schema detection."""

    def _make_obj(self):
        from nlp_xrs.data.dataset import BaseDataset

        obj = SimpleNamespace()
        obj._iter_sid_entities = BaseDataset._iter_sid_entities.__get__(obj, type(obj))
        return obj

    def test_nested_item_only(self):
        obj = self._make_obj()
        item_sub = SimpleNamespace(config_name="x", mode="replace")
        sid_cfg = SimpleNamespace(item=item_sub)
        pairs = list(obj._iter_sid_entities(sid_cfg))
        self.assertEqual(len(pairs), 1)
        entity, sub = pairs[0]
        self.assertEqual(entity, Key.ITEM)
        self.assertIs(sub, item_sub)

    def test_nested_user_only(self):
        obj = self._make_obj()
        user_sub = SimpleNamespace(config_name="y", mode="replace")
        sid_cfg = SimpleNamespace(user=user_sub)
        pairs = list(obj._iter_sid_entities(sid_cfg))
        self.assertEqual(len(pairs), 1)
        entity, sub = pairs[0]
        self.assertEqual(entity, Key.USER)
        self.assertIs(sub, user_sub)

    def test_nested_both(self):
        obj = self._make_obj()
        item_sub = SimpleNamespace(config_name="x", mode="replace")
        user_sub = SimpleNamespace(config_name="y", mode="replace")
        sid_cfg = SimpleNamespace(item=item_sub, user=user_sub)
        pairs = list(obj._iter_sid_entities(sid_cfg))
        self.assertEqual(len(pairs), 2)
        entities = {e for e, _ in pairs}
        self.assertIn(Key.ITEM, entities)
        self.assertIn(Key.USER, entities)

    def test_legacy_flat_treated_as_item(self):
        obj = self._make_obj()
        # Legacy schema: config_name at the top level (no item/user sub-keys)
        sid_cfg = SimpleNamespace(config_name="old_config", mode="replace")
        pairs = list(obj._iter_sid_entities(sid_cfg))
        self.assertEqual(len(pairs), 1)
        entity, sub = pairs[0]
        self.assertEqual(entity, Key.ITEM)
        self.assertIs(sub, sid_cfg)


class TestBuildSidCache(unittest.TestCase):
    """_build_sid_cache: cache is built correctly for each entity."""

    def setUp(self):
        self.item_ids = [0, 1, 2]
        self.user_ids = [10, 11]
        self.n_codebooks = 3
        self.codebook_size = 16

    def _make_obj_with_tokenizer(self, entity, ids):
        tok = _make_stub_tokenizer(self.n_codebooks, self.codebook_size, ids)
        obj = SimpleNamespace(sid_tokenizers={entity: tok}, _sid_caches={})
        from nlp_xrs.data.dataset import BaseDataset

        obj._build_sid_cache = BaseDataset._build_sid_cache.__get__(obj, type(obj))
        return obj, tok

    def test_item_cache_keys_and_length(self):
        obj, tok = self._make_obj_with_tokenizer(Key.ITEM, self.item_ids)
        obj._build_sid_cache(Key.ITEM)
        cache = obj._sid_caches[Key.ITEM]
        self.assertIsInstance(cache, pd.Series)
        self.assertEqual(set(cache.index.tolist()), set(self.item_ids))
        # Each entry should be a list of n_codebooks ints
        for codes in cache:
            self.assertEqual(len(codes), self.n_codebooks)

    def test_item_cache_stored_in_sid_caches(self):
        """Item codes live in _sid_caches. There is no singular _sid_cache alias."""
        obj, tok = self._make_obj_with_tokenizer(Key.ITEM, self.item_ids)
        obj._build_sid_cache(Key.ITEM)
        self.assertIn(Key.ITEM, obj._sid_caches)
        self.assertFalse(hasattr(obj, "_sid_cache"))

    def test_user_cache_no_alias(self):
        """_sid_cache is NOT set for Key.USER."""
        obj, tok = self._make_obj_with_tokenizer(Key.USER, self.user_ids)
        obj._build_sid_cache(Key.USER)
        self.assertIn(Key.USER, obj._sid_caches)
        self.assertFalse(hasattr(obj, "_sid_cache"))


class TestApplySidReplace(unittest.TestCase):
    """apply_sid_replace: raw IDs are replaced with flattened SID codes."""

    N_CODEBOOKS = 3
    CODEBOOK_SIZE = 16

    def _build_obj_and_cache(self, entity, ids):
        tok = _make_stub_tokenizer(self.N_CODEBOOKS, self.CODEBOOK_SIZE, ids)
        from nlp_xrs.data.dataset import BaseDataset

        obj = SimpleNamespace(
            sid_tokenizers={entity: tok},
            _sid_caches={},
        )
        obj._build_sid_cache = BaseDataset._build_sid_cache.__get__(obj, type(obj))
        obj.apply_sid_replace = BaseDataset.apply_sid_replace.__get__(obj, type(obj))
        obj._build_sid_cache(entity)
        return obj, tok

    def test_item_replace_single(self):
        obj, tok = self._build_obj_and_cache(Key.ITEM, [0, 1, 2])
        data = {Key.ITEM: torch.LongTensor([0])}
        obj.apply_sid_replace(data, entity=Key.ITEM)
        result = data[Key.ITEM]
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.numel(), self.N_CODEBOOKS)

    def test_item_replace_sequence(self):
        item_ids = [0, 1, 2]
        obj, tok = self._build_obj_and_cache(Key.ITEM, item_ids)
        # Sequence of 3 items → 3 * N_CODEBOOKS codes
        data = {Key.ITEM: torch.LongTensor(item_ids)}
        obj.apply_sid_replace(data, entity=Key.ITEM)
        self.assertEqual(data[Key.ITEM].numel(), len(item_ids) * self.N_CODEBOOKS)

    def test_user_replace(self):
        user_ids = [10, 11]
        obj, tok = self._build_obj_and_cache(Key.USER, user_ids)
        data = {Key.USER: torch.LongTensor([10])}
        obj.apply_sid_replace(data, entity=Key.USER)
        self.assertEqual(data[Key.USER].numel(), self.N_CODEBOOKS)

    def test_replace_codes_are_offset_applied(self):
        """Codes in the output should include codebook offsets (i.e. > raw values)."""
        ids = [0]
        n_cb, cb_size = 2, 8
        tok = _make_stub_tokenizer(n_cb, cb_size, ids)
        from nlp_xrs.data.dataset import BaseDataset

        obj = SimpleNamespace(sid_tokenizers={Key.ITEM: tok}, _sid_caches={})
        obj._build_sid_cache = BaseDataset._build_sid_cache.__get__(obj, type(obj))
        obj.apply_sid_replace = BaseDataset.apply_sid_replace.__get__(obj, type(obj))
        obj._build_sid_cache(Key.ITEM)

        data = {Key.ITEM: torch.LongTensor([0])}
        obj.apply_sid_replace(data, entity=Key.ITEM)
        codes = data[Key.ITEM].tolist()
        # Second codebook code must be >= cb_size (offset applied)
        self.assertGreaterEqual(codes[1], cb_size)

    def test_default_entity_is_item(self):
        """apply_sid_replace with no entity kwarg defaults to Key.ITEM."""
        obj, _ = self._build_obj_and_cache(Key.ITEM, [0, 1])
        data = {Key.ITEM: torch.LongTensor([0])}
        obj.apply_sid_replace(data)  # no entity arg
        self.assertEqual(data[Key.ITEM].numel(), self.N_CODEBOOKS)


class TestFinishInitSidRegistration(unittest.TestCase):
    """_finish_init: lengths are multiplied and partials are registered."""

    N_CB_ITEM = 3
    N_CB_USER = 2
    CODEBOOK_SIZE = 16

    def _make_obj(self, entities):
        """Build a minimal BaseDataset-shell with the given active entities."""
        from nlp_xrs.data.dataset import BaseDataset

        obj = SimpleNamespace(
            sid_tokenizers={},
            _sid_caches={},
            _sid_entity_cfgs={},
            data_info=SimpleNamespace(lengths={Key.ITEM: 1, Key.USER: 1}),
            data_transform_fns=[],
            # Stub _build_sid_cache so _finish_init doesn't need a real cache
            apply_sid_replace=BaseDataset.apply_sid_replace,
        )

        for entity, n_cb in entities.items():
            tok = _make_stub_tokenizer(n_cb, self.CODEBOOK_SIZE, [0])
            obj.sid_tokenizers[entity] = tok
            obj._sid_caches[entity] = pd.Series([tok.tokenize([0], n_reserved=0)[0]], index=[0])
            obj._sid_entity_cfgs[entity] = SimpleNamespace(mode="replace")

        # Bind _finish_init — but we only call the SID part, not handle_encodings
        def run_sid_only(self_=obj):
            for entity, tokenizer in self_.sid_tokenizers.items():
                mode = getattr(self_._sid_entity_cfgs[entity], "mode", "replace")
                if mode == "replace":
                    self_.data_info.lengths[entity] *= tokenizer.n_codebooks
                    self_.data_transform_fns.append(partial(self_.apply_sid_replace, entity=entity))

        obj._run_sid_only = run_sid_only
        return obj

    def test_item_length_multiplied(self):
        obj = self._make_obj({Key.ITEM: self.N_CB_ITEM})
        obj._run_sid_only()
        self.assertEqual(obj.data_info.lengths[Key.ITEM], self.N_CB_ITEM)

    def test_user_length_multiplied(self):
        obj = self._make_obj({Key.USER: self.N_CB_USER})
        obj._run_sid_only()
        self.assertEqual(obj.data_info.lengths[Key.USER], self.N_CB_USER)

    def test_both_lengths_multiplied(self):
        obj = self._make_obj({Key.ITEM: self.N_CB_ITEM, Key.USER: self.N_CB_USER})
        obj._run_sid_only()
        self.assertEqual(obj.data_info.lengths[Key.ITEM], self.N_CB_ITEM)
        self.assertEqual(obj.data_info.lengths[Key.USER], self.N_CB_USER)

    def test_one_partial_per_entity(self):
        obj = self._make_obj({Key.ITEM: self.N_CB_ITEM, Key.USER: self.N_CB_USER})
        obj._run_sid_only()
        self.assertEqual(len(obj.data_transform_fns), 2)

    def test_partial_entity_kwarg(self):
        """Each registered partial should carry the correct entity keyword."""
        obj = self._make_obj({Key.ITEM: self.N_CB_ITEM, Key.USER: self.N_CB_USER})
        obj._run_sid_only()
        registered_entities = {fn.keywords.get("entity") for fn in obj.data_transform_fns}
        self.assertIn(Key.ITEM, registered_entities)
        self.assertIn(Key.USER, registered_entities)


class TestSEQUERTextFusionSIDData(unittest.TestCase):
    def test_stash_raw_items_preserves_sequence_before_sid_replacement(self):
        data = {Key.ITEM: torch.tensor([3, 4, 5])}

        SEQUERTextFusionDataset.stash_raw_items(data)
        data[Key.ITEM][0] = 99

        self.assertEqual(data[Key.ITEM_RAW].tolist(), [3, 4, 5])

    def test_collator_pads_raw_items_to_unflattened_sequence_length(self):
        mapper = SimpleNamespace(get_idx=lambda token: 12)
        collator = SEQUERTextFusionCollator(
            tok=None,
            field_handler=None,
            mappers={Key.ITEM: mapper},
            item_seq_len=6,
            n_codebooks=2,
            raw_item_pad_idx=0,
        )
        batch = {
            Key.ITEM: [torch.tensor([1, 2, 3, 4])],
            Key.ITEM_RAW: [torch.tensor([3, 4])],
        }

        padded = collator.pad_sequences(batch)

        self.assertEqual(padded[Key.ITEM].tolist(), [[1, 2, 3, 4, 12, 12]])
        self.assertEqual(padded[Key.ITEM_RAW].tolist(), [[3, 4, 0]])


if __name__ == "__main__":
    unittest.main()
