"""
Unit tests for nlp_xrs.data.utils.

Covers: Pipeline, Mapper, EntityTokenizer, FieldManager,
        pad2longest, pad2len, sliding_w_hist,
        get_transform_mapping, postprocessing.
"""

import unittest

import numpy as np
import pandas as pd
import torch

from nlp_xrs.data.utils import (
    EntityTokenizer,
    FieldManager,
    Mapper,
    Pipeline,
    get_transform_mapping,
    pad2len,
    pad2longest,
    postprocessing,
    sliding_w_hist,
)
from nlp_xrs.utils.constants import Key

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class TestPipelineConstruction(unittest.TestCase):
    def test_empty_pipeline_has_zero_length(self):
        p = Pipeline()
        self.assertEqual(len(p), 0)

    def test_empty_pipeline_repr(self):
        self.assertEqual(repr(Pipeline()), "Pipeline[<empty>]")

    def test_empty_pipeline_names_returns_empty_list(self):
        self.assertEqual(Pipeline().names(), [])

    def test_empty_pipeline_iteration_yields_nothing(self):
        self.assertEqual(list(Pipeline()), [])


class TestPipelineAppend(unittest.TestCase):
    def _fn(self, name):
        def f():
            pass

        f.__name__ = name
        return f

    def test_append_increases_length(self):
        p = Pipeline()
        p.append("a", self._fn("a"))
        self.assertEqual(len(p), 1)

    def test_append_order_preserved(self):
        p = Pipeline()
        p.append("first", self._fn("first"))
        p.append("second", self._fn("second"))
        self.assertEqual(p.names(), ["first", "second"])

    def test_repr_shows_step_names(self):
        p = Pipeline()
        p.append("get_basic", self._fn("get_basic"))
        p.append("sid_replace", self._fn("sid_replace"))
        self.assertIn("get_basic", repr(p))
        self.assertIn("sid_replace", repr(p))
        self.assertIn("→", repr(p))

    def test_iteration_yields_callables_only(self):
        calls = []

        def step_a():
            calls.append("a")

        def step_b():
            calls.append("b")

        p = Pipeline()
        p.append("a", step_a)
        p.append("b", step_b)
        for fn in p:
            fn()
        self.assertEqual(calls, ["a", "b"])


class TestPipelinePrepend(unittest.TestCase):
    def test_prepend_inserts_at_front(self):
        p = Pipeline()
        p.append("b", lambda: None)
        p.prepend("a", lambda: None)
        self.assertEqual(p.names(), ["a", "b"])

    def test_prepend_to_empty_is_first_element(self):
        p = Pipeline()
        p.prepend("only", lambda: None)
        self.assertEqual(p.names(), ["only"])


class TestPipelineInsertBefore(unittest.TestCase):
    def test_insert_before_middle_step(self):
        p = Pipeline()
        p.append("a", lambda: None)
        p.append("c", lambda: None)
        p.insert_before("c", "b", lambda: None)
        self.assertEqual(p.names(), ["a", "b", "c"])

    def test_insert_before_first_step(self):
        p = Pipeline()
        p.append("b", lambda: None)
        p.insert_before("b", "a", lambda: None)
        self.assertEqual(p.names(), ["a", "b"])

    def test_insert_before_nonexistent_raises_key_error(self):
        p = Pipeline()
        p.append("a", lambda: None)
        with self.assertRaises(KeyError):
            p.insert_before("z", "x", lambda: None)


class TestPipelineInsertAfter(unittest.TestCase):
    def test_insert_after_first_step(self):
        p = Pipeline()
        p.append("a", lambda: None)
        p.append("c", lambda: None)
        p.insert_after("a", "b", lambda: None)
        self.assertEqual(p.names(), ["a", "b", "c"])

    def test_insert_after_last_step(self):
        p = Pipeline()
        p.append("a", lambda: None)
        p.insert_after("a", "b", lambda: None)
        self.assertEqual(p.names(), ["a", "b"])

    def test_insert_after_nonexistent_raises_key_error(self):
        p = Pipeline()
        p.append("a", lambda: None)
        with self.assertRaises(KeyError):
            p.insert_after("z", "x", lambda: None)


class TestPipelineRemove(unittest.TestCase):
    def test_remove_existing_step(self):
        p = Pipeline()
        p.append("a", lambda: None)
        p.append("b", lambda: None)
        p.remove("a")
        self.assertEqual(p.names(), ["b"])

    def test_remove_only_step_leaves_empty(self):
        p = Pipeline()
        p.append("a", lambda: None)
        p.remove("a")
        self.assertEqual(len(p), 0)

    def test_remove_nonexistent_raises_key_error(self):
        p = Pipeline()
        with self.assertRaises(KeyError):
            p.remove("ghost")


class TestPipelineReplace(unittest.TestCase):
    def test_replace_swaps_callable(self):
        calls = []

        def old():
            calls.append("old")

        def new():
            calls.append("new")

        p = Pipeline()
        p.append("step", old)
        p.replace("step", new)
        for fn in p:
            fn()
        self.assertEqual(calls, ["new"])

    def test_replace_preserves_position(self):
        p = Pipeline()
        p.append("a", lambda: None)
        p.append("b", lambda: None)
        p.append("c", lambda: None)
        p.replace("b", lambda: None)
        self.assertEqual(p.names(), ["a", "b", "c"])

    def test_replace_nonexistent_raises_key_error(self):
        p = Pipeline()
        with self.assertRaises(KeyError):
            p.replace("ghost", lambda: None)


class TestPipelineClear(unittest.TestCase):
    def test_clear_empties_pipeline(self):
        p = Pipeline()
        p.append("a", lambda: None)
        p.append("b", lambda: None)
        p.clear()
        self.assertEqual(len(p), 0)
        self.assertEqual(p.names(), [])

    def test_clear_then_re_append(self):
        p = Pipeline()
        p.append("old", lambda: None)
        p.clear()
        p.append("new", lambda: None)
        self.assertEqual(p.names(), ["new"])


class TestPipelineContains(unittest.TestCase):
    def test_contains_existing_step(self):
        p = Pipeline()
        p.append("a", lambda: None)
        self.assertIn("a", p)

    def test_does_not_contain_absent_step(self):
        p = Pipeline()
        self.assertNotIn("a", p)


class TestPipelineOrderingGuarantee(unittest.TestCase):
    """Verify the critical ordering invariant: merge_hist before sid_replace.

    This reproduces the registration pattern used by SEQUERDataset and
    BaseDataset._finish_init without requiring a full dataset instantiation.
    """

    def _simulate_sequer_pipeline(self):
        """Return a Stage-3 Pipeline built with the SEQUER registration order:
        SEQUERDataset appends merge_hist first, then super() (BaseDataset)
        appends sid_replace last.
        """
        p = Pipeline()
        p.append("merge_hist", lambda data: None)  # SEQUERDataset before super()
        p.append("sid_replace", lambda data: None)  # BaseDataset._finish_init end
        return p

    def test_merge_hist_index_less_than_sid_replace(self):
        p = self._simulate_sequer_pipeline()
        names = p.names()
        self.assertLess(names.index("merge_hist"), names.index("sid_replace"))

    def test_merge_hist_callable_runs_before_sid_replace(self):
        execution_order = []
        p = Pipeline()
        p.append("merge_hist", lambda data: execution_order.append("merge_hist"))
        p.append("sid_replace", lambda data: execution_order.append("sid_replace"))
        for fn in p:
            fn({})
        self.assertEqual(execution_order, ["merge_hist", "sid_replace"])

    def test_sasrec_pipeline_has_same_ordering_guarantee(self):
        """SASRecDataset uses the identical registration pattern."""
        p = self._simulate_sequer_pipeline()
        self.assertLess(p.names().index("merge_hist"), p.names().index("sid_replace"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mapper(items):
    """Build a Mapper from a plain list of item strings/ints."""
    return Mapper(pd.Series(items, index=range(len(items))))


# ---------------------------------------------------------------------------
# Mapper
# ---------------------------------------------------------------------------


class TestMapperConstruction(unittest.TestCase):
    def test_valid_monotonic_index_succeeds(self):
        m = _make_mapper(["a", "b", "c"])
        self.assertEqual(len(m), 3)

    def test_non_monotonic_but_complete_index_is_sorted_and_accepted(self):
        """Mapper sorts the index first, so a permuted-but-complete index is valid."""
        s = pd.Series(["c", "a", "b"], index=[2, 0, 1])
        m = Mapper(s)
        self.assertEqual(len(m), 3)
        # After sort_index: 0->"a", 1->"b", 2->"c"
        self.assertEqual(m.get_item(0), "a")
        self.assertEqual(m.get_item(1), "b")
        self.assertEqual(m.get_item(2), "c")

    def test_missing_index_value_raises_assertion_error(self):
        s = pd.Series(["a", "b"], index=[0, 2])  # skips 1
        with self.assertRaises(AssertionError):
            Mapper(s)

    def test_single_item_mapper(self):
        m = _make_mapper(["only"])
        self.assertEqual(len(m), 1)
        self.assertEqual(m.get_item(0), "only")

    def test_integer_items(self):
        m = _make_mapper([10, 20, 30])
        self.assertEqual(m.get_idx(20), 1)


class TestMapperLookup(unittest.TestCase):
    def setUp(self):
        self.items = ["apple", "banana", "cherry"]
        self.m = _make_mapper(self.items)

    def test_get_item_roundtrip(self):
        for i, item in enumerate(self.items):
            self.assertEqual(self.m.get_item(i), item)

    def test_get_idx_roundtrip(self):
        for i, item in enumerate(self.items):
            self.assertEqual(self.m.get_idx(item), i)

    def test_get_items_returns_list(self):
        result = self.m.get_items([0, 2])
        self.assertEqual(result, ["apple", "cherry"])

    def test_get_idxs_returns_list(self):
        result = self.m.get_idxs(["banana", "apple"])
        self.assertEqual(result, [1, 0])

    def test_get_vocab_contains_all_items(self):
        vocab = self.m.get_vocab()
        self.assertIsInstance(vocab, list)
        self.assertEqual(set(vocab), set(self.items))

    def test_len_equals_number_of_items(self):
        self.assertEqual(len(self.m), 3)

    def test_get_item_out_of_range_raises(self):
        with self.assertRaises(KeyError):
            self.m.get_item(99)

    def test_get_idx_unknown_item_raises(self):
        with self.assertRaises(KeyError):
            self.m.get_idx("durian")


class TestMapperAddItems(unittest.TestCase):
    def test_add_new_items_increases_vocab(self):
        m = _make_mapper(["a", "b"])
        m.add_items(["c", "d"])
        self.assertEqual(len(m), 4)
        self.assertIn("c", m.get_vocab())
        self.assertIn("d", m.get_vocab())

    def test_add_items_assigns_sequential_indices(self):
        m = _make_mapper(["a", "b"])
        m.add_items(["c"])
        self.assertEqual(m.get_idx("c"), 2)

    def test_add_duplicate_items_ignored(self):
        m = _make_mapper(["a", "b"])
        m.add_items(["a", "b"])  # already present
        self.assertEqual(len(m), 2)

    def test_add_items_partial_overlap(self):
        m = _make_mapper(["a", "b"])
        m.add_items(["b", "c"])  # "b" exists, "c" is new
        self.assertEqual(len(m), 3)
        self.assertEqual(m.get_idx("c"), 2)

    def test_existing_indices_preserved_after_add(self):
        m = _make_mapper(["x", "y"])
        m.add_items(["z"])
        self.assertEqual(m.get_idx("x"), 0)
        self.assertEqual(m.get_idx("y"), 1)


class TestMapperReplaceItems(unittest.TestCase):
    def test_replace_updates_item2idx(self):
        m = _make_mapper(["old_a", "b"])
        m.replace_items({"old_a": "new_a"})
        self.assertEqual(m.get_idx("new_a"), 0)

    def test_replace_updates_idx2item(self):
        m = _make_mapper(["old_a", "b"])
        m.replace_items({"old_a": "new_a"})
        self.assertEqual(m.get_item(0), "new_a")

    def test_replace_unknown_key_is_noop(self):
        m = _make_mapper(["a", "b"])
        m.replace_items({"nonexistent": "x"})
        self.assertEqual(m.get_vocab(), ["a", "b"])


class TestMapperEncLen(unittest.TestCase):
    def test_enc_len_returns_one_when_no_enc_column(self):
        m = _make_mapper(["a", "b"])
        self.assertEqual(m.enc_len(), 1)

    def test_enc_len_returns_max_token_count_when_enc_column_present(self):
        m = _make_mapper(["a", "b", "c"])
        m.idx2item["enc"] = pd.Series([[1, 2], [3, 4, 5], [6]], index=[0, 1, 2])
        self.assertEqual(m.enc_len(), 3)


class TestMapperDropEmbed(unittest.TestCase):
    def test_drop_embed_noop_when_no_embs(self):
        m = _make_mapper(["a"])
        m.drop_embed()  # should not raise

    def test_drop_embed_removes_embs_attribute(self):
        m = _make_mapper(["a"])
        m.embs = np.array([[0.1, 0.2]])
        m.drop_embed()
        self.assertFalse(hasattr(m, "embs"))


# ---------------------------------------------------------------------------
# EntityTokenizer
# ---------------------------------------------------------------------------


class TestEntityTokenizer(unittest.TestCase):
    def test_empty_init_has_zero_entities(self):
        et = EntityTokenizer()
        self.assertEqual(len(et), 0)

    def test_special_tokens_added_on_init(self):
        et = EntityTokenizer(special_tokens={"pad": "<pad>", "unk": "<unk>"})
        self.assertEqual(len(et), 2)
        self.assertEqual(et.encode("<pad>"), 0)
        self.assertEqual(et.encode("<unk>"), 1)

    def test_add_entity_increments_length(self):
        et = EntityTokenizer()
        et.add_entity("user_1")
        self.assertEqual(len(et), 1)

    def test_add_entity_duplicate_ignored(self):
        et = EntityTokenizer()
        et.add_entity("e")
        et.add_entity("e")
        self.assertEqual(len(et), 1)

    def test_encode_returns_correct_index(self):
        et = EntityTokenizer()
        et.add_entity("apple")
        et.add_entity("banana")
        self.assertEqual(et.encode("apple"), 0)
        self.assertEqual(et.encode("banana"), 1)

    def test_add_entities_bulk(self):
        et = EntityTokenizer()
        et.add_entities(["a", "b", "c"])
        self.assertEqual(len(et), 3)

    def test_add_entities_no_duplicates(self):
        et = EntityTokenizer()
        et.add_entity("a")
        et.add_entities(["a", "b"])  # "a" already exists
        self.assertEqual(len(et), 2)

    def test_batch_encode(self):
        et = EntityTokenizer()
        et.add_entities(["x", "y", "z"])
        result = et.batch_encode(["z", "x"])
        self.assertEqual(list(result), [et.encode("z"), et.encode("x")])

    def test_batch_decode_returns_correct_entities(self):
        et = EntityTokenizer()
        et.add_entities(["cat", "dog", "fish"])
        idxs = [et.encode("dog"), et.encode("cat")]
        decoded = et.batch_decode(idxs)
        self.assertEqual(list(decoded), ["dog", "cat"])

    def test_add_entities_with_special_tokens_offsets_correctly(self):
        et = EntityTokenizer(special_tokens={"pad": "<pad>"})
        et.add_entities(["item_1", "item_2"])
        # <pad> is at 0; item_1 and item_2 follow
        self.assertEqual(et.encode("<pad>"), 0)
        self.assertIn(et.encode("item_1"), {1, 2})
        self.assertIn(et.encode("item_2"), {1, 2})


# ---------------------------------------------------------------------------
# FieldManager
# ---------------------------------------------------------------------------


class TestFieldManager(unittest.TestCase):
    def setUp(self):
        self.fm = FieldManager()

    def test_default_orig_contains_user_and_item(self):
        self.assertIn(Key.USER, self.fm.orig)
        self.assertIn(Key.ITEM, self.fm.orig)

    def test_add_original_appends_new_column(self):
        self.fm.add_original("new_col")
        self.assertIn("new_col", self.fm.orig)

    def test_add_original_no_duplicates(self):
        self.fm.add_original(Key.USER)  # already present
        count = self.fm.orig.count(Key.USER)
        self.assertEqual(count, 1)

    def test_add_original_accepts_list(self):
        self.fm.add_original(["col_a", "col_b"])
        self.assertIn("col_a", self.fm.orig)
        self.assertIn("col_b", self.fm.orig)

    def test_add_float_appends_new_column(self):
        self.fm.add_float("score")
        self.assertIn("score", self.fm.float)

    def test_add_float_no_duplicates(self):
        self.fm.add_float("score")
        self.fm.add_float("score")
        self.assertEqual(self.fm.float.count("score"), 1)

    def test_add_encoding_adds_to_enc_dict(self):
        self.fm.add_encoding("text_col", "text_mapper")
        self.assertIn("text_col", self.fm.enc)
        self.assertEqual(self.fm.enc["text_col"], "text_mapper")

    def test_add_encoding_overwrites_existing_mapper(self):
        self.fm.add_encoding("text_col", "mapper_v1")
        self.fm.add_encoding("text_col", "mapper_v2")
        self.assertEqual(self.fm.enc["text_col"], "mapper_v2")

    def test_add_embedding_adds_to_emb_dict(self):
        self.fm.add_embedding("item_col", "item_emb_mapper")
        self.assertIn("item_col", self.fm.emb)

    def test_add_processed_appends_column(self):
        self.fm.add_processed("graph_feat")
        self.assertIn("graph_feat", self.fm.processed)

    def test_add_processed_no_duplicates(self):
        self.fm.add_processed("x")
        self.fm.add_processed("x")
        self.assertEqual(self.fm.processed.count("x"), 1)

    def test_add_pad_mask_appends_column(self):
        self.fm.add_pad_mask("input_ids")
        self.assertIn("input_ids", self.fm.pad_mask)

    def test_add_pad_mask_no_duplicates(self):
        self.fm.add_pad_mask("ids")
        self.fm.add_pad_mask("ids")
        self.assertEqual(self.fm.pad_mask.count("ids"), 1)

    def test_get_all_columns_includes_defaults(self):
        cols = self.fm.get_all_columns()
        self.assertIn(Key.USER, cols)
        self.assertIn(Key.ITEM, cols)
        # keep_cols defaults
        self.assertIn(Key.EXP, cols)
        self.assertIn(Key.RATING, cols)

    def test_get_all_columns_includes_added_orig(self):
        self.fm.add_original("extra")
        self.assertIn("extra", self.fm.get_all_columns())

    def test_get_all_columns_includes_enc_keys(self):
        self.fm.add_encoding("text_enc", "mapper")
        self.assertIn("text_enc", self.fm.get_all_columns())

    def test_get_all_columns_includes_emb_keys(self):
        self.fm.add_embedding("item_emb", "mapper")
        self.assertIn("item_emb", self.fm.get_all_columns())

    def test_to_dict_has_expected_keys(self):
        d = self.fm.to_dict()
        for key in ("orig", "float", "enc", "emb", "processed", "pad_mask"):
            with self.subTest(key=key):
                self.assertIn(key, d)

    def test_to_dict_enc_reflects_added_encoding(self):
        self.fm.add_encoding("col", "mapper")
        d = self.fm.to_dict()
        self.assertEqual(d["enc"]["col"], "mapper")


# ---------------------------------------------------------------------------
# pad2longest
# ---------------------------------------------------------------------------


class TestPad2Longest(unittest.TestCase):
    def test_equal_length_tensors_stack_unchanged(self):
        ts = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
        result = pad2longest(ts, side="left")
        self.assertEqual(result.shape, (2, 3))
        torch.testing.assert_close(result[0], torch.tensor([1, 2, 3]))

    def test_left_padding_shorter_tensor(self):
        ts = [torch.tensor([1, 2, 3]), torch.tensor([4, 5])]
        result = pad2longest(ts, side="left", value=0)
        self.assertEqual(result.shape, (2, 3))
        self.assertEqual(result[1].tolist(), [0, 4, 5])

    def test_right_padding_shorter_tensor(self):
        ts = [torch.tensor([1, 2, 3]), torch.tensor([4, 5])]
        result = pad2longest(ts, side="right", value=0)
        self.assertEqual(result.shape, (2, 3))
        self.assertEqual(result[1].tolist(), [4, 5, 0])

    def test_custom_pad_value(self):
        ts = [torch.tensor([1, 2, 3]), torch.tensor([7])]
        result = pad2longest(ts, side="right", value=-1)
        self.assertEqual(result[1].tolist(), [7, -1, -1])

    def test_single_tensor_returns_stacked(self):
        ts = [torch.tensor([5, 6])]
        result = pad2longest(ts, side="left")
        self.assertEqual(result.shape, (1, 2))

    def test_2d_tensors_padded_along_last_dim(self):
        # Tensors differ only along the last dim; pass dim=-1 explicitly.
        ts = [torch.ones(2, 3), torch.ones(2, 5)]
        result = pad2longest(ts, dim=-1, side="right", value=0)
        self.assertEqual(result.shape, (2, 2, 5))
        # First tensor: last 2 cols of each row should be 0
        self.assertEqual(result[0, 0, 3:].tolist(), [0.0, 0.0])

    def test_non_last_dim_padding_dim_0(self):
        # Each tensor has shape (len, 4); pad along dim=0
        ts = [torch.ones(3, 4), torch.ones(5, 4)]
        result = pad2longest(ts, dim=0, side="right", value=0)
        self.assertEqual(result.shape, (2, 5, 4))
        # First tensor rows 3-4 should be zero
        self.assertEqual(result[0, 3:].sum().item(), 0.0)

    def test_three_tensors_different_lengths(self):
        ts = [torch.tensor([1]), torch.tensor([1, 2, 3]), torch.tensor([1, 2])]
        result = pad2longest(ts, side="right", value=0)
        self.assertEqual(result.shape, (3, 3))
        self.assertEqual(result[0].tolist(), [1, 0, 0])
        self.assertEqual(result[1].tolist(), [1, 2, 3])
        self.assertEqual(result[2].tolist(), [1, 2, 0])

    def test_bool_tensors_padded_with_zero(self):
        ts = [torch.BoolTensor([True, True, True]), torch.BoolTensor([True, False])]
        result = pad2longest(ts, side="right", value=0)
        self.assertEqual(result.shape, (2, 3))
        self.assertFalse(result[1, 2].item())


# ---------------------------------------------------------------------------
# pad2len
# ---------------------------------------------------------------------------


class TestPad2Len(unittest.TestCase):
    def test_left_padding_to_target_length(self):
        ts = [torch.tensor([1, 2]), torch.tensor([3])]
        result = pad2len(ts, length=4, side="left", value=0)
        self.assertEqual(result.shape, (2, 4))
        self.assertEqual(result[0].tolist(), [0, 0, 1, 2])
        self.assertEqual(result[1].tolist(), [0, 0, 0, 3])

    def test_right_padding_to_target_length(self):
        ts = [torch.tensor([1, 2]), torch.tensor([3, 4, 5])]
        result = pad2len(ts, length=5, side="right", value=0)
        self.assertEqual(result.shape, (2, 5))
        self.assertEqual(result[0].tolist(), [1, 2, 0, 0, 0])
        self.assertEqual(result[1].tolist(), [3, 4, 5, 0, 0])

    def test_tensor_already_at_target_length(self):
        ts = [torch.tensor([1, 2, 3])]
        result = pad2len(ts, length=3, side="right", value=0)
        self.assertEqual(result.shape, (1, 3))
        self.assertEqual(result[0].tolist(), [1, 2, 3])

    def test_custom_pad_value_filled_correctly(self):
        ts = [torch.tensor([5])]
        result = pad2len(ts, length=3, side="right", value=99)
        self.assertEqual(result[0].tolist(), [5, 99, 99])

    def test_single_element_tensors(self):
        ts = [torch.tensor([7]), torch.tensor([8])]
        result = pad2len(ts, length=2, side="right", value=0)
        self.assertEqual(result.shape, (2, 2))

    def test_2d_tensors_padded_along_last_dim(self):
        # Tensors differ only along the last dim; pass dim=-1 explicitly.
        ts = [torch.ones(2, 2), torch.ones(2, 3)]
        result = pad2len(ts, length=4, dim=-1, side="right", value=0)
        self.assertEqual(result.shape, (2, 2, 4))
        # Second tensor (2x3) padded to (2x4): last col should be 0
        self.assertEqual(result[1, 0, 3].item(), 0.0)

    def test_non_last_dim_padding(self):
        ts = [torch.ones(2, 4), torch.ones(3, 4)]
        result = pad2len(ts, length=5, dim=0, side="right", value=0)
        self.assertEqual(result.shape, (2, 5, 4))
        self.assertEqual(result[0, 2:].sum().item(), 0.0)


# ---------------------------------------------------------------------------
# sliding_w_hist
# ---------------------------------------------------------------------------


class TestSlidingWHist(unittest.TestCase):
    def _make_group(self, n):
        return pd.DataFrame({"val": range(n)}, index=range(n))

    def test_first_row_always_has_empty_history(self):
        g = self._make_group(5)
        hist = sliding_w_hist(g, max_l=10)
        self.assertEqual(hist[0], [])

    def test_second_row_contains_first_index(self):
        g = self._make_group(5)
        hist = sliding_w_hist(g, max_l=10)
        self.assertEqual(hist[1], [0])

    def test_history_grows_up_to_max_l(self):
        g = self._make_group(5)
        hist = sliding_w_hist(g, max_l=10)
        self.assertEqual(hist[4], [0, 1, 2, 3])

    def test_history_clipped_at_max_l(self):
        g = self._make_group(6)
        hist = sliding_w_hist(g, max_l=3)
        # Row 5: indices [2, 3, 4] (window of 3 before row 5)
        self.assertEqual(hist[5], [2, 3, 4])

    def test_max_l_zero_always_empty_history(self):
        g = self._make_group(4)
        hist = sliding_w_hist(g, max_l=0)
        for h in hist:
            self.assertEqual(h, [])

    def test_max_l_one_only_immediate_predecessor(self):
        g = self._make_group(4)
        hist = sliding_w_hist(g, max_l=1)
        self.assertEqual(hist[0], [])
        self.assertEqual(hist[1], [0])
        self.assertEqual(hist[2], [1])
        self.assertEqual(hist[3], [2])

    def test_length_equals_group_size(self):
        g = self._make_group(7)
        hist = sliding_w_hist(g, max_l=5)
        self.assertEqual(len(hist), 7)

    def test_single_row_group(self):
        g = self._make_group(1)
        hist = sliding_w_hist(g, max_l=10)
        self.assertEqual(len(hist), 1)
        self.assertEqual(hist[0], [])

    def test_history_contains_dataframe_index_values(self):
        """History entries are actual index values, not positional ints."""
        g = pd.DataFrame({"val": [10, 20, 30]}, index=[100, 200, 300])
        hist = sliding_w_hist(g, max_l=5)
        self.assertEqual(hist[0], [])
        self.assertEqual(hist[1], [100])
        self.assertEqual(hist[2], [100, 200])


# ---------------------------------------------------------------------------
# get_transform_mapping
# ---------------------------------------------------------------------------


class TestGetTransformMapping(unittest.TestCase):
    def test_simple_categorical_series_zero_indexed(self):
        s = pd.Series(["b", "a", "c", "a"])
        transformed, mapper = get_transform_mapping(s)
        # "a"->0, "b"->1, "c"->2  (sorted alphabetically)
        self.assertEqual(mapper[0], "a")
        self.assertEqual(mapper[1], "b")
        self.assertEqual(mapper[2], "c")

    def test_transformed_data_consistent_with_mapper(self):
        s = pd.Series(["cat", "dog", "cat"])
        transformed, mapper = get_transform_mapping(s)
        # verify each transformed value maps back to original
        for orig, enc in zip(s, transformed):
            self.assertEqual(mapper[enc], orig)

    def test_mapper_series_indexed_by_encoding(self):
        s = pd.Series([10, 20, 30, 10])
        _, mapper = get_transform_mapping(s)
        self.assertIn(0, mapper.index)
        self.assertIn(1, mapper.index)
        self.assertIn(2, mapper.index)
        self.assertEqual(len(mapper), 3)

    def test_nested_list_series_all_elements_encoded(self):
        s = pd.Series([[1, 3], [2, 1]])
        transformed, mapper = get_transform_mapping(s)
        # All three unique values should be in mapper
        unique_vals = set()
        for lst in transformed:
            unique_vals.update(lst)
        self.assertEqual(unique_vals, set(mapper.index.tolist()))

    def test_nested_list_preserves_structure(self):
        s = pd.Series([[10, 20], [30]])
        transformed, _ = get_transform_mapping(s)
        self.assertIsInstance(transformed.iloc[0], list)
        self.assertEqual(len(transformed.iloc[0]), 2)
        self.assertIsInstance(transformed.iloc[1], list)
        self.assertEqual(len(transformed.iloc[1]), 1)

    def test_single_unique_value_mapped_to_zero(self):
        s = pd.Series(["x", "x", "x"])
        transformed, mapper = get_transform_mapping(s)
        self.assertEqual(set(transformed.tolist()), {0})
        self.assertEqual(mapper[0], "x")

    def test_integer_series_sorted_mapping(self):
        s = pd.Series([5, 3, 1, 3])
        transformed, mapper = get_transform_mapping(s)
        # sorted: 1->0, 3->1, 5->2
        self.assertEqual(mapper[0], 1)
        self.assertEqual(mapper[1], 3)
        self.assertEqual(mapper[2], 5)


# ---------------------------------------------------------------------------
# postprocessing
# ---------------------------------------------------------------------------


class TestPostprocessing(unittest.TestCase):
    def test_possessive_s_spaced(self):
        self.assertEqual(postprocessing("it's"), "it 's")

    def test_apostrophe_m_spaced(self):
        self.assertIn(" 'm", postprocessing("I'm happy"))

    def test_apostrophe_ve_spaced(self):
        self.assertIn(" 've", postprocessing("I've done"))

    def test_negation_contraction_spaced(self):
        self.assertIn(" n't", postprocessing("won't"))

    def test_apostrophe_re_spaced(self):
        self.assertIn(" 're", postprocessing("they're"))

    def test_apostrophe_d_spaced(self):
        self.assertIn(" 'd", postprocessing("I'd"))

    def test_apostrophe_ll_spaced(self):
        self.assertIn(" 'll", postprocessing("I'll"))

    def test_parentheses_spaced(self):
        result = postprocessing("(hello)")
        self.assertIn("( hello )", result)

    def test_multiple_commas_collapsed(self):
        result = postprocessing("a,,b")
        self.assertNotIn(",,", result)
        self.assertIn(",", result)

    def test_colon_replaced_by_comma_space(self):
        result = postprocessing("a:b")
        self.assertNotIn(":", result)
        self.assertIn(",", result)

    def test_semicolons_replaced_by_period(self):
        result = postprocessing("a;b")
        self.assertNotIn(";", result)
        self.assertIn(".", result)

    def test_multiple_exclamation_collapsed(self):
        result = postprocessing("wow!!!")
        self.assertNotIn("!!!", result)
        self.assertIn("!", result)

    def test_multiple_question_marks_collapsed(self):
        result = postprocessing("really???")
        self.assertNotIn("???", result)
        self.assertIn("?", result)

    def test_multiple_periods_collapsed(self):
        result = postprocessing("end...")
        self.assertNotIn("...", result)
        self.assertIn(".", result)

    def test_leading_trailing_whitespace_stripped(self):
        result = postprocessing("  hello  ")
        self.assertEqual(result, result.strip())

    def test_multiple_spaces_collapsed_to_one(self):
        result = postprocessing("hello   world")
        self.assertNotIn("  ", result)

    def test_clean_string_passes_through_mostly_unchanged(self):
        s = "the product is great"
        result = postprocessing(s)
        self.assertIn("great", result)

    def test_empty_string_returns_empty(self):
        result = postprocessing("")
        self.assertEqual(result, "")


if __name__ == "__main__":
    unittest.main()
