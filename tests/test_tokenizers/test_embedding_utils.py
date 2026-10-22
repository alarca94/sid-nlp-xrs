"""
Unit tests for item-embedding utility functions in
nlp_xrs.semantic_ids.utils.

Covers: map_text_mode, _get_entity_review_texts, _build_metadata_texts,
_encode_texts_to_item_embeddings, _embed_and_combine_per_item.
"""

import unittest
from unittest.mock import MagicMock

import pandas as pd
import torch

from nlp_xrs.semantic_ids.utils import (
    _build_metadata_texts,
    _embed_and_combine_per_item,
    _encode_texts_to_item_embeddings,
    _get_entity_review_texts,
    map_text_mode,
)
from nlp_xrs.utils.constants import Key

# ---------------------------------------------------------------------------
# Minimal Mapper stub – avoids importing the real Mapper (heavy deps)
# ---------------------------------------------------------------------------


class _StubMapper:
    """Lightweight stand-in for nlp_xrs.data.utils.Mapper."""

    def __init__(self, vocab):
        """
        Args:
            vocab: list of decoded strings; position i is the decoded value
                   for encoded index i.
        """
        self._vocab = list(vocab)
        self.item2idx = pd.Series(range(len(self._vocab)), index=self._vocab, dtype="int64")
        self.idx2item = pd.Series(self._vocab, dtype="object").to_frame("raw")

    def get_item(self, idx: int):
        return self._vocab[idx]

    def get_items(self, idxs):
        return [self._vocab[i] for i in idxs]

    def __len__(self):
        return len(self._vocab)


# ---------------------------------------------------------------------------
# map_text_mode
# ---------------------------------------------------------------------------


class TestMapTextMode(unittest.TestCase):
    """Tests for map_text_mode using the 'beauty' template."""

    TEMPLATE = "Title: title; Brand: brand; Categories: categories; Price: price"

    def _beauty_row(self, **overrides):
        """Return a single-row DataFrame with default beauty columns."""
        row = {
            "title": "Widget Pro",
            "brand": "Acme",
            "categories": "Tools",
            "price": "9.99",
        }
        row.update(overrides)
        return pd.DataFrame([row])

    # --- exact-value tests --------------------------------------------------

    def test_basic_substitution(self):
        df = self._beauty_row()
        result = map_text_mode("beauty", df)
        self.assertEqual(
            result.iloc[0],
            "Title: Widget Pro; Brand: Acme; Categories: Tools; Price: 9.99",
        )

    def test_nan_value_becomes_empty_string(self):
        df = self._beauty_row(brand=float("nan"))
        result = map_text_mode("beauty", df)
        self.assertIn("Brand: ;", result.iloc[0])
        self.assertNotIn("nan", result.iloc[0])

    def test_none_value_becomes_empty_string(self):
        df = self._beauty_row(price=None)
        result = map_text_mode("beauty", df)
        self.assertIn("Price: ", result.iloc[0])
        # The trailing empty-string replacement leaves "Price: " at end
        self.assertFalse(result.iloc[0].endswith("None"))

    def test_list_value_is_joined_with_comma(self):
        df = self._beauty_row(categories=["Electronics", "Gadgets"])
        result = map_text_mode("beauty", df)
        self.assertIn("Categories: Electronics, Gadgets", result.iloc[0])

    def test_list_with_none_entries_skips_none(self):
        df = self._beauty_row(categories=["Electronics", None, "Gadgets"])
        result = map_text_mode("beauty", df)
        self.assertIn("Categories: Electronics, Gadgets", result.iloc[0])

    def test_multiple_rows_returns_one_string_per_row(self):
        df = pd.DataFrame(
            [
                {
                    "title": "Item A",
                    "brand": "B1",
                    "categories": "C1",
                    "price": "1.00",
                },
                {
                    "title": "Item B",
                    "brand": "B2",
                    "categories": "C2",
                    "price": "2.00",
                },
            ]
        )
        result = map_text_mode("beauty", df)
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), 2)
        self.assertIn("Item A", result.iloc[0])
        self.assertIn("Item B", result.iloc[1])

    # --- error cases --------------------------------------------------------

    def test_unknown_dataset_raises_value_error(self):
        with self.assertRaises(ValueError):
            map_text_mode("no_such_dataset", pd.DataFrame([{"x": 1}]))

    def test_no_matching_columns_raises_value_error(self):
        # DataFrame columns do not appear in the beauty template
        df = pd.DataFrame([{"irrelevant": "foo"}])
        with self.assertRaises(ValueError):
            map_text_mode("beauty", df)


# ---------------------------------------------------------------------------
# _get_entity_review_texts
# ---------------------------------------------------------------------------


class TestGetEntityReviewTexts(unittest.TestCase):
    """Tests for _get_entity_review_texts."""

    def test_scalar_text_values_decoded_and_grouped_by_item(self):
        """Scalar encoded ids are decoded and grouped into per-item lists."""
        # vocab: 0->"good", 1->"great", 2->"bad"
        mapper = _StubMapper(["good", "great", "bad"])
        inter_df = pd.DataFrame({Key.ITEM: [0, 0, 1], Key.EXP: [0, 1, 2]})
        result = _get_entity_review_texts(inter_df, {Key.EXP: mapper}, entity="item")
        self.assertEqual(result[0], ["good", "great"])
        self.assertEqual(result[1], ["bad"])

    def test_list_text_values_decoded_and_joined_into_string(self):
        """List-of-token-id values are decoded then joined into one string."""
        mapper = _StubMapper(["hello", "world", "foo"])
        inter_df = pd.DataFrame({Key.ITEM: [0, 1], Key.EXP: [[0, 1], [2]]})
        result = _get_entity_review_texts(inter_df, {Key.EXP: mapper}, entity="item")
        self.assertEqual(result[0], ["hello world"])
        self.assertEqual(result[1], ["foo"])

    def test_auto_selects_review_column_when_present(self):
        """Key.REV is used automatically when it exists in the DataFrame."""
        rev_mapper = _StubMapper(["review_word", "other_word"])
        exp_mapper = _StubMapper(["exp_word"])
        inter_df = pd.DataFrame({Key.ITEM: [0], Key.REV: [0], Key.EXP: [0]})
        result = _get_entity_review_texts(inter_df, {Key.REV: rev_mapper, Key.EXP: exp_mapper}, entity="item")
        self.assertEqual(result[0], ["review_word"])

    def test_falls_back_to_exp_column_when_review_absent(self):
        """Key.EXP is used when Key.REV is not present in the DataFrame."""
        mapper = _StubMapper(["exp_text"])
        inter_df = pd.DataFrame({Key.ITEM: [0], Key.EXP: [0]})
        result = _get_entity_review_texts(inter_df, {Key.EXP: mapper}, entity="item")
        self.assertEqual(result[0], ["exp_text"])

    def test_multiple_reviews_per_item_all_captured(self):
        """All reviews for an item are returned, preserving order."""
        mapper = _StubMapper(["w0", "w1", "w2", "w3"])
        inter_df = pd.DataFrame({Key.ITEM: [5, 5, 5], Key.EXP: [0, 1, 2]})
        result = _get_entity_review_texts(inter_df, {Key.EXP: mapper}, entity="item")
        self.assertEqual(result[5], ["w0", "w1", "w2"])


# ---------------------------------------------------------------------------
# _build_metadata_texts
# ---------------------------------------------------------------------------


class TestBuildMetadataTexts(unittest.TestCase):
    """Tests for _build_metadata_texts."""

    def _beauty_df_with_item_col(self, item_ids):
        """DataFrame with integer item IDs and beauty columns."""
        return pd.DataFrame(
            {
                Key.ITEM: item_ids,
                "title": [f"T{i}" for i in item_ids],
                "brand": [f"B{i}" for i in item_ids],
                "categories": [f"C{i}" for i in item_ids],
                "price": [f"{i}.00" for i in item_ids],
            }
        )

    # --- exact-value tests --------------------------------------------------

    def test_integer_item_ids_indexed_by_item_value(self):
        """Output Series is indexed by the integer item-ID values."""
        df = self._beauty_df_with_item_col([2, 5, 7])
        result = _build_metadata_texts(df, mappers={}, dataset_name="beauty")
        self.assertIsInstance(result, pd.Series)
        for iid in (2, 5, 7):
            self.assertIn(iid, result.index)
        self.assertIn("T2", result.loc[2])

    def test_item_ids_taken_from_dataframe_index_when_no_item_column(self):
        """When Key.ITEM is absent, the DataFrame index is used as item IDs."""
        df = pd.DataFrame(
            {
                "title": ["Widget"],
                "brand": ["ACME"],
                "categories": ["Tools"],
                "price": ["4.99"],
            },
            index=pd.Index([3]),
        )
        result = _build_metadata_texts(df, mappers={}, dataset_name="beauty")
        print(result)
        self.assertIn(3, result.index)
        self.assertIn("Widget", result.loc[3])

    def test_string_item_ids_mapped_to_encoded_integers(self):
        """String item IDs are replaced with their encoded integer indices."""
        raw_items = ["item_A", "item_B", "item_C"]
        item_mapper = _StubMapper(raw_items)
        df = pd.DataFrame(
            {
                Key.ITEM: ["item_A", "item_B", "item_C"],
                "title": ["The brand, categories and price are awesome", "TB", "TC"],
                "brand": ["BA", "BB", "BC"],
                "categories": ["CA", "CB", "CC"],
                "price": ["1.0", "2.0", "3.0"],
            }
        )
        result = _build_metadata_texts(df, mappers={Key.ITEM: item_mapper}, dataset_name="beauty")
        # item_A -> 0, item_B -> 1, item_C -> 2
        self.assertEqual(set(result.index), {0, 1, 2})
        self.assertEqual(
            result.loc[0],
            "Title: The brand, categories and price are awesome; Brand: BA; Categories: CA; Price: 1.0",
        )
        self.assertEqual(
            result.loc[1],
            "Title: TB; Brand: BB; Categories: CB; Price: 2.0",
        )

    def test_items_absent_from_mapper_are_filtered_out(self):
        """Items not in the mapper's vocabulary are excluded from the result."""
        item_mapper = _StubMapper(["item_A", "item_B"])
        df = pd.DataFrame(
            {
                Key.ITEM: ["item_A", "item_B", "item_UNKNOWN"],
                "title": ["TA", "TB", "TU"],
                "brand": ["B", "B", "B"],
                "categories": ["C", "C", "C"],
                "price": ["1.0", "2.0", "3.0"],
            }
        )
        result = _build_metadata_texts(df, mappers={Key.ITEM: item_mapper}, dataset_name="beauty")
        self.assertEqual(len(result), 2)
        # The encoded IDs should only be 0 and 1
        self.assertEqual(set(result.index), {0, 1})

    # --- error cases --------------------------------------------------------

    def test_missing_dataset_name_raises_value_error(self):
        df = self._beauty_df_with_item_col([0])
        with self.assertRaises(ValueError):
            _build_metadata_texts(df, mappers={})


# ---------------------------------------------------------------------------
# _encode_texts_to_item_embeddings
# ---------------------------------------------------------------------------


class TestEncodeTextsToItemEmbeddings(unittest.TestCase):
    """Tests for _encode_texts_to_item_embeddings."""

    def _mock_model(self, emb_dim: int = 16):
        model = MagicMock()
        model.encode.side_effect = lambda texts, **kw: torch.randn(len(texts), emb_dim)
        return model

    def test_output_shape(self):
        model = self._mock_model(emb_dim=16)
        item_texts = pd.Series(["text a", "text b"], index=[1, 3])
        result = _encode_texts_to_item_embeddings(model, item_texts, num_items=5)
        self.assertEqual(result.shape, (5, 16))

    def test_items_not_in_series_are_zero_vectors(self):
        """Rows for item IDs absent from item_texts must be all-zero."""
        emb_dim = 4
        model = MagicMock()
        model.encode.return_value = torch.ones(1, emb_dim) * 7.0
        item_texts = pd.Series(["text"], index=[2])
        result = _encode_texts_to_item_embeddings(model, item_texts, num_items=5)
        for idx in [0, 1, 3, 4]:
            self.assertTrue(
                torch.all(result[idx] == 0).item(),
                f"Row {idx} should be a zero vector",
            )

    def test_embeddings_placed_at_correct_row_indices(self):
        """Each text's embedding is stored at the row matching its item index."""
        emb_a = torch.tensor([1.0, 2.0, 3.0, 4.0])
        emb_b = torch.tensor([5.0, 6.0, 7.0, 8.0])
        model = MagicMock()
        model.encode.return_value = torch.stack([emb_a, emb_b])
        item_texts = pd.Series(["text a", "text b"], index=[0, 3])
        result = _encode_texts_to_item_embeddings(model, item_texts, num_items=5)
        torch.testing.assert_close(result[0], emb_a)
        torch.testing.assert_close(result[3], emb_b)


# ---------------------------------------------------------------------------
# _embed_and_combine_per_item
# ---------------------------------------------------------------------------


class TestEmbedAndCombinePerItem(unittest.TestCase):
    """Tests for _embed_and_combine_per_item."""

    def test_mean_combine_output_shape(self):
        model = MagicMock()
        model.encode.side_effect = lambda texts, **kw: torch.randn(len(texts), 8)
        item_texts = pd.Series([["r1", "r2"], ["r3"]], index=[0, 2])
        result = _embed_and_combine_per_item(model, item_texts, num_items=4, combine="mean")
        self.assertEqual(result.shape, (4, 8))

    def test_max_combine_output_shape(self):
        model = MagicMock()
        model.encode.side_effect = lambda texts, **kw: torch.randn(len(texts), 8)
        item_texts = pd.Series([["r1", "r2"], ["r3"]], index=[0, 2])
        result = _embed_and_combine_per_item(model, item_texts, num_items=4, combine="max")
        self.assertEqual(result.shape, (4, 8))

    def test_items_with_no_reviews_remain_zero(self):
        """Items absent from item_texts keep their zero-initialised embedding."""
        model = MagicMock()
        model.encode.side_effect = lambda texts, **kw: torch.ones(len(texts), 4)
        item_texts = pd.Series([["r"]], index=[1])
        result = _embed_and_combine_per_item(model, item_texts, num_items=3, combine="mean")
        self.assertTrue(torch.all(result[0] == 0).item())
        self.assertTrue(torch.all(result[2] == 0).item())

    def test_mean_is_arithmetic_mean_of_review_embeddings(self):
        """Mean mode must equal the row-wise average of per-review embeddings."""
        emb0 = torch.tensor([1.0, 0.0, 0.0, 0.0])
        emb1 = torch.tensor([0.0, 1.0, 0.0, 0.0])
        model = MagicMock()
        model.encode.side_effect = lambda texts, **kw: torch.stack([emb0, emb1])
        item_texts = pd.Series([["text0", "text1"]], index=[0])
        result = _embed_and_combine_per_item(model, item_texts, num_items=1, combine="mean")
        torch.testing.assert_close(result[0], (emb0 + emb1) / 2)

    def test_max_is_elementwise_max_of_review_embeddings(self):
        """Max mode must equal the element-wise maximum across review embeddings."""
        emb0 = torch.tensor([1.0, 0.0, 3.0, 0.0])
        emb1 = torch.tensor([0.0, 2.0, 0.0, 4.0])
        expected = torch.tensor([1.0, 2.0, 3.0, 4.0])
        model = MagicMock()
        model.encode.side_effect = lambda texts, **kw: torch.stack([emb0, emb1])
        item_texts = pd.Series([["text0", "text1"]], index=[0])
        result = _embed_and_combine_per_item(model, item_texts, num_items=1, combine="max")
        torch.testing.assert_close(result[0], expected)

    def test_empty_review_texts_raises_value_error(self):
        """Passing a Series with no reviews at all must raise ValueError."""
        model = MagicMock()
        with self.assertRaises(ValueError):
            _embed_and_combine_per_item(
                model,
                pd.Series([], dtype=object),
                num_items=5,
                combine="mean",
            )


if __name__ == "__main__":
    unittest.main()
