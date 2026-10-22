import unittest

import pandas as pd
import torch

from nlp_xrs.data.utils import compute_concept_targets
from nlp_xrs.utils.constants import Key

CODEBOOK = 6


def _acts_mapper(rows):
    """rows: review_id -> (concept_ids, acts), with the layout produced by ``get_acts_mapper``."""
    return pd.DataFrame(
        {
            "review_id": list(rows),
            "concept_id": [c for c, _ in rows.values()],
            "act": [a for _, a in rows.values()],
        }
    ).set_index("review_id")


def _expected(*reviews):
    dist = torch.zeros(CODEBOOK)
    for concepts, acts in reviews:
        row = torch.zeros(CODEBOOK)
        row[concepts] = torch.tensor(acts)
        dist += row / len(reviews)
    return dist / dist.sum()


class TestComputeConceptTargets(unittest.TestCase):
    def setUp(self):
        self.acts = {
            10: ([0, 1], [1.0, 3.0]),
            11: ([1, 2], [2.0, 2.0]),
            12: ([3, 4], [1.0, 1.0]),
            13: ([5, 0], [4.0, 1.0]),
            # Test-split reviews: item 2 also has a test review, item 3 only appears in test.
            20: ([5, 4], [9.0, 9.0]),
            21: ([2, 3], [5.0, 5.0]),
        }
        self.data = pd.DataFrame(
            {
                Key.ITEM: [2, 0, 0, 1, 2, 3],
                Key.REV: [13, 10, 11, 12, 20, 21],
            },
            index=[100, 101, 102, 103, 104, 105],
        )
        self.train_ixs = [100, 101, 102, 103]

    def _targets(self, num_rows=5):
        train_data = self.data.loc[self.train_ixs]
        return compute_concept_targets(train_data, Key.ITEM, _acts_mapper(self.acts), num_rows, CODEBOOK)

    def test_rows_are_indexed_by_item_id(self):
        targets = self._targets()
        torch.testing.assert_close(targets[0], _expected(self.acts[10], self.acts[11]))
        torch.testing.assert_close(targets[1], _expected(self.acts[12]))
        torch.testing.assert_close(targets[2], _expected(self.acts[13]))

    def test_test_reviews_are_not_used(self):
        targets = self._targets()
        torch.testing.assert_close(targets[2], _expected(self.acts[13]))
        self.assertEqual(targets[2, 4].item(), 0.0)

    def test_cold_start_and_unseen_items_get_zero_rows(self):
        targets = self._targets()
        self.assertTrue(torch.all(targets[3] == 0))
        self.assertTrue(torch.all(targets[4] == 0))
        self.assertTrue(torch.isfinite(targets).all())

    def test_training_slice_matches_tokenizer_selection(self):
        targets = self._targets()
        train_data = self.data.loc[self.train_ixs]
        # Same selection as SemanticIDTokenizer.train: unique training ids, first-seen order.
        train_items = train_data[Key.ITEM].unique().tolist()
        self.assertEqual(train_items, [2, 0, 1])
        sliced = targets[train_items]
        for row, item in zip(sliced, train_items):
            torch.testing.assert_close(row, targets[item])
            self.assertAlmostEqual(row.sum().item(), 1.0, places=5)

    def test_list_valued_key_uses_last_element(self):
        train_data = self.data.loc[self.train_ixs].copy()
        train_data[Key.ITEM] = [[4, 2], [4, 0], [3, 0], [1]]
        targets = compute_concept_targets(train_data, Key.ITEM, _acts_mapper(self.acts), 5, CODEBOOK)
        torch.testing.assert_close(targets[0], _expected(self.acts[10], self.acts[11]))
        self.assertTrue(torch.all(targets[4] == 0))

    def test_input_frame_is_not_modified(self):
        train_data = self.data.loc[self.train_ixs].copy()
        train_data[Key.ITEM] = [[2], [0], [0], [1]]
        before = train_data[Key.ITEM].tolist()
        compute_concept_targets(train_data, Key.ITEM, _acts_mapper(self.acts), 5, CODEBOOK)
        self.assertEqual(train_data[Key.ITEM].tolist(), before)


if __name__ == "__main__":
    unittest.main()
