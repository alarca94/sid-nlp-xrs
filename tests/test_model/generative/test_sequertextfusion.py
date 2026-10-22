"""Semantic-ID coverage for all SEQUERTextFusion modes."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from nlp_xrs.models.generative.sequertextfusion import SEQUERTextFusion
from nlp_xrs.utils.constants import PAD_TOK, Key, Task

N_CODEBOOKS = 2
N_USERS = 5
N_RAW_ITEMS = 8
SID_VOCAB_SIZE = 13
N_WTOKENS = 20
EMSIZE = 8
ITEMS_PER_SEQUENCE = 3
TEXT_LENGTH = 4


def _make_tok():
    tok = MagicMock()
    tok.pad_token_id = 0
    tok.bos_token_id = 1
    tok.eos_token_id = 2
    return tok


def _make_data_info(use_concept=False):
    item_mapper = MagicMock()
    item_mapper.get_idx.side_effect = lambda token: SID_VOCAB_SIZE - 1 if token == PAD_TOK else 0
    sid_tokenizer = SimpleNamespace(n_codebooks=N_CODEBOOKS, codebook_sizes=[6, 6])
    data_info = SimpleNamespace(
        n_users=N_USERS,
        n_items=N_RAW_ITEMS,
        n_wtokens=N_WTOKENS,
        lengths={
            Key.USER: 1,
            Key.ITEM: ITEMS_PER_SEQUENCE * N_CODEBOOKS,
            Key.FEAT: 0,
            Key.EXP: TEXT_LENGTH + 1,
        },
        tok=_make_tok(),
        mappers={Key.ITEM: item_mapper},
        special_items={PAD_TOK: 0},
        sid_tokenizer=sid_tokenizer,
        sid_vocab_size=SID_VOCAB_SIZE,
        n_codebooks=N_CODEBOOKS,
        item_text_emb=torch.randn(N_RAW_ITEMS, EMSIZE),
    )
    if use_concept:
        data_info.sae_n_concepts = 7
        data_info.sae_topk = 2
    return data_info


def _make_cfg(fusion_type, use_concept=False):
    return SimpleNamespace(
        emsize=EMSIZE,
        nhead=2,
        nhid=16,
        nlayers=1,
        dropout=0.0,
        fusion_type=fusion_type,
        sid_agg="mean",
        mask_type="peter",
        greedy=True,
        use_feature=False,
        use_concept=use_concept,
        use_pretrained_lm=False,
        tied_weights=False,
        init_codebooks=False,
        mtl_aggregator=SimpleNamespace(name="avg"),
    )


def _make_batch():
    # The last raw item is padding; its paired SID codes use the SID PAD token.
    item = torch.tensor(
        [
            [1, 7, 2, 8, SID_VOCAB_SIZE - 1, SID_VOCAB_SIZE - 1],
            [3, 9, 4, 10, SID_VOCAB_SIZE - 1, SID_VOCAB_SIZE - 1],
        ]
    )
    return {
        Key.USER: torch.tensor([[1], [2]]),
        Key.ITEM: item,
        Key.ITEM_RAW: torch.tensor([[2, 3, 0], [4, 5, 0]]),
        Key.EXP: torch.tensor([[1, 3, 4, 0], [1, 5, 6, 0]]),
        Key.SEQ_LEN: torch.tensor([[ITEMS_PER_SEQUENCE], [ITEMS_PER_SEQUENCE]]),
    }


class TestSEQUERTextFusionSemanticIDs(unittest.TestCase):
    def test_sid_forward_all_fusion_modes(self):
        for fusion_type in ("gating", "concat-mlp", "attention", "nofusion"):
            with self.subTest(fusion_type=fusion_type):
                model = SEQUERTextFusion(_make_data_info(), _make_cfg(fusion_type))
                model.eval()
                output = model(_make_batch())

                self.assertEqual(output[Task.EXPLANATION].shape[:2], (TEXT_LENGTH, 2))
                self.assertEqual(output[Task.NEXT_ITEM].shape, ((ITEMS_PER_SEQUENCE - 1) * N_CODEBOOKS, 2, EMSIZE))
                self.assertEqual(output[Task.RATING].shape[0], ITEMS_PER_SEQUENCE)

    def test_sid_collaborative_embeddings_are_mean_pooled_per_item(self):
        model = SEQUERTextFusion(_make_data_info(), _make_cfg("gating"))
        batch = _make_batch()

        _, item_embeddings, text_embeddings = model.get_ui_text_embeddings(
            batch[Key.USER].T, batch[Key.ITEM].T, batch[Key.ITEM_RAW].T
        )

        self.assertEqual(item_embeddings.shape, (ITEMS_PER_SEQUENCE, 2, EMSIZE))
        self.assertEqual(text_embeddings.shape, (ITEMS_PER_SEQUENCE, 2, EMSIZE))

    def test_unpack_batch_returns_collaborative_and_raw_item_ids(self):
        model = SEQUERTextFusion(_make_data_info(), _make_cfg("gating"))
        batch = _make_batch()

        _, item, _, _ = model.unpack_batch(batch)

        self.assertIsInstance(item, list)
        self.assertEqual(len(item), 2)
        self.assertTrue(torch.equal(item[0], batch[Key.ITEM].T))
        self.assertTrue(torch.equal(item[1], batch[Key.ITEM_RAW].T))

    def test_nofusion_forwards_and_masks_concept_embeddings(self):
        model = SEQUERTextFusion(_make_data_info(use_concept=True), _make_cfg("nofusion", use_concept=True))
        batch = _make_batch()
        batch[Key.CONCEPT_ID] = torch.tensor([[1, 2], [3, 4]])

        output = model(batch)
        self.assertEqual(output[Task.EXPLANATION].shape[:2], (TEXT_LENGTH, 2))

        user, item, text, _ = model.unpack_batch(batch)
        concepts = batch[Key.CONCEPT_ID].T
        src = model.get_embedding_input(user, item, text, concepts)
        _, mask = model.get_masks(user, item, text, src.size(0), concepts)
        concept_start = model.ui_len
        self.assertFalse(mask[:, concept_start : concept_start + concepts.shape[0]].any().item())


if __name__ == "__main__":
    unittest.main()
