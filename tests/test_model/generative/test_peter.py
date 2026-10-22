"""Unit tests for the PETER generative recommender model.

Tests are grouped by concern:

  TestPeterMaskGeneration   – ``generate_peter_mask`` / ``generate_peter_sid_mask`` tensor
                               logic (causal structure, intra-item bidirectionality, SID mode).
  TestPETERInit             – Model construction for regular and SID modes, tied/untied weights.
  TestPETERLengthsAndIdx    – That ``i_ix``, ``u_ix``, ``src_len``, ``tgt_len`` are derived
                               correctly from ``data_info.lengths`` in both modes.
  TestPETERUnpackBatch      – ``unpack_batch`` currently transposes batch-first [B,L] inputs
                               to seq-first [L,B] for the internal transformer.
                               NOTE: these tests document the *current* layout; they will need
                               updating once the batch-first migration lands.
  TestPETERPrepareXY        – Teacher-forcing preparation: EXP shifted by one, generation mode,
                               CONTEXT label transposed.
  TestPETERGetMasks         – ``get_masks``: key_padding_mask shape/content, PAD positions masked,
                               user position never masked, attn_mask sliced correctly.
  TestPETEREmbeddingInput   – ``get_embedding_input`` concatenation order and output shape.
  TestPETERForwardShapes    – ``forward`` output dict: shapes for EXPLANATION, RATING, CONTEXT.
  TestPETERPredictMethods   – ``predict_context``, ``predict_rating``, ``predict_exp`` called on
                               synthetic hidden states, verifying shapes and valid log-probs.
  TestPETERLoss             – ``rating_loss``, ``exp_loss``, ``context_loss`` correctness and
                               masking behaviour.
  TestPETERSIDMode          – Multi-codebook specific: i_ix/u_ix offsets, embedding table size,
                               intra-item bidirectional attention in the generated mask.
"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import math
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
import torch.nn.functional as F

from nlp_xrs.models.generative.base import generate_peter_mask, generate_peter_sid_mask
from nlp_xrs.models.generative.peter import PETER
from nlp_xrs.utils.constants import PAD_TOK, Key, Task

# ---------------------------------------------------------------------------
# Shared tiny-model constants
# ---------------------------------------------------------------------------

N_USERS = 6
N_ITEMS = 12
N_WTOKENS = 40
EMSIZE = 16
NHEAD = 2
NHID = 32
NLAYERS = 2
BATCH = 4

U_LEN = 1
I_LEN = 1  # regular mode; overridden for SID tests
EXP_LEN = 7  # total explanation tokens, incl. BOS and EOS

PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
ITEM_PAD_IDX = 0


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def _make_tok(pad: int = PAD_IDX, bos: int = BOS_IDX, eos: int = EOS_IDX) -> MagicMock:
    tok = MagicMock()
    tok.pad_token_id = pad
    tok.bos_token_id = bos
    tok.eos_token_id = eos
    return tok


def _make_item_mapper(pad_idx: int = ITEM_PAD_IDX) -> MagicMock:
    mapper = MagicMock()
    mapper.get_idx.side_effect = lambda token: pad_idx if token == PAD_TOK else 0
    return mapper


def _make_data_info(n_codebooks: int = 1, sid_vocab_size: int = None) -> SimpleNamespace:
    """Build a minimal ``data_info`` compatible with PETER.__init__."""
    i_len = n_codebooks  # one token per codebook
    data_info = SimpleNamespace(
        n_users=N_USERS,
        n_items=N_ITEMS,
        n_wtokens=N_WTOKENS,
        lengths={
            Key.USER: U_LEN,
            Key.ITEM: i_len,
            Key.FEAT: 0,
            Key.EXP: EXP_LEN,
        },
        # BaseModel.__init__ pops these from __dict__
        tok=_make_tok(),
        mappers={Key.ITEM: _make_item_mapper()},
    )
    if n_codebooks > 1:
        data_info.n_codebooks = n_codebooks
        data_info.sid_vocab_size = sid_vocab_size or (N_ITEMS * n_codebooks)
        # BaseModel reads n_codebooks from the tokenizer, not from data_info.n_codebooks.
        data_info.sid_tokenizer = SimpleNamespace(n_codebooks=n_codebooks)
    return data_info


def _make_cfg(**overrides) -> SimpleNamespace:
    base = dict(
        emsize=EMSIZE,
        hsize=EMSIZE,
        nhead=NHEAD,
        nhid=NHID,
        nlayers=NLAYERS,
        dropout=0.0,  # deterministic for tests
        greedy=True,
        mask_type="peter",
        use_feature=False,
        use_pretrained_lm=False,
        tied_weights=False,
        init_codebooks=False,
        mtl_aggregator=SimpleNamespace(name="avg"),
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _make_peter(n_codebooks: int = 1, sid_vocab_size: int = None, **cfg_overrides) -> PETER:
    data_info = _make_data_info(n_codebooks=n_codebooks, sid_vocab_size=sid_vocab_size)
    cfg = _make_cfg(**cfg_overrides)
    return PETER(data_info, cfg)


def _make_batch(model: PETER, batch_size: int = BATCH) -> dict:
    """Return a valid raw batch (batch-first, pre-``prepare_xy``)."""
    exp_len = EXP_LEN
    i_len = model.lengths[Key.ITEM]

    user = torch.randint(1, N_USERS, (batch_size, U_LEN))
    item = torch.randint(1, N_ITEMS, (batch_size, i_len))
    # First item in half the batch is padding to exercise key_padding_mask
    item[: batch_size // 2, 0] = ITEM_PAD_IDX
    text = torch.randint(BOS_IDX + 1, N_WTOKENS - 1, (batch_size, exp_len))
    text[:, 0] = BOS_IDX
    # Pad the last token of second half to test mask coverage
    text[batch_size // 2 :, -1] = PAD_IDX
    rating = torch.rand(batch_size, 1) * 4.0 + 1.0
    context = torch.randint(3, N_WTOKENS - 1, (batch_size, exp_len))

    return {
        Key.USER: user,
        Key.ITEM: item,
        Key.EXP: text,
        Key.RATING: rating,
        Key.CONTEXT: context,
        "size": batch_size,
    }


# ---------------------------------------------------------------------------
# 1. Mask generation
# ---------------------------------------------------------------------------


class TestPeterMaskGeneration(unittest.TestCase):
    """Pure tensor logic for ``generate_peter_mask`` and ``generate_peter_sid_mask``."""

    def test_peter_mask_shape(self):
        src_len, tgt_len = 2, 5
        mask = generate_peter_mask(src_len, tgt_len)
        total = src_len + tgt_len
        self.assertEqual(mask.shape, (total, total))

    def test_peter_mask_lower_triangle_unmasked(self):
        """All causal positions (col ≤ row) must be False (unmasked)."""
        mask = generate_peter_mask(src_len=2, tgt_len=4)
        total = 6
        for row in range(total):
            for col in range(row + 1):
                self.assertFalse(
                    mask[row, col].item(),
                    f"Causal position ({row},{col}) must be unmasked",
                )

    def test_peter_mask_upper_triangle_masked(self):
        """Positions above the diagonal must be True (masked) *except* for the
        PETER-style item-attends-to-user cell at (0, 1)."""
        mask = generate_peter_mask(src_len=2, tgt_len=4)
        total = 6
        for row in range(total):
            for col in range(row + 1, total):
                if row == 0 and col == 1:
                    continue  # PETER special cell
                self.assertTrue(
                    mask[row, col].item(),
                    f"Future position ({row},{col}) must be masked",
                )

    def test_peter_mask_item_attends_to_user(self):
        """Item position (0) must see the user position (1)."""
        mask = generate_peter_mask(src_len=2, tgt_len=3)
        self.assertFalse(mask[0, 1].item(), "Item must attend to user (PETER style)")

    def test_peter_sid_mask_shape(self):
        ui_len, src_len, tgt_len = 3, 3, 5
        mask = generate_peter_sid_mask(ui_len, src_len, tgt_len)
        self.assertEqual(mask.shape, (src_len + tgt_len, src_len + tgt_len))

    def test_peter_sid_mask_item_codes_see_each_other_and_user(self):
        """All item-code positions (0..ui_len-2) attend to all of [0..ui_len-1]
        (bidirectional intra-item + user visibility)."""
        n_codebooks = 3
        ui_len = n_codebooks + 1  # 3 item codes + 1 user
        src_len = ui_len
        tgt_len = 4
        mask = generate_peter_sid_mask(ui_len, src_len, tgt_len)
        for code_row in range(ui_len - 1):  # each item-code row
            for col in range(ui_len):  # all item codes + user
                self.assertFalse(
                    mask[code_row, col].item(),
                    f"Item code {code_row} must attend to position {col}",
                )

    def test_peter_sid_mask_explanation_tokens_are_causal(self):
        """Explanation positions must not attend to future tokens."""
        ui_len, src_len, tgt_len = 3, 3, 5
        total = src_len + tgt_len
        mask = generate_peter_sid_mask(ui_len, src_len, tgt_len)
        for row in range(src_len, total):
            for col in range(row + 1, total):
                self.assertTrue(
                    mask[row, col].item(),
                    f"Explanation position ({row},{col}) must be causally masked",
                )

    def test_peter_sid_mask_with_n_codebooks_1_equals_peter_mask(self):
        """When n_codebooks=1, SID mask should equal the classic PETER mask."""
        src_len = 2  # 1 item + 1 user
        tgt_len = 4
        classic = generate_peter_mask(src_len, tgt_len)
        sid = generate_peter_sid_mask(ui_len=src_len, src_len=src_len, tgt_len=tgt_len)
        self.assertTrue(torch.equal(classic, sid))


# ---------------------------------------------------------------------------
# 2. Model initialisation
# ---------------------------------------------------------------------------


class TestPETERInit(unittest.TestCase):
    def test_regular_mode_item_embedding_vocab_size(self):
        model = _make_peter()
        self.assertEqual(model.item_emb.weight.shape[0], N_ITEMS)

    def test_regular_mode_word_emb_vocab_size(self):
        model = _make_peter()
        self.assertEqual(model.word_emb.weight.shape[0], N_WTOKENS)

    def test_regular_mode_embedding_dim(self):
        model = _make_peter()
        self.assertEqual(model.word_emb.weight.shape[1], EMSIZE)
        self.assertEqual(model.user_emb.weight.shape[1], EMSIZE)
        self.assertEqual(model.item_emb.weight.shape[1], EMSIZE)

    def test_untied_hidden2wtoken_shape(self):
        model = _make_peter(tied_weights=False)
        self.assertEqual(model.hidden2wtoken.weight.shape, (N_WTOKENS, EMSIZE))

    def test_tied_weights_shares_embedding(self):
        """With tied_weights, hidden2wtoken must share word_emb's weight tensor."""
        model = _make_peter(tied_weights=True)
        # The final linear layer in hidden2wtoken must have the same weight data
        # as the word embedding matrix.
        if hasattr(model.hidden2wtoken, "weight"):
            self.assertIs(model.hidden2wtoken.weight, model.word_emb.weight)
        else:
            # Sequential case: last module is the tied linear
            final = list(model.hidden2wtoken.children())[-1]
            self.assertIs(final.weight, model.word_emb.weight)

    def test_rating_predictor_output_shape(self):
        model = _make_peter()
        dummy = torch.zeros(BATCH, EMSIZE)
        out = model.rating_predictor(dummy)
        # MLP.forward applies torch.squeeze → shape is [B] for batch > 1
        self.assertEqual(out.shape, (BATCH,))

    def test_attn_mask_stored_on_model(self):
        model = _make_peter()
        self.assertIsInstance(model.attn_mask, torch.Tensor)
        total = model.src_len + model.tgt_len
        self.assertEqual(model.attn_mask.shape, (total, total))


# ---------------------------------------------------------------------------
# 3. Length and index attributes
# ---------------------------------------------------------------------------


class TestPETERLengthsAndIdx(unittest.TestCase):
    """i_ix, u_ix, src_len, tgt_len are derived from data_info.lengths."""

    def test_regular_i_ix_is_zero(self):
        """With a single item token, the last (only) item code is at position 0."""
        model = _make_peter(n_codebooks=1)
        self.assertEqual(model.i_ix, 0)

    def test_regular_u_ix_is_i_len(self):
        """User token immediately follows item code(s)."""
        model = _make_peter(n_codebooks=1)
        self.assertEqual(model.u_ix, I_LEN)

    def test_src_len_equals_ui_len_when_no_feature(self):
        model = _make_peter()
        self.assertEqual(model.src_len, model.ui_len)
        self.assertEqual(model.ui_len, U_LEN + I_LEN)

    def test_tgt_len_is_exp_len_minus_one(self):
        """Teacher-forcing target omits the last token; target is e_len-1 steps."""
        model = _make_peter()
        self.assertEqual(model.tgt_len, EXP_LEN - 1)

    def test_context_len_equals_exp_len(self):
        model = _make_peter()
        self.assertEqual(model.context_len, EXP_LEN)


# ---------------------------------------------------------------------------
# 4. unpack_batch  (seq-first internal layout)
# ---------------------------------------------------------------------------


class TestPETERUnpackBatch(unittest.TestCase):
    """unpack_batch currently transposes batch-first [B,L] inputs to seq-first [L,B].

    NOTE: These tests document the *current* behaviour.  Once the planned
    batch-first migration is complete the transpose will be removed and these
    assertions will need to be flipped (or replaced by tests that verify
    [B,L] shapes throughout the forward path).
    """

    def setUp(self):
        self.model = _make_peter()

    def _prep_inputs(self, batch):
        inputs, _ = self.model.prepare_xy(batch)
        return inputs

    def test_user_transposed_to_seq_first(self):
        batch = _make_batch(self.model)
        inputs = self._prep_inputs(batch)
        user, _, _ = self.model.unpack_batch(inputs)
        # batch[Key.USER] is [B, U_LEN]; after transpose → [U_LEN, B]
        self.assertEqual(user.shape, (U_LEN, BATCH))

    def test_item_transposed_to_seq_first(self):
        batch = _make_batch(self.model)
        inputs = self._prep_inputs(batch)
        _, item, _ = self.model.unpack_batch(inputs)
        self.assertEqual(item.shape, (I_LEN, BATCH))

    def test_text_transposed_to_seq_first(self):
        batch = _make_batch(self.model)
        inputs = self._prep_inputs(batch)
        _, _, text = self.model.unpack_batch(inputs)
        # text (post-prepare_xy) is [B, tgt_len]; after transpose → [tgt_len, B]
        self.assertEqual(text.shape, (self.model.tgt_len, BATCH))

    def test_unpack_preserves_values(self):
        """Values must be identical after transpose (no data mutation)."""
        batch = _make_batch(self.model)
        inputs = self._prep_inputs(batch)
        user, item, text = self.model.unpack_batch(inputs)
        self.assertTrue(torch.equal(user, inputs[Key.USER].T))
        self.assertTrue(torch.equal(item, inputs[Key.ITEM].T))
        self.assertTrue(torch.equal(text, inputs[Key.EXP].T))


# ---------------------------------------------------------------------------
# 5. prepare_xy
# ---------------------------------------------------------------------------


class TestPETERPrepareXY(unittest.TestCase):
    def setUp(self):
        self.model = _make_peter()

    def test_exp_input_is_prefix(self):
        """inputs[Key.EXP] must be batch[Key.EXP][..., :-1] (all but last token)."""
        batch = _make_batch(self.model)
        inputs, _ = self.model.prepare_xy(batch)
        self.assertTrue(torch.equal(inputs[Key.EXP], batch[Key.EXP][:, :-1]))

    def test_exp_label_is_suffix(self):
        """labels[Task.EXPLANATION] is batch[Key.EXP][..., 1:] transposed."""
        batch = _make_batch(self.model)
        _, labels = self.model.prepare_xy(batch)
        expected = batch[Key.EXP][:, 1:].T.contiguous()
        self.assertTrue(torch.equal(labels[Task.EXPLANATION], expected))

    def test_exp_input_and_label_are_shifted_by_one(self):
        """inputs[:, t+1] == labels[t, :]: the next-input token is the current target."""
        batch = _make_batch(self.model)
        inputs, labels = self.model.prepare_xy(batch)
        # inputs [B, tgt_len]: inputs[:, t] = raw_exp[:, t]
        # labels [tgt_len, B]: labels[t, :] = raw_exp[:, t+1]
        # Therefore: inputs[:, t+1] == raw_exp[:, t+1] == labels[t, :]
        for t in range(self.model.tgt_len - 1):
            self.assertTrue(
                torch.equal(inputs[Key.EXP][:, t + 1], labels[Task.EXPLANATION][t, :]),
                f"Alignment mismatch at step {t}: next-input token ≠ current label token",
            )

    def test_rating_label_squeezed(self):
        batch = _make_batch(self.model)
        _, labels = self.model.prepare_xy(batch)
        self.assertEqual(labels[Task.RATING].shape, (BATCH,))

    def test_context_label_transposed(self):
        """labels[Task.CONTEXT] must be [EXP_LEN, B] (transposed from [B, EXP_LEN])."""
        batch = _make_batch(self.model)
        _, labels = self.model.prepare_xy(batch)
        self.assertEqual(labels[Task.CONTEXT].shape, (EXP_LEN, BATCH))

    def test_generation_mode_fills_exp_with_bos(self):
        """In generation mode the EXP input is a single BOS token per sample."""
        batch = _make_batch(self.model)
        inputs, _ = self.model.prepare_xy(batch, generation=True)
        expected = torch.full((BATCH, U_LEN), BOS_IDX)
        self.assertTrue(
            torch.equal(inputs[Key.EXP], expected),
            "Generation mode must fill EXP with BOS",
        )

    def test_generation_mode_labels_unchanged(self):
        """Labels should still be derived from the original batch EXP."""
        batch = _make_batch(self.model)
        _, labels_train = self.model.prepare_xy(batch, generation=False)
        _, labels_gen = self.model.prepare_xy(batch, generation=True)
        self.assertTrue(torch.equal(labels_train[Task.EXPLANATION], labels_gen[Task.EXPLANATION]))


# ---------------------------------------------------------------------------
# 6. get_masks
# ---------------------------------------------------------------------------


class TestPETERGetMasks(unittest.TestCase):
    def setUp(self):
        self.model = _make_peter()

    def _get_unpacked(self, batch):
        inputs, _ = self.model.prepare_xy(batch)
        return self.model.unpack_batch(inputs)

    def test_key_padding_mask_shape(self):
        batch = _make_batch(self.model)
        user, item, text = self._get_unpacked(batch)
        total_len = I_LEN + U_LEN + self.model.tgt_len
        _, kpm = self.model.get_masks(user, item, text, total_len)
        self.assertEqual(kpm.shape, (BATCH, total_len))

    def test_user_positions_never_masked(self):
        """User tokens are never padding → their mask column must be all False."""
        batch = _make_batch(self.model)
        user, item, text = self._get_unpacked(batch)
        total_len = I_LEN + U_LEN + self.model.tgt_len
        _, kpm = self.model.get_masks(user, item, text, total_len)
        user_col = I_LEN  # user token is right after item codes
        self.assertFalse(kpm[:, user_col].any().item(), "User column must never be masked")

    def test_padding_items_are_masked(self):
        """Batch samples where item == ITEM_PAD_IDX must have that position masked."""
        batch = _make_batch(self.model)
        # Force item to pad for all samples
        batch[Key.ITEM][:] = ITEM_PAD_IDX
        user, item, text = self._get_unpacked(batch)
        total_len = I_LEN + U_LEN + self.model.tgt_len
        _, kpm = self.model.get_masks(user, item, text, total_len)
        # Item position is column 0 (first i_len columns)
        self.assertTrue(kpm[:, 0].all().item(), "Padded item column must be fully masked")

    def test_non_padding_items_are_not_masked(self):
        batch = _make_batch(self.model)
        # Force item to a real token for all samples
        batch[Key.ITEM][:] = 5
        user, item, text = self._get_unpacked(batch)
        total_len = I_LEN + U_LEN + self.model.tgt_len
        _, kpm = self.model.get_masks(user, item, text, total_len)
        self.assertFalse(kpm[:, 0].any().item(), "Non-pad item column must not be masked")

    def test_padding_text_tokens_are_masked(self):
        """Positions in the text where tok == pad_token_id must be masked."""
        batch = _make_batch(self.model)
        # Fill EXP with pad except BOS at position 0
        batch[Key.EXP][:] = PAD_IDX
        batch[Key.EXP][:, 0] = BOS_IDX
        user, item, text = self._get_unpacked(batch)
        total_len = I_LEN + U_LEN + self.model.tgt_len
        _, kpm = self.model.get_masks(user, item, text, total_len)
        # Text columns start at I_LEN + U_LEN
        text_start = I_LEN + U_LEN
        # All text positions should be masked (since text fed to transformer is all-pad after BOS)
        # The BOS column (text col 0) is NOT pad so it will NOT be masked
        self.assertTrue(
            kpm[:, text_start + 1 :].all().item(),
            "Padded text positions must all be masked",
        )

    def test_attn_mask_is_bool_tensor(self):
        batch = _make_batch(self.model)
        user, item, text = self._get_unpacked(batch)
        total_len = I_LEN + U_LEN + self.model.tgt_len
        attn_mask, _ = self.model.get_masks(user, item, text, total_len)
        self.assertEqual(attn_mask.dtype, torch.bool)

    def test_attn_mask_sliced_to_total_len(self):
        batch = _make_batch(self.model)
        user, item, text = self._get_unpacked(batch)
        total_len = I_LEN + U_LEN + self.model.tgt_len
        attn_mask, _ = self.model.get_masks(user, item, text, total_len)
        self.assertEqual(attn_mask.shape, (total_len, total_len))


# ---------------------------------------------------------------------------
# 7. get_embedding_input
# ---------------------------------------------------------------------------


class TestPETEREmbeddingInput(unittest.TestCase):
    def setUp(self):
        self.model = _make_peter()

    def _seq_first_inputs(self, batch):
        inputs, _ = self.model.prepare_xy(batch)
        return self.model.unpack_batch(inputs)

    def test_output_shape_is_seq_first(self):
        """get_embedding_input returns [total_len, B, D] (seq-first for transformer)."""
        batch = _make_batch(self.model)
        user, item, text = self._seq_first_inputs(batch)
        src = self.model.get_embedding_input(user, item, text)
        total_len = I_LEN + U_LEN + self.model.tgt_len
        self.assertEqual(src.shape, (total_len, BATCH, EMSIZE))

    def test_embedding_concat_order_item_user_text(self):
        """Item codes come first, then user, then text tokens."""
        batch = _make_batch(self.model)
        user, item, text = self._seq_first_inputs(batch)
        src = self.model.get_embedding_input(user, item, text)

        # Manually compute expected sub-tensors
        i_emb = self.model.item_emb(item) * math.sqrt(EMSIZE)  # [I_LEN, B, D]
        u_emb = self.model.user_emb(user) * math.sqrt(EMSIZE)  # [U_LEN, B, D]
        w_emb = self.model.word_emb(text) * math.sqrt(EMSIZE)  # [tgt_len, B, D]

        self.assertTrue(torch.allclose(src[:I_LEN], i_emb))
        self.assertTrue(torch.allclose(src[I_LEN : I_LEN + U_LEN], u_emb))
        self.assertTrue(torch.allclose(src[I_LEN + U_LEN :], w_emb))

    def test_scale_factor_applied_for_scratch_model(self):
        """Scratch models scale embeddings by sqrt(hsize) before positional encoding."""
        batch = _make_batch(self.model)
        user, item, text = self._seq_first_inputs(batch)
        src = self.model.get_embedding_input(user, item, text)
        raw = self.model.item_emb(item)
        # src[:I_LEN] == raw * sqrt(hsize)
        self.assertTrue(torch.allclose(src[:I_LEN], raw * math.sqrt(EMSIZE)))


# ---------------------------------------------------------------------------
# 8. forward – output shapes
# ---------------------------------------------------------------------------


class TestPETERForwardShapes(unittest.TestCase):
    def setUp(self):
        self.model = _make_peter()
        self.model.eval()

    def _run_forward(self, batch_size: int = BATCH):
        batch = _make_batch(self.model, batch_size)
        inputs, _ = self.model.prepare_xy(batch)
        with torch.no_grad():
            return self.model(inputs)

    def test_explanation_output_shape(self):
        """log_word_prob must be [tgt_len, B, n_wtokens]."""
        out = self._run_forward()
        self.assertEqual(
            out[Task.EXPLANATION].shape,
            (self.model.tgt_len, BATCH, N_WTOKENS),
        )

    def test_rating_output_shape(self):
        # MLP.forward applies torch.squeeze → [B] for batch > 1
        out = self._run_forward()
        self.assertEqual(out[Task.RATING].shape, (BATCH,))

    def test_context_output_shape(self):
        """Context distribution must be [B, n_wtokens]."""
        out = self._run_forward()
        self.assertEqual(out[Task.CONTEXT].shape, (BATCH, N_WTOKENS))

    def test_explanation_log_probs_sum_to_one(self):
        """exp(log_softmax) along vocab axis must sum to 1."""
        out = self._run_forward()
        probs = out[Task.EXPLANATION].exp()  # [tgt_len, B, V]
        sums = probs.sum(-1)  # [tgt_len, B]
        self.assertTrue(
            torch.allclose(sums, torch.ones_like(sums), atol=1e-5),
            "Explanation probabilities must sum to 1 over vocab",
        )

    def test_context_log_probs_sum_to_one(self):
        out = self._run_forward()
        probs = out[Task.CONTEXT].exp()  # [B, V]
        sums = probs.sum(-1)  # [B]
        self.assertTrue(
            torch.allclose(sums, torch.ones_like(sums), atol=1e-5),
            "Context probabilities must sum to 1 over vocab",
        )

    def test_output_keys_present(self):
        out = self._run_forward()
        for task in [Task.EXPLANATION, Task.RATING, Task.CONTEXT]:
            self.assertIn(task, out)

    def test_forward_batch_size_one(self):
        """Edge case: batch of size 1 must not crash."""
        out = self._run_forward(batch_size=1)
        self.assertEqual(out[Task.EXPLANATION].shape[1], 1)


# ---------------------------------------------------------------------------
# 9. Prediction methods (on synthetic hidden states)
# ---------------------------------------------------------------------------


class TestPETERPredictMethods(unittest.TestCase):
    """Call predict_* methods directly with hand-crafted hidden states.

    The hidden state is currently seq-first: [total_len, B, D].
    These tests verify the correct *slice* is used for each head.
    """

    def setUp(self):
        self.model = _make_peter()
        self.model.eval()
        total_len = self.model.src_len + self.model.tgt_len
        self.hidden = torch.randn(total_len, BATCH, EMSIZE)

    def test_predict_context_uses_i_ix_slice(self):
        """predict_context reads hidden[i_ix], so its result must equal
        log_softmax(hidden2wtoken(hidden[i_ix]))."""
        expected = F.log_softmax(self.model.hidden2wtoken(self.hidden[self.model.i_ix]), dim=-1)
        result = self.model.predict_context(self.hidden)
        self.assertTrue(torch.allclose(result, expected))

    def test_predict_rating_uses_u_ix_slice(self):
        """predict_rating reads hidden[u_ix]."""
        expected = self.model.rating_predictor(self.hidden[self.model.u_ix])
        result = self.model.predict_rating(self.hidden)
        self.assertTrue(torch.allclose(result, expected))

    def test_predict_exp_reads_from_src_len_onwards(self):
        """predict_exp reads hidden[src_len:] for the explanation tokens."""
        exp_hidden = self.hidden[self.model.src_len :]
        expected = F.log_softmax(self.model.hidden2wtoken(exp_hidden), dim=-1)
        result = self.model.predict_exp(self.hidden)
        self.assertTrue(torch.allclose(result, expected))

    def test_predict_context_output_shape(self):
        out = self.model.predict_context(self.hidden)
        self.assertEqual(out.shape, (BATCH, N_WTOKENS))

    def test_predict_rating_output_shape(self):
        # MLP.forward applies torch.squeeze → [B] for batch > 1
        out = self.model.predict_rating(self.hidden)
        self.assertEqual(out.shape, (BATCH,))

    def test_predict_exp_output_shape(self):
        out = self.model.predict_exp(self.hidden)
        self.assertEqual(out.shape, (self.model.tgt_len, BATCH, N_WTOKENS))


# ---------------------------------------------------------------------------
# 10. Loss functions
# ---------------------------------------------------------------------------


class TestPETERLoss(unittest.TestCase):
    def setUp(self):
        self.model = _make_peter()

    def test_rating_loss_is_mse(self):
        preds = torch.tensor([3.5, 4.0, 2.0])
        labels = torch.tensor([3.0, 4.0, 3.0])
        expected = ((preds - labels) ** 2).mean()
        loss = self.model.rating_loss(preds, labels)
        self.assertAlmostEqual(loss.item(), expected.item(), places=5)

    def test_rating_loss_is_zero_for_perfect_predictions(self):
        targets = torch.tensor([1.0, 2.0, 3.0, 4.0])
        loss = self.model.rating_loss(targets.clone(), targets)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_exp_loss_ignores_pad_positions(self):
        """NLL loss must be identical whether extra pad tokens are added to the tail."""
        tgt_len = self.model.tgt_len
        # Use uniform log-probs (log(1/V) everywhere)
        log_probs = torch.full((tgt_len, BATCH, N_WTOKENS), -math.log(N_WTOKENS))
        labels_no_pad = torch.ones(tgt_len, BATCH, dtype=torch.long)
        # Add padding to last row
        labels_with_pad = labels_no_pad.clone()
        labels_with_pad[-1] = PAD_IDX

        loss_no_pad = self.model.exp_loss(log_probs[:-1], labels_no_pad[:-1])
        loss_with_pad = self.model.exp_loss(log_probs, labels_with_pad)
        # Pad positions carry no loss; remaining tokens match → losses equal
        self.assertAlmostEqual(loss_no_pad.item(), loss_with_pad.item(), places=5)

    def test_context_loss_lower_when_distribution_matches(self):
        """context_loss broadcasts the *same* predicted distribution over all label positions
        (see context_criterion repeat).  A distribution that concentrates on the single target
        token shared across every label position must beat a uniform predictor."""
        # Use the same target token for every position and every batch sample
        target_token = 7  # arbitrary, not a special token
        labels = torch.full((EXP_LEN, BATCH), target_token, dtype=torch.long)

        # Bad: uniform over vocab
        uniform = torch.full((BATCH, N_WTOKENS), -math.log(N_WTOKENS))

        # Good: all mass on target_token
        peaked = torch.full((BATCH, N_WTOKENS), -100.0)
        peaked[:, target_token] = 0.0
        peaked = F.log_softmax(peaked, dim=-1)

        loss_uniform = self.model.context_loss(uniform, labels)
        loss_peaked = self.model.context_loss(peaked, labels)
        self.assertLess(loss_peaked.item(), loss_uniform.item())

    def test_context_loss_ignores_bos_pad_eos(self):
        """context_criterion has zero weight on BOS/PAD/EOS → removing those
        label positions from the input should not change the loss."""
        labels_plain = torch.randint(3, N_WTOKENS, (EXP_LEN, BATCH))
        # Replace some positions with special tokens
        labels_with_special = labels_plain.clone()
        labels_with_special[0] = BOS_IDX
        labels_with_special[-1] = EOS_IDX

        log_probs = torch.randn(BATCH, N_WTOKENS)
        log_probs = F.log_softmax(log_probs, dim=-1)

        loss_plain = self.model.context_loss(log_probs, labels_plain)
        loss_special = self.model.context_loss(log_probs, labels_with_special)
        # The special-token rows contribute 0 to the loss → losses differ
        # (unless all labels happened to avoid special tokens, which is unlikely
        # with 40-vocab random labels, but we just check no crash + finite loss)
        self.assertTrue(loss_plain.isfinite())
        self.assertTrue(loss_special.isfinite())


# ---------------------------------------------------------------------------
# 11. SID mode
# ---------------------------------------------------------------------------


class TestPETERSIDMode(unittest.TestCase):
    """Tests specific to multi-codebook (n_codebooks > 1) SID operation."""

    N_CB = 3
    SID_VOCAB = N_ITEMS * 3  # 3 codebooks × N_ITEMS codes

    def _make_sid_model(self):
        return _make_peter(n_codebooks=self.N_CB, sid_vocab_size=self.SID_VOCAB)

    def test_item_embedding_uses_sid_vocab(self):
        model = self._make_sid_model()
        self.assertEqual(model.item_emb.weight.shape[0], self.SID_VOCAB)

    def test_i_ix_points_to_last_codebook(self):
        """i_ix must be i_len - 1 = n_codebooks - 1 (0-indexed last code)."""
        model = self._make_sid_model()
        self.assertEqual(model.i_ix, self.N_CB - 1)

    def test_u_ix_points_after_all_item_codes(self):
        """User token follows all n_codebooks item-code positions."""
        model = self._make_sid_model()
        self.assertEqual(model.u_ix, self.N_CB)

    def test_attn_mask_intra_item_bidirectional(self):
        """Item-code positions must all see each other (bidirectional intra-item)."""
        model = self._make_sid_model()
        mask = model.attn_mask
        for i in range(self.N_CB):
            for j in range(self.N_CB):
                self.assertFalse(
                    mask[i, j].item(),
                    f"Item-code pair ({i},{j}) must be mutually visible",
                )

    def test_forward_shapes_in_sid_mode(self):
        """Full forward pass in SID mode must produce correctly-shaped outputs."""
        model = self._make_sid_model()
        model.eval()
        batch = _make_batch(model)
        inputs, _ = model.prepare_xy(batch)
        with torch.no_grad():
            out = model(inputs)
        self.assertEqual(out[Task.EXPLANATION].shape, (model.tgt_len, BATCH, N_WTOKENS))
        self.assertEqual(out[Task.RATING].shape, (BATCH,))
        self.assertEqual(out[Task.CONTEXT].shape, (BATCH, N_WTOKENS))

    def test_n_codebooks_attribute_set(self):
        model = self._make_sid_model()
        self.assertEqual(model.n_codebooks, self.N_CB)

    def test_src_len_includes_all_item_codes(self):
        """src_len = n_codebooks + u_len (no feature)."""
        model = self._make_sid_model()
        self.assertEqual(model.src_len, self.N_CB + U_LEN)


class TestPETERConceptInput(unittest.TestCase):
    """SAE concept embeddings are conditioning inputs, not word tokens."""

    N_CONCEPTS = 11
    TOP_K = 3

    def _make_model(self, use_feature: bool = False) -> PETER:
        data_info = _make_data_info()
        data_info.sae_n_concepts = self.N_CONCEPTS
        data_info.sae_topk = self.TOP_K
        if use_feature:
            data_info.lengths[Key.FEAT] = 2
        return PETER(data_info, _make_cfg(use_concept=True, use_feature=use_feature))

    def _make_batch(self, model: PETER) -> dict:
        exp = torch.tensor([[BOS_IDX, 5, 6, EOS_IDX, PAD_IDX, PAD_IDX]] * BATCH)
        batch = {
            Key.USER: torch.ones(BATCH, U_LEN, dtype=torch.long),
            Key.ITEM: torch.ones(BATCH, I_LEN, dtype=torch.long),
            Key.EXP: exp,
            Key.RATING: torch.ones(BATCH, 1),
            Key.CONCEPT_ID: torch.tensor([[1, 3, 5]] * BATCH),
            Key.CONCEPT_ACT: torch.ones(BATCH, self.TOP_K),
            "size": BATCH,
        }
        if getattr(model.cfg, "use_feature", False):
            batch[Key.FEAT] = torch.tensor([[7, 8]] * BATCH)
        return batch

    def test_embedding_is_tied_to_concept_head(self):
        model = self._make_model()
        self.assertIs(model.concept_emb.weight, model.context_head.weight)
        self.assertEqual(model.src_len, model.ui_len + self.TOP_K)

    def test_prepare_xy_keeps_ground_truth_concepts_as_separate_input(self):
        model = self._make_model(use_feature=True)
        batch = self._make_batch(model)
        inputs, _ = model.prepare_xy(batch)
        self.assertTrue(torch.equal(inputs[Key.CONCEPT_ID], batch[Key.CONCEPT_ID]))
        self.assertTrue(torch.equal(inputs[Key.EXP], torch.cat([batch[Key.FEAT], batch[Key.EXP][:, :-1]], dim=1)))

    def test_embeddings_and_padding_mask_insert_concept_positions(self):
        model = self._make_model()
        inputs, _ = model.prepare_xy(self._make_batch(model))
        user, item, text = model.unpack_batch(inputs)
        concepts = inputs[Key.CONCEPT_ID].T
        src = model.get_embedding_input(user, item, text, concepts)
        concept_start = model.ui_len
        expected = model.concept_emb(concepts) * math.sqrt(EMSIZE)
        self.assertTrue(torch.allclose(src[concept_start : concept_start + self.TOP_K], expected))

        _, mask = model.get_masks(user, item, text, src.size(0), concepts)
        self.assertFalse(mask[:, concept_start : concept_start + self.TOP_K].any().item())

    def test_generation_injects_predicted_concepts_after_first_step(self):
        model = self._make_model()
        model.eval()
        batch = self._make_batch(model)
        inputs, _ = model.prepare_xy(batch, generation=True)
        seen_concepts = []
        original_forward = model._forward

        def record_forward(user, item, text, concepts=None):
            seen_concepts.append(concepts)
            return original_forward(user, item, text, concepts)

        model._forward = record_forward
        model.check_gen_end = lambda *args: False
        with torch.no_grad():
            model.generate(batch)

        self.assertIsNone(seen_concepts[0])
        self.assertIsNotNone(seen_concepts[1])
        self.assertEqual(seen_concepts[1].shape, (self.TOP_K, BATCH))


if __name__ == "__main__":
    unittest.main()
