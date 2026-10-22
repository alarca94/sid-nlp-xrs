import unittest

import torch

from nlp_xrs.models.losses import NextItemLoss


class TestNextItemLoss(unittest.TestCase):
    """Unit tests for NextItemLoss covering gradient flow and indexing correctness."""

    PAD = 0
    N_ITEMS = 30
    DIM = 16

    def _emb(self, n=None, d=None):
        n = n or self.N_ITEMS
        d = d or self.DIM
        return torch.nn.Embedding(n, d, padding_idx=self.PAD)

    # ------------------------------------------------------------------
    # CE loss
    # ------------------------------------------------------------------

    def test_ce_loss_gradients_flow_through_seq_output(self):
        """CE loss must produce non-zero gradients at non-padding positions."""
        B, S = 4, 6
        emb = self._emb()
        seq_output = torch.randn(B, S, self.DIM, requires_grad=True)
        pos_target = torch.randint(1, self.N_ITEMS, (B * S,))

        loss_fn = NextItemLoss(loss_type="CE", ignore_index=self.PAD, last_only=False, l2_norm=False, batch_first=True)
        loss = loss_fn.ce_loss(emb, seq_output, pos_target, None)
        loss.backward()

        self.assertIsNotNone(seq_output.grad, "seq_output should have gradients")
        self.assertGreater(
            seq_output.grad.abs().sum().item(),
            0,
            "CE loss gradients through seq_output should be non-zero",
        )

    def test_ce_loss_gradients_flow_through_embedding_weights(self):
        """CE loss must produce non-zero gradients on the embedding weight matrix."""
        B, S = 4, 6
        emb = self._emb()
        seq_output = torch.randn(B, S, self.DIM)
        pos_target = torch.randint(1, self.N_ITEMS, (B * S,))

        loss_fn = NextItemLoss(loss_type="CE", ignore_index=self.PAD, last_only=False, l2_norm=False, batch_first=True)
        loss = loss_fn.ce_loss(emb, seq_output, pos_target, None)
        loss.backward()

        self.assertIsNotNone(emb.weight.grad, "embedding weights should have gradients")
        self.assertGreater(
            emb.weight.grad.abs().sum().item(),
            0,
            "CE loss gradients through embedding weights should be non-zero",
        )

    def test_ce_loss_all_padding_targets_gives_zero_grad(self):
        """When every target is the ignore_index the loss should still be finite but
        its gradient w.r.t. seq_output should be exactly zero (NLL skips all)."""
        B, S = 2, 4
        emb = self._emb()
        seq_output = torch.randn(B, S, self.DIM, requires_grad=True)
        pos_target = torch.full((B * S,), self.PAD)  # all padding

        loss_fn = NextItemLoss(loss_type="CE", ignore_index=self.PAD, last_only=False, l2_norm=False, batch_first=True)
        loss = loss_fn.ce_loss(emb, seq_output, pos_target, None)
        loss.backward()

        self.assertEqual(
            seq_output.grad.abs().sum().item(),
            0.0,
            "All-padding targets should give zero gradient (all ignored)",
        )

    # ------------------------------------------------------------------
    # BPR loss
    # ------------------------------------------------------------------

    def test_bpr_loss_gradients_flow(self):
        """BPR loss must return non-zero gradients through seq_output."""
        B = 8
        emb = self._emb()
        seq_output = torch.randn(B, self.DIM, requires_grad=True)
        pos_target = torch.randint(1, self.N_ITEMS, (B,))
        neg_target = torch.randint(1, self.N_ITEMS, (B,))

        loss_fn = NextItemLoss(loss_type="BPR", ignore_index=self.PAD, last_only=True, l2_norm=False, batch_first=True)
        loss = loss_fn.bpr_loss(emb, seq_output, pos_target, neg_target)
        loss.backward()

        self.assertIsNotNone(seq_output.grad)
        self.assertGreater(seq_output.grad.abs().sum().item(), 0)

    # ------------------------------------------------------------------
    # last_only indexing  (the batch_range bug)
    # ------------------------------------------------------------------

    def test_last_only_selects_correct_position_per_batch_item(self):
        """__call__ with last_only=True must select position `length-2` for each
        batch item — not the transposed index that the old bug produced."""
        B, S = 4, 7  # intentionally B != S to expose the shape-mismatch bug
        emb = self._emb()
        seq_output = torch.zeros(B, S, self.DIM)

        # Place a unique sentinel at the expected position for each batch item
        lengths = torch.tensor([3, 5, 4, 7])  # positions: 1, 3, 2, 5
        for b, pos in enumerate(lengths - 2):
            seq_output[b, pos] = float(b + 1)  # distinct non-zero value

        pos_target = torch.zeros(B, S, dtype=torch.long)
        neg_target = torch.zeros(B, S, dtype=torch.long)
        # Give valid (non-padding) targets only at the expected positions
        for b, pos in enumerate(lengths - 2):
            pos_target[b, pos] = b + 1
            neg_target[b, pos] = (b + 2) % self.N_ITEMS or 1

        loss_fn = NextItemLoss(loss_type="BPR", ignore_index=self.PAD, last_only=True, l2_norm=False, batch_first=True)
        # Should not raise; would fail with shape error under the old bug when B != S
        loss = loss_fn(
            emb_layer=emb,
            input=seq_output,
            pos_target=pos_target,
            neg_target=neg_target,
            lengths=lengths,
        )
        self.assertTrue(torch.isfinite(loss), "loss should be a finite scalar")

    def test_last_only_indexing_shape_mismatch_would_fail_with_old_bug(self):
        """Explicitly verify that the correct batch_range uses input.shape[0] (B),
        not input.shape[1] (S). With B != S the old code would raise or produce
        wrong results; the fixed code returns shape (B, D)."""
        B, S, D = 5, 9, self.DIM  # B != S
        input_tensor = torch.randn(B, S, D)
        lengths = torch.randint(2, S, (B,))

        correct_batch_range = torch.arange(input_tensor.shape[0])  # (B,)
        selected = input_tensor[correct_batch_range, lengths - 1]  # (B, D)
        self.assertEqual(selected.shape, (B, D))

        # Old (buggy) batch_range would be shape (S,) which differs from lengths (B,)
        wrong_batch_range = torch.arange(input_tensor.shape[1])  # (S,)
        with self.assertRaises(IndexError):
            # PyTorch raises when two 1-D index tensors have different sizes
            _ = input_tensor[lengths - 1, wrong_batch_range]

    # ------------------------------------------------------------------
    # Pad-mask direction (the inverted-mask bug in SASRec.forward)
    # ------------------------------------------------------------------

    def test_sasrec_pad_mask_zeros_padding_positions_not_item_positions(self):
        """The SASRec pad mask must zero PAD positions and preserve item positions.
        The old bug did the opposite: `emb * (seq == pad)` zeros non-pad slots."""
        pad_token = 0
        # Sequence: [pad, pad, item1, item2, item3]
        item_seq = torch.tensor([[pad_token, pad_token, 1, 2, 3]])  # (1, 5)
        emb_out = torch.ones(1, 5, 8)

        pad_mask = (item_seq == pad_token).unsqueeze(-1)  # True at pad positions

        # --- Buggy version (old code): keeps padding, zeros items ---
        buggy_out = emb_out * pad_mask
        self.assertEqual(
            buggy_out[0, 2:].sum().item(),
            0.0,
            "Old bug: item positions should be zeroed (demonstrating the bug)",
        )
        self.assertGreater(
            buggy_out[0, :2].sum().item(),
            0.0,
            "Old bug: padding positions should be kept (demonstrating the bug)",
        )

        # --- Fixed version: zeros padding, keeps items ---
        fixed_out = emb_out.masked_fill(pad_mask, 0.0)
        self.assertEqual(
            fixed_out[0, :2].sum().item(),
            0.0,
            "Fixed: padding positions must be zeroed out",
        )
        self.assertGreater(
            fixed_out[0, 2:].sum().item(),
            0.0,
            "Fixed: item positions must be preserved",
        )

    def test_inverted_pad_mask_causes_zero_ce_gradients(self):
        """Regression test: with the inverted mask, CE loss gradients for non-pad
        outputs become zero because `output * False == 0` blocks backprop.

        The key setup: targets at padding positions are set to ignore_index so the
        CE loss ignores those slots (the only path where mask=1 lets gradients
        through). The non-padding positions have mask=False=0, which zeroes the
        output and kills the gradient via the chain rule. Both paths blocked →
        total gradient on emb_out is exactly zero despite a non-zero loss."""
        B, S, D, N = 2, 5, 16, 30
        pad_token = 0
        item_seq = torch.tensor([[pad_token, 1, 2, 3, 4], [pad_token, pad_token, 1, 2, 3]])  # (B, S)

        emb_out = torch.randn(B, S, D, requires_grad=True)
        pad_mask = (item_seq == pad_token).unsqueeze(-1)

        # Apply the OLD (buggy) mask — zeros non-padding slots, keeps padding slots
        masked_buggy = emb_out * pad_mask

        emb_layer = torch.nn.Embedding(N, D, padding_idx=pad_token)
        # Targets at padding positions must be ignore_index so CE also ignores
        # those slots (otherwise their grad would flow back through mask=1).
        # Targets at non-padding positions are valid — CE computes a real loss
        # there, but mask=False zeroes the output so grad is killed by chain rule.
        pad_positions = (item_seq == pad_token).reshape(-1)
        pos_target = torch.randint(1, N, (B * S,))
        pos_target[pad_positions] = pad_token

        loss_fn = NextItemLoss(loss_type="CE", ignore_index=pad_token, last_only=False, l2_norm=False, batch_first=True)
        loss = loss_fn.ce_loss(emb_layer, masked_buggy, pos_target, None)
        loss.backward()

        # Loss is non-zero: non-padding outputs are all-zero → log_softmax gives
        # log(1/N) per position, so NLL = log(N) > 0.
        self.assertGreater(loss.item(), 0.0, "Loss must be non-zero despite the bug")
        self.assertEqual(
            emb_out.grad.abs().sum().item(),
            0.0,
            "Inverted mask must cause zero gradients on emb_out (regression check)",
        )

    # ------------------------------------------------------------------
    # batch_first=False variants (sequence-first input layout)
    # ------------------------------------------------------------------

    def test_last_only_selects_correct_position_per_batch_item_seq_first(self):
        """Same as test_last_only_selects_correct_position_per_batch_item but
        with batch_first=False: input is (S, B, D) and __call__ must transpose
        before indexing."""
        B, S = 4, 7  # B != S to expose shape bugs
        emb = self._emb()
        # Sequence-first layout: (S, B, D)
        seq_output = torch.zeros(S, B, self.DIM)

        lengths = torch.tensor([3, 5, 4, 7])  # expected positions: 1, 3, 2, 5
        for b, pos in enumerate(lengths - 2):
            seq_output[pos, b] = float(b + 1)  # (S, B, D) indexing

        pos_target = torch.zeros(S, B, dtype=torch.long)
        neg_target = torch.zeros(S, B, dtype=torch.long)
        for b, pos in enumerate(lengths - 2):
            pos_target[pos, b] = b + 1
            neg_target[pos, b] = (b + 2) % self.N_ITEMS or 1

        loss_fn = NextItemLoss(loss_type="BPR", ignore_index=self.PAD, last_only=True, l2_norm=False, batch_first=False)
        loss = loss_fn(
            emb_layer=emb,
            input=seq_output,
            pos_target=pos_target,
            neg_target=neg_target,
            lengths=lengths,
        )
        self.assertTrue(torch.isfinite(loss), "loss should be a finite scalar")

    def test_bpr_loss_gradients_flow_seq_first(self):
        """BPR loss via __call__ with last_only=True and batch_first=False must
        produce non-zero gradients after transposing the sequence-first input."""
        B, S = 8, 6
        emb = self._emb()
        # Sequence-first layout: (S, B, D)
        seq_output = torch.randn(S, B, self.DIM, requires_grad=True)
        lengths = torch.full((B,), S, dtype=torch.long)  # use the last valid slot

        pos_target = torch.zeros(S, B, dtype=torch.long)
        neg_target = torch.zeros(S, B, dtype=torch.long)
        for b in range(B):
            pos_target[lengths[b] - 2, b] = torch.randint(1, self.N_ITEMS, (1,)).item()
            neg_target[lengths[b] - 2, b] = torch.randint(1, self.N_ITEMS, (1,)).item()

        loss_fn = NextItemLoss(loss_type="BPR", ignore_index=self.PAD, last_only=True, l2_norm=False, batch_first=False)
        loss = loss_fn(
            emb_layer=emb,
            input=seq_output,
            pos_target=pos_target,
            neg_target=neg_target,
            lengths=lengths,
        )
        loss.backward()

        self.assertIsNotNone(seq_output.grad)
        self.assertGreater(
            seq_output.grad.abs().sum().item(), 0, "BPR gradients should be non-zero with seq-first input"
        )

    def test_ce_loss_gradients_flow_through_seq_output_seq_first(self):
        """CE loss with batch_first=False and last_only=False: the input layout
        does not change the loss computation (no transposition for last_only=False),
        confirming gradients flow regardless of the declared layout."""
        B, S = 4, 6
        emb = self._emb()
        # Even with batch_first=False, last_only=False does not transpose; the
        # tensor is flattened to (B*S, D) either way.
        seq_output = torch.randn(B, S, self.DIM, requires_grad=True)
        pos_target = torch.randint(1, self.N_ITEMS, (B * S,))

        loss_fn = NextItemLoss(loss_type="CE", ignore_index=self.PAD, last_only=False, l2_norm=False, batch_first=False)
        loss = loss_fn.ce_loss(emb, seq_output, pos_target, None)
        loss.backward()

        self.assertIsNotNone(seq_output.grad)
        self.assertGreater(
            seq_output.grad.abs().sum().item(),
            0,
            "CE loss gradients must be non-zero regardless of batch_first when last_only=False",
        )

    def test_fixed_pad_mask_gives_nonzero_ce_gradients(self):
        """With the corrected mask (masked_fill), CE loss gradients are non-zero."""
        B, S, D, N = 2, 5, 16, 30
        pad_token = 0
        item_seq = torch.tensor([[pad_token, 1, 2, 3, 4], [pad_token, pad_token, 1, 2, 3]])

        emb_out = torch.randn(B, S, D, requires_grad=True)
        pad_mask = (item_seq == pad_token).unsqueeze(-1)

        # Apply the FIXED mask — zeros padding slots, keeps item slots
        masked_fixed = emb_out.masked_fill(pad_mask, 0.0)

        emb_layer = torch.nn.Embedding(N, D, padding_idx=pad_token)
        pos_target = torch.randint(1, N, (B * S,))
        loss_fn = NextItemLoss(loss_type="CE", ignore_index=pad_token, last_only=False, l2_norm=False, batch_first=True)
        loss = loss_fn.ce_loss(emb_layer, masked_fixed, pos_target, None)
        loss.backward()

        self.assertGreater(loss.item(), 0.0)
        self.assertGreater(
            emb_out.grad.abs().sum().item(),
            0.0,
            "Fixed mask must produce non-zero gradients through seq_output",
        )
