from typing import Optional

import torch
import torch.nn.functional as F


class NextItemLoss:
    def __init__(
        self,
        loss_type: str = "CE",
        ignore_index: int = -100,
        last_only: bool = False,
        l2_norm: bool = False,
        batch_first: bool = False,
    ):
        self.ignore_index = ignore_index
        self.last_only = last_only
        self.loss_type = loss_type
        self.l2_norm = l2_norm
        self.batch_first = batch_first

        if loss_type == "CE":
            self.loss_fn = self.ce_loss
        elif loss_type == "BCE":
            self.loss_fn = self.bce_loss
        elif loss_type == "BPR":
            self.loss_fn = self.bpr_loss

    def ce_loss(
        self,
        emb_layer: torch.nn.Module,
        seq_output: torch.Tensor,
        pos_target: torch.Tensor,
        neg_target: torch.Tensor,
        emb_layer_ixs: Optional[slice] = None,
    ):
        emb_weights = emb_layer.weight[emb_layer_ixs] if emb_layer_ixs is not None else emb_layer.weight
        if self.l2_norm:
            seq_output = F.normalize(seq_output, p=2, dim=-1)
            emb_weights = F.normalize(emb_weights, p=2, dim=-1)

        logits = F.log_softmax(torch.matmul(seq_output, emb_weights.T), dim=-1)
        logits = logits.view(-1, emb_weights.shape[0])
        return F.nll_loss(logits, pos_target, reduction="mean", ignore_index=self.ignore_index)

    def bce_loss(
        self,
        emb_layer: torch.nn.Module,
        seq_output: torch.Tensor,
        pos_target: torch.Tensor,
        neg_target: torch.Tensor,
        **kwargs,
    ):
        eps = 1e-24
        pos_emb = emb_layer(pos_target)
        neg_emb = emb_layer(neg_target)
        if self.l2_norm:
            pos_emb = F.normalize(pos_emb, p=2, dim=-1)
            neg_emb = F.normalize(neg_emb, p=2, dim=-1)

        pos_logits = torch.sum(pos_emb.view(-1, seq_output.shape[-1]) * seq_output, -1)
        neg_logits = torch.sum(neg_emb.view(-1, seq_output.shape[-1]) * seq_output, -1)
        loss = torch.mean(-torch.log(torch.sigmoid(pos_logits) + eps) - torch.log(1 - torch.sigmoid(neg_logits) + eps))
        return loss

    def bpr_loss(
        self,
        emb_layer: torch.nn.Module,
        seq_output: torch.Tensor,
        pos_target: torch.Tensor,
        neg_target: torch.Tensor,
        **kwargs,
    ):
        gamma = 1e-10
        pos_items_emb = emb_layer(pos_target)
        neg_items_emb = emb_layer(neg_target)
        if self.l2_norm:
            pos_items_emb = F.normalize(pos_items_emb, p=2, dim=-1)
            neg_items_emb = F.normalize(neg_items_emb, p=2, dim=-1)

        pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
        neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
        loss = -torch.log(gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return loss

    def __call__(
        self,
        emb_layer: torch.nn.Module,
        input: torch.Tensor,
        pos_target: torch.Tensor,
        neg_target: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        emb_layer_ixs: Optional[slice] = None,
    ):
        if self.last_only:
            assert lengths is not None
            if not self.batch_first:
                input = input.transpose(0, 1)
                pos_target = pos_target.transpose(0, 1)
                neg_target = neg_target.transpose(0, 1)
            batch_range = torch.arange(input.shape[0])
            # Filter cold-start cases NOTE: not necessary if we add UNK/MASK token for cold-start cases
            # mask = lengths > 1
            # batch_range = batch_range[mask]
            # lengths = lengths[mask]
            # return self.loss_fn(embeddings, input[-1], pos_target[-1], neg_target[-1])
            # As lengths include the whole item_seq length (with cand_item)
            return self.loss_fn(
                emb_layer=emb_layer,
                seq_output=input[batch_range, lengths - 2],
                pos_target=pos_target[batch_range, lengths - 2],
                neg_target=neg_target[batch_range, lengths - 2],
                emb_layer_ixs=emb_layer_ixs,
            )
        return self.loss_fn(
            emb_layer, input, pos_target.reshape((-1,)), neg_target.reshape((-1,)), emb_layer_ixs=emb_layer_ixs
        )


class ConceptKLLoss:
    """
    KL divergence loss for concept-distribution supervision.

    Used during quantizer training to align learned representations with the
    SAE concept space.  The loss is:

        λ · KL(P_groundtruth ‖ Q_predicted)

    where ``Q_predicted = softmax(projection_head(encoder_output))`` and
    ``P_groundtruth`` is the L1-normalised SAE concept distribution.

    Args:
        kl_weight: Scaling factor λ (0 disables the loss).
        reduction: Reduction mode passed to :func:`torch.nn.functional.kl_div`
                   (default: ``"batchmean"``).
    """

    def __init__(self, kl_weight: float = 1.0, reduction: str = "batchmean"):
        self.kl_weight = kl_weight
        self.reduction = reduction

    def __call__(
        self,
        log_pred: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute weighted KL divergence.

        Args:
            log_pred: Log-probability predictions ``(B, concept_dim)``
                      (output of ``F.log_softmax`` applied to the projection head).
            targets:  L1-normalised ground-truth concept distributions ``(B, concept_dim)``.

        Returns:
            Scalar KL loss scaled by ``kl_weight``.
        """
        if self.kl_weight == 0:
            return torch.tensor(0.0, device=log_pred.device)
        kl = F.kl_div(log_pred, targets.to(log_pred.device), reduction=self.reduction)
        return self.kl_weight * kl
