"""Code adapted from https://github.com/alarca94/sequer-recsys23"""

from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from nlp_xrs.models.generative.base import generate_sequer_mask
from nlp_xrs.models.generative.peter import PETER
from nlp_xrs.models.losses import NextItemLoss
from nlp_xrs.utils.constants import (
    PAD_TOK,
    TOP_KS,
    InputType,
    Key,
    ModelType,
    SeqMode,
    Task,
)
from nlp_xrs.utils.funcs import predict


class ResBlock(nn.Module):
    """Residual block with a zero-initialized linear layer and SiLU activation.

    Produces an identity mapping at initialization, which lets per-codebook
    prediction heads start from a neutral state before training updates them.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.zeros_(self.linear.weight)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.linear(x))


class SEQUER(PETER):
    TASKS = [Task.RATING, Task.EXPLANATION, Task.CONTEXT, Task.NEXT_ITEM]
    INPUT_TYPE = InputType.CUSTOM
    MODEL_TYPE = ModelType.GENERATIVE
    SEQ_MODE = SeqMode.HIST_ITEM

    def __init__(self, data_info, cfg, **kwargs):
        super().__init__(data_info, cfg, **kwargs)

        nextitem_loss = getattr(cfg, "nextitem_loss", "CE")
        last_only = nextitem_loss.lower() in ["bpr", "bce"]

        self.attn_mask = generate_sequer_mask(
            self.src_len, self.tgt_len, self.mask_type, lengths=self.data_info.lengths
        )

        self.nextit_criterion = NextItemLoss(
            loss_type=nextitem_loss, ignore_index=self.mappers[Key.ITEM].get_idx(PAD_TOK), last_only=last_only
        )
        # self.nexitem_criterion = nn.NLLLoss(ignore_index=self.mappers[Key.ITEM].get_idx(UNK_TOK))

        if self.sid_tokenizer is not None and self.sid_agg is not None:
            self.codebook_heads = nn.ModuleList([ResBlock(self.hsize) for _ in range(self.n_codebooks)])
        else:
            self.codebook_heads = None

        self.init_weights()

    def prepare_xy(self, batch, generation=False):
        if generation:
            batch.pop(Key.NEG_OPT, None)
        else:
            batch.pop(Key.NEG_EVAL, None)

        inputs, labels = super().prepare_xy(batch, generation)
        labels[Task.RATING] = labels[Task.RATING].T.contiguous()
        if self.sid_tokenizer is not None and self.sid_agg is not None:
            shift_ix = self.n_codebooks
        else:
            shift_ix = 1
        labels[Task.NEXT_ITEM] = batch[Key.ITEM][:, shift_ix:].T.contiguous()
        if generation:
            labels[Task.RATING] = torch.gather(labels[Task.RATING], 0, batch[Key.SEQ_LEN] - 1).detach().clone()
            labels[Task.NEXT_ITEM] = torch.gather(labels[Task.NEXT_ITEM], 0, batch[Key.SEQ_LEN] - 2).detach().clone()
        else:
            labels[Task.NEXT_ITEM] = (
                labels[Task.NEXT_ITEM],
                batch.get(Key.NEG_OPT, torch.tensor([]))[:, shift_ix:].T.contiguous(),
            )

        return inputs, labels

    def predict_context(self, hidden, **kwargs):
        context_prob = self.context_head(hidden[self.u_ix])  # (batch_size, n_token)
        log_context_dis = F.log_softmax(context_prob, dim=-1)
        return log_context_dis

    def predict_rating(self, hidden, **kwargs):
        if self.SEQ_MODE not in [SeqMode.NO_SEQ, SeqMode.HIST_ITEM_RATING_LAST]:
            rating = self.rating_predictor(hidden[: self.u_ix])
        else:
            # rating = self.rating_predictor(hidden[self.lens[Key.ITEM] - 1:self.lens[Key.ITEM]])
            batch_ixs = torch.arange(hidden.shape[1], device=hidden.device)
            rating = self.rating_predictor(hidden[kwargs[Key.SEQ_LEN] - 1, batch_ixs])
        return rating

    def predict_next(self, hidden, **kwargs):
        item_preds = hidden[: self.u_ix - 1]  # (S, B, H)
        if self.codebook_heads is not None:
            S, B, H = item_preds.shape
            # Apply each codebook's head, then interleave along dim=0 so that
            # row t*C+i = codebook_heads[i](item_preds[t]), matching the label
            # interleaving produced by repeat_interleave in prepare_xy.
            stacked = torch.stack([head(item_preds) for head in self.codebook_heads], dim=1)  # (S, C, B, H)
            item_preds = stacked.reshape(S * self.n_codebooks, B, H)

        return item_preds

    def unpack_batch(self, batch):
        u, i, e = super().unpack_batch(batch)
        return u, i, e, batch[Key.SEQ_LEN].squeeze(1)

    def predict_tasks(self, hidden, **kwargs):
        preds = super().predict_tasks(hidden, **kwargs)
        preds[Task.NEXT_ITEM] = self.predict_next(hidden, **kwargs)
        return preds

    def nextitem_loss(self, preds, labels, **kwargs):
        pos_labels = labels[0]
        neg_labels = labels[1]
        if self.sid_tokenizer is not None and self.sid_agg is not None:
            # Parallel SID prediction. Each codebook predicts the next item code of the same codebook independently.
            # Therefore, per each codebook, compute the loss of predicting the next item code using the corresponding item embeddings.
            # CHECKME: Is it properly matching the predictions and labels to the correct codebook?
            loss = 0
            end_ixs = np.cumsum(self.sid_tokenizer.codebook_sizes)
            start_ixs = end_ixs - self.sid_tokenizer.codebook_sizes
            label_mask = pos_labels != self.mappers[Key.ITEM].get_idx(PAD_TOK)
            for i in range(self.n_codebooks):
                label_mask_i = label_mask[i :: self.n_codebooks]
                preds_i = preds[i :: self.n_codebooks][label_mask_i]
                pos_labels_i = pos_labels[i :: self.n_codebooks][label_mask_i] - start_ixs[i]
                neg_labels_i = neg_labels[i :: self.n_codebooks][label_mask_i] - start_ixs[i]
                loss += self.nextit_criterion(
                    emb_layer=self.item_emb,
                    input=preds_i,
                    pos_target=pos_labels_i,
                    neg_target=neg_labels_i,
                    lengths=kwargs[Key.SEQ_LEN],
                    emb_layer_ixs=slice(start_ixs[i], end_ixs[i]),
                )
            return loss / self.n_codebooks
        return self.nextit_criterion(
            emb_layer=self.item_emb,
            input=preds,
            pos_target=pos_labels,
            neg_target=neg_labels,
            lengths=kwargs[Key.SEQ_LEN],
        )

    def rating_loss(self, preds, labels, **kwargs):
        if preds.shape[0] == labels.shape[0] and len(preds.shape) < len(labels.shape):
            preds = preds.unsqueeze(-1)
        elif preds.shape[0] < labels.shape[0]:
            # This is most likely due to SID aggregation in the prediction and outdated labels
            labels = labels[: preds.shape[0]]
        return self.rating_criterion(preds, labels).mean()

    def get_loss_fns(self):
        loss_fns = super().get_loss_fns()

        loss_fns[Task.NEXT_ITEM] = self.nextitem_loss
        return loss_fns

    def set_batchwise_loss_args(self, inputs, labels):
        super().set_batchwise_loss_args(inputs, labels)
        self.loss_args[Key.SEQ_LEN] = inputs[Key.SEQ_LEN].squeeze(1)

    def forward(self, batch, **kwargs):
        """
        :param user: (batch_size,), torch.int64
        :param item: (batch_size, seq_len), torch.int64
        :param text: (total_len - ui_len - hist_len, batch_size), torch.int64
        :return log_word_prob: target tokens (tgt_len, batch_size, ntoken) if gen_mode=False; the last token (batch_size, ntoken) otherwise.
        :return log_context_dis: (batch_size, ntoken) if context_prediction=True; None otherwise.
        :return rating: (batch_size,) if rating_prediction=True; None otherwise.
        :return attns: (nlayers, batch_size, total_len, total_len)
        :return log_next_item: (batch_size, hist_len, nitem) if next_item_prediction=True; None otherwise
        :return
        """
        user, item, text, seq_len = self.unpack_batch(batch)
        concepts = batch.get(Key.CONCEPT_ID) if self.use_concept else None
        if concepts is not None:
            concepts = concepts.T

        hidden, attns = self._forward(user, item, text, concepts)

        src_len = self.ui_len + (0 if concepts is None else concepts.shape[0])
        if getattr(self.cfg, "use_feature", False):
            src_len += self.lengths[Key.FEAT]
        res = self.predict_tasks(hidden, **{Key.SEQ_LEN: seq_len, "src_len": src_len})
        res["attns"] = attns
        return res

    def process_first_genstep(
        self,
        pred: Dict[Task, Any],
        batch: Dict[str, torch.Tensor],
        inputs: Dict[str, torch.Tensor] | None = None,
    ) -> Dict[Task, List]:
        batch_ixs = torch.arange(batch[Key.USER].shape[0])
        seq_len = batch[Key.SEQ_LEN].squeeze(1)

        if len(pred[Task.RATING].shape) > 1:
            pred[Task.RATING] = pred[Task.RATING][seq_len - 1, batch_ixs]

        if self.sid_tokenizer is not None and self.sid_agg is not None:
            # Filter last predictions for each codebook.
            _, B, E = pred[Task.NEXT_ITEM].shape
            last_preds = pred[Task.NEXT_ITEM].reshape(-1, self.n_codebooks, B, E)[seq_len - 2, :, batch_ixs]
            # Get code logits
            end_ixs = np.cumsum(self.sid_tokenizer.codebook_sizes)
            start_ixs = end_ixs - self.sid_tokenizer.codebook_sizes
            logits = []
            for i in range(self.n_codebooks):
                preds_i = last_preds[:, i]
                logits.append(
                    F.log_softmax(torch.matmul(preds_i, self.item_emb.weight[start_ixs[i] : end_ixs[i]].T), dim=-1)
                )
            # Stack logits and aggregate per item.
            logits = torch.cat(logits, dim=-1)
            item_to_tokens = (torch.tensor(self.sid_tokenizer.item_to_tokens.tolist()) + start_ixs).to(
                last_preds.device
            )
            pred[Task.NEXT_ITEM] = torch.gather(
                input=logits.unsqueeze(-2).expand(-1, item_to_tokens.shape[0], -1),  # (batch_size, n_items, n_tokens)
                dim=-1,
                index=item_to_tokens.unsqueeze(0).expand(logits.shape[0], -1, -1),  # (batch_size, n_items, code_dim)
            ).mean(dim=-1)
        else:
            pred[Task.NEXT_ITEM] = F.log_softmax(
                torch.matmul(pred[Task.NEXT_ITEM][seq_len - 2, batch_ixs], self.item_emb.weight.T), dim=-1
            ).squeeze(0)
        # [BSZ, HSZ] * [HSZ, NI] = BSZ, NI
        pred[Task.NEXT_ITEM_SAMPLE] = torch.gather(pred[Task.NEXT_ITEM], 1, batch[Key.NEG_EVAL]).tolist()
        pred[Task.NEXT_ITEM] = predict(pred[Task.NEXT_ITEM], topk=max(TOP_KS)).tolist()
        return super().process_first_genstep(pred, batch, inputs)
