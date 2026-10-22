"""Code adapted from https://github.com/lileipisces/PETER

The PETER class supports both raw item IDs (1 token per item) and Semantic IDs
(C tokens per item, one per RQ-VAE codebook level). SID mode is activated
automatically when ``data_info`` carries ``n_codebooks``, ``sid_tokenizer``, and
``sid_vocab_size`` attributes (set by the dataset class upon SID tokenizer init).
"""

import math
from collections import defaultdict
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModel

from nlp_xrs.utils.constants import (
    PAD_TOK,
    InputType,
    Key,
    ModelType,
    SeqMode,
    Task,
)
from nlp_xrs.utils.funcs import predict

from ..base import MLP, BaseModel, PositionalEncoding, TransformerEncoder, TransformerEncoderLayer
from .base import generate_peter_sid_mask, generate_square_subsequent_mask


class PETER(BaseModel):
    TASKS = [Task.RATING, Task.EXPLANATION, Task.CONTEXT]
    INPUT_TYPE = InputType.REGULAR
    MODEL_TYPE = ModelType.GENERATIVE
    SEQ_MODE = SeqMode.NO_SEQ

    def __init__(self, data_info: SimpleNamespace, cfg: SimpleNamespace, **kwargs: Any) -> None:
        self.use_pretrained_lm = getattr(cfg, "use_pretrained_lm", False)

        super().__init__(data_info, cfg, **kwargs)

        self.dropout = getattr(cfg, "dropout", 0.5)
        self.mask_type = getattr(cfg, "mask_type", "peter")
        self.pretrained_model_name = getattr(cfg, "pretrained_model_name", None)
        self.tied_weights = getattr(cfg, "tied_weights", False) or self.use_pretrained_lm

        self.use_sae_context = hasattr(self.data_info, "sae_n_concepts") and self.data_info.sae_n_concepts > 0
        self.use_concept = getattr(cfg, "use_concept", False)
        if self.use_concept and not self.use_sae_context:
            raise ValueError("use_concept requires SAE concept targets (data.sae_cfg).")

        init_cb = getattr(cfg, "init_codebooks", False)
        if init_cb and self.sid_tokenizer is not None:
            self._configure_sid_item_weights()

        if self.use_pretrained_lm and self.pretrained_model_name is not None:
            self._init_pretrained_model(cfg)
        else:
            self._init_scratch_model(cfg, data_info)

        if self.tied_weights:
            pred_layer = nn.Linear(self.word_emb.embedding_dim, self.data_info.n_wtokens, bias=False)
            pred_layer.weight = self.word_emb.weight
            if self.hsize != self.word_emb.embedding_dim:
                # Apply projection and linear layer tied to initial embedding weights
                self.hidden2wtoken = nn.Sequential(
                    nn.Linear(self.hsize, self.word_emb.embedding_dim), nn.ReLU(), pred_layer
                )
            else:
                self.hidden2wtoken = pred_layer
        else:
            self.hidden2wtoken = nn.Linear(self.hsize, self.data_info.n_wtokens)
        self.rating_predictor = MLP(hidden_sizes=(self.hsize, self.def_emsize))

        # NOTE: Model input is {item, user, text}. item can be either a single item ID or a sequence of semantic codes.
        self.lengths = self.data_info.lengths

        if self.sid_agg is not None:
            assert self.sid_agg in ["mean"], "Invalid SID aggregation method"
            # NOTE: Update depending on the aggregation method
            self.lengths[Key.ITEM] = self.lengths[Key.ITEM] // self.n_codebooks

        # User SID: collapse the user sequence length back to 1 because
        # get_embedding_input always mean-pools user SID codes unconditionally.
        if self.user_sid_tokenizer is not None and self.sid_agg is not None:
            assert self.sid_agg in ["mean"], "Invalid SID aggregation method"
            self.lengths[Key.USER] = self.lengths[Key.USER] // self.user_n_codebooks

        self.ui_len = self.lengths[Key.USER] + self.lengths[Key.ITEM]
        self.src_len = self.ui_len
        self.concept_len = self.data_info.sae_topk if self.use_concept else 0
        self.src_len += self.concept_len
        if getattr(cfg, "use_feature", False):
            self.src_len += self.lengths[Key.FEAT]
        self.tgt_len = self.lengths[Key.EXP] - 1
        if self.use_sae_context:
            self.context_len = self.data_info.sae_topk
        else:
            self.context_len = self.lengths[Key.EXP]

        # Index of the last item code. For regular items n_codebooks=1 → i_ix=0; for SID n_codebooks=C → i_ix=C-1.
        self.i_ix = self.lengths[Key.ITEM] - 1
        self.u_ix = self.lengths[Key.ITEM] + self.lengths[Key.USER] - 1  # i_len accounts for SID via data_info.lengths

        if self.mask_type.lower() == "peter":
            # PETER-style mask adapted allows for all item codes to attend to each other and to the user.
            self.attn_mask = generate_peter_sid_mask(self.ui_len, self.src_len, self.tgt_len)
        else:
            self.attn_mask = generate_square_subsequent_mask(self.src_len + self.tgt_len)

        # Initialize loss functions
        # Create special token mask for the extended vocabulary
        special_tok_mask = torch.ones((self.word_emb.weight.size(0),), dtype=torch.float)
        ignore_idxs = [self.tok.bos_token_id, self.tok.pad_token_id, self.tok.eos_token_id]
        assert all([ix is not None for ix in ignore_idxs])
        special_tok_mask[ignore_idxs] = 0
        if not self.use_sae_context:
            self.context_head = self.hidden2wtoken
            self.context_criterion = nn.NLLLoss(weight=special_tok_mask)
        else:
            self.context_head = nn.Linear(self.hsize, self.data_info.sae_n_concepts)
            if self.use_concept:
                self.concept_emb = nn.Embedding(self.data_info.sae_n_concepts, self.hsize)
                self.concept_emb.weight = self.context_head.weight

        self.init_weights()

        self.rating_criterion = nn.MSELoss(reduction="none")
        self.text_criterion = nn.NLLLoss(ignore_index=self.tok.pad_token_id)
        self.loss_fns = self.get_loss_fns()

    def _configure_sid_item_weights(self):
        """
        Item Embedding will be initialized from the RQ-VAE codebook centroids.
        This method configures `self.data_info` according to `get_embeddings` method.
        """
        codebooks = self.sid_tokenizer.model.quantizer.codebooks  # (C, K, D_rqvae)

        # Concatenate codebooks along the first dimension
        codebooks = codebooks.view(-1, codebooks.shape[-1])
        setattr(self.data_info, f"{Key.ITEM}_emb", codebooks)
        setattr(self.data_info, f"load_{Key.ITEM}_embeddings", True)

    def _init_pretrained_model(self, cfg: SimpleNamespace) -> None:
        """Initialize pre-trained language model from Huggingface"""
        self.logger.info(f"Loading pre-trained model: {self.pretrained_model_name}")

        # Load config and model
        hf_config = AutoConfig.from_pretrained(self.pretrained_model_name)
        self.transformer = AutoModel.from_pretrained(self.pretrained_model_name)

        if hf_config.vocab_size < self.data_info.n_wtokens:
            self.logger.info(f"Resizing token embeddings from {hf_config.vocab_size} to {self.data_info.n_wtokens}")
            self.transformer.resize_token_embeddings(self.data_info.n_wtokens)

        # Get hidden size from pre-trained model
        self.hsize = hf_config.hidden_size

        # Get the word embedding layer from pre-trained model
        if hasattr(self.transformer, "embeddings"):
            self.word_emb = self.transformer.embeddings.word_embeddings
        elif hasattr(self.transformer, "wte"):
            self.word_emb = self.transformer.wte
        elif hasattr(self.transformer, "embed_tokens"):
            self.word_emb = self.transformer.embed_tokens
        else:
            raise ValueError("Could not find embedding layer in pre-trained model")

        # Create separate embeddings for users and items at the pre-trained model's hidden size.
        item_vocab_size = self.sid_vocab_size if self.sid_vocab_size is not None else self.data_info.n_items
        user_vocab_size = getattr(self.data_info, "user_sid_vocab_size", None) or self.data_info.n_users
        self.user_emb = self.get_embeddings(Key.USER, user_vocab_size)
        self.item_emb = self.get_embeddings(Key.ITEM, item_vocab_size)

        # Freeze pre-trained weights if specified
        if getattr(cfg, "freeze_pretrained", False):
            for param in self.transformer.parameters():
                param.requires_grad = False
            # Word embeddings will be frozen, but user/item embeddings stay trainable

        self.pos_encoder = None  # Pre-trained models have their own positional encoding

    def _init_scratch_model(self, cfg: SimpleNamespace, data_info: SimpleNamespace):
        """Initialize model from scratch."""
        nhead = cfg.nhead
        nhid = cfg.nhid
        nlayers = cfg.nlayers

        self.word_emb = self.get_embeddings(Key.EXP, data_info.n_wtokens, self.tok.pad_token_id)

        # SID mode: codes from C codebooks share one embedding table of size sid_vocab_size,
        # with each codebook level occupying its own offset range (set by the dataset).
        # Regular mode: a single embedding per item/user.
        item_vocab_size = self.sid_vocab_size if self.sid_vocab_size is not None else data_info.n_items
        self.item_emb = self.get_embeddings(Key.ITEM, item_vocab_size)

        # User SID vocab (ablation conditions 3/4/5): if the dataset sets a
        # user_sid_vocab_size on data_info, the user embedding table is sized
        # accordingly (same offset-range scheme as item SIDs).
        user_vocab_size = getattr(data_info, "user_sid_vocab_size", None) or data_info.n_users
        self.user_emb = self.get_embeddings(Key.USER, user_vocab_size)

        self.pos_encoder = PositionalEncoding(self.hsize, self.dropout)
        encoder_layers = TransformerEncoderLayer(self.hsize, nhead, nhid, self.dropout)
        self.transformer = TransformerEncoder(encoder_layers, nlayers)

    def requires_common_blocks(self) -> bool:
        return False

    def init_weights(self):
        initrange = 0.1

        if self.use_pretrained_lm:
            # Initialize user and item embeddings
            self.user_emb.weight.data.uniform_(-initrange, initrange)
            self.item_emb.weight.data.uniform_(-initrange, initrange)
            # Word embeddings from pre-trained model are already initialized
        else:
            self.user_emb.weight.data.uniform_(-initrange, initrange)
            self.item_emb.weight.data.uniform_(-initrange, initrange)
            self.word_emb.weight.data.uniform_(-initrange, initrange)

        if not self.tied_weights:
            self.hidden2wtoken.weight.data.uniform_(-initrange, initrange)
            self.hidden2wtoken.bias.data.zero_()
        elif isinstance(self.hidden2wtoken, nn.Sequential):
            for layer in self.hidden2wtoken:
                if isinstance(layer, nn.Linear):
                    layer.weight.data.uniform_(-initrange, initrange)
                    if layer.bias is not None:
                        layer.bias.data.zero_()

        if self.use_sae_context:
            self.context_head.weight.data.uniform_(-initrange, initrange)
            self.context_head.bias.data.zero_()

    def _cat_src(
        self,
        ui_src: torch.Tensor,
        text_src: torch.Tensor,
        concepts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Concatenate UI, optional concept, and word representations."""
        srcs = [ui_src]
        if concepts is not None:
            srcs.append(self.concept_emb(concepts))
        srcs.append(text_src)
        return torch.cat(srcs, 0)

    def get_embedding_input(
        self,
        user: torch.Tensor,
        item: torch.Tensor,
        text: torch.Tensor,
        concepts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Item SID: shape (C, B, D) when SID tokens, (1, B, D) otherwise.
        i_src = self.item_emb(item)
        if i_src.shape[0] > 1 and self.sid_tokenizer is not None and self.sid_agg is not None:
            IxC, B, D = i_src.shape
            i_src = i_src.reshape(IxC // self.n_codebooks, self.n_codebooks, B, D).mean(dim=1)

        # User SID: shape (n_codebooks, B, D) when user SIDs are enabled and
        # user has been replaced with an SID token sequence; (1, B, D) otherwise.
        u_src = self.user_emb(user)
        if u_src.shape[0] > 1 and self.user_sid_tokenizer is not None and self.sid_agg is not None:
            # Multi-token user: mean-pool codebook dimension
            u_src = u_src.mean(dim=0, keepdim=True)  # → (1, B, D)

        w_src = self.word_emb(text)  # (seq_len, B, D)

        src = self._cat_src(torch.cat([i_src, u_src], 0), w_src, concepts)

        if not self.use_pretrained_lm:
            # Only scale for scratch models
            src = src * math.sqrt(self.hsize)

        return src

    def get_val_loss(self, results: Dict[Task, float]) -> float:
        return results[Task.EXPLANATION] + results[Task.RATING] * int(self.task_agg.task_w[Task.RATING] != 0)

    def prepare_xy(
        self, batch: Dict[str, torch.Tensor], generation: bool = False
    ) -> Tuple[Dict[str, torch.Tensor], Dict[Task, torch.Tensor]]:
        inputs = batch.copy()
        if generation:
            B = batch[Key.USER].shape[0]
            inputs[Key.EXP] = torch.full((B, 1), self.tok.bos_token_id, dtype=torch.long, device=batch[Key.USER].device)
            # The first generation step predicts concepts; subsequent calls
            # receive its predicted IDs through process_first_genstep.
            inputs.pop(Key.CONCEPT_ID, None)
        else:
            inputs[Key.EXP] = inputs[Key.EXP][..., :-1].contiguous()

        if self.use_sae_context:
            context_labels = batch[Key.CONCEPT_ID]
        else:
            context_labels = batch[Key.CONTEXT].T.contiguous()

        labels = {
            Task.RATING: batch[Key.RATING].squeeze(1),
            Task.EXPLANATION: batch[Key.EXP][..., 1:].contiguous(),
            Task.CONTEXT: context_labels,
        }

        if getattr(self.cfg, "use_feature", False):
            inputs[Key.EXP] = torch.cat([inputs[Key.FEAT], inputs[Key.EXP]], 1)
        if self.use_concept and not generation:
            inputs[Key.CONCEPT_ID] = batch[Key.CONCEPT_ID]

        labels[Task.EXPLANATION] = labels[Task.EXPLANATION].T.contiguous()
        return inputs, labels

    def unpack_batch(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return batch[Key.USER].T, batch[Key.ITEM].T, batch[Key.EXP].T

    def get_masks(
        self,
        user: torch.Tensor,
        item: torch.Tensor,
        text: torch.Tensor,
        total_len: int,
        concepts: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_mask = self.attn_mask[:total_len, :total_len].to(user.device)
        middle = torch.zeros((1, user.shape[1]), dtype=torch.bool, device=user.device)
        left = item == self.mappers[Key.ITEM].get_idx(PAD_TOK)
        if item.shape[0] > 1 and self.sid_tokenizer is not None and self.sid_agg is not None:
            IxC, B = item.shape
            left = left.reshape(IxC // self.n_codebooks, self.n_codebooks, B).any(dim=1)
        right = text == self.tok.pad_token_id
        masks = [left, middle]
        if concepts is not None:
            masks.append(torch.zeros_like(concepts, dtype=torch.bool))
        masks.append(right)
        key_padding_mask = torch.cat(masks, 0).t()
        return attn_mask, key_padding_mask

    def predict_context(self, hidden: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        context_prob = self.context_head(hidden[self.i_ix])  # (batch_size, ntoken)
        log_context_dis = F.log_softmax(context_prob, dim=-1)
        return log_context_dis

    def predict_rating(self, hidden: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        rating = self.rating_predictor(hidden[self.u_ix])  # (batch_size,)
        return rating

    def predict_exp(self, hidden: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        src_len = kwargs.get("src_len", self.src_len)
        word_prob = self.hidden2wtoken(hidden[src_len:])  # (tgt_len, batch_size, ntoken)
        log_word_prob = F.log_softmax(word_prob, dim=-1)
        return log_word_prob

    def predict_tasks(self, hidden: torch.Tensor, **kwargs: Any) -> Dict[Task, torch.Tensor]:
        return {
            Task.EXPLANATION: self.predict_exp(hidden, **kwargs),
            Task.RATING: self.predict_rating(hidden, **kwargs),
            Task.CONTEXT: self.predict_context(hidden, **kwargs),
        }

    def rating_loss(self, preds: torch.Tensor, labels: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.rating_criterion(preds, labels).mean()

    def exp_loss(self, preds: torch.Tensor, labels: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.text_criterion(preds.view(-1, self.data_info.n_wtokens), labels.view(-1))

    def context_loss(self, preds: torch.Tensor, labels: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        labels = labels.T  # [B, vs]

        # Equivalent to selecting NLLLoss inputs after the repeat/view.
        nll = -preds.gather(dim=1, index=labels)  # [B, vs]

        # NLLLoss(weight=..., reduction="mean") divides by the
        # sum of selected class weights, not simply by the number of labels.
        weights = self.context_criterion.weight[labels]  # [B, vs]

        return (nll * weights).sum() / weights.sum().clamp_min(1e-12)

    def concept_loss(self, preds: torch.Tensor, labels: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        activations = kwargs[Key.CONCEPT_ACT] / kwargs[Key.CONCEPT_ACT].norm(p=1, dim=1, keepdim=True).clamp(min=1e-10)
        label_dist = torch.zeros(labels.shape[0], self.data_info.sae_n_concepts, device=labels.device)
        label_dist.scatter_(1, labels, activations)
        return F.cross_entropy(preds, label_dist)

    def set_batchwise_loss_args(self, inputs, labels):
        super().set_batchwise_loss_args(inputs, labels)
        if self.use_sae_context:
            self.loss_args[Key.CONCEPT_ACT] = inputs[Key.CONCEPT_ACT]

    def get_loss_fns(self) -> Dict[Task, Any]:
        loss_fns = {Task.EXPLANATION: self.exp_loss, Task.RATING: self.rating_loss}
        if self.use_sae_context:
            loss_fns[Task.CONTEXT] = self.concept_loss
        else:
            loss_fns[Task.CONTEXT] = self.context_loss
        return loss_fns

    def _forward(
        self,
        user: torch.Tensor,
        item: torch.Tensor,
        text: torch.Tensor,
        concepts: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        src = self.get_embedding_input(user, item, text, concepts)

        if self.use_pretrained_lm:
            # For pre-trained models, use their forward pass
            # Convert to (batch_size, seq_len, hidden_size)
            src = src.transpose(0, 1)

            _, key_padding_mask = self.get_masks(user, item, text, total_len=src.size(1), concepts=concepts)

            # Get outputs from pre-trained model
            outputs = self.transformer(
                inputs_embeds=src, attention_mask=(~key_padding_mask).float(), return_dict=True, output_attentions=True
            )

            # Convert back to (seq_len, batch_size, hidden_size)
            hidden = outputs.last_hidden_state.transpose(0, 1)
            attns = torch.stack(outputs.attentions, dim=0) if outputs.attentions else None
        else:
            # Original transformer encoder
            src = self.pos_encoder(src)
            attn_mask, key_padding_mask = self.get_masks(user, item, text, total_len=src.size(0), concepts=concepts)
            hidden, attns = self.transformer(src, attn_mask, key_padding_mask)
        return hidden, attns

    def forward(self, batch: Dict[str, torch.Tensor], **kwargs: Any) -> Dict[Any, Any]:
        """
        :param user: (batch_size, 1), torch.int64
        :param item: (batch_size, 1), torch.int64
        :param text: (total_len - ui_len, batch_size), torch.int64
        :return log_word_prob: target tokens (tgt_len, batch_size, ntoken)
        :return log_context_dis: (batch_size, ntoken)
        :return rating: (batch_size,)
        :return attns: (nlayers, batch_size, total_len, total_len)
        """
        # NOTE: All texts start with BOS token. Only texts shorter than tgt_len end with EOS token.
        user, item, text = self.unpack_batch(batch)
        concepts = batch.get(Key.CONCEPT_ID) if self.use_concept else None
        if concepts is not None:
            concepts = concepts.T

        hidden, attns = self._forward(user, item, text, concepts)

        src_len = self.ui_len + (0 if concepts is None else concepts.shape[0])
        if getattr(self.cfg, "use_feature", False):
            src_len += self.lengths[Key.FEAT]
        res = self.predict_tasks(hidden, src_len=src_len)
        res["attns"] = attns
        return res

    def process_first_genstep(
        self,
        pred: Dict[Task, Any],
        batch: Dict[str, torch.Tensor],
        inputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[Task, List]:
        pred[Task.RATING] = pred[Task.RATING].tolist()
        concept_ids = predict(pred[Task.CONTEXT], topk=self.context_len)
        if self.use_concept and inputs is not None:
            inputs[Key.CONCEPT_ID] = concept_ids
            pred[Task.EXPLANATION] = self(inputs)[Task.EXPLANATION]
        pred[Task.CONTEXT] = concept_ids.tolist()
        return pred

    def generate(
        self, batch: Dict[str, torch.Tensor], **kwargs: Any
    ) -> Tuple[Dict[Task, List], Dict[Task, torch.Tensor]]:
        inputs, labels = self.prepare_xy(batch, generation=True)

        preds = defaultdict(list)
        words = []
        words_lens = torch.zeros(batch["size"], dtype=torch.long, device=inputs[Key.USER].device)
        for idx in range(self.tgt_len):
            pred = self(inputs)
            if idx == 0:
                for t, v in self.process_first_genstep(pred, batch, inputs).items():
                    preds[t].extend(v)

            word_probs = pred[Task.EXPLANATION][-1].exp()
            _, word_var = word_probs.max(-1)

            words.append(word_var)

            if self.check_gen_end(word_var, words_lens, idx):
                break

            inputs[Key.EXP] = torch.cat((inputs[Key.EXP], word_var.unsqueeze(1)), dim=1)

        words = torch.stack(words, dim=1)

        preds[Task.EXPLANATION] = words.cpu().numpy().tolist()
        return preds, labels
