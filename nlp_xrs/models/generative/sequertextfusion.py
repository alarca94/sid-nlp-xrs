from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from nlp_xrs.utils.constants import PAD_TOK, Key, SeqMode

from .sequer import SEQUER


class SEQUERTextFusion(SEQUER):
    """
    SEQUERTextFusion extends SEQUER to fuse collaborative item embeddings with precomputed item text embeddings
    via different fusion mechanisms:
    - gating: Uses a learnable gate to combine embeddings
    - concat-mlp: Concatenates embeddings and passes through MLP
    - attention: Uses attention mechanism to weight and combine embeddings
    """

    def __init__(self, data_info: SimpleNamespace, cfg: SimpleNamespace, **kwargs):
        if cfg.fusion_type.lower() == "nofusion":
            # Adapt data_info.lengths so the mask is created for the correct sequence length
            data_info.lengths[Key.ITEM] *= 2  # [item_cf_1...N, item_text_1...N]

        super().__init__(data_info, cfg, **kwargs)
        # Get the item text embedding matrix from data_info
        item_text_emb = getattr(data_info, "item_text_emb", None)
        if item_text_emb is None:
            raise ValueError("SEQUERTextFusion requires item_text_emb in data_info. Use SEQUERTextFusionDataset.")
        # Ensure item_text_emb is a torch.Tensor and on the correct device
        if not isinstance(item_text_emb, torch.Tensor):
            item_text_emb = torch.tensor(item_text_emb, dtype=torch.float)
        self.item_text_emb = nn.Embedding.from_pretrained(
            item_text_emb, padding_idx=self.data_info.special_items[PAD_TOK]
        )
        if self.item_text_emb.embedding_dim != self.hsize:
            self.item_text_emb = nn.Sequential(
                self.item_text_emb, nn.Linear(self.item_text_emb.embedding_dim, self.hsize)
            )

        if cfg.fusion_type.lower() == "gating":
            # Gating mechanism: simple linear gate (configurable)
            self.gate_layer = nn.Linear(self.hsize * 3, 1)  # [user_emb, item_emb, text_emb] -> gate
            self.get_embedding_input = self.get_embedding_input_gating
        elif cfg.fusion_type.lower() == "concat-mlp":
            # Concatenation + MLP mechanism
            mlp_hidden_size = getattr(cfg, "fusion_mlp_hidden_size", self.hsize * 2)
            self.mlp = nn.Sequential(
                nn.Linear(self.hsize * 2, mlp_hidden_size),  # [item_emb, text_emb]
                nn.ReLU(),
                nn.Dropout(getattr(cfg, "fusion_mlp_dropout", self.dropout)),
                nn.Linear(mlp_hidden_size, self.hsize),
            )
            self.get_embedding_input = self.get_embedding_input_concat_mlp
        elif cfg.fusion_type.lower() == "attention":
            # Attention mechanism
            attention_dim = getattr(cfg, "fusion_attention_dim", self.hsize)
            self.attention_query = nn.Linear(self.hsize, attention_dim)
            self.attention_key = nn.Linear(self.hsize, attention_dim)
            self.attention_value = nn.Linear(self.hsize, attention_dim)
            self.attention_output = nn.Linear(attention_dim, self.hsize)
            self.attention_dropout = nn.Dropout(getattr(cfg, "fusion_attention_dropout", self.dropout))
            self.get_embedding_input = self.get_embedding_input_attention
        elif cfg.fusion_type.lower() == "nofusion":
            # New mode: concatenate item_cf and item_text embeddings in sequence
            self.get_embedding_input = self.get_embedding_input_cf_text_seq
            self.predict_rating = self.predict_rating_extended
            self.predict_next = self.predict_next_extended
            self.get_masks = self.get_masks_extended
            self.use_text_for_nextitem = getattr(cfg, "use_text_for_nextitem", False)
        else:
            raise NotImplementedError(
                f"Fusion {cfg.fusion_type.lower()} not implemented. Available types are: {{gating, concat-mlp, attention, nofusion}}"
            )

    def unpack_batch(self, batch):  # type: ignore[override]
        """Return collaborative and text-lookup item IDs together."""
        user, item_cf, text, seq_len = super().unpack_batch(batch)
        item_raw = batch.get(Key.ITEM_RAW)
        item_text_ids = item_raw.T if item_raw is not None else item_cf
        return user, [item_cf, item_text_ids], text, seq_len

    @staticmethod
    def _split_item(item):
        """Split collaborative and text-lookup item IDs."""
        if isinstance(item, (list, tuple)):
            return item
        return item, item

    def get_ui_text_embeddings(
        self, user: torch.Tensor, item: torch.Tensor, item_raw: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build user, collaborative-item, and raw-item-text embeddings."""
        i_src_cf = self.item_emb(item)
        if i_src_cf.shape[0] > 1 and self.sid_tokenizer is not None and self.sid_agg is not None:
            item_len, batch_size, embedding_dim = i_src_cf.shape
            i_src_cf = i_src_cf.reshape(item_len // self.n_codebooks, self.n_codebooks, batch_size, embedding_dim).mean(
                dim=1
            )

        u_src = self.user_emb(user)
        if u_src.shape[0] > 1 and self.user_sid_tokenizer is not None and self.sid_agg is not None:
            u_src = u_src.mean(dim=0, keepdim=True)

        return u_src, i_src_cf, self.item_text_emb(item_raw)

    def get_embedding_input_gating(
        self, user: torch.Tensor, item: torch.Tensor, text: torch.Tensor, concepts: torch.Tensor | None = None
    ) -> torch.Tensor:
        item_cf, item_raw = self._split_item(item)
        u_src, i_src_cf, text_emb = self.get_ui_text_embeddings(user, item_cf, item_raw)
        # Expand user embedding to match item sequence
        u_src_exp = u_src.expand(i_src_cf.shape[0], -1, -1)  # (seq_len, batch_size, emsize)
        # Gating: concatenate user and item emb, pass through gate_layer
        gate_input = torch.cat([u_src_exp, i_src_cf, text_emb], dim=-1)  # (seq_len, batch_size, 3*emsize)
        gate = torch.sigmoid(self.gate_layer(gate_input))  # (seq_len, batch_size, 1)
        # Fuse
        fused_emb = gate * i_src_cf + (1 - gate) * text_emb  # (seq_len, batch_size, emsize)
        w_src = self.word_emb(text)  # (total_len - ui_len - hist_len, batch_size, emsize)
        src = self._cat_src(torch.cat([fused_emb, u_src], 0), w_src, concepts)
        src = src * torch.sqrt(torch.tensor(self.hsize, dtype=src.dtype, device=src.device))
        return src

    def get_embedding_input_concat_mlp(
        self, user: torch.Tensor, item: torch.Tensor, text: torch.Tensor, concepts: torch.Tensor | None = None
    ) -> torch.Tensor:
        item_cf, item_raw = self._split_item(item)
        u_src, i_src_cf, text_emb = self.get_ui_text_embeddings(user, item_cf, item_raw)
        # Concatenate all embeddings and pass through MLP
        concat_input = torch.cat([i_src_cf, text_emb], dim=-1)  # (seq_len, batch_size, 3*emsize)
        fused_emb = self.mlp(concat_input)  # (seq_len, batch_size, emsize)
        w_src = self.word_emb(text)  # (total_len - ui_len - hist_len, batch_size, emsize)
        src = self._cat_src(torch.cat([fused_emb, u_src], 0), w_src, concepts)
        src = src * torch.sqrt(torch.tensor(self.hsize, dtype=src.dtype, device=src.device))
        return src

    def get_embedding_input_attention(
        self, user: torch.Tensor, item: torch.Tensor, text: torch.Tensor, concepts: torch.Tensor | None = None
    ) -> torch.Tensor:
        item_cf, item_raw = self._split_item(item)
        u_src, i_src_cf, text_emb = self.get_ui_text_embeddings(user, item_cf, item_raw)
        # Expand user embedding to match item sequence
        u_src_exp = u_src.expand(i_src_cf.shape[0], -1, -1)  # (seq_len, batch_size, emsize)

        # Use user embedding as query, item and text embeddings as key-value pairs
        # Query: user embedding
        query = self.attention_query(u_src_exp)  # (seq_len, batch_size, attention_dim)

        # Keys and Values: concatenate item and text embeddings
        # Create a sequence of [item_emb, text_emb] for each position
        keys_values = torch.stack([i_src_cf, text_emb], dim=1)  # (seq_len, 2, batch_size, emsize)
        keys_values = keys_values.transpose(1, 2)  # (seq_len, batch_size, 2, emsize)

        # Apply key and value transformations
        keys = self.attention_key(keys_values)  # (seq_len, batch_size, 2, attention_dim)
        values = self.attention_value(keys_values)  # (seq_len, batch_size, 2, attention_dim)

        # Compute attention scores
        # query: (seq_len, batch_size, attention_dim)
        # keys: (seq_len, batch_size, 2, attention_dim)
        attention_scores = torch.matmul(keys, query.unsqueeze(-1)).squeeze(-1)  # (seq_len, batch_size, 2)
        attention_weights = F.softmax(attention_scores, dim=-1)  # (seq_len, batch_size, 2)
        attention_weights = self.attention_dropout(attention_weights)

        # Apply attention weights to values
        attended_values = torch.sum(
            attention_weights.unsqueeze(-1) * values, dim=2
        )  # (seq_len, batch_size, attention_dim)

        # Project back to embedding dimension
        fused_emb = self.attention_output(attended_values)  # (seq_len, batch_size, emsize)

        w_src = self.word_emb(text)  # (total_len - ui_len - hist_len, batch_size, emsize)
        src = self._cat_src(torch.cat([fused_emb, u_src], 0), w_src, concepts)
        src = src * torch.sqrt(torch.tensor(self.hsize, dtype=src.dtype, device=src.device))
        return src

    def get_embedding_input_cf_text_seq(
        self, user: torch.Tensor, item: torch.Tensor, text: torch.Tensor, concepts: torch.Tensor | None = None
    ) -> torch.Tensor:
        item_cf, item_raw = self._split_item(item)
        u_src, i_src_cf, text_emb = self.get_ui_text_embeddings(user, item_cf, item_raw)
        S, B, E = i_src_cf.shape
        i_src = torch.stack([i_src_cf, text_emb], dim=1)  # (seq_len, 2, batch_size, emsize)
        i_src = i_src.view(S * 2, B, E)  # (2*seq_len, batch_size, emsize)
        # Stack: [item_cf_1...N, item_text_1...N, user, concepts?, text]
        src = self._cat_src(torch.cat([i_src, u_src], 0), self.word_emb(text), concepts)
        src = src * torch.sqrt(torch.tensor(self.hsize, dtype=src.dtype, device=src.device))
        return src

    def predict_rating_extended(self, hidden, **kwargs):
        if self.SEQ_MODE not in [SeqMode.NO_SEQ, SeqMode.HIST_ITEM_RATING_LAST]:
            rating = self.rating_predictor(hidden[0 : self.u_ix : 2])
        else:
            # rating = self.rating_predictor(hidden[self.lens[Key.ITEM] - 1:self.lens[Key.ITEM]])
            batch_ixs = torch.arange(hidden.shape[1], device=hidden.device)
            rating = self.rating_predictor(hidden[(kwargs[Key.SEQ_LEN] - 1) * 2, batch_ixs])
        return rating

    def predict_next_extended(self, hidden, **kwargs):
        item_preds = hidden[self.use_text_for_nextitem : self.u_ix - 2 : 2]
        if self.codebook_heads is not None:
            sequence_len, batch_size, hidden_size = item_preds.shape
            item_preds = torch.stack([head(item_preds) for head in self.codebook_heads], dim=1).reshape(
                sequence_len * self.n_codebooks, batch_size, hidden_size
            )
        return item_preds

    def get_masks(self, user, item, text, total_len, concepts=None):
        item_cf, _ = self._split_item(item)
        return super().get_masks(user, item_cf, text, total_len, concepts)

    def get_masks_extended(self, user, item, text, total_len, concepts=None):
        attn_mask = self.attn_mask[:total_len, :total_len].to(user.device)
        middle = torch.zeros((1, user.shape[1]), dtype=torch.bool, device=user.device)
        _, item_raw = self._split_item(item)
        left = item_raw == self.data_info.special_items[PAD_TOK]
        left = torch.repeat_interleave(left, 2, dim=0)  # torch.zeros_like(item, dtype=torch.bool)
        right = text == self.tok.pad_token_id  # self.data_info.special_toks[PAD_TOK]
        masks = [left, middle]
        if concepts is not None:
            masks.append(torch.zeros_like(concepts, dtype=torch.bool))
        masks.append(right)
        key_padding_mask = torch.cat(masks, 0).t()
        return attn_mask, key_padding_mask
