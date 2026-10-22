from __future__ import annotations

import gc
import logging
import os
import re
import shutil
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from tokenizers import Tokenizer, decoders, trainers
from tokenizers.models import BPE, WordLevel
from tokenizers.pre_tokenizers import ByteLevel, WhitespaceSplit
from tokenizers.processors import TemplateProcessing
from torch.nn import functional as F
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from nlp_xrs.semantic_ids.utils import get_sae_dists
from nlp_xrs.utils.constants import (
    BOS_TOK,
    EOS_TOK,
    MAX_HIST_LEN,
    UNK_TOK,
    VOCAB_SIZE,
    Key,
    Suffix,
)
from nlp_xrs.utils.in_out import get_acts_mapper, load_safe_parquet

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


def get_context(docs, pos_tags=("NOUN", "ADJ")):
    import spacy

    nlp = spacy.load("en_core_web_sm", exclude=["parser", "lemmatizer", "ner"])

    contexts = [" ".join(set([tok.text for tok in s if tok.pos_ in pos_tags])) for s in nlp.pipe(docs)]

    return contexts


class Pipeline:
    """Ordered, named sequence of pipeline step functions.

    Each step is stored as a ``(name, fn)`` pair so that subclasses can
    insert, remove, or replace steps by name rather than relying on the
    implicit ordering produced by cooperative ``super()`` / MRO calls.

    Iteration yields only the callables, keeping the existing ``__getitem__``
    loops in :class:`~nlp_xrs.data.dataset.BaseDataset` fully compatible::

        for fn in self.data_transform_fns:   # still works unchanged
            fn(data)

    Examples::

        p = Pipeline()
        p.append("get_basic", build_fn)
        p.append("merge_hist", merge_fn)
        p.insert_before("merge_hist", "my_prep", prep_fn)
        print(p)  # Pipeline[get_basic → my_prep → merge_hist]
    """

    def __init__(self):
        self._steps: List[Tuple[str, Any]] = []

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def append(self, name: str, fn: Any) -> None:
        """Append *fn* under *name* at the end of the pipeline."""
        self._steps.append((name, fn))

    def prepend(self, name: str, fn: Any) -> None:
        """Insert *fn* under *name* at the beginning of the pipeline."""
        self._steps.insert(0, (name, fn))

    def insert_before(self, target: str, name: str, fn: Any) -> None:
        """Insert *fn* immediately before the step named *target*.

        Raises :class:`KeyError` if *target* is not found.
        """
        idx = self._index_of(target)
        self._steps.insert(idx, (name, fn))

    def insert_after(self, target: str, name: str, fn: Any) -> None:
        """Insert *fn* immediately after the step named *target*.

        Raises :class:`KeyError` if *target* is not found.
        """
        idx = self._index_of(target)
        self._steps.insert(idx + 1, (name, fn))

    def remove(self, name: str) -> None:
        """Remove the step named *name*.

        Raises :class:`KeyError` if *name* is not found.
        """
        self._index_of(name)  # raises if absent
        self._steps = [(n, f) for n, f in self._steps if n != name]

    def replace(self, name: str, fn: Any) -> None:
        """Replace the callable of the step named *name* with *fn*.

        Raises :class:`KeyError` if *name* is not found.
        """
        idx = self._index_of(name)
        self._steps[idx] = (name, fn)

    def clear(self) -> None:
        """Remove all steps."""
        self._steps = []

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def names(self) -> List[str]:
        """Return the ordered list of step names."""
        return [n for n, _ in self._steps]

    def __contains__(self, name: str) -> bool:
        return any(n == name for n, _ in self._steps)

    def __len__(self) -> int:
        return len(self._steps)

    def __iter__(self):
        """Yield callables only, so existing ``for fn in pipeline`` loops work."""
        return (fn for _, fn in self._steps)

    def __repr__(self) -> str:
        chain = " → ".join(self.names()) if self._steps else "<empty>"
        return f"Pipeline[{chain}]"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _index_of(self, name: str) -> int:
        for i, (n, _) in enumerate(self._steps):
            if n == name:
                return i
        raise KeyError(f"Pipeline step {name!r} not found. Available steps: {self.names()}")


class Mapper:
    def __init__(self, idx2item: pd.Series):
        idx2item.sort_index(inplace=True)
        # Assert the index is monotonically sorted with +1 increase each time
        assert idx2item.index.tolist() == list(range(len(idx2item)))
        self.item2idx: pd.Series = pd.Series(idx2item.index.values, index=idx2item.values)
        self.idx2item: pd.DataFrame = idx2item.to_frame("raw")

    def add_items(self, items):
        items = list(set(items).difference(self.idx2item["raw"]))
        offset = len(self.item2idx)
        new_item2idx = {item: i + offset for i, item in enumerate(items)}
        self.item2idx = pd.to_numeric(pd.concat([self.item2idx, pd.Series(new_item2idx)]), downcast="integer")
        self.idx2item = pd.concat(
            (
                self.idx2item,
                pd.Series(list(new_item2idx.keys()), index=list(new_item2idx.values())).to_frame(name="raw"),
            )
        )

    def replace_items(self, item2new: Dict[Any, Any]):
        self.item2idx.rename(index=item2new, inplace=True)
        self.idx2item.replace(item2new, inplace=True)

    def get_vocab(self) -> List:
        return self.item2idx.index.tolist()

    def get_item(self, idx: int) -> Union[int, str]:
        return self.idx2item["raw"].loc[idx]

    def get_items(self, idxs: List[int]) -> List[Union[int, str]]:
        return self.idx2item["raw"].loc[idxs].tolist()

    def get_idx(self, item: Union[int, str]) -> int:
        return self.item2idx.loc[item]

    def get_idxs(self, items: List[Union[int, str]]) -> List[int]:
        return self.item2idx.loc[items].tolist()

    def get_enc(self, idxs: List[int]) -> List[List[int]]:
        return list(self.idx2item["enc"].loc[idxs])

    def get_emb(self, idxs: List[int]) -> List[List[float]]:
        return self.embs[list(idxs)].tolist()

    def del_embs(self):
        del self.embs
        gc.collect()

    def enc_len(self) -> int:
        if "enc" in self.idx2item.columns:
            return self.idx2item["enc"].str.len().max()
        return 1

    def encode(self, tok: PreTrainedTokenizerFast, **kwargs):
        self.idx2item["enc"] = tok(self.idx2item["raw"].tolist(), **kwargs)["input_ids"]

    def embed(self, sent_emb: SentenceTransformer, **kwargs):
        batch_size = kwargs.get("batch_size", 64)
        chunk_size = kwargs.get("chunk_size", 10000)
        if torch.cuda.device_count() > 1 or not torch.cuda.is_available():
            self.embs = self.multigpu_embed(sent_emb, batch_size, chunk_size)
        else:
            self.embs = self.chunk_embed(sent_emb, batch_size, chunk_size)

    def multigpu_embed(self, embedder: SentenceTransformer, batch_size: int, chunk_size: int) -> np.ndarray:
        pool = embedder.start_multi_process_pool()
        embs = embedder.encode_multi_process(self.idx2item["raw"], pool, batch_size=batch_size, chunk_size=chunk_size)
        embedder.stop_multi_process_pool(pool)
        return embs

    def chunk_embed(self, embedder: SentenceTransformer, batch_size: int, chunk_size: int) -> np.ndarray:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        texts = self.idx2item["raw"].tolist()
        n_chunks = int(np.ceil(len(texts) / chunk_size))
        embs = []
        for i in range(n_chunks):
            embs.append(
                embedder.encode(
                    texts[i * chunk_size : (i + 1) * chunk_size],
                    batch_size=batch_size,
                    device=device,
                    show_progress_bar=False,
                )
            )
        return np.concatenate(embs, axis=0)

    def save_embed(self, path_to_file: str):
        np.save(path_to_file, self.embs)

    def load_embed(self, path_to_file: str):
        self.embs = np.load(path_to_file)

    def drop_embed(self):
        if hasattr(self, "embs"):
            del self.embs

    def __len__(self) -> int:
        return len(self.item2idx)


class EntityTokenizer:
    def __init__(self, special_tokens=None):
        self.idx2entity = []
        if special_tokens is not None:
            self.idx2entity += list(special_tokens.values())
        self.entity2idx = pd.Series({e: i for i, e in enumerate(self.idx2entity)}, dtype=int)

    def add_entity(self, e):
        if e not in self.entity2idx:
            self.entity2idx[e] = len(self.idx2entity)
            self.idx2entity.append(e)

    def add_entities(self, es: list):
        es = list(set(es).difference(self.idx2entity))
        offset = len(self.entity2idx)
        self.entity2idx = pd.concat([self.entity2idx, pd.Series({es: i + offset for i, es in enumerate(es)})])
        self.idx2entity.extend(es)

    def encode(self, entity):
        return self.entity2idx[entity]

    def batch_encode(self, entities):
        return self.entity2idx[entities].values

    def batch_decode(self, idxs):
        return np.array(self.idx2entity)[idxs]

    def __len__(self):
        return len(self.idx2entity)


class FieldManager:
    """Manages dataset column configuration and metadata."""

    def __init__(self):
        self.orig: List[str] = [Key.USER, Key.ITEM]  # torch.LongTensor() - no encoding needed
        self.float: List[str] = []  # torch.FloatTensor()
        self.enc: Dict[str, str] = {}  # {field: mapper} - requires encoding
        self.emb: Dict[str, str] = {}  # {field: mapper} - requires embedding
        self.processed: List[str] = []  # Already processed as torch.tensor()
        self.pad_mask: List[str] = []  # Columns requiring padding mask
        self.keep_cols: List[str] = [Key.USER, Key.ITEM, Key.EXP, Key.FEAT, Key.RATING, Key.TIMESTEP]

    def add_original(self, col: Union[str, List[str]]):
        """Add a column that does not require encoding."""
        if isinstance(col, str):
            col = [col]
        for c in col:
            if c not in self.orig:
                self.orig.append(c)

    def add_float(self, col: Union[str, List[str]]):
        """Add a column that does not require encoding."""
        if isinstance(col, str):
            col = [col]
        for c in col:
            if c not in self.float:
                self.float.append(c)

    def add_encoding(self, col: str, mapper_col: str):
        """Add a column that needs encoding."""
        self.enc[col] = mapper_col

    def add_embedding(self, col: str, mapper_col: str):
        """Add a column that needs embedding."""
        self.emb[col] = mapper_col

    def add_processed(self, col: Union[str, List[str]]):
        """Add any column that has been processed."""
        if isinstance(col, str):
            col = [col]
        for c in col:
            if c not in self.processed:
                self.processed.append(c)

    def add_pad_mask(self, col: Union[str, List[str]]):
        """Add a column that needs padding mask."""
        if isinstance(col, str):
            col = [col]
        for c in col:
            if c not in self.pad_mask:
                self.pad_mask.append(c)

    def get_all_columns(self):
        """Get all columns that should be kept in the dataset."""
        all_cols = set()
        all_cols.update(self.orig)
        all_cols.update(self.float)
        all_cols.update(self.processed)
        all_cols.update(self.enc.keys())
        all_cols.update(self.emb.keys())
        all_cols.update(self.keep_cols)
        return all_cols

    def to_dict(self):
        """Convert to dictionary format for backward compatibility."""
        return {
            "orig": self.orig,
            "float": self.float,
            "enc": self.enc,
            "emb": self.emb,
            "processed": self.processed,
            "pad_mask": self.pad_mask,
        }


def get_word_tokenizer(
    path,
    data,
    tokenizer_name="default",
    special_tokens=None,
    vocab_size=VOCAB_SIZE,
    retrain=False,
    padding_side="right",
    truncation_side="right",
    test=False,
    use_auth_token=False,
):
    if special_tokens is None:
        special_tokens = dict()
    if test:
        tok_path = os.path.join(path, f"{tokenizer_name}{Suffix.TEST}.json")
    else:
        tok_path = os.path.join(path, f"{tokenizer_name}.json")
    special_tok_vals = list(special_tokens.values())

    if os.path.exists(tok_path) and not retrain:
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tok_path, padding_side=padding_side, truncation_side=truncation_side
        )
        special_tokens = {
            k: v if getattr(tokenizer, k) is None else getattr(tokenizer, k) for k, v in special_tokens.items()
        }
        tokenizer.add_special_tokens(special_tokens)
        if test:
            os.remove(tok_path)
    elif tokenizer_name == "bpe":
        tokenizer = Tokenizer(BPE(unk_token=UNK_TOK))
        # tokenizer.pre_tokenizer = WhitespaceSplit()
        tokenizer.pre_tokenizer = ByteLevel()
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = TemplateProcessing(
            single=f"{BOS_TOK} $0 {EOS_TOK}",
            special_tokens=[(BOS_TOK, special_tok_vals.index(BOS_TOK)), (EOS_TOK, special_tok_vals.index(EOS_TOK))],
        )
        trainer = trainers.BpeTrainer(
            special_tokens=special_tok_vals, show_progress=True, vocab_size=vocab_size + len(special_tokens)
        )
        tokenizer.train_from_iterator(iter(data), trainer=trainer, length=len(data))
        tokenizer.save(tok_path)
        tokenizer = get_word_tokenizer(
            path, data, tokenizer_name, special_tokens, padding_side="right", truncation_side="right", retrain=False
        )
    elif tokenizer_name == "default":
        tokenizer = Tokenizer(WordLevel(unk_token=UNK_TOK))
        # tokenizer.normalizer = Lowercase()
        tokenizer.pre_tokenizer = WhitespaceSplit()
        tokenizer.post_processor = TemplateProcessing(
            single=f"{BOS_TOK} $0 {EOS_TOK}",
            special_tokens=[(BOS_TOK, special_tok_vals.index(BOS_TOK)), (EOS_TOK, special_tok_vals.index(EOS_TOK))],
        )
        trainer = trainers.WordLevelTrainer(
            special_tokens=special_tok_vals, show_progress=True, vocab_size=vocab_size + len(special_tokens)
        )
        tokenizer.train_from_iterator(iter(data), trainer=trainer, length=len(data))
        tokenizer.save(tok_path)
        tokenizer = get_word_tokenizer(
            path, data, tokenizer_name, special_tokens, padding_side="right", truncation_side="right", retrain=False
        )
    else:
        # if tokenizer_name in ['gpt2', 'meta-llama/Llama-2-7b-hf']:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                use_fast=True,  # **special_tokens,
                use_auth_token=use_auth_token,
            )
            special_tokens = {k: v for k, v in special_tokens.items() if k not in tokenizer.special_tokens_map}
            tokenizer.add_special_tokens(special_tokens)
        except Exception:
            raise ValueError(f"Tokenizer not implemented: {tokenizer_name}")

    return tokenizer


def compute_concept_targets(
    train_data: pd.DataFrame,
    key: str,
    acts_mapper: pd.DataFrame,
    num_rows: int,
    sae_codebook_size: int,
) -> torch.Tensor:
    """Per-entity SAE concept distribution, indexed by encoded entity id.

    Row ``i`` is the L1-normalised mean SAE activation over the reviews of entity ``i`` in
    ``train_data``. Entities without reviews in ``train_data`` (e.g. cold-start items) keep an
    all-zero row; the tokenizer only reads the rows of entities present in its training data.
    """
    targets = torch.zeros(num_rows, sae_codebook_size)
    reviews = train_data[[key, Key.REV]].copy()
    if isinstance(reviews[key].iloc[0], list):
        reviews[key] = reviews[key].str[-1]
    dists = reviews.groupby(key)[Key.REV].apply(get_sae_dists, acts_mapper=acts_mapper, sae_codebook_size=sae_codebook_size)
    targets[dists.index.values] = torch.vstack(dists.values.tolist())
    return targets / targets.norm(p=1, dim=1, keepdim=True).clamp(min=1e-10)


def get_sid_tokenizer(
    data: pd.DataFrame,
    train_ixs: List[int],
    metadata_df: Optional[pd.DataFrame],
    mappers: Dict[str, Mapper],
    sid_cfg: SimpleNamespace,
    use_train_only: bool = True,
    dedup_items_on_load: bool = True,
    entity: str = Key.ITEM,
    **kwargs,
) -> SemanticIDTokenizer:
    """Load (or train) a SemanticIDTokenizer for *entity* (item or user).

    For the pre-trained path ``tokenizer.load()`` finds ``is_trained=True``
    and returns immediately, so ``entity`` has no effect in the common case.

    The fallback training path mirrors ``run_training`` in
    ``train_tokenizer.py``:

    * ``entity == Key.ITEM``: extract item embeddings and train directly on
      ``data`` (existing behaviour, unchanged).
    * ``entity == Key.USER``: extract user embeddings via
      :func:`~nlp_xrs.semantic_ids.utils.extract_user_embeddings` and remap
      user IDs into the ``Key.ITEM`` slot of a proxy DataFrame before calling
      ``tokenizer.train``, because the tokenizer is entity-agnostic.

    Args:
        data:              Full interaction DataFrame.
        train_ixs:         Row indices for the train (+ optionally valid) split.
        metadata_df:       Item metadata DataFrame (required for item text /
                           user taste embeddings).
        mappers:           Mapper dict (keyed by ``Key.*`` constants).
        sid_cfg:           Resolved tokenizer config with ``tokenizer_path``
                           and ``embedding_path`` already set.
        use_train_only:    Whether to restrict training to ``train_ixs``.
        dedup_items_on_load: Deduplicate SID codes at load time.
        entity:            ``Key.ITEM`` (default) or ``Key.USER``.
        **kwargs:          Extra kwargs forwarded to embedding extraction
                           (``dataset_name``, ``metadata_path``, …).
    """
    from pathlib import Path

    from nlp_xrs.semantic_ids.semantic_ids import SemanticIDTokenizer
    from nlp_xrs.semantic_ids.utils import extract_item_embeddings, extract_user_embeddings

    tok_path = Path(sid_cfg.tokenizer_path)
    logging.info(
        "Initializing %s semantic ID tokenizer from: %s  (exists=%s)",
        entity,
        tok_path,
        tok_path.exists(),
    )
    if not tok_path.exists():
        logging.info("Tokenizer checkpoint not found – will train and save to: %s", tok_path)

    tokenizer = SemanticIDTokenizer(sid_cfg, dedup_items_on_load=dedup_items_on_load)
    tokenizer.load(sid_cfg.tokenizer_path)

    if not tokenizer.is_trained:
        if entity == Key.USER:
            tgt_key = Key.USER
            all_embeddings = extract_user_embeddings(
                inter_df=data,
                metadata_df=metadata_df,
                mappers=mappers,
                train_ixs=train_ixs,
                save_dir=sid_cfg.embedding_path,
                emb_config=sid_cfg.emb_config,
                **{k: v for k, v in kwargs.items() if k != "metadata_path"},
            )
        else:
            tgt_key = Key.ITEM
            if metadata_df is None:
                metadata_df = load_safe_parquet(kwargs.get("metadata_path"))

            all_embeddings = extract_item_embeddings(
                inter_df=data,
                metadata_df=metadata_df,
                train_ixs=train_ixs,
                mappers=mappers,
                save_dir=sid_cfg.embedding_path,
                emb_config=sid_cfg.emb_config,
                **kwargs,
            )

        item_features = {"embeddings": all_embeddings}
        train_data = data.loc[train_ixs] if use_train_only else data
        if hasattr(sid_cfg, "sae_preset"):
            emb_model_alias = sid_cfg.emb_config.embedding_model.replace("/", "--")
            acts_path = os.path.join(
                Path(sid_cfg.embedding_path).parent.parent.parent.parent,
                "SAE",
                f"acts_{emb_model_alias}_{sid_cfg.sae_preset}_{sid_cfg.seed}.pt",
            )
            item_features["concept_targets"] = compute_concept_targets(
                train_data,
                tgt_key,
                get_acts_mapper(acts_path),
                num_rows=all_embeddings.shape[0],
                sae_codebook_size=tokenizer.model.sae_codebook_size,
            )

        if isinstance(train_data[tgt_key].iloc[0], (list, tuple)):
            train_data[tgt_key] = train_data[tgt_key].str[0]
        tokenizer.train(data=train_data, item_features=item_features, key=tgt_key)

        tokenizer.save(sid_cfg.tokenizer_path)
        logging.info("Saved %s tokenizer to %s", entity, sid_cfg.tokenizer_path)
        if getattr(sid_cfg, "discard_embedding_cache", False):
            cache = Path(sid_cfg.embedding_path)
            shutil.rmtree(cache, ignore_errors=True)
            if cache.parent.name == "_emb_cache" and cache.parent.is_dir() and not any(cache.parent.iterdir()):
                cache.parent.rmdir()

        tokenizer.load(sid_cfg.tokenizer_path)  # Needed for deduplication codebook

    return tokenizer


def get_semantic_ids(self, sample: Dict[str, Any], data: Dict[str, Any]):
    """
    Getter function to add semantic IDs to samples.

    Args:
        sample: Sample dictionary
        data: Data dictionary to populate
    """
    if not hasattr(self, "tokenizer"):
        return

    # Get item IDs
    item_ids = sample[Key.ITEM]
    if not isinstance(item_ids, list):
        item_ids = [item_ids]

    # Tokenize items
    semantic_ids = self.tokenizer.tokenize(item_ids)

    # Add to data
    key = Key.ITEM if self.cfg.semantic_id_mode == "replace" else Key.ITEM + "_semid"
    data[key] = torch.LongTensor(semantic_ids)

    return sample, data


def postprocessing(string):
    """
    adopted from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub("'s", " 's", string)
    string = re.sub("'m", " 'm", string)
    string = re.sub("'ve", " 've", string)
    string = re.sub("n't", " n't", string)
    string = re.sub("'re", " 're", string)
    string = re.sub("'d", " 'd", string)
    string = re.sub("'ll", " 'll", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(",+", " , ", string)
    string = re.sub(":+", " , ", string)
    string = re.sub(";+", " . ", string)
    string = re.sub(r"\.+", " . ", string)
    string = re.sub("!+", " ! ", string)
    string = re.sub(r"\?+", " ? ", string)
    string = re.sub(" +", " ", string).strip()
    return string


# class GloVe:
#     def __init__(self, f_name):
#         if not os.path.isfile(os.path.join(WE_PATH, f'{f_name}.txt')):
#             self._download_we(f_name)
#
#         w2ix = {}
#         vectors = []
#
#         # Read in the data.
#         with open(os.path.join(WE_PATH, f'{f_name}.txt'), 'r', encoding='utf-8') as savefile:
#             for i, line in enumerate(savefile):
#                 tokens = line.split(' ')
#
#                 word = tokens[0]
#                 entries = tokens[1:]
#
#                 w2ix[word] = i
#                 vectors.append(np.array(float(x) for x in entries))
#
#         self.w2ix = pd.Series(w2ix)
#         self.vectors = np.concatenate(vectors, axis=0)
#
#     def
#
#     @staticmethod
#     def _download_we(f_name):
#         with urllib.request.urlopen(f'http://nlp.stanford.edu/data/wordvecs/{f_name}.zip') \
#                 as response, open(os.path.join(WE_PATH, f'{f_name}.zip'), 'wb') as out_file:
#             shutil.copyfileobj(response, out_file)
#
#         with zipfile.ZipFile(os.path.join(WE_PATH, f'{f_name}.zip'), 'r') as zip_ref:
#             zip_ref.extractall(WE_PATH)
#
#         os.remove(os.path.join(WE_PATH, f'{f_name}.zip'))


def pad2longest(tensors, dim=0, side="left", value=0):
    """
    Adds padding to the LEFT of each tensor up to the longest length in the batch
    """
    last_dim = len(tensors[0].shape) - 1
    if dim not in [-1, last_dim]:
        tensors = [tensor.transpose(dim, last_dim) for tensor in tensors]
        return pad2longest(tensors, dim=-1, side=side, value=value).transpose(dim + 1, last_dim + 1)

    max_l = max([tensor.shape[-1] for tensor in tensors])
    if side == "left":
        return torch.stack([F.pad(tensor, pad=(max_l - tensor.shape[-1], 0), value=value) for tensor in tensors])
    else:
        return torch.stack([F.pad(tensor, pad=(0, max_l - tensor.shape[-1]), value=value) for tensor in tensors])


def pad2len(tensors, length, dim=0, side="left", value=0):
    """
    Adds padding to the LEFT of each tensor up to the given length
    """
    last_dim = len(tensors[0].shape) - 1
    if dim not in [-1, last_dim]:
        tensors = [tensor.transpose(dim, last_dim) for tensor in tensors]
        return pad2len(tensors, length, dim=-1, side=side, value=value).transpose(dim + 1, last_dim + 1)

    if side == "left":
        return torch.stack([F.pad(tensor, pad=(length - tensor.shape[-1], 0), value=value) for tensor in tensors])
    else:
        return torch.stack([F.pad(tensor, pad=(0, length - tensor.shape[-1]), value=value) for tensor in tensors])
    # return torch.stack([F.pad(tensor, pad=(length - tensor.shape[-1], 0), value=0) for tensor in tensors])


def sliding_w_hist(g: pd.DataFrame, max_l: int = MAX_HIST_LEN):
    return [g.index.values[max(i - max_l, 0) : i].tolist() for i in range(g.shape[0])]


def get_transform_mapping(col_data: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Returns the encoding mapper and the transformed data
    """

    def replace(r):
        if isinstance(r[0], list):
            return [replace(ri) for ri in r]
        else:
            return mapper.loc[r].tolist()

    unique_vals = col_data.copy()
    is_nested = False
    while isinstance(unique_vals.iloc[0], list):
        is_nested = True
        unique_vals = unique_vals.explode()
    unique_vals = sorted(unique_vals.unique().tolist())
    # unique_vals = col_data.unique().tolist()
    mapper = pd.to_numeric(pd.Series(range(len(unique_vals)), index=unique_vals), downcast="integer")

    col_data = col_data.apply(replace) if is_nested else col_data.map(mapper).astype(mapper.dtype)

    return col_data, pd.Series(mapper.index.values, index=mapper.values)
