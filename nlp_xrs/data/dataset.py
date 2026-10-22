"""
nlp_xrs.data.dataset
##################################

Module description:

This module contains the Base dataset classes and several custom datasets.

Dataset classes for the generation-based backbones used in the paper: NRT and PETER
(GenDataset), SEQUER (SEQUERDataset), and TF-SEQUER (SEQUERTextFusionDataset).

"""

import copy
import logging
import numbers
import os
import random
import re
import time
from collections import defaultdict
from collections.abc import Iterable
from functools import partial
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader, Dataset, Subset

from nlp_xrs.data.collators import (
    BaseCollator,
    SEQUERCollator,
    SEQUERTextFusionCollator,
)
from nlp_xrs.data.utils import (
    FieldManager,
    Mapper,
    get_context,
    get_sid_tokenizer,
    get_transform_mapping,
    get_word_tokenizer,
    sliding_w_hist,
)
from nlp_xrs.semantic_ids.utils import get_tok_emb_paths, set_nested_attr
from nlp_xrs.utils.constants import (
    BASE_COLS,
    DEFAULT_HIST_LEN,
    MAX_HIST_LEN,
    PACKAGE_NAME,
    PAD_TOK,
    SPECIAL_TOKENS,
    UNK_TOK,
    HistKey,
    Key,
    Prefix,
    SeqMode,
    Suffix,
    Task,
)
from nlp_xrs.utils.in_out import (
    get_acts_mapper,
    load_auxiliary_config,
    load_mapper_parquet,
    load_partition,
    load_safe_parquet,
)


def _apply_mini_tokenizer_budget(tokenizer_config) -> None:
    """Keep a ``_mini`` SID fit short enough for an end-to-end test."""
    tokenizer_config.num_epochs = 1
    tokenizer_config.num_steps = 2
    if hasattr(tokenizer_config, "opq_niter"):
        tokenizer_config.opq_niter = 1
    if hasattr(tokenizer_config, "faiss_omp_num_threads"):
        tokenizer_config.faiss_omp_num_threads = min(tokenizer_config.faiss_omp_num_threads, 4)
    if hasattr(tokenizer_config, "init_kwargs"):
        tokenizer_config.init_kwargs.max_iter = 2
    emb_config = tokenizer_config.emb_config
    current_length = getattr(emb_config, "max_seq_length", None)
    if current_length is None or current_length > 64:
        emb_config.max_seq_length = 64
    if getattr(emb_config, "batch_size", 8) > 8:
        emb_config.batch_size = 8


def _fn_name(fn) -> str:
    """Return a human-readable name for a callable (handles partials and bound methods)."""
    if hasattr(fn, "__name__"):
        return fn.__name__
    inner = getattr(fn, "func", fn)
    return getattr(inner, "__name__", repr(fn))


def get_default_args(config_name, dataset="yelp", test=True, base_path=None):
    if base_path is None:
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return SimpleNamespace(
        config_id=config_name,
        dataset=dataset,
        fold="0",
        base_path=base_path,
        cuda=False,
        test=test,
        save_results=False,
        no_generate=True,
        resume_training=False,
        train_only=False,
        eval_only=False,
        log_to_file=False,
        seed=1111,
    )


class BaseDataset(Dataset):
    """:class:`BaseDataset` stores the original dataset in memory.
    The dataset init logic follows the next steps:
    1) Variable setting/initialization (`self._reset_stage`).
    2) Data/Split/Mappers reading (`self.read_data`).
    3) Handle Auxiliary columns i.e. modify existing column data or adding new columns (`self.handle_aux_cols`).
    4) Assert Test items appear in Train (`self.check_test_in_train` -- Optional by `cfg.assert_test_in_train`).
    5) Get/Create Tokenizer (`~nlp_xrs.data.utils.get_tokenizers`).
    6) SeqDataset extension: Handle Past Information (`self.handle_past_info`).
    7) Finish initialization (`self.finish_init`):
        a) Obtain encodings/embeddings (`self.handle_encodings`).
        b) Get field lengths (`self.get_required_lengths`).
        c) Remove unnecessary columns (according to `self.field_handler`).

    Additional Methods:
        `get_basic`: it follows rules based on `self.field_handler` to get necessary information from a sample of the dataset
            in order to create a batch.
        `get_data_info`: it stores certain data statistics/features useful for the model creation.
        `get_dataloader`: it returns a DataLoader for the specified data subset (train, valid, test).
        `get_emb_model`: it return a text embedding model (as of now, it only creates SentenceTransformers).
        `get_gen_labels`: it gets the labels (explanation, features, etc) for the specified data subset.
        `set_stage`: it sets the stage for Multi-Stage Training

    Args:
        data_cfg: Data Configuration (SimpleNamespace).
        model_cfg: Model Configuration (SimpleNamespace).
        device: Device to run experiments (torch.device("cuda:0")).
        data_file: Data that contains the review user-item interactions. Defaults to "reviews.pkl".
        **kwargs:
            logger: Defaults to logging.getLogger(PACKAGE_NAME).
            seed: Random seed for reproducible experiments.
            paths: SimpleNamespace containing the paths to the data, auxiliary data and tokenizers.

    Custom Attributes:
        getter_fns: list of callables to build the data instance from the index dict sample at __getitem__
            (e.g. `self.get_basic`)
        mod_fns: list of callables to modify the data instance once assembled
        cols: dict necessary for `self.get_basic` and `self.collate`. Keys description:
            "orig": list of long-type fields that do not require encoding -- `torch.LongTensor()`
            "float": list of float-type fields -- `torch.FloatTensor()`
            "enc": dict of {field: mapper} for encodings -- `torch.LongTensor()`
            "emb": dict of {field: mapper} for embeddings -- `torch.FloatTensor()`
            "processed": list of already processed field -- `torch.tensor()`
            "pad_mask": list of field that require padding mask at `self.collate`
    """

    def __init__(self, data_cfg, model_cfg, device, data_file="reviews.parquet", **kwargs):
        # Step 1: Initialize configuration and basic attributes
        self._initialize_config(data_cfg, model_cfg, device, kwargs)
        self.paths = kwargs["paths"]

        # Step 2: Set up paths and file handling
        data_file = self._initialize_paths(data_file, kwargs)
        self._validate_config()

        # Step 3: Initialize stage tracking
        self._reset_stage()

        # Step 4: Load data, mappers, and splits
        load_cols = BASE_COLS + getattr(self.cfg, "load_cols", [])
        load_metadata = getattr(self.cfg, "load_metadata", False)
        self._load_data(data_file, load_cols, load_metadata)

        # Step 5: Validate and prepare data
        self._validate_and_prepare_data()

        # Step 6: Initialize tokenizer
        self._initialize_tokenizers()

        # Step 7: Process past information (for sequential datasets)
        self._process_past_info()

        # Step 8: Finalize initialization
        self._finish_init()
        self.collator = self._create_collator()
        self.train()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if self._bench_enabled:
            return self._getitem_timed(idx)

        sample = self.data.loc[idx].to_dict()
        sample["index"] = idx
        data = {}

        # Stage 1: Sample preparation (modify sample only)
        for prep_fn in self.sample_prep_fns:
            prep_fn(sample)

        # Stage 2: Data building (read sample, write data)
        for build_fn in self.data_builder_fns:
            build_fn(sample, data)

        # Stage 3: Data transformation (read/modify data only)
        for transform_fn in self.data_transform_fns:
            transform_fn(data)

        return data

    def _getitem_timed(self, idx):
        """Benchmarking variant of __getitem__ that accumulates per-fn timings into self._bench."""
        t = time.perf_counter()
        sample = self.data.loc[idx].to_dict()
        self._bench["pandas_lookup"] += time.perf_counter() - t
        sample["index"] = idx
        data = {}

        for prep_fn in self.sample_prep_fns:
            name = _fn_name(prep_fn)
            t = time.perf_counter()
            prep_fn(sample)
            self._bench[name] += time.perf_counter() - t

        for build_fn in self.data_builder_fns:
            name = _fn_name(build_fn)
            t = time.perf_counter()
            build_fn(sample, data)
            self._bench[name] += time.perf_counter() - t

        for transform_fn in self.data_transform_fns:
            name = _fn_name(transform_fn)
            t = time.perf_counter()
            transform_fn(data)
            self._bench[name] += time.perf_counter() - t

        return data

    def _create_collator(self) -> BaseCollator:
        """Create the default collator."""
        return BaseCollator(self.tok, self.field_handler, self.mappers, bench=self._bench)

    def _initialize_config(self, data_cfg, model_cfg, device, kwargs):
        """Initialize configuration and basic attributes."""
        self.cfg = copy.deepcopy(data_cfg)
        self.model_cfg = model_cfg
        self.logger = kwargs.get("logger", logging.getLogger(PACKAGE_NAME))
        self.cfg.seed = kwargs.get("seed", None)
        self.cfg.hist_len = getattr(self.cfg, "hist_len", 0)
        self.do_eval = getattr(self.cfg, "do_eval", True)
        self.add_special_toks = False
        self.device = device
        self.phase = "train"

        # Initialize workers and batch sizes
        self.workers = {
            "train": getattr(data_cfg, "train_workers", 0),
            "valid": getattr(data_cfg, "eval_workers", 5),
            "test": getattr(data_cfg, "eval_workers", 5),
        }
        self.batch_size = {
            "train": getattr(data_cfg, "train_batch_size", data_cfg.batch_size),
            "valid": getattr(data_cfg, "eval_batch_size", data_cfg.batch_size),
            "test": getattr(data_cfg, "eval_batch_size", data_cfg.batch_size),
        }

        # Fine-grained data load benchmarking (enabled by trainer when benchmark_steps is set)
        self._bench_enabled: bool = False
        self._bench: Dict[str, float] = defaultdict(float)

    def _initialize_paths(self, data_file, kwargs):
        """Set up file paths and handle test suffix."""
        suffix = ""
        if self.cfg.test_flag:
            suffix = Suffix.TEST
            data_file = f"{suffix}.".join(data_file.split("."))
            self.cfg.fold = self.cfg.fold + suffix

        self.suffix = suffix
        self.aux_path = kwargs["paths"].aux_data
        self.data_path = kwargs["paths"].data
        self.tokenizers_path = kwargs["paths"].tokenizers

        return data_file

    def _load_data(self, data_file, load_cols, load_metadata=False):
        """Load data, mappers, and split indexes."""
        self.read_data(data_file, self.suffix, load_cols=load_cols, load_metadata=load_metadata)

    def _validate_and_prepare_data(self):
        """Validate loaded data and prepare feature mappings."""
        # Validate data integrity
        assert not (self.data[Key.EXP].apply(len) == 0).any(), "Found empty explanations in dataset"
        assert not (self.data[Key.FEAT].apply(len) == 0).any(), "Found empty features in dataset"

        # Prepare feature set
        self.feature_set = self.mappers[Key.FEAT].get_vocab()
        self.mappers[Key.FEAT].replace_items({"": UNK_TOK})
        self.mappers[Key.FEAT].add_items([UNK_TOK, PAD_TOK])

        # Get dataset information
        self.data_info = self.get_data_info()
        self.item_set = set(self.mappers[Key.ITEM].item2idx.tolist())

        # Handle auxiliary columns
        self.handle_aux_cols()

        # Check test items in train (if required)
        if self.cfg.assert_test_in_train:
            train_ixs = self.split_ixs["train"]
            self.check_test_in_train(train_ixs)

    def _validate_config(self):
        """Validate configuration before initialization."""
        errors = []

        # Check required attributes
        required_attrs = ["dataset", "fold", "word_tok"]
        for attr in required_attrs:
            if not hasattr(self.cfg, attr):
                errors.append(f"Missing required config attribute: {attr}")

        # Validate embedding models configuration
        if hasattr(self.cfg, "emb_models") and self.cfg.emb_models:
            for i, emb_model in enumerate(self.cfg.emb_models):
                if not hasattr(emb_model, "name"):
                    errors.append(f"emb_models[{i}] missing 'name' attribute")
                if not hasattr(emb_model, "type"):
                    errors.append(f"emb_models[{i}] missing 'type' attribute")
                if not hasattr(emb_model, "apply_to"):
                    errors.append(f"emb_models[{i}] missing 'apply_to' attribute")

        # Validate paths
        if not hasattr(self, "data_path") or not os.path.exists(self.data_path):
            errors.append(f"Data path does not exist: {getattr(self, 'data_path', 'not set')}")

        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            raise ValueError(error_msg)

    def _initialize_tokenizers(self):
        """Initialize word tokenizer."""
        self.logger.info(f"Getting {self.cfg.word_tok} tokenizer...")
        train_ixs = self.split_ixs["train"]
        train_exps = self.mappers[Key.EXP].get_items(self.data.loc[train_ixs, Key.EXP].explode().unique())

        self.tok = get_word_tokenizer(
            self.paths.tokenizers,
            train_exps,
            self.cfg.word_tok,
            SPECIAL_TOKENS,
            self.cfg.vocab_sz,
            self.cfg.retrain_tok,
            test=self.cfg.test_flag,
            use_auth_token=self.cfg.hf_auth_token,
        )

        # Per-entity SID state.  Keys are Key.ITEM or Key.USER.
        self.sid_tokenizers: Dict[str, Any] = {}
        self._sid_caches: Dict[str, Any] = {}
        self._sid_entity_cfgs: Dict[str, Any] = {}

        if hasattr(self.cfg, "semantic_id"):
            for entity, sub_cfg in self._iter_sid_entities(self.cfg.semantic_id):
                self._init_sid_entity(entity, sub_cfg)

        self.data_info.n_wtokens = len(self.tok)
        self.data_info.tok = self.tok
        self.data_info.mappers = self.mappers

    def _process_past_info(self):
        """Process past information for sequential datasets."""
        if self.cfg.seq_mode.requires_past_info():
            self.handle_past_info()

    def _reset_stage(self):
        self.field_handler = FieldManager()
        self.sample_prep_fns = []  # Stage 1: Modify sample only
        self.data_builder_fns = [self.get_basic]  # Stage 2: Read sample, write data
        self.data_transform_fns = []  # Stage 3: Read/modify data only

    def eval(self, phase):
        assert phase in ["train", "valid", "test"]
        self.phase = phase.lower()

    def train(self):
        self.phase = "train"

    def get_hist_len(self):
        if not hasattr(self.cfg, "hist_len"):
            self.cfg.hist_len = 0

    def get_hist_data(self, data_file, suffix=""):
        raise NotImplementedError()

    def set_stage(self, stage, model=None):
        # By default, data does not change from one training-eval stage to another
        pass

    def get_data_info(self):
        return SimpleNamespace(
            dataset=f"{self.cfg.dataset}_{self.cfg.fold}",
            n_users=len(self.mappers[Key.USER]),
            n_items=len(self.mappers[Key.ITEM]),
            n_feats=len(self.feature_set),
            n_sents=len(self.mappers[Key.EXP]),
            max_rating=self.data[Key.RATING].max(skipna=True),
            mean_rating=self.data[Key.RATING].mean(skipna=True),
            min_rating=self.data[Key.RATING].min(skipna=True),
            special_items={v: self.mappers[Key.ITEM].get_idx(v) for v in SPECIAL_TOKENS.values()},
        )

    # ------------------------------------------------------------------
    # SID helpers
    # ------------------------------------------------------------------

    def _iter_sid_entities(self, sid_cfg):
        """Yield ``(entity, sub_cfg)`` pairs from ``data.semantic_id``.

        Supports two schemas:

        * **Nested** (new): ``sid_cfg`` has ``item`` and/or ``user`` sub-keys.
          Each sub-key is itself a ``SimpleNamespace`` with ``config_name`` /
          ``config_id``, ``mode``, etc.
        * **Legacy flat** (backward-compatible): ``sid_cfg`` has ``config_name``
          or ``config_id`` at the top level → treated as ``Key.ITEM``-only.
        """
        has_item = hasattr(sid_cfg, "item")
        has_user = hasattr(sid_cfg, "user")
        if has_item or has_user:
            if has_item:
                yield Key.ITEM, sid_cfg.item
            if has_user:
                yield Key.USER, sid_cfg.user
        else:
            # Legacy flat schema – treat as item-only.
            yield Key.ITEM, sid_cfg

    def _init_sid_entity(self, entity: str, sub_cfg) -> None:
        """Load (or train) the SID tokenizer for one entity and update state.

        Args:
            entity:  ``Key.ITEM`` or ``Key.USER``.
            sub_cfg: The entity-specific sub-config ``SimpleNamespace`` with
                     ``config_name`` / ``config_id``, ``mode``,
                     ``dedup_items_on_load``, optional ``overrides``, ``seed``,
                     and ``use_experiment_seed`` fields.
        """
        # Resolve seed priority (same logic as the old flat block):
        #   1. sub_cfg.seed             – explicit tokenizer seed
        #   2. sub_cfg.use_experiment_seed – forward experiment seed
        #   3. None                     – seed-agnostic path
        if hasattr(sub_cfg, "seed"):
            tok_seed = sub_cfg.seed
        elif getattr(sub_cfg, "use_experiment_seed", True):
            tok_seed = self.cfg.seed
        else:
            tok_seed = None

        # Registry stem differs by entity (mirrors train_tokenizer.py line 482).
        registry_stem = "user_registry" if entity == Key.USER else "registry"

        if hasattr(sub_cfg, "config_name"):
            self.logger.info("Getting tokenizer config for %s Semantic IDs: %s...", entity, sub_cfg.config_name)
            tokenizer_config = load_auxiliary_config(
                os.path.join(self.paths.config, "tokenizers", sub_cfg.config_name + ".yaml")
            )
        elif hasattr(sub_cfg, "config_id"):
            self.logger.info("Getting tokenizer config for %s Semantic IDs: %s...", entity, sub_cfg.config_id)
            seed_suffix = f"_seed{tok_seed}" if tok_seed is not None else ""
            tokenizer_config = load_auxiliary_config(
                os.path.join(self.paths.results, "tokenizers", self.cfg.dataset),
                file_prefix=f"{registry_stem}{seed_suffix}",
                file_extension=".json",
                config_hash=sub_cfg.config_id,
                sid_tok_path=os.path.join(self.tokenizers_path, "semantic_ids"),
            )
        else:
            raise ValueError(f"data.semantic_id.{entity} must have either 'config_name' or 'config_id'.")

        # Apply dot-notation overrides declared in the experiment YAML.
        if hasattr(sub_cfg, "overrides"):
            for key, value in vars(sub_cfg.overrides).items():
                self.logger.info("Applying %s tokenizer config override: %s = %s", entity, key, value)
                set_nested_attr(tokenizer_config, key, value)

        if self.cfg.test_flag:
            # One checkpoint per tokenizer config, separate from the full-data tree.
            _apply_mini_tokenizer_budget(tokenizer_config)
            tok_dir = os.path.join(
                self.tokenizers_path, "semantic_ids", f"{sub_cfg.config_name}{Suffix.TEST}.pt"
            )
            emb_dir = os.path.join(self.tokenizers_path, "semantic_ids", "_emb_cache", sub_cfg.config_name)
            tokenizer_config.discard_embedding_cache = True
        else:
            emb_dir, tok_dir, _ = get_tok_emb_paths(self.paths, tokenizer_config, seed=tok_seed)
            tokenizer_config.discard_embedding_cache = False
        tokenizer_config.embedding_path = emb_dir
        tokenizer_config.tokenizer_path = tok_dir
        tokenizer_config.seed = tok_seed

        tokenizer = get_sid_tokenizer(
            data=self.data,
            train_ixs=self.split_ixs["train"] + self.split_ixs["valid"],
            metadata_df=self.metadata,
            mappers=self.mappers,
            sid_cfg=tokenizer_config,
            use_train_only=getattr(self.cfg, "use_train_only", True),
            dedup_items_on_load=getattr(sub_cfg, "dedup_items_on_load", True),
            dataset_name=self.cfg.dataset,
            metadata_path=os.path.join(self.data_path, f"metadata{self.suffix}.parquet"),
            entity=entity,
        )

        self.sid_tokenizers[entity] = tokenizer
        self._sid_entity_cfgs[entity] = sub_cfg

        # Set entity-specific data_info fields and mapper special-token offsets.
        # Item:  n_codebooks, sid_tokenizer, sid_vocab_size  (legacy names kept).
        # User:  user_n_codebooks, user_sid_tokenizer, user_sid_vocab_size.
        sid_raw_vocab = tokenizer.vocab_size  # K*C (e.g. 1024)

        if entity == Key.ITEM:
            sid_vocab_size = sid_raw_vocab + len(SPECIAL_TOKENS)  # adds PAD slot

            self.data_info.n_codebooks = tokenizer.n_codebooks
            self.data_info.sid_tokenizer = tokenizer
            self.data_info.sid_vocab_size = sid_vocab_size
            #  Option 1: Either create a new mapper for the SID tokenizer
            #  Option 2: Offset existing mapper items by the number of SID + special tokens.
            #  Option 3: Add a new column to the data with the SID codes.
            for i, tok in enumerate(SPECIAL_TOKENS.values()):
                self.mappers[entity].item2idx[tok] = sid_raw_vocab + i
        else:
            self.data_info.user_n_codebooks = tokenizer.n_codebooks
            self.data_info.user_sid_tokenizer = tokenizer
            self.data_info.user_sid_vocab_size = sid_raw_vocab

        self._build_sid_cache(entity)

    def _finish_init(self):
        self.handle_encodings()
        self.data_info.lengths = self.get_required_lengths()

        # Register SID transforms for every active entity.  SID transforms run
        # last (Stage 3) so sequence-assembly transforms (e.g. merge_hist) have
        # already concatenated history + target IDs before the SID lookup runs.
        for entity, tokenizer in self.sid_tokenizers.items():
            mode = getattr(self._sid_entity_cfgs[entity], "mode", "replace")
            if mode == "replace":
                self.data_info.lengths[entity] *= tokenizer.n_codebooks
                self.data_transform_fns.append(partial(self.apply_sid_replace, entity=entity))
            elif mode == "augment":
                if entity != Key.ITEM:
                    raise NotImplementedError("Augment mode is only supported for Key.ITEM.")
                # Augment mode: original item-ID column is preserved at its
                # current length; the SID codes live in a *separate*
                # "item_semid" column whose length we register alongside.
                # Model code that wants both must read Key.ITEM (original ids)
                # and "item_semid" (SID codes) independently.
                self.data_info.lengths[f"{Key.ITEM}_semid"] = self.data_info.n_codebooks
                self.data_transform_fns.append(self.apply_sid_augment)
            else:
                raise ValueError(f"Unknown semantic_id_mode: {mode!r}")

        # Remove unnecessary columns
        rm_cols = set(self.data.columns).difference(
            self.field_handler.get_all_columns() | set(getattr(self.cfg, "load_cols", []))
        )
        self.data.drop(list(rm_cols), axis=1, inplace=True)

        # Wire SAE concept activations into the batch when Key.REV is loaded and
        # the model uses context prediction (requires_context).  The acts file must
        # exist at the canonical path; its absence is treated as a silent no-op so
        # that configs with load_cols=[review] still work on datasets without SAE.
        if hasattr(self.cfg, "sae_cfg"):
            emb_model = getattr(self.cfg.sae_cfg, "emb_model", "Qwen/Qwen3-Embedding-0.6B")
            sae_mode = getattr(self.cfg.sae_cfg, "mode", "medium")
            seed = getattr(self.cfg.sae_cfg, "seed", self.cfg.seed)
            assert Key.REV in self.data.columns
            emb_model_alias = emb_model.replace("/", "--")
            acts_path = os.path.join(
                self.data_path,
                self.cfg.fold,
                "SAE",
                f"acts_{emb_model_alias}_{sae_mode}_{seed}.pt",
            )
            if os.path.exists(acts_path):
                self.acts_mapper = get_acts_mapper(acts_path)
                self.data_info.sae_n_concepts = self.acts_mapper[Key.CONCEPT_ID].explode().max() + 1
                self.data_info.sae_topk = len(self.acts_mapper[Key.CONCEPT_ID].iloc[0])
                self.data_builder_fns.append(self.get_acts)
                self.logger.info("SAE activations loaded from %s; get_acts added to data_builder_fns.", acts_path)
            else:
                raise FileNotFoundError(f"SAE activations file not found: {acts_path}")

    def _build_sid_cache(self, entity: str = Key.ITEM) -> None:
        """Pre-compute offset-applied SID code lists keyed by entity ID.

        The cache is stored in ``self._sid_caches[entity]``.

        The user tokenizer's ``item_to_tokens`` is indexed by *user IDs*
        because training maps user IDs into the ``Key.ITEM`` slot of the
        proxy DataFrame (see ``train_tokenizer.py`` lines 262-268).
        """
        tokenizer = self.sid_tokenizers[entity]
        cache = pd.Series(
            tokenizer.tokenize(tokenizer.item_to_tokens.index.tolist(), n_reserved=0),
            index=tokenizer.item_to_tokens.index,
        )
        self._sid_caches[entity] = cache

    def get_basic(self, sample, data):
        for c, v in sample.items():
            if c in self.field_handler.processed:
                data[c] = v
            elif c in self.field_handler.orig:
                data[c] = torch.tensor(v, dtype=torch.long)
            elif c in self.field_handler.enc:
                m = self.field_handler.enc[c]
                if isinstance(v, (list, tuple)):
                    data[c] = [torch.LongTensor(self.mappers[m].get_enc(si)) for si in v]
                else:
                    data[c] = torch.LongTensor(self.mappers[m].get_enc(v))
            elif c in self.field_handler.emb:
                m = self.field_handler.emb[c]
                if isinstance(v, (list, tuple)):
                    data[f"{Prefix.EMB}{c}"] = [torch.LongTensor(self.mappers[m].get_emb(si)) for si in v]
                else:
                    data[f"{Prefix.EMB}{c}"] = torch.LongTensor(self.mappers[m].get_emb(v))
            elif c in self.field_handler.float:
                data[c] = torch.FloatTensor(v)

    def get_acts(self, sample, data):
        """Stage 2 builder: look up SAE concept IDs and activation values for a review.

        Requires ``self.acts_mapper`` to have been loaded during ``_finish_init``
        (activated when ``Key.REV`` is in ``cfg.load_cols`` and the acts file exists).
        Writes fixed-length tensors of shape ``(top_k,)`` — no padding needed since
        all reviews share the same top-k by construction.
        """
        row = self.acts_mapper.loc[sample[Key.REV]]
        data[Key.CONCEPT_ID] = torch.LongTensor(row["concept_id"])
        data[Key.CONCEPT_ACT] = torch.FloatTensor(row["act"])

    def apply_sid_augment(self, data):
        """
        Augment mode (Stage 3): keep ``data[Key.ITEM]`` as raw item IDs *and*
        add the flattened SID codes as a separate ``data["item_semid"]`` tensor.

        The SID codes are stored without per-codebook offsets (raw indices in
        ``[0, codebook_size)``) so that downstream model code can apply its own
        offsets if needed.

        Operates on the already-built ``data`` dict so that sequence-assembly
        transforms (e.g. ``merge_hist``) have already concatenated history +
        target item IDs before the SID lookup runs.
        """
        raise NotImplementedError("Augment mode is not implemented yet.")

    def apply_sid_replace(self, data, entity: str = Key.ITEM):
        """
        Replace mode (Stage 3): substitute ``data[entity]`` raw IDs with
        flattened SID codes.

        Operating in Stage 3 means that sequence-assembly transforms such as
        ``merge_hist`` in :class:`SEQUERDataset` runs
        first and produce a single contiguous tensor of raw entity IDs.  This
        method then maps every ID in that sequence to its ``n_codebooks`` SID
        codes and concatenates them into a new 1-D tensor.

        Uses the pre-computed ``_sid_caches[entity]`` (built in
        ``_build_sid_cache``) for an O(n_entities) lookup rather than
        per-sample arithmetic.

        Args:
            data:   Sample data dict (modified in-place).
            entity: The column to replace (``Key.ITEM`` or ``Key.USER``).
        """
        raw_ids = data[entity].tolist()
        sid_codes = list(sum(self._sid_caches[entity][raw_ids], []))
        data[entity] = torch.LongTensor(sid_codes)

    def read_data(self, data_file, suffix="", load_cols=BASE_COLS, load_hist=False, load_metadata=False):
        self.logger.info("Reading data, mappers and split indexes...")
        ixs_dir = os.path.join(self.data_path, str(self.cfg.fold))

        if not os.path.exists(self.data_path):
            if self.cfg.test_flag:
                raise NotImplementedError(
                    "Dataset for testing purposes does not exist and sample function is not implemented yet."
                )
                # self.sample_dataset()
            else:
                raise FileNotFoundError(f"Path {self.data_path} not found.")
        data: pd.DataFrame = load_safe_parquet(os.path.join(self.data_path, data_file), columns=load_cols)
        metadata = None
        metadata_path = os.path.join(self.data_path, f"metadata{suffix}.parquet")
        if load_metadata and os.path.exists(metadata_path):
            metadata = load_safe_parquet(metadata_path)
        split_ixs = load_partition(ixs_dir)

        # Assert uniqueness of indexes in dataset and partitions
        assert data.index.is_unique
        for _, v in split_ixs.items():
            assert len(np.unique(v)) == len(v)

        # Ensure monotonic index by steps of 1
        self.monotonic_ix = (data.sort_index().index.values == np.arange(stop=data.shape[0], dtype=int)).all()
        if not self.monotonic_ix:
            data.reset_index(inplace=True)
            mapping = pd.Series(data.index.values, index=data["index"])
            for c, v in split_ixs.items():
                split_ixs[c] = mapping.loc[split_ixs[c]].tolist()
            data.drop("index", inplace=True, axis=1)

        if Key.RATING in load_cols and not pd.api.types.is_float_dtype(data[Key.RATING].dtype):
            data[Key.RATING] = pd.to_numeric(data[Key.RATING], downcast="float")

        # NOTE: P5 does subword tokenization of user and item encoded IDs.
        mapper_path = os.path.join(self.data_path, "mappers")

        def get_value(x):
            return get_value(x[0]) if isinstance(x, list) else x

        def nested_map(x, mapper):
            if isinstance(x[0], list):
                return list(map(lambda x: nested_map(x, mapper), x))
            else:
                return mapper.get_idxs(x)

        mappers: Dict[str, Mapper] = {}
        for c in set(load_cols).intersection([Key.USER, Key.ITEM, Key.EXP, Key.FEAT, Key.REV]):
            if os.path.isfile(os.path.join(mapper_path, f"{c}{suffix}.parquet")):
                mapper = load_mapper_parquet(os.path.join(mapper_path, f"{c}{suffix}.parquet"))
            else:
                data[c], mapper = get_transform_mapping(data[c])
                mapper.reset_index().to_parquet(os.path.join(mapper_path, f"{c}{suffix}.parquet"))

            mappers[c] = Mapper(mapper)
            if not isinstance(get_value(data[c].iloc[0]), numbers.Integral):
                if isinstance(data[c].iloc[0], list):
                    data[c] = data[c].apply(lambda x: nested_map(x, mappers[c]))
                else:
                    data[c] = mappers[c].get_idxs(data[c].tolist())

        mappers[Key.ITEM].add_items(SPECIAL_TOKENS.values())
        mappers[Key.EXP].add_items([""])

        self.data, self.split_ixs, self.mappers, self.metadata = data, split_ixs, mappers, metadata

        unrolled = self.data[[Key.ITEM, Key.USER, Key.FEAT]].explode(Key.FEAT).explode(Key.FEAT)
        self.u2f = unrolled.groupby(Key.USER)[Key.FEAT].agg(set)
        self.i2f = unrolled.groupby(Key.ITEM)[Key.FEAT].agg(set)

    def get_required_lengths(self):
        return {c: self.mappers[c].enc_len() for c in [Key.USER, Key.ITEM, Key.FEAT, Key.EXP]}

    def handle_aux_cols(self):
        if self.cfg.requires_rating:
            self.data[Key.RATING] = self.data[Key.RATING].values[:, None].tolist()
            self.field_handler.add_float(Key.RATING)
        else:
            self.data.drop(Key.RATING, axis=1, inplace=True)

        if self.cfg.requires_context and self.cfg.mod_context_flag:
            self.logger.info("Getting modified context")
            aux = self.mappers[Key.EXP].get_items(range(len(self.mappers[Key.EXP])))
            aux = pd.Series(get_context(aux, pos_tags=["NOUN", "ADJ"]))
            aux = self.data[Key.EXP].explode().map(aux).groupby(level=0).agg(" ".join)
            aux = aux.str.split().apply(lambda l: " ".join(set(l)))

            self.data[Key.CONTEXT], mapper = get_transform_mapping(aux)
            self.mappers[Key.CONTEXT] = Mapper(mapper)

    def handle_past_info(self):
        pass

    def handle_encodings(self, ui_as_list=True):
        """Main entry point for encoding and embedding setup."""
        self._setup_ui_columns(ui_as_list)
        self._configure_encoding_columns()
        self._encode_text_fields()
        self._handle_embeddings()

    def _setup_ui_columns(self, ui_as_list):
        """Convert User and Item columns to lists if needed."""
        if ui_as_list:
            self.data[Key.ITEM] = self.data[Key.ITEM].values[:, None].tolist()
            self.data[Key.USER] = self.data[Key.USER].values[:, None].tolist()

    def _configure_encoding_columns(self):
        """Determine which columns need encoding based on config."""
        if self.cfg.requires_feat:
            self.field_handler.add_encoding(Key.FEAT, Key.FEAT)

        if self.cfg.requires_exp:
            self.field_handler.add_encoding(Key.EXP, Key.EXP)

        if self.cfg.requires_context:
            if self.cfg.mod_context_flag:
                assert Key.CONTEXT in self.mappers, f"Modified context requires {Key.CONTEXT} mapper"
                self.field_handler.add_encoding(Key.CONTEXT, Key.CONTEXT)
            else:
                self.field_handler.add_encoding(Key.CONTEXT, Key.EXP)

    def _encode_text_fields(self):
        """Perform actual text encoding for configured columns."""
        self.logger.info("Encoding necessary fields...")

        for mapper_col in set(self.field_handler.enc.values()):
            if mapper_col == Key.EXP:
                self.mappers[mapper_col].encode(
                    self.tok,
                    truncation=True,
                    max_length=self.cfg.txt_len,
                    add_special_tokens=self.add_special_toks,
                )
            elif mapper_col == Key.CONTEXT:
                self.mappers[mapper_col].encode(
                    self.tok,
                    truncation=True,
                    max_length=self.cfg.txt_len,
                    add_special_tokens=False,
                )
            elif mapper_col in [Key.FEAT, Key.ASP]:
                self.mappers[mapper_col].encode(self.tok, add_special_tokens=False)

    def _handle_embeddings(self):
        """Handle embedding computation and loading."""
        if self.cfg.emb_models is None:
            return

        self.logger.info("Obtaining necessary embeddings...")
        if not hasattr(self, "precomputed_emb_dim"):
            self.precomputed_emb_dim = {}

        for emb_model in self.cfg.emb_models:
            embedder = self._get_or_create_embedder(emb_model)
            self._process_embedding_model(emb_model, embedder)

    def _get_or_create_embedder(self, emb_model_cfg: SimpleNamespace) -> Optional[SentenceTransformer]:
        """Get or create embedder for a model if needed."""
        c_req_m = [
            c
            for c in emb_model_cfg.apply_to
            if not os.path.isfile(
                os.path.join(
                    self.data_path,
                    "embeddings",
                    f"{Prefix.EMB}{emb_model_cfg.name.replace('/', '--')}_{c}{self.suffix}.npy",
                )
            )
        ]
        if c_req_m:
            return self.get_emb_model(emb_model_cfg)
        return None

    def _process_embedding_model(self, emb_model_cfg, embedder):
        """Process embeddings for all columns in an embedding model."""
        for c in emb_model_cfg.apply_to:
            self.field_handler.add_embedding(c, c)
            emb_path = self._get_embedding_path(c, emb_model_cfg)

            if os.path.isfile(emb_path):
                self._load_embedding(c, emb_path, emb_model_cfg)
            else:
                self._compute_and_save_embedding(c, embedder, emb_path, emb_model_cfg)

            self.precomputed_emb_dim[c] = self.mappers[c].embs.shape[1]

    def _get_embedding_path(self, col, emb_model_cfg):
        """Get the path for an embedding file."""
        f_name = f"{Prefix.EMB}{emb_model_cfg.name.replace('/', '--')}_{col}{self.suffix}.npy"
        emb_dir = os.path.join(self.data_path, "embeddings")
        if not os.path.isdir(emb_dir):
            os.makedirs(emb_dir)
        return os.path.join(emb_dir, f_name)

    def _load_embedding(self, col, emb_path, emb_model_cfg):
        """Load precomputed embedding."""
        self.logger.info(f"Loading {emb_model_cfg.name} embeddings for {col} field...")
        self.mappers[col].load_embed(emb_path)

    def _compute_and_save_embedding(self, col, embedder, emb_path, emb_model_cfg):
        """Compute and save embedding."""
        self.logger.info(f"Computing {emb_model_cfg.name} embeddings for {col} field...")
        self.mappers[col].embed(embedder, batch_size=getattr(emb_model_cfg, "batch_size", self.batch_size["train"]))
        self.logger.info(f"Saving {emb_model_cfg.name} embeddings for {col} field...")
        self.mappers[col].save_embed(emb_path)

    def get_emb_model(self, emb_cfg):
        if emb_cfg.type == "huggingface":
            model = SentenceTransformer(emb_cfg.name, device=self.device, use_auth_token=self.cfg.hf_auth_token)
            max_length = getattr(emb_cfg, "max_seq_length", None)
            if max_length is not None:
                model.max_seq_length = max_length
            return model
        # elif cfg.type == 'glove':
        #     f_name = getattr(cfg, 'f_name', 'glove.42B.300d')
        #     raise NotImplementedError('GloVe embeddings are not implemented into the framework yet')
        else:
            raise NotImplementedError(f"{emb_cfg.type} embeddings are not implemented into the framework yet")

    def neg_sample(self, item_seq: set, n_neg=99, replace=False):
        neg_items = list(self.item_set.difference(item_seq))
        if n_neg >= len(neg_items):
            random.shuffle(neg_items)
        else:
            neg_items = np.random.choice(neg_items, n_neg, replace=replace).tolist()
        return neg_items

    def get_gen_labels(self, gen_steps, n_batches, batch_size, pred_tasks, phase):
        labels = {}
        # Labels for EXP and FEAT assume the split was not shuffled
        select_ixs = self.split_ixs[phase]
        if 0 < gen_steps < n_batches:
            select_ixs = select_ixs[: gen_steps * batch_size]

        for meta_c in [Key.USER, Key.ITEM]:
            labels[f"metadata-{meta_c}"] = self.data.loc[select_ixs, meta_c].explode().tolist()
        labels[Task.EXPLANATION] = self.data.loc[select_ixs, Key.EXP].tolist()
        labels[Task.FEAT] = self.data.loc[select_ixs, Key.FEAT]
        if isinstance(labels[Task.FEAT].iloc[0][0], Iterable):
            labels[Task.FEAT] = labels[Task.FEAT].apply(sum, start=[]).apply(lambda l: list(set(l))).tolist()
        else:
            labels[Task.FEAT] = labels[Task.FEAT].apply(lambda l: list(set(l))).tolist()

        if Task.RATING in pred_tasks:
            labels[Task.RATING] = self.data.loc[select_ixs, Key.RATING].tolist()
        for t in [Task.NEXT_ITEM, Task.TOPN]:
            if t in pred_tasks:
                labels[t] = labels[f"metadata-{Key.ITEM}"]

        if getattr(self.data_info, "sae_n_concepts", 0) > 0:
            labels[Task.CONTEXT] = self.acts_mapper[Key.CONCEPT_ID].loc[self.data.loc[select_ixs, Key.REV]].tolist()
        return labels

    def get_subset(self, split):
        return Subset(dataset=self, indices=self.split_ixs[split])

    def get_dataloaders(self, drop_last=False):
        dls = []
        for split in ["train", "valid", "test"]:
            dls.append(self.get_dataloader(split, drop_last=(drop_last if split == "train" else False)))
        return dls

    def get_dataloader(self, split=None, drop_last=False):
        n_workers = self.workers[split]
        return DataLoader(
            self.get_subset(split),
            self.batch_size[split],
            num_workers=n_workers,
            collate_fn=self.collate,
            shuffle=(split == "train"),
            drop_last=drop_last,
            pin_memory=(self.device.type == "cuda"),
            persistent_workers=(n_workers > 0),
            prefetch_factor=(2 if n_workers > 0 else None),
        )

    def log_data_benchmark(self, n_steps: int):
        """Log fine-grained per-step timing breakdown for __getitem__ + collate.

        Intended to be called from the trainer's _log_benchmark after n_steps batches.
        Per-sample keys (everything except collate_*) are reported both per sample and
        per batch; collate keys are per-batch only.
        """
        if not self._bench or n_steps == 0:
            return

        bs = self.batch_size["train"]
        total = sum(self._bench.values())
        if total == 0:
            return

        collate_prefix = "collate_"
        lines = [f"[Data breakdown] {n_steps} steps  (batch={bs})"]
        lines.append(f"  {'Step':<32} {'ms/sample':>10}  {'ms/batch':>10}  {'%':>6}")
        lines.append("  " + "-" * 64)
        for key, acc in self._bench.items():
            ms_per_batch = 1000 * acc / n_steps
            pct = 100 * acc / total
            if key.startswith(collate_prefix):
                lines.append(f"  {key:<32} {'—':>10}  {ms_per_batch:>10.2f}  {pct:>5.1f}%")
            else:
                ms_per_sample = ms_per_batch / bs
                lines.append(f"  {key:<32} {ms_per_sample:>10.3f}  {ms_per_batch:>10.2f}  {pct:>5.1f}%")
        self.logger.info("\n".join(lines))

    def check_test_in_train(self, train_ixs):
        def clean_histories(hist_series: pd.Series):
            return (
                hist_series.explode()
                .astype(int)
                .replace(unk_items, self.mappers[Key.ITEM].get_idx(UNK_TOK))
                .groupby(level=0)
                .agg(list)
            )

        self.logger.info("Checking test items are present in the train+valid set...")
        train_items = set(self.data.loc[train_ixs, Key.ITEM].tolist())
        train_users = set(self.data.loc[train_ixs, Key.USER].tolist())

        val_tst_ixs = self.split_ixs["valid"] + self.split_ixs["test"]
        mask = self.data.loc[val_tst_ixs, Key.ITEM].isin(train_items)

        unk_items = self.data.loc[np.array(val_tst_ixs)[np.argwhere(~mask.values)[:, 0]], Key.ITEM].unique()
        self.item_set.difference_update(unk_items)

        # Replace unseen items from the item history column with the UNK_ITEM
        if HistKey(Key.ITEM, Key.USER) in self.data.columns and len(unk_items) > 0:
            ixs = np.intersect1d(
                self.data.index[self.data[HistKey(Key.ITEM, Key.USER)].str.len() > 0].values,
                val_tst_ixs,
                assume_unique=True,
            )
            self.data.loc[ixs, HistKey(Key.ITEM, Key.USER)] = clean_histories(
                self.data.loc[ixs, HistKey(Key.ITEM, Key.USER)]
            )

        mask = mask & self.data.loc[val_tst_ixs, Key.USER].isin(train_users)

        for phase in ["valid", "test"]:
            n_samples = len(self.split_ixs[phase])
            keep_ixs = np.argwhere(mask.values[:n_samples])[:, 0]
            self.split_ixs[phase] = np.array(self.split_ixs[phase])[keep_ixs].tolist()
            self.logger.info(
                f"Out of {n_samples} {phase} samples, {n_samples - len(self.split_ixs[phase])} were removed "
                f"as either user or item was not present in the training set"
            )
            mask = mask[n_samples:]

        self.data = self.data.loc[sum(self.split_ixs.values(), start=[])].sort_values(Key.TIMESTEP)

    def flatten_cols(self, filter_unique=False, preserve_order=False):
        self.data[Key.FEAT] = self.data[Key.FEAT].apply(partial(sum, start=[]))

        if preserve_order:
            # NOTE 1: ~x3 slower but preserves temporal order of sentences (unique starts from most recent sentences)
            # NOTE 2: pandas .unique() states "Uniques are returned in order of appearance"
            unique_fn = lambda l: pd.Series(l[::-1]).unique().tolist()[::-1]
        else:
            unique_fn = lambda l: list(set(l))

        if filter_unique:
            for c in [Key.FEAT, Key.EXP]:
                self.data[c] = self.data[c].apply(unique_fn)

    def collate(self, batch) -> Dict[str, torch.Tensor]:
        return self.collator(batch)


class GenDataset(BaseDataset):
    """:class:`GenDataset` extends :class:`~nlp_xrs.data.dataset.BaseDataset`."""

    def _finish_init(self):
        gen_sampling = getattr(self.cfg, "gen_sampling", "alternate")
        if gen_sampling == "alternate":
            self.sample_prep_fns.append(self.sample_exps)
        elif gen_sampling == "fixed":
            self.sample_fixed()
            self.sample_prep_fns.append(self.sample_exps_fixed)
            self.field_handler.keep_cols += [f"sampled_{Key.FEAT}", f"sampled_{Key.EXP}"]
        elif gen_sampling == "alternate_multiple":
            self.sample_prep_fns.append(self.sample_exps_multiple)
            self.data_transform_fns.append(self.merge_exps)
        else:
            raise NotImplementedError(
                f'Generative sampling "{gen_sampling}" not implemented. Use "fixed" or "alternate" instead.'
            )

        self.add_special_toks = True
        super()._finish_init()

    def sample_fixed(self):
        """
        Sample a single feature and explanation for each item.
        """
        fe = self.data[[Key.FEAT, Key.EXP]].explode([Key.FEAT, Key.EXP]).sample(frac=1)
        fe = fe[~fe.index.duplicated(keep="first")]
        fe[Key.FEAT] = fe[Key.FEAT].map(random.choice)
        fe.rename(lambda c: f"sampled_{c}", inplace=True, axis=1)
        self.data = self.data.join(fe)
        assert not any(self.data.index.duplicated())
        assert not self.data[[f"sampled_{Key.FEAT}", f"sampled_{Key.EXP}"]].isna().values.any()

    def sample_exps_fixed(self, sample, map_to_str=False):
        sample[f"{Key.EXP}_full"] = sample[Key.EXP]
        sample[f"{Key.FEAT}_full"] = sample[Key.FEAT]

        sample[Key.EXP] = sample.pop(f"sampled_{Key.EXP}")
        sample[Key.FEAT] = sample.pop(f"sampled_{Key.FEAT}")
        if self.cfg.requires_context and not self.cfg.mod_context_flag:
            sample[f"{Key.CONTEXT}_full"] = sample.get(Key.CONTEXT, sample[f"{Key.EXP}_full"])
            # NOTE: In the next getter_fn, the CONTEXT mapper will be used to map the index to the actual context
            sample[Key.CONTEXT] = sample.get(Key.CONTEXT, sample[Key.EXP])

        if map_to_str:
            sample[Key.FEAT] = self.mappers[Key.FEAT].get_item(sample[Key.FEAT])
            sample[Key.EXP] = self.mappers[Key.EXP].get_item(sample[Key.EXP])

    def sample_exps(self, sample, map_to_str=False):
        sample[f"{Key.EXP}_full"] = sample[Key.EXP]
        sample[f"{Key.FEAT}_full"] = sample[Key.FEAT]
        # Sample exp and corresponding feats
        ix = random.randint(0, len(sample[Key.EXP]) - 1)

        if self.cfg.requires_context and not self.cfg.mod_context_flag:
            sample[f"{Key.CONTEXT}_full"] = sample.get(Key.CONTEXT, sample[f"{Key.EXP}_full"])
            # NOTE: In the next getter_fn, the CONTEXT mapper will be used to map the index to the actual context
            sample[Key.CONTEXT] = sample.get(Key.CONTEXT, sample[Key.EXP])[ix]

        sample[Key.EXP] = sample[Key.EXP][ix]
        sample[Key.FEAT] = sample[Key.FEAT][ix]
        # Sample a single feat from all possibilities
        ix = random.randint(0, len(sample[Key.FEAT]) - 1)
        sample[Key.FEAT] = sample[Key.FEAT][ix]
        if map_to_str:
            sample[Key.FEAT] = self.mappers[Key.FEAT].get_item(sample[Key.FEAT])
            sample[Key.EXP] = self.mappers[Key.EXP].get_item(sample[Key.EXP])

    def sample_exps_multiple(self, sample, map_to_str=False):
        sample[f"{Key.EXP}_full"] = sample[Key.EXP]
        sample[f"{Key.FEAT}_full"] = sample[Key.FEAT]
        # Sample exp and corresponding feats
        n_exps = random.randint(1, len(sample[Key.EXP]))
        ixs = random.sample(range(len(sample[Key.EXP])), n_exps)

        if self.cfg.requires_context and not self.cfg.mod_context_flag:
            sample[f"{Key.CONTEXT}_full"] = sample.get(Key.CONTEXT, sample[f"{Key.EXP}_full"])
            # NOTE: In the next getter_fn, the CONTEXT mapper will be used to map the index to the actual context
            sample[Key.CONTEXT] = list(map(sample.get(Key.CONTEXT, sample[Key.EXP]).__getitem__, ixs))

        sample[Key.EXP] = list(map(sample[Key.EXP].__getitem__, ixs))
        sample[Key.FEAT] = sum(list(map(sample[Key.FEAT].__getitem__, ixs)), start=[])
        # Sample a single feat from all possibilities
        ix = random.randint(0, len(sample[Key.FEAT]) - 1)
        sample[Key.FEAT] = sample[Key.FEAT][ix]
        if map_to_str:
            sample[Key.FEAT] = self.mappers[Key.FEAT].get_item(sample[Key.FEAT])
            sample[Key.EXP] = " ".join(self.mappers[Key.EXP].get_items(sample[Key.EXP]))

    def merge_exps(self, data):
        data[Key.EXP] = torch.cat(data[Key.EXP], axis=0)
        mask = torch.isin(
            data[Key.EXP], torch.tensor([self.tok.pad_token_id, self.tok.eos_token_id, self.tok.bos_token_id])
        )
        mask[[0, -1]] = False  # Keep the BOS and EOS token if no truncation is needed
        data[Key.EXP] = data[Key.EXP][~mask][: self.cfg.txt_len]

        if self.cfg.requires_context and not self.cfg.mod_context_flag:
            data[Key.CONTEXT] = torch.cat(data[Key.CONTEXT], axis=0)
            data[Key.CONTEXT] = data[Key.CONTEXT][~mask]  # [:self.cfg.txt_len]


class SeqDataset(BaseDataset):
    def _finish_init(self):
        if self.cfg.requires_nextitem:
            self.data_builder_fns.append(self.get_neg_samples)

        super()._finish_init()

        # Set the length of the item history in the input sequence
        if HistKey(Key.ITEM, Key.USER) in self.data.columns and hasattr(self.data_info, "lengths"):
            self.data_info.lengths[Key.ITEM] *= (
                min(self.data[HistKey(Key.ITEM, Key.USER)].apply(len).max(), self.cfg.hist_len) + 1
            )

    def read_data(
        self,
        data_file: str,
        suffix: str = "",
        load_cols: List[str] = BASE_COLS,
        load_hist: bool = True,
        load_metadata: bool = False,
    ):
        super().read_data(data_file, suffix, load_cols=load_cols, load_metadata=load_metadata)

        if load_hist:
            self.get_hist_data(data_file, suffix)
            if self.cfg.requires_nextitem:
                self.get_negative_samples(suffix)

    def get_hist_len(self):
        if not hasattr(self.cfg, "hist_len"):
            self.cfg.hist_len = DEFAULT_HIST_LEN

    def get_profile_hist(self, hist_data_path, save=True):
        orig_index = self.data.index.values
        hist_cols = []
        for groupby in [Key.USER, Key.ITEM]:
            self.logger.info(f"Extracting history for {groupby} column")
            hist_cols.append(f"{groupby}_{Prefix.HIST[:-1]}")
            self.data.sort_values([groupby, Key.TIMESTEP], inplace=True)
            histories = self.data[[groupby]].groupby(groupby)[groupby].apply(sliding_w_hist)
            self.data[hist_cols[-1]] = sum(histories.tolist(), [])
        self.data = self.data.loc[orig_index]
        if save:
            self.data[hist_cols].to_pickle(hist_data_path)

    def get_hist_data(self, data_file, suffix=""):
        def getter(idxr: pd.Series, select: str) -> List[Any]:
            # NOTE: it assumes the self.data index is monotonically increasing with steps of 1
            values = self.data[select].values

            return [values[ixs].tolist() for ixs in idxr]

        self.get_hist_len()
        cols = self.get_profile_hist_cols()

        # If, for any reason, the sequential model requires no past info (hist_len == 0), skip historical retrieval
        if self.cfg.hist_len == 0:
            for groupby, select_cols in cols.items():
                idxr_col = f"{groupby}_{Prefix.HIST[:-1]}"
                for c in select_cols:
                    new_c = f"{idxr_col}_{c}"
                    self.data[new_c] = np.empty((len(self.data), 0)).tolist()
            return

        hist_data_path = os.path.join(self.data_path, f"{Prefix.HIST}{data_file}")
        if os.path.isfile(hist_data_path) and self.monotonic_ix:
            self.data = pd.concat(
                [
                    self.data,
                    load_safe_parquet(hist_data_path, columns=[f"{c}_{Prefix.HIST[:-1]}" for c in cols.keys()]),
                ],
                axis=1,
            )
        else:
            self.get_profile_hist(hist_data_path, save=False)  # not self.cfg.test_flag)

        self.data.sort_index(inplace=True)

        for groupby, select_cols in cols.items():
            max_l = getattr(self.cfg, f"{groupby}_hist_len", self.cfg.hist_len)
            idxr_col = f"{groupby}_{Prefix.HIST[:-1]}"
            has_hist_mask = self.data[idxr_col].str.len() > 0
            idxr = self.data.loc[has_hist_mask, idxr_col]
            if max_l < MAX_HIST_LEN:
                idxr = idxr.str[-max_l:]
            for c in select_cols:
                new_c = f"{idxr_col}_{c}"
                self.data[new_c] = np.empty((len(self.data), 0)).tolist()
                self.data.loc[has_hist_mask, new_c] = pd.Series(
                    getter(idxr, c), index=has_hist_mask[has_hist_mask].index
                )

            self.data.drop(idxr_col, axis=1, inplace=True)

    def get_profile_hist_cols(self):
        """
        Return a dict of {profile_src: retrieve_cols}, where profile_src is the groupby column and retrieve_cols are the
        columns retrieved by the sliding window approach. For instance, {Key.USER: [Key.ITEM, Key.RATING]} will retrieve the
        historical items and rating from the user profile history.
        """
        cols = defaultdict(list)
        if SeqMode.HIST_ITEM.value <= self.cfg.seq_mode.value <= SeqMode.HIST_ITEM_U_EXP_EMB.value:
            cols[Key.USER].append(Key.ITEM)
            if self.cfg.requires_rating:
                cols[Key.USER].append(Key.RATING)
        if SeqMode.HIST_ITEM_U_EXP.value <= self.cfg.seq_mode.value < SeqMode.CUSTOM.value:
            cols[Key.USER].append(Key.EXP)
        if self.cfg.seq_mode.value == SeqMode.HIST_UI_EXP.value:
            cols[Key.ITEM].append(Key.EXP)
        return cols

    def get_negative_samples(self, suffix=""):
        neg_sample_path = os.path.join(self.data_path, f"user_neg_samples{suffix}.parquet")
        self.u2i = self.data[[Key.USER, Key.ITEM]].groupby(Key.USER)[Key.ITEM].agg(set)  # .str[-1]
        if os.path.exists(neg_sample_path):
            self.user_neg_samples = load_mapper_parquet(neg_sample_path)
            if not set(self.u2i.index.tolist()).issubset(set(self.user_neg_samples.keys())):
                # os.remove(neg_sample_path)
                raise Exception(
                    f'Missing users in "user_neg_samples{suffix}.parquet". Run process_data.get_aux_data(model_name='
                    f'"{self.model_cfg.name}, "dataset="{self.cfg.dataset}") to get the  auxiliary data '
                    f"for this model"
                )
        else:
            raise FileNotFoundError(
                f'Run process_data.get_aux_data(model_name="{self.model_cfg.name}, '
                f'"dataset="{self.cfg.dataset}") to get the  auxiliary data for this model'
            )

    def flatten_cols(self, filter_unique=False, preserve_order=False):
        #  Current version is a global flattening
        super().flatten_cols(filter_unique, preserve_order)

        hist_cols = [col for col in self.data.columns if Prefix.HIST in col]
        if hist_cols:
            from pandarallel import pandarallel

            pandarallel.initialize(nb_workers=min(os.cpu_count(), 3), progress_bar=False)

        for c in hist_cols:
            mask = self.data[c].str.len() > 0
            # Handle those columns with no history
            if mask.any():
                if Key.FEAT in c:
                    self.data.loc[mask, c] = self.data.loc[mask, c].parallel_apply(
                        lambda ls: sum(map(partial(sum, start=[]), ls), [])
                    )
                elif Key.EXP in c:
                    self.data.loc[mask, c] = self.data.loc[mask, c].apply(sum, start=[])
                if filter_unique:
                    if preserve_order:
                        self.data.loc[mask, c] = self.data.loc[mask, c].parallel_apply(
                            lambda l: pd.Series(l[::-1]).unique().tolist()[::-1]
                        )
                    else:
                        self.data.loc[mask, c] = self.data.loc[mask, c].apply(lambda l: list(set(l)))

    def ensure_max_len(self):
        for c in self.data.columns:
            if Prefix.HIST in c and self.cfg.hist_len != "all":
                self.data[c] = self.data[c].apply(lambda h: h[-self.cfg.hist_len :])

    def handle_past_info(self):
        super().handle_past_info()

        # Limit context length
        self.ensure_max_len()

        if self.cfg.requires_nextitem:
            # NOTE: Add UNK_TOKEN for cold_start recommendation (to estimate one item as context)
            if self.cfg.coldstart_fill:
                fill_lambda = lambda h: h if h else [self.mappers[Key.ITEM].get_idx(UNK_TOK)]
                self.data[HistKey(Key.ITEM, Key.USER)] = self.data[HistKey(Key.ITEM, Key.USER)].apply(fill_lambda)
            self.field_handler.add_original(HistKey(Key.ITEM, Key.USER))

            if (
                self.cfg.seq_mode != SeqMode.HIST_ITEM_RATING_LAST
                and HistKey(Key.RATING, Key.USER) in self.data.columns
            ):
                if self.cfg.coldstart_fill:
                    fill_lambda = lambda h: h if h else [self.data_info.mean_rating]
                    self.data[HistKey(Key.RATING, Key.USER)] = self.data[HistKey(Key.RATING, Key.USER)].apply(
                        fill_lambda
                    )
                self.field_handler.add_float(HistKey(Key.RATING, Key.USER))

        if self.cfg.seq_mode.requires_past_user_exp():
            if self.cfg.coldstart_fill:
                self.data[HistKey(Key.EXP, Key.USER)] = self.data[HistKey(Key.EXP, Key.USER)].apply(
                    lambda h: h if h else [[self.mappers[Key.EXP].get_idx("")]]
                )
            self.field_handler.add_encoding(HistKey(Key.EXP, Key.USER), Key.EXP)
            self.field_handler.add_encoding(HistKey(Key.FEAT, Key.USER), Key.FEAT)
            if self.cfg.emb_models is not None:
                self.field_handler.add_embedding(HistKey(Key.EXP, Key.USER), Key.EXP)
                self.field_handler.add_embedding(HistKey(Key.FEAT, Key.USER), Key.FEAT)

        if self.cfg.seq_mode.requires_past_ui_exp():
            if self.cfg.coldstart_fill:
                self.data[HistKey(Key.EXP, Key.ITEM)] = self.data[HistKey(Key.EXP, Key.ITEM)].apply(
                    lambda h: h if h else [[self.mappers[Key.EXP].get_idx("")]]
                )
            self.field_handler.add_encoding(HistKey(Key.EXP, Key.ITEM), Key.EXP)
            self.field_handler.add_encoding(HistKey(Key.FEAT, Key.ITEM), Key.FEAT)
            if self.cfg.emb_models is not None:
                self.field_handler.add_embedding(HistKey(Key.EXP, Key.ITEM), Key.EXP)
                self.field_handler.add_embedding(HistKey(Key.FEAT, Key.ITEM), Key.FEAT)

    def get_neg_samples(self, sample, data, only_past_pos=True):
        if only_past_pos:
            seq = sample[HistKey(Key.ITEM, Key.USER)] + sample[Key.ITEM]
        else:
            seq = self.u2i[sample[Key.USER]]
        # Sample negative items for optimization with BCE (1 per item in the sequence - except cand. item)
        data[Key.NEG_OPT] = torch.LongTensor(self.neg_sample(seq, len(seq), replace=True))
        # Sample negative items for evaluation metrics computation
        data[Key.NEG_EVAL] = torch.LongTensor([sample[Key.ITEM][-1]] + self.user_neg_samples[sample[Key.USER][0]])


class SEQUERDataset(SeqDataset, GenDataset):
    def _finish_init(self):
        self.data_transform_fns.append(self.merge_hist)
        super()._finish_init()

    def merge_hist(self, data):
        data[Key.ITEM] = torch.cat((data.pop(HistKey(Key.ITEM, Key.USER)), data[Key.ITEM]), dim=0)
        data[Key.RATING] = torch.cat((data.pop(HistKey(Key.RATING, Key.USER)), data[Key.RATING]), dim=0)
        data[Key.SEQ_LEN] = torch.LongTensor([len(data[Key.ITEM])])

    def _create_collator(self) -> BaseCollator:
        item_seq_max_len = getattr(self.data_info, "lengths", {}).get(Key.ITEM, 512)
        return SEQUERCollator(
            tok=self.tok, field_handler=self.field_handler, mappers=self.mappers, item_seq_len=item_seq_max_len
        )


class SEQUERTextFusionDataset(SEQUERDataset):
    """
    SEQUERTextFusionDataset extends SEQUERDataset to provide precomputed item text embeddings (e.g., from BERT) for use in fusion models.
    It loads the item text embeddings and aligns them with item_ids, making them available via self.item_text_emb and self.data_info.item_text_emb.
    Usage: Used with SEQUERTextFusion model for collaborative+text fusion.
    """

    def _finish_init(self):
        super()._finish_init()

        item_sid_cfg = self._sid_entity_cfgs.get(Key.ITEM)
        if item_sid_cfg is not None and getattr(item_sid_cfg, "mode", "replace") == "replace":
            # SID replacement runs after merge_hist and overwrites Key.ITEM.
            # Preserve the assembled raw sequence so text embeddings can still
            # be indexed by the original item vocabulary.
            replace_ix = next(
                (
                    ix
                    for ix, transform_fn in enumerate(self.data_transform_fns)
                    if getattr(transform_fn, "keywords", {}).get("entity") == Key.ITEM
                ),
                None,
            )
            if replace_ix is None:
                raise RuntimeError("Expected an item SID replacement transform for SEQUERTextFusionDataset.")
            self.data_transform_fns.insert(replace_ix, self.stash_raw_items)
            self.data_info.lengths[Key.ITEM_RAW] = self.data_info.lengths[Key.ITEM] // self.data_info.n_codebooks

        # Determine which text source to use for item text embeddings
        use_review_text = getattr(self.cfg, "use_review_text", False)
        if use_review_text:
            assert Key.REV in getattr(self.cfg, "load_cols", [])
            text_col = Key.REV
            logger_text = "review"
            self.field_handler.emb.pop(Key.REV)
        else:
            text_col = Key.EXP
            logger_text = "explanation"

        self.logger.info(f"Aggregating item text embeddings from {logger_text} texts...")

        # Ensure the required embeddings are loaded
        if text_col not in self.mappers or not hasattr(self.mappers[text_col], "embs"):
            raise ValueError(
                f"SEQUERTextFusionDataset requires {logger_text} embeddings to be precomputed. "
                f"Make sure to configure emb_models with apply_to=[{text_col}]."
            )

        # Get aggregation method from config
        aggregation_method = getattr(self.cfg, "item_text_aggregation", "mean")

        # Get all unique items in the vocabulary
        all_items = list(range(len(self.mappers[Key.ITEM])))

        # Aggregate item text embeddings
        split_data = self.data.loc[self.split_ixs["train"] + self.split_ixs["valid"]]
        if aggregation_method.startswith("tf-idf"):
            if use_review_text:
                corpus = split_data[text_col]
            else:
                corpus = split_data[Key.EXP].explode().explode()
            self.tfidf_weights = self.get_tfidf_weights(
                vocab_sz=20000,
                stopwords="english",
                lowercase=True,
                min_df=5,
                deduplicate_corpus=False,
                corpus=corpus,
                col=text_col,
            )
            self.tfidf_weights = np.maximum(self.tfidf_weights, 1e-3)
        if use_review_text:
            # Each entry in Key.FEAT is a review id (int or list of int)
            item_text_embs = (
                split_data.explode(Key.ITEM)
                .groupby(Key.ITEM)[Key.REV]
                .apply(
                    lambda x: self.aggregate_item_text_embeddings_single(
                        review_ixs=x.values.tolist(),
                        aggregation_method=aggregation_method,
                        mapper=self.mappers[Key.REV],
                    )
                )
            )
        else:
            # Each entry in Key.EXP is a list of explanation ids
            item_text_embs = (
                split_data.explode(Key.ITEM)
                .groupby(Key.ITEM)[Key.EXP]
                .agg(
                    lambda x: self.aggregate_item_text_embeddings(
                        explanation_ixs=x.values.tolist(),
                        aggregation_method=aggregation_method,
                        mapper=self.mappers[Key.EXP],
                    )
                )
            )

        # NOTE(aar): Not all items will have text embeddings as some items are not present in train/valid sets
        # assert len(item_text_embs) == len(all_items) - len(SPECIAL_TOKENS)
        self.item_text_emb = torch.zeros(len(all_items), item_text_embs.iloc[0].shape[-1], dtype=torch.float)
        self.item_text_emb[item_text_embs.index.tolist()] = torch.tensor(
            np.vstack(item_text_embs.values), dtype=torch.float
        )
        assert not self.item_text_emb.isnan().any()
        self.data_info.item_text_emb = self.item_text_emb

        self.logger.info(
            f"Created item text embeddings matrix with shape: {self.item_text_emb.shape} "
            f"using {aggregation_method} aggregation from {logger_text}"
        )
        self.logger.info(f"Number of items with text embeddings: {len(item_text_embs)} / {len(all_items)}")

    @staticmethod
    def stash_raw_items(data):
        """Keep the merged raw item sequence before SID replacement."""
        data[Key.ITEM_RAW] = data[Key.ITEM].clone()

    def _create_collator(self) -> BaseCollator:
        item_seq_max_len = getattr(self.data_info, "lengths", {}).get(Key.ITEM, 512)
        return SEQUERTextFusionCollator(
            tok=self.tok,
            field_handler=self.field_handler,
            mappers=self.mappers,
            item_seq_len=item_seq_max_len,
            n_codebooks=getattr(self.data_info, "n_codebooks", 1),
            raw_item_pad_idx=self.data_info.special_items[PAD_TOK],
            bench=self._bench if self._bench_enabled else None,
        )

    def get_tfidf_weights(
        self,
        vocab_sz: int = 20000,
        stopwords: str = "english",
        lowercase: bool = True,
        min_df: int = 5,
        deduplicate_corpus: bool = True,
        corpus: Optional[pd.Series] = None,
        col: str = Key.EXP,
    ) -> np.ndarray:
        """
        It returns a CSR matrix with the TF-IDF representation of each sentence in the corpus
        """
        # Remove numbers (with) punctuation from text (floats, dates/times)
        p = r"\b\d+([:\.]\d+)?(\w+)?\b"

        if deduplicate_corpus:
            corpus = self.mappers[col].idx2item["raw"]
        elif corpus is None:
            corpus = self.data[col]

        texts = list(map(lambda s: re.sub(p, "", s), self.mappers[col].get_items(corpus)))

        tfidf_m = TfidfVectorizer(max_features=vocab_sz, stop_words=stopwords, lowercase=lowercase, min_df=min_df)
        tfidf_m = tfidf_m.fit(texts)
        unique_texts = list(map(lambda s: re.sub(p, "", s), self.mappers[col].idx2item["raw"].tolist()))
        return tfidf_m.transform(unique_texts).sum(axis=1).A1

    def aggregate_item_text_embeddings(
        self, explanation_ixs: List[List[int]], aggregation_method: str, mapper: Mapper
    ) -> np.ndarray:
        # explanation_ixs: list of list of explanation ids
        if aggregation_method == "mean":
            return np.mean(list(map(lambda ixs: np.mean(mapper.get_emb(ixs), axis=0), explanation_ixs)), axis=0)
        elif aggregation_method == "sum":
            return np.sum(list(map(lambda ixs: np.sum(mapper.get_emb(ixs), axis=0), explanation_ixs)), axis=0)
        elif aggregation_method == "tf-idf-sum":
            return np.sum(
                list(map(lambda ixs: self.weighted_sum(ixs, self.tfidf_weights, mapper), explanation_ixs)), axis=0
            )
        elif aggregation_method == "tf-idf-mean":
            return np.mean(
                list(map(lambda ixs: self.weighted_mean(ixs, self.tfidf_weights, mapper), explanation_ixs)), axis=0
            )
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}. Supported methods: 'mean', 'sum'")

    def weighted_sum(self, ixs: List[int], weights: np.ndarray, mapper: Mapper) -> np.ndarray:
        return np.sum(np.multiply(mapper.get_emb(ixs), weights[ixs].reshape(-1, 1)), axis=0)

    def weighted_mean(self, ixs: List[int], weights: np.ndarray, mapper: Mapper) -> np.ndarray:
        return self.weighted_sum(ixs, weights, mapper) / sum(weights[ixs])

    def aggregate_item_text_embeddings_single(
        self, review_ixs: List[int], aggregation_method: str, mapper: Mapper
    ) -> np.ndarray:
        # review_ixs: list of review ids (int)
        if aggregation_method == "mean":
            return np.mean(mapper.get_emb(review_ixs), axis=0)
        elif aggregation_method == "sum":
            return np.sum(mapper.get_emb(review_ixs), axis=0)
        elif aggregation_method == "tf-idf-sum":
            return self.weighted_sum(review_ixs, self.tfidf_weights, mapper)
        elif aggregation_method == "tf-idf-mean":
            return self.weighted_mean(review_ixs, self.tfidf_weights, mapper)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}. Supported methods: 'mean', 'sum'")
