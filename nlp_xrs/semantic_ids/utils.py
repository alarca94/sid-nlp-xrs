"""
Utility functions for tokenizer integration with datasets.
"""

from __future__ import annotations

import hashlib
import html
import json
import logging
import os
import re
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from transformers import AutoConfig

from nlp_xrs.training.scheduler import WarmupLinearScheduler
from nlp_xrs.utils.constants import DATA_MODE, PACKAGE_NAME, Key
from nlp_xrs.utils.funcs import namespace2dict

logger = logging.getLogger(PACKAGE_NAME)

TEXT_COMBINE_METHODS = ("concat_text", "concat_embed", "mean", "weighted_mean")
TEXT_COMPONENTS = ("metadata", "description")
USER_PROFILE_COMPONENTS = ("taste", "technical")

# ---------------------------------------------------------------------------
# Dataset-specific metadata templates
# Keys must match the dataset name used in cfg.data.dataset.
# A value of None means the dataset uses a row-level formatter instead of a template.
# ---------------------------------------------------------------------------
FIELD_SEPARATOR = "; "

TEXT_TEMPLATES = {
    # Opinion-only datasets – Yelp uses a row-level formatter (see DATASET_FORMATTERS).
    "yelp": None,
    # Amazon product datasets.
    "beauty": FIELD_SEPARATOR.join(["Title: {title}", "Brand: {brand}", "Categories: {categories}", "Price: {price}"]),
    "cell": FIELD_SEPARATOR.join(["Title: {title}", "Brand: {brand}", "Categories: {categories}", "Price: {price}"]),
}

# Columns that carry description text, per dataset.
DESCRIPTION_COLS: Dict[str, List[str]] = {
    "beauty": ["description"],
    "cell": ["description"],
}

DEFAULT_LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"


# ---------------------------------------------------------------------------
# Yelp row-level metadata formatter
# Mirrors the processing logic from the genrec Yelp dataset class.
# ---------------------------------------------------------------------------


def _yelp_clean_text(val) -> str:
    """Minimal text cleaning: unescape HTML entities and collapse whitespace."""
    if val is None:
        return ""
    text = html.unescape(str(val)).strip()
    text = " ".join(text.split())
    return text


def _yelp_process_attributes(attributes) -> List[str]:
    """Convert a business attributes dict into a list of descriptive sentences."""
    if not isinstance(attributes, dict):
        try:
            attributes = dict(attributes)
        except Exception:
            return []

    sentences: List[str] = []

    for attr_name, attr_value in attributes.items():
        if attr_value in ("False", False, "'False'", None, "None"):
            continue

        # Price-range shorthand (1-4 stars → $ symbols)
        clean_val = _yelp_clean_text(attr_value)
        if (
            re.search(r"price|range", attr_name, re.IGNORECASE)
            and re.match(r"^\d+(\.\d+)?$", clean_val)
            and 1 <= int(float(clean_val)) <= 4
        ):
            sentences.append(f"{attr_name}: {'$' * int(float(clean_val))}.")
            continue

        # Nested dict (or its string representation)
        nested = attr_value
        if isinstance(attr_value, str) and attr_value.startswith("{") and attr_value.endswith("}"):
            try:
                clean_str = attr_value.replace("u'", "'").replace('u"', '"')
                nested = eval(clean_str)
            except Exception:
                sentences.append(f"{attr_name} is {clean_val}.")
                continue

        if isinstance(nested, dict):
            true_keys: List[str] = []
            for k, v in nested.items():
                if v in (True, "True", "'True'"):
                    true_keys.append(k.replace("_", " "))
                elif v not in (False, "False", "'False'", None, "None"):
                    norm = _yelp_clean_text(v)
                    if norm:
                        sentences.append(f"{attr_name} {k.replace('_', ' ')} is {norm}.")
            if true_keys:
                keys_str = true_keys[0] if len(true_keys) == 1 else ", ".join(true_keys[:-1]) + f", and {true_keys[-1]}"
                sentences.append(f"{attr_name} has {keys_str}.")
            continue

        if attr_value in (True, "True", "'True'"):
            sentences.append(f"Has {attr_name.replace('_', ' ')}.")
            continue

        norm = _yelp_clean_text(attr_value)
        if norm:
            sentences.append(f"{attr_name.replace('_', ' ')} is {norm}.")

    return sentences


def _yelp_format_hours(hours) -> str:
    """Convert a business hours dict into a readable sentence."""
    if not isinstance(hours, dict) or not hours:
        return ""

    parts: List[str] = []
    for day, time_range in hours.items():
        if time_range == "0:0-0:0":
            continue
        formatted = time_range
        if re.match(r"\d+:\d+-\d+:\d+", time_range):
            try:
                start, end = time_range.split("-")
                sh, sm = map(int, start.split(":"))
                eh, em = map(int, end.split(":"))
                s_ampm = "am" if sh < 12 else "pm"
                e_ampm = "am" if eh < 12 else "pm"
                sh = sh % 12 or 12
                eh = eh % 12 or 12
                formatted = f"{sh}:{sm:02d}{s_ampm}-{eh}:{em:02d}{e_ampm}"
            except Exception:
                pass
        parts.append(f"{day}: {formatted}")

    return f"Hours: {', '.join(parts)}." if parts else ""


_YELP_FEATURES_NEEDED = [
    "name",
    "categories",
    "stars",
    "review_count",
    "city",
    "state",
    "attributes",
    "hours",
]

_YELP_SIMPLE_FORMATTERS = {
    "name": lambda v: f"Name: {_yelp_clean_text(v)}.",
    "city": lambda v: f"Located in {_yelp_clean_text(v)}.",
    "state": lambda v: f"State: {_yelp_clean_text(v)}.",
    "categories": lambda v: f"Categories: {', '.join(v.split(', '))}." if v else "",
    "stars": lambda v: f"Rating: {v} stars.",
    "review_count": lambda v: f"Number of reviews: {v}.",
}


def _yelp_format_row(row: pd.Series) -> str:
    """
    Build a descriptive text string from a single Yelp business metadata row.

    Covers: name, city, state, categories, stars, review_count, attributes, hours.
    Missing or null fields are silently skipped.
    """
    parts: List[str] = []

    for feature in _YELP_FEATURES_NEEDED:
        val = row.get(feature) if hasattr(row, "get") else getattr(row, feature, None)
        if val is None:
            continue
        try:
            if pd.isna(val):
                continue
        except (TypeError, ValueError):
            pass

        if feature == "attributes":
            attr_sentences = _yelp_process_attributes(val)
            if attr_sentences:
                parts.append(" ".join(attr_sentences))
        elif feature == "hours":
            h = _yelp_format_hours(val)
            if h:
                parts.append(h)
        else:
            fmt = _YELP_SIMPLE_FORMATTERS.get(feature)
            if fmt:
                text = fmt(val)
                if text:
                    parts.append(text)

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Registry: row-level formatters for datasets that cannot be expressed as a
# simple str.format_map template (e.g. because they contain nested dicts).
# A formatter receives a pd.Series (one metadata row) and returns a str.
# ---------------------------------------------------------------------------
DATASET_FORMATTERS: Dict[str, Callable[[pd.Series], str]] = {
    "yelp": _yelp_format_row,
}


def set_nested_attr(obj: SimpleNamespace, dotted_key: str, value) -> None:
    """Set an attribute on a (possibly nested) SimpleNamespace via dot notation.

    Examples::

        set_nested_attr(cfg, "seed", 42)
        set_nested_attr(cfg, "model.n_codebooks", 4)
        set_nested_attr(cfg, "emb_config.components", ["metadata", "description"])
    """
    parts = dotted_key.split(".")
    node = obj
    for part in parts[:-1]:
        node = getattr(node, part)
    setattr(node, parts[-1], value)


def _format_value(val) -> str:
    """Normalize a single cell value to a clean string."""
    if val is None:
        return ""
    try:
        if pd.isna(val):
            return ""
    except (ValueError, TypeError):
        pass
    if isinstance(val, (list, tuple)):
        return ", ".join(str(v) for v in val if v is not None)
    return str(val)


def _emb_config_slug(emb_config: SimpleNamespace) -> str:
    """Return a filesystem-safe slug that uniquely identifies a text embedding config.

    Examples:
        components=['metadata'], text_combine_method='concat_text'
        → 'text__metadata'

        components=['metadata', 'description'], text_combine_method='concat_text'
        → 'text__metadata_description__concat_text'

        entity='user', profiles=['taste', 'technical'], text_combine_method='concat_text'
        → 'user__taste_technical__concat_text'
    """
    entity = getattr(emb_config, "entity", "item")
    if entity == "user":
        profiles = getattr(emb_config, "profiles", list(USER_PROFILE_COMPONENTS))
        profiles_str = "_".join(profiles)
        combine = getattr(emb_config, "text_combine_method", "concat_text")
        if len(profiles) > 1:
            return f"user__{profiles_str}__{combine}"
        return f"user__{profiles_str}"

    method = getattr(emb_config, "source", "text")
    if method not in ("text", "hybrid"):
        return method

    components_str = "_".join(emb_config.components)
    if len(emb_config.components) > 1:
        combine = getattr(emb_config, "text_combine_method", "concat_text")
        return f"{method}__{components_str}__{combine}"
    else:
        return f"{method}__{components_str}"


def get_tok_emb_paths(
    paths: SimpleNamespace,
    sid_config: SimpleNamespace,
    seed: Optional[int] = None,
) -> Tuple[Path, Path, str]:
    """Get paths to tokenizer (directory) and embeddings (file).

    The directory layout is::

        tokenizers/semantic_ids/<slug>/<emb_model>/
        ├── embeddings.pt          ← shared embedding cache (emb_dir)
        └── <run_id>/              ← quantiser config hash
            ├── ...                ← seed-agnostic run (seed=None)
            └── seed_<N>/          ← per-seed subdirectory (when seed is given)
                ├── model_rqvae.pt
                ├── mapping_rqvae.pt
                └── run_info.json

    When *seed* is provided, a ``seed_<N>`` subdirectory is appended to
    ``tok_dir`` so that runs with different seeds coexist under the same config
    hash without overwriting each other.  Passing ``seed=None`` (default)
    reproduces the original flat layout for full backward compatibility.

    Args:
        paths:      Resolved filesystem paths (from ``handle_dirs``).
        sid_config: Tokenizer config ``SimpleNamespace`` (quantiser + emb).
        seed:       Optional integer seed.  When set, appends ``seed_<N>/``
                    beneath the ``<run_id>/`` directory.  The *run_id* hash is
                    never affected by the seed value.
    """
    emb_config = sid_config.emb_config
    emb_model_name = getattr(emb_config, "embedding_model", None).replace("/", "--")
    slug = _emb_config_slug(emb_config)

    run_id = generate_run_id(sid_config)
    emb_dir = Path(paths.tokenizers, "semantic_ids", slug, emb_model_name)
    tok_dir = get_tokenizer_run_dir(emb_dir, run_id)

    if seed is not None:
        tok_dir = tok_dir / f"seed_{seed}"

    return emb_dir, tok_dir, run_id


def generate_run_id(config: SimpleNamespace, exclude_keys: List[str] = ["base_config"]) -> str:
    """Return a 16-char deterministic hex ID for a tokenizer training config.

    Only quantizer hyper-parameters are hashed.

    The same config always produces the same ID, so re-running the same
    experiment lands in the same directory rather than creating a duplicate.

    Ignoring base_config by default due to being specific of explore_tokenizer.py and not relevant to the config.
    """
    config_dict = namespace2dict(config)
    for key in exclude_keys:
        config_dict.pop(key, None)
    config_str = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def get_tokenizer_run_dir(base_emb_path: Path, run_id: str) -> Path:
    """Return the per-run subdirectory beneath the shared embedding cache dir.

    Layout::

        <base_emb_path>/
        ├── embeddings.pt        ← shared, no run ID
        └── <run_id>/            ← returned by this function
            ├── model_rqvae.pt
            ├── mapping_rqvae.pt
            └── run_info.json
    """
    return Path(base_emb_path) / run_id


def save_run_registry(
    base_path: str,
    dataset: str,
    run_id: str,
    entry: Dict,
    seed: int,
    registry_stem: str = "registry",
) -> None:
    """Upsert a run entry into ``results/tokenizers/<dataset>/<registry_stem>_seed<N>.json``.

    The registry is a JSON object keyed by run ID.  Missing files are created
    automatically.  Concurrent writes are not protected — use only from a
    single process.

    Args:
        base_path:      Repository root.
        dataset:        Dataset name (e.g. ``'beauty'``).
        run_id:         16-char hex run identifier.
        entry:          Dict of metadata to store for this run.
        seed:           Random seed used for this run (appended to filename).
        registry_stem:  Filename stem (default ``'registry'``).  Use
                        ``'user_registry'`` for user-entity tokenizer runs so
                        item and user runs are stored in separate files.
    """
    machine_suffix = f"_{os.environ.get('MACHINE_ID', '')}" if "MACHINE_ID" in os.environ else ""
    registry_path = Path(
        base_path, "results", "tokenizers", dataset, f"{registry_stem}_seed{seed}{machine_suffix}.json"
    )
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    registry: Dict = {}
    if registry_path.exists():
        try:
            with open(registry_path) as f:
                registry = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    entry.setdefault("created_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    registry[run_id] = entry

    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=4)

    logger.info(f"Registry updated: {registry_path}")


def map_text_mode(dataset_name: str, df: pd.DataFrame) -> pd.Series:
    """
    Map metadata DataFrame columns into text using a dataset-specific template
    or row-level formatter.

    Two dispatch paths:

    1. **Row-level formatter** (``DATASET_FORMATTERS``): datasets with complex or
       nested metadata fields (e.g. Yelp ``attributes`` / ``hours``).  The
       callable receives a ``pd.Series`` (one row) and returns a ``str``.

    2. **String template** (``TEXT_TEMPLATES``): datasets where all referenced
       columns are flat strings.  Values are resolved via ``str.format_map`` in
       a single pass so column values are never re-substituted.

    Args:
        dataset_name: Key into ``DATASET_FORMATTERS`` or ``TEXT_TEMPLATES``
            (e.g. ``"yelp"``, ``"beauty"``).
        df: Metadata DataFrame whose columns are referenced in the template or
            consumed by the formatter.

    Returns:
        A pd.Series with one text string per row in df.
    """
    # Path 1: row-level formatter (complex / nested metadata)
    formatter = DATASET_FORMATTERS.get(dataset_name)
    if formatter is not None:
        return df.apply(formatter, axis=1)

    # Path 2: simple str.format_map template
    template = TEXT_TEMPLATES.get(dataset_name)
    if template is None:
        raise ValueError(
            f"No text template or formatter defined for dataset: '{dataset_name}'. "
            f"Add an entry to TEXT_TEMPLATES or DATASET_FORMATTERS in "
            f"nlp_xrs/semantic_ids/utils.py."
        )

    referenced_cols = [col for col in df.columns if f"{{{col}}}" in template]
    if not referenced_cols:
        raise ValueError(
            f"No DataFrame columns match template for '{dataset_name}'. "
            f"Template: '{template}', Columns: {list(df.columns)}"
        )

    clean_df = df[referenced_cols].apply(lambda col: col.map(_format_value))
    return clean_df.apply(lambda row: template.format_map(row), axis=1)


def load_tokenizer(tokenizer_path: Union[str, Path], config):
    """
    Load a trained tokenizer.

    Args:
        tokenizer_path: Path to tokenizer directory
        config: Configuration object

    Returns:
        Loaded tokenizer
    """
    from nlp_xrs.semantic_ids import SemanticIDTokenizer

    tokenizer = SemanticIDTokenizer(config)
    tokenizer.load(tokenizer_path)

    return tokenizer


def extract_item_embeddings(
    inter_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    train_ixs: List[int],
    mappers: Dict[str, Mapper],
    save_dir: Union[str, Path],
    emb_config: SimpleNamespace,
    **kwargs,
) -> torch.Tensor:
    """
    Extract item embeddings for tokenizer training.

    Args:
        inter_df:    Interaction DataFrame.
        metadata_df: Item metadata DataFrame.
        train_ixs:   Row indices that belong to the training split.
        mappers:     Mapper dict (Key.ITEM → Mapper, Key.EXP → Mapper, etc.).
        save_dir:    Directory used to cache computed embeddings on disk.
        emb_config:  Text-embedding settings. A :class:`~types.SimpleNamespace` with:

            * ``source`` – ``'text'``.
            * ``components`` *(List[str])* – ``'metadata'`` and/or ``'description'``.
            * ``text_combine_method`` *(str)* – ``'concat_text'``.
            * ``embedding_model`` *(str)* – sentence-transformer model name.

        **kwargs: Forwarded to the underlying extractor.  Notable keys:
            ``dataset_name`` *(str)* – required.
            ``device`` *(str)* – torch device string.

    Returns:
        Float tensor of shape ``(num_items, embedding_dim)``.
    """
    save_path = Path(save_dir) / "embeddings.pt"
    method = emb_config.source
    if method != "text":
        raise ValueError(f"Unknown embedding method: '{method}'. Valid option: 'text'.")
    if metadata_df is None:
        raise ValueError("metadata_df is required for text embedding method")
    return extract_text_embeddings(
        metadata_df=metadata_df,
        inter_df=inter_df,
        mappers=mappers,
        save_path=save_path,
        emb_config=emb_config,
        **kwargs,
    )


# ---------------------------------------------------------------------------
#  Private helpers for extract_text_embeddings
# ---------------------------------------------------------------------------


def _idx_to_raw(mapper: pd.Series | Mapper) -> pd.Series:
    """Return decoded strings indexed by encoded id.

    Accepts a ``Mapper`` (via ``idx2item['raw']``), that column as a DataFrame,
    or a Series that is already indexed by encoded id.
    """
    if isinstance(mapper, pd.Series):
        return mapper
    if isinstance(mapper, pd.DataFrame):
        if "raw" not in mapper.columns:
            raise TypeError("DataFrame mapper must contain a 'raw' column")
        return mapper["raw"]

    idx2item = getattr(mapper, "idx2item", None)
    if isinstance(idx2item, pd.DataFrame):
        if "raw" not in idx2item.columns:
            raise TypeError("mapper.idx2item must contain a 'raw' column")
        return idx2item["raw"]
    if isinstance(idx2item, pd.Series):
        return idx2item
    raise TypeError(f"Cannot decode texts from mapper of type {type(mapper).__name__}")


def _raw_to_idx(mapper: pd.Series | Mapper) -> pd.Series:
    """Return encoded ids indexed by raw item id.

    Prefers ``Mapper.item2idx``. A decoded Series (encoded id → raw string) is inverted.
    """
    item2idx = getattr(mapper, "item2idx", None)
    if isinstance(item2idx, pd.Series):
        return item2idx
    decoded = _idx_to_raw(mapper)
    return pd.Series(decoded.index, index=decoded.to_numpy())


def _get_entity_review_texts(
    inter_df: pd.DataFrame,
    mappers: Dict[str, pd.Series | Mapper],
    text_col: Optional[str] = None,
    entity: str = "item",
) -> pd.Series:
    """
    Decode review / explanation texts and group them by item.

    Returns:
        pd.Series indexed by encoded item-id, values are *lists* of decoded
        text strings (one per interaction).
    """
    entity_col = Key.ITEM if entity == "item" else Key.USER
    if text_col is None:
        text_col = Key.REV if Key.REV in inter_df.columns else Key.EXP
    mapper_key = text_col if text_col in mappers else Key.EXP
    decoded = _idx_to_raw(mappers[mapper_key])

    df = inter_df[[entity_col, text_col]].copy()

    if len(df) > 0 and isinstance(df[entity_col].iloc[0], (list, tuple)):
        df[entity_col] = df[entity_col].str[-1]

    if isinstance(df[text_col].iloc[0], list):
        # One joined string per interaction, then a list of those strings per entity.
        df[text_col] = df[text_col].map(
            lambda token_ids: " ".join(decoded.loc[list(token_ids)].astype(str).tolist()) if token_ids else ""
        )
        return df.groupby(entity_col)[text_col].agg(list)

    df[text_col] = decoded.loc[df[text_col].to_numpy()].tolist()
    texts = df.groupby(entity_col)[text_col].agg(list)
    texts[texts.isna()] = [[]] * texts.isna().sum()
    return texts


# Rating → text label mapping (shared across datasets; all use 1–5 scale).
_RATING_LABELS: Dict[int, str] = {
    1: "very low",
    2: "low",
    3: "neutral",
    4: "high",
    5: "very high",
}


def _get_user_item_metadata_texts(
    inter_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    dataset_name: str,
    max_items: Optional[int] = None,
) -> pd.Series:
    """
    For each user, collect the formatted metadata strings of their interacted
    items — annotated with a user-facing rating label — and return them as a
    list sorted chronologically (oldest → most recent), optionally capped at
    *max_items* (keeping the most recent ones).

    Each entry has the form::

        "<metadata text> (Rating: <label>)"

    where ``<label>`` is one of ``very low / low / neutral / high / very high``
    mapped from the raw 1–5 numeric rating.  Rows without a rating value
    (column missing or NaN) are included without the rating annotation.

    Args:
        inter_df:     Interaction DataFrame.  Must contain ``Key.USER``,
                      ``Key.ITEM``, and ``Key.RATING`` (ratings are required
                      for user taste profile generation).  ``Key.TIMESTEP``
                      is used for chronological ordering when present.
        metadata_df:  Item metadata DataFrame indexed by encoded item-id.
        dataset_name: Dataset key passed to :func:`map_text_mode`.
        max_items:    Maximum number of items per user (most recent).  ``None``
                      keeps all.

    Returns:
        pd.Series indexed by encoded user-id, values are *lists* of annotated
        metadata text strings (one per interacted item, chronological order).
    """
    has_timestep = Key.TIMESTEP in inter_df.columns

    assert Key.RATING in inter_df.columns, "Rating column is required for user taste profile generation"
    keep_cols = [Key.USER, Key.ITEM, Key.RATING]

    df = inter_df[keep_cols].copy()

    if has_timestep:
        df = inter_df.sort_values(Key.TIMESTEP, ascending=True)[keep_cols].copy()

    if len(df) > 0 and isinstance(df[Key.USER].iloc[0], (list, tuple)):
        df[Key.USER] = df[Key.USER].str[0]
    if len(df) > 0 and isinstance(df[Key.ITEM].iloc[0], (list, tuple)):
        df[Key.ITEM] = df[Key.ITEM].str[0]

    if len(df) > 0 and isinstance(df[Key.RATING].iloc[0], (list, tuple)):
        df[Key.RATING] = df[Key.RATING].str[0]

    df[Key.RATING] = df[Key.RATING].astype(float).clip(1, 5).round(0).astype(int).map(_RATING_LABELS)

    item_meta_texts = map_text_mode(dataset_name, metadata_df)
    if Key.ITEM in metadata_df.columns:
        item_meta_texts = pd.Series(item_meta_texts.values, index=metadata_df[Key.ITEM].values)

    valid_item_ids = set(item_meta_texts.index.tolist())

    def _collect_item_texts(group: pd.DataFrame) -> List[str]:
        group = group[group[Key.ITEM].isin(valid_item_ids)]
        if max_items is not None and group.shape[0] > max_items:
            group = group.iloc[-max_items:]
        meta = item_meta_texts.loc[group[Key.ITEM].values] + "; Rating: " + group[Key.RATING].values
        return meta.tolist()

    return df.groupby(Key.USER).apply(_collect_item_texts)


def _build_metadata_texts(
    metadata_df: pd.DataFrame,
    mappers: Dict[str, pd.Series | Mapper],
    dataset_name: Optional[str] = None,
    **kwargs,
) -> pd.Series:
    """
    Build a text string per item from metadata columns via the dataset template.

    Returns:
        pd.Series indexed by *encoded* item-id, values are template-filled
        text strings.
    """
    if dataset_name is None:
        raise ValueError("dataset_name kwarg is required for 'metadata' text embedding mode")

    texts = map_text_mode(dataset_name, metadata_df)

    if Key.ITEM in metadata_df.columns:
        item_ids = metadata_df[Key.ITEM]
        texts = pd.concat([texts, item_ids], axis=1).set_index(Key.ITEM).squeeze()

    # Integer item ids are already encoded. String ids are mapped through item2idx.
    if Key.ITEM in mappers and texts.index.dtype == object:
        item2idx = _raw_to_idx(mappers[Key.ITEM])
        texts = texts.loc[texts.index.isin(item2idx.index)]
        texts.index = texts.index.map(item2idx)

    return texts


def _encode_texts_to_item_embeddings(
    model,
    item_texts: pd.Series,
    num_items: int,
    batch_size: int = 64,
) -> torch.Tensor:
    """
    Encode a Series of text strings into a dense item-embedding tensor.

    Args:
        model: ``SentenceTransformer`` instance.
        item_texts: pd.Series indexed by item-id, values are text strings.
        num_items: Total number of items (tensor rows).

    Returns:
        ``torch.Tensor`` of shape ``(num_items, embedding_dim)``.
    """
    texts_list = item_texts.values.tolist()
    devices = None
    if torch.cuda.is_available():
        # NOTE: Multi-GPU support is added in newer versions of sentence-transformers
        # devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        devices = "cuda:0"

    embeddings = model.encode(
        texts_list,
        show_progress_bar=True,
        convert_to_tensor=True,
        batch_size=batch_size,
        device=devices,
        # chunk_size=max(batch_size, 5000),
    )

    item_embeddings = torch.zeros(num_items, embeddings.shape[1], dtype=embeddings.dtype)
    item_embeddings[item_texts.index.tolist()] = embeddings.cpu()
    return item_embeddings


def _embed_and_combine_per_item(
    model,
    item_review_texts: pd.Series,
    num_items: int,
    combine: str = "mean",
) -> torch.Tensor:
    """
    Embed each review individually and combine per item.

    Args:
        model: ``SentenceTransformer`` instance.
        item_review_texts: pd.Series  (item_id -> list[str]).
        num_items: Total number of items.
        combine: ``"mean"`` or ``"max"``.

    Returns:
        ``torch.Tensor`` of shape ``(num_items, embedding_dim)``.
    """
    all_texts: List[str] = []
    item_ids: List[int] = []
    for item_id, texts in item_review_texts.items():
        for text in texts:
            all_texts.append(text)
            item_ids.append(item_id)

    if not all_texts:
        raise ValueError("No review texts found for any item")

    all_embeddings = model.encode(all_texts, show_progress_bar=True, convert_to_tensor=True).cpu()
    emb_dim = all_embeddings.shape[1]

    item_embeddings = torch.zeros(num_items, emb_dim)
    item_counts = torch.zeros(num_items)

    for idx, item_id in enumerate(item_ids):
        if combine == "mean":
            item_embeddings[item_id] += all_embeddings[idx]
            item_counts[item_id] += 1
        elif combine == "max":
            if item_counts[item_id] == 0:
                item_embeddings[item_id] = all_embeddings[idx]
                item_counts[item_id] = 1
            else:
                item_embeddings[item_id] = torch.max(item_embeddings[item_id], all_embeddings[idx])

    if combine == "mean":
        nonzero = item_counts > 0
        item_embeddings[nonzero] /= item_counts[nonzero].unsqueeze(1)

    return item_embeddings


# ---------------------------------------------------------------------------
#  Main text-embedding extraction
# ---------------------------------------------------------------------------


def extract_text_embeddings(
    metadata_df: pd.DataFrame,
    inter_df: pd.DataFrame,
    mappers: Dict[str, Mapper],
    save_path: Union[str, Path],
    emb_config: SimpleNamespace,
    overwrite: bool = False,
    **kwargs,
) -> torch.Tensor:
    """
    Extract text-based item embeddings driven by a *emb_config* sub-config.

    The exact text content and how multiple sources are merged is controlled
    entirely by *emb_config* (see :func:`extract_item_embeddings` for the
    full field description).

    Args:
        metadata_df: Item metadata DataFrame.
        inter_df:    Interaction DataFrame.
        mappers:     Mapper dict for decoding encoded columns.
        save_path:   Path to cache computed embeddings.
        emb_config: Sub-config controlling which text components to use and
            how to combine them.  See :func:`extract_item_embeddings`.
        overwrite:   Re-compute even if a cache file already exists.
        **kwargs:
            dataset_name (str): Required for metadata and description text.
            test (bool): Append a test suffix to the cache filename.

    Returns:
        ``torch.Tensor`` of shape ``(num_items, embedding_dim)``.
    """
    from sentence_transformers import SentenceTransformer

    model_name = emb_config.embedding_model
    text_combine_method = getattr(emb_config, "text_combine_method", "concat_text")

    if os.path.isfile(save_path) and not overwrite:
        logging.info("Loading existing text embeddings from %s", save_path)
        return torch.load(save_path, weights_only=True, map_location="cpu")

    logging.info(
        "Computing text embeddings: components=%s, combine=%s, model=%s",
        emb_config.components,
        text_combine_method,
        model_name,
    )
    num_items = len(mappers[Key.ITEM])

    dataset_name = kwargs.pop("dataset_name", None)

    if text_combine_method == "concat_text":
        # Concatenate all component texts into one string per item, embed once.
        model = SentenceTransformer(model_name)
        if (max_length := getattr(emb_config, "max_seq_length", None)) is not None:
            model.max_seq_length = max_length
        texts_path = save_path.parent / "concat_text.json"
        texts_path.parent.mkdir(parents=True, exist_ok=True)
        item_texts = build_item_texts_from_components(
            components=emb_config.components,
            metadata_df=metadata_df,
            inter_df=inter_df,
            mappers=mappers,
            dataset_name=dataset_name,
            **{k: v for k, v in namespace2dict(emb_config).items() if k != "components"},
            **kwargs,
        )
        item_texts.to_json(texts_path)
        item_embeddings = _encode_texts_to_item_embeddings(
            model, item_texts, num_items, batch_size=getattr(emb_config, "batch_size", 64)
        )

    elif text_combine_method in ("concat_embed", "mean", "weighted_mean"):
        raise NotImplementedError("concat_embed, mean, and weighted_mean are not implemented yet")

    else:
        raise ValueError(
            f"Unknown text_combine_method: '{text_combine_method}'. Valid options: {TEXT_COMBINE_METHODS}."
        )

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(item_embeddings, save_path)
        logging.info("Saved text embeddings to %s", save_path)

    return item_embeddings


def get_scheduler(scheduler_cfg: SimpleNamespace, optimizer: torch.optim.Optimizer):
    """
    Get scheduler.

    Args:
        scheduler_cfg: Scheduler configuration
        optimizer: Optimizer

    Returns:
        Scheduler
    """
    if scheduler_cfg.name.lower() == "warmup_linear":
        warmup_steps = getattr(scheduler_cfg, "warmup_steps", 10)
        total_steps = getattr(scheduler_cfg, "total_steps", 100)
        min_ratio = getattr(scheduler_cfg, "min_ratio", 0.1)
        return WarmupLinearScheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps, min_ratio=min_ratio)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_cfg.name}")


def get_hidden_dim(model_id: str) -> Optional[int]:
    config = AutoConfig.from_pretrained(model_id)
    # Check common keys in order of prevalence
    for key in ["hidden_size", "d_model", "n_embd"]:
        if hasattr(config, key):
            return getattr(config, key)
    return None


# ---------------------------------------------------------------------------
# Component-based item-text builders
# ---------------------------------------------------------------------------


def _build_description_texts(
    metadata_df: pd.DataFrame,
    dataset_name: str,
    ref_index: Optional[pd.Index] = None,
) -> pd.Series:
    """
    Build a description text string per item from free-text metadata columns.

    Args:
        metadata_df:  Item metadata DataFrame.
        dataset_name: Dataset key used to look up ``DESCRIPTION_COLS``.
        ref_index:    Optional index to align the result (e.g. the index of
                      the metadata texts Series).

    Returns:
        pd.Series of description strings.  Items without a description get
        an empty string.
    """
    desc_cols = DESCRIPTION_COLS.get(dataset_name, [])
    avail = [c for c in desc_cols if c in metadata_df.columns]

    if not avail:
        logging.warning("No description columns found for '%s'; returning empty texts.", dataset_name)
        idx = ref_index if ref_index is not None else metadata_df.index
        return pd.Series("", index=idx)

    def _join(row):
        parts = [str(row[c]).strip() for c in avail if pd.notna(row[c]) and str(row[c]).strip()]
        return " ".join(parts)

    raw = metadata_df.apply(_join, axis=1)
    if ref_index is not None and len(raw) == len(ref_index):
        raw = pd.Series(raw.values, index=ref_index)
    not_empty = raw.notna() & (raw != "")
    raw[not_empty] = "Description: " + raw[not_empty]
    return raw


def build_item_texts_from_components(
    components: List[str],
    metadata_df: pd.DataFrame,
    inter_df: pd.DataFrame,
    mappers: Dict[str, Mapper],
    dataset_name: str,
    **kwargs,
) -> pd.Series:
    """
    Build one text string per item by concatenating metadata and description text.

    * ``'metadata'``    – structured metadata via the dataset template.
    * ``'description'`` – free-text description column(s).

    Component texts share the same item-id index and are joined with
    :data:`FIELD_SEPARATOR` (``"; "``).

    Returns:
        pd.Series indexed by encoded item-id, values are text strings.
    """
    if not components:
        raise ValueError("At least one component must be listed in emb_config.components")
    unknown = set(components) - set(TEXT_COMPONENTS)
    if unknown:
        raise ValueError(f"Unknown text component(s): {unknown}. Valid: {TEXT_COMPONENTS}")

    parts: List[pd.Series] = []

    for component in components:
        if component == "metadata":
            parts.append(_build_metadata_texts(metadata_df, mappers, dataset_name=dataset_name))

        elif component == "description":
            ref_idx = parts[0].index if parts else None
            parts.append(_build_description_texts(metadata_df, dataset_name, ref_index=ref_idx))

        else:
            raise ValueError(f"Unknown text component: '{component}'. Valid: {TEXT_COMPONENTS}")

    combined = parts[0]
    for p in parts[1:]:
        combined = combined.str.cat(p.reindex(combined.index).fillna(""), sep=FIELD_SEPARATOR)
    return combined


# ---------------------------------------------------------------------------
# User profile text builders
# ---------------------------------------------------------------------------


def _user_profile_cache_path(
    cache_dir: Union[str, Path],
    profile: str,
    llm_model: str,
    max_items_or_reviews: int,
    max_tokens_per_review: int,
    max_output_tokens: int,
    temperature: float,
) -> Path:
    """Return a deterministic JSON path for a cached user profile component.

    Args:
        cache_dir:              Directory under which cache files are stored.
        profile:                ``'taste'`` or ``'technical'``.
        llm_model:              HF model id or local path.
        max_items_or_reviews:   ``max_items`` (taste) or ``max_reviews``
                                (technical) — whichever applies.
        max_tokens_per_review:    Max input tokens per prompt.
        max_output_tokens:      Max new tokens generated per output.
        temperature:            Sampling temperature.

    Returns:
        :class:`pathlib.Path` pointing to the cache JSON file.
    """
    model_slug = llm_model.replace("/", "--")
    filename = f"llm_{profile}__user__{model_slug}__{max_items_or_reviews}__{str(temperature).replace('.', ',')}__{max_tokens_per_review}__{max_output_tokens}.json"
    return Path(cache_dir) / filename


def _load_user_profile_cache(cache_path: Union[str, Path]) -> Optional[pd.Series]:
    """Load a cached user profile Series from *cache_path*, or return ``None``.

    Args:
        cache_path: Path to a parquet file produced by ``prepare_ui_llm_desc.py``.

    Returns:
        Restored ``pd.Series`` (user-id index, string values) or ``None``.
    """
    path = Path(cache_path)
    if not path.is_file():
        return None

    with open(path, "r") as f:
        data = json.load(f)
    series = pd.Series({int(k): v for k, v in data.items()})
    logger.info("Loaded user profile cache from %s (%d users)", path, len(series))
    return series


def build_user_profile_texts(
    profiles: List[str],
    inter_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    mappers: Dict[str, Mapper],
    dataset_name: str,
    llm_model: Optional[str] = None,
    max_items: int = 20,
    max_reviews: int = 20,
    llm_cache_dir: Optional[Union[str, Path]] = None,
    **kwargs,
) -> pd.Series:
    """
    Build one text string per user by concatenating the requested profile
    components.

    Each *profile* adds a piece of text to the user representation:

    * ``'taste'``     – LLM-generated product-preference profile based on the
                        user's interaction history metadata.
    * ``'technical'`` – LLM-generated reviewing-style profile based on the
                        user's raw review texts.

    Profile texts are joined with :data:`FIELD_SEPARATOR` (``"; "``).  Each
    component is cached independently as a JSON file under *llm_cache_dir*
    (when provided), keyed by the LLM model and generation parameters so the
    same output is reused across embedding-model or quantizer ablation runs.

    Args:
        profiles:       Ordered list of profile names (``'taste'``,
                        ``'technical'``).
        inter_df:       Interaction DataFrame (must include ``Key.USER``,
                        ``Key.ITEM``, ``Key.REV`` or ``Key.EXP``, and
                        ``Key.TIMESTEP`` if available).
        metadata_df:    Item metadata DataFrame for building item text strings.
        mappers:        Mapper dict for decoding encoded columns.
        dataset_name:   Dataset key (used by :func:`map_text_mode`).
        llm_model:      HF model id; used to construct the expected cache path.
        max_items:      Max interacted items per user for the ``'taste'``
                        profile (used for cache-path construction).
        max_reviews:    Max reviews per user for the ``'technical'`` profile
                        (used for cache-path construction).
        llm_cache_dir:  Required directory containing the pre-generated JSON
                        cache files (produced by ``prepare_ui_llm_desc.py``).
                        A :exc:`FileNotFoundError` is raised for any missing
                        file.
        **kwargs:       Generation parameters used when constructing the
                        expected cache path (``max_prompt_tokens``,
                        ``max_output_tokens``, ``temperature``).

    Returns:
        pd.Series indexed by encoded user-id, values are text strings.
    """
    if not profiles:
        raise ValueError("At least one profile must be listed in emb_config.profiles")
    unknown = set(profiles) - set(USER_PROFILE_COMPONENTS)
    if unknown:
        raise ValueError(f"Unknown user profile(s): {unknown}. Valid: {USER_PROFILE_COMPONENTS}")

    if llm_model is None:
        llm_model = DEFAULT_LLM_MODEL

    _max_tokens_per_review: int = kwargs.get("max_tokens_per_review", 200)
    _max_output_tokens: int = kwargs.get("max_output_tokens", 512)
    _temperature: float = kwargs.get("temperature", 0.7)

    parts: List[pd.Series] = []

    for profile in profiles:
        _max_count = max_items if profile == "taste" else max_reviews
        if llm_cache_dir is None:
            raise ValueError(
                f"Profile '{profile}' requires 'llm_cache_dir' to be set in emb_config. "
                f"Add it to your config, then generate the descriptions by running:\n"
                f"  python prepare_ui_llm_desc.py"
                f" --entity-type user --desc-type {profile}"
                f" --llm-model {llm_model} --max-reviews-per-entity {_max_count}"
                f" --temperature {_temperature}"
                f" --max-tokens-per-review {_max_tokens_per_review}"
                f" --max-output-tokens {_max_output_tokens}"
                f" --dataset {dataset_name} [--base-path <base> --fold <fold> --test]"
            )
        _cache_path = _user_profile_cache_path(
            llm_cache_dir,
            profile,
            llm_model,
            _max_count,
            _max_tokens_per_review,
            _max_output_tokens,
            _temperature,
        )
        cached = _load_user_profile_cache(_cache_path)
        if cached is None:
            raise FileNotFoundError(
                f"Cache file '{_cache_path}' not found. Generate it first by running:\n"
                f"  python prepare_ui_llm_desc.py"
                f" --entity-type user --desc-type {profile}"
                f" --llm-model {llm_model} --max-reviews-per-entity {_max_count}"
                f" --temperature {_temperature}"
                f" --max-tokens-per-review {_max_tokens_per_review}"
                f" --max-output-tokens {_max_output_tokens}"
                f" --dataset {dataset_name} [--base-path <base> --fold <fold> --test]\n"
                f"which writes {_cache_path.name} under <base>/data/<Dataset>/{DATA_MODE}/<fold>/tokenizers/semantic_ids/"
            )
        parts.append(cached)

    combined = parts[0]
    for p in parts[1:]:
        combined = combined.str.cat(p.reindex(combined.index).fillna(""), sep=FIELD_SEPARATOR)
    return combined


def extract_user_embeddings(
    inter_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    train_ixs: List[int],
    mappers: Dict[str, Mapper],
    save_dir: Union[str, Path],
    emb_config: SimpleNamespace,
    overwrite: bool = False,
    **kwargs,
) -> torch.Tensor:
    """
    Extract user embeddings for tokenizer training.

    User embeddings are derived from LLM-generated profile texts (taste and/or
    technical) that are encoded with a SentenceTransformer model.  The result
    is cached at ``save_dir/user_embeddings.pt`` and shared across quantizer
    runs that use the same embedding config.

    Profile texts must be pre-generated by ``prepare_ui_llm_desc.py`` and cached
    as JSON files.  They live at::

        <save_dir>/../../user_<profile>_texts__<llm_slug>__<params>.json

    i.e. one level above the embedding-model-specific directory, shared across
    embedding-model ablation runs.

    Args:
        inter_df:    Interaction DataFrame.
        metadata_df: Item metadata DataFrame (required for the ``'taste'``
                     profile).
        train_ixs:   Row indices belonging to the training split (reserved for
                     future use; user embeddings currently use all interactions).
        mappers:     Mapper dict for decoding encoded columns.
        save_dir:    Directory used to cache the computed embedding tensor.
                     ``user_embeddings.pt`` is written here.
        emb_config: Sub-config controlling which profiles to use.
            Required fields:

            * ``profiles`` *(List[str])* – ``'taste'`` and/or ``'technical'``.
            * ``embedding_model`` *(str)* – SentenceTransformer model name.
            * ``text_combine_method`` *(str)* – only ``'concat_text'`` is
              supported; the two profile texts are concatenated before encoding.
            * ``llm_model`` *(str, optional)* – HF model id for profile
              generation; defaults to :data:`DEFAULT_LLM_MODEL`.
            * ``max_items`` *(int, optional)* – max items per user for the
              ``'taste'`` profile (default 20).
            * ``max_reviews`` *(int, optional)* – max reviews per user for
              the ``'technical'`` profile (default 20).

        overwrite:   Re-compute even if the embedding cache file already exists.
        **kwargs:    Forwarded to :func:`build_user_profile_texts`
                     (``dataset_name``, ``max_prompt_tokens``,
                     ``max_output_tokens``, ``temperature``, etc.).

    Returns:
        Float tensor of shape ``(num_users, embedding_dim)``.
    """
    from sentence_transformers import SentenceTransformer

    save_dir = Path(save_dir)
    save_path = save_dir / "user_embeddings.pt"

    if save_path.is_file() and not overwrite:
        logger.info("Loading existing user embeddings from %s", save_path)
        return torch.load(save_path, weights_only=True, map_location="cpu")

    model_name = emb_config.embedding_model
    profiles = getattr(emb_config, "profiles", list(USER_PROFILE_COMPONENTS))
    llm_model = getattr(emb_config, "llm_model", None)
    max_items = getattr(emb_config, "max_items", 20)
    max_reviews = getattr(emb_config, "max_reviews", 20)
    max_tokens_per_review = getattr(emb_config, "max_tokens_per_review", 200)
    max_output_tokens = getattr(emb_config, "max_output_tokens", 512)
    temperature = getattr(emb_config, "temperature", 0.7)

    # Cache profile texts one level above the emb-model dir so they are
    # shared across embedding-model ablation runs.
    llm_cache_dir = save_dir.parent.parent

    logger.info(
        "Computing user embeddings: profiles=%s, model=%s, llm=%s",
        profiles,
        model_name,
        llm_model,
    )

    user_texts = build_user_profile_texts(
        profiles=profiles,
        inter_df=inter_df,
        metadata_df=metadata_df,
        mappers=mappers,
        llm_model=llm_model,
        max_items=max_items,
        max_reviews=max_reviews,
        llm_cache_dir=llm_cache_dir,
        max_tokens_per_review=max_tokens_per_review,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        **kwargs,
    )

    num_users = len(mappers[Key.USER])
    model = SentenceTransformer(model_name)
    if (max_length := getattr(emb_config, "max_seq_length", None)) is not None:
        model.max_seq_length = max_length

    user_embeddings = _encode_texts_to_item_embeddings(
        model,
        user_texts,
        num_users,
        batch_size=getattr(emb_config, "batch_size", 64),
    )

    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(user_embeddings, save_path)
    logger.info("Saved user embeddings to %s", save_path)

    return user_embeddings


def get_sae_dists(reviews: pd.Series, acts_mapper: pd.DataFrame, sae_codebook_size: int) -> torch.Tensor:
    review_acts = acts_mapper.loc[reviews.values]
    label_dist = torch.zeros(review_acts.shape[0], sae_codebook_size, device="cpu")
    label_dist.scatter_(
        1, torch.tensor(review_acts[Key.CONCEPT_ID].tolist()), torch.tensor(review_acts["act"].tolist())
    )
    return label_dist.mean(0).cpu()
