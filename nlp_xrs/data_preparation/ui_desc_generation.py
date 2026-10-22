"""
vLLM-based LLM text generation for pre-computing user-profile caches.

This module is intentionally isolated from the main training framework so that
it can be run in a separate conda environment that has vLLM installed.

Entry-point: :func:`generate_ui_descriptions` — called by ``prepare_ui_llm_desc.py``.

Supported description types:

* ``'taste'``      – product-preference profile from interaction history.
* ``'technical'``  – reviewing-style profile from raw review texts.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from nlp_xrs.utils.in_out import load_mapper_parquet, load_partition, load_safe_parquet

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from nlp_xrs.semantic_ids.utils import (
    _get_entity_review_texts,
    _get_user_item_metadata_texts,
)
from nlp_xrs.utils.constants import DATA_MAP, DATA_MODE, PACKAGE_NAME, Key, Suffix

logger = logging.getLogger(PACKAGE_NAME)

USER_DESC_TYPES = ("taste", "technical")

FEW_SHOT_EXAMPLES = {
    "beauty": {
        "taste": (
            "BAD: 'This user enjoys beauty products and rates most items highly.'\n"
            "GOOD: 'This user consistently gravitates toward hydrating skincare — vitamin C serums, "
            "oil-based moisturizers, and sheet masks — rated 4-5 stars, while avoiding color cosmetics "
            "under $10 which received low ratings. Brand loyalty is low but ingredient loyalty is high.'"
        ),
        "technical": (
            "BAD: 'This user focuses on quality and usability with a positive sentiment.'\n"
            'GOOD: \'This user repeatedly flags "pilling under makeup" and "broke me out" as dealbreakers, '
            'while praising "sinks in fast" and "fragrance-free". They qualify positives with skin-type '
            'caveats ("works for dry skin but...") and rarely recommend without conditions.\''
        ),
    },
    "cell": {
        "taste": (
            "BAD: 'This user buys phone accessories and prefers affordable options.'\n"
            "GOOD: 'This user focuses on protective cases and charging accessories in the $10-$25 range, "
            'consistently rating MFi-certified cables highly while penalizing items that "stopped working '
            "after 2 weeks\". Strong preference for Android-compatible products over Apple ecosystem.'"
        ),
        "technical": (
            "BAD: 'This user comments on build quality and value for money.'\n"
            'GOOD: \'This user anchors critiques to durability timelines ("lasted 3 months", "still going '
            'strong after a year") and uses hands-on phrasing like "feels flimsy" or "snaps in perfectly". '
            "They compare products to previous purchases and flag compatibility issues specifically.'"
        ),
    },
    "yelp": {
        "taste": (
            "BAD: 'This user enjoys dining out and prefers good value restaurants.'\n"
            "GOOD: 'This user favors casual Asian fusion and brunch spots in the $15-$30/person range, "
            "rating highly when portions are large and service is fast. They penalize long waits heavily "
            "regardless of food quality, and show loyalty to 2-3 recurring venues.'"
        ),
        "technical": (
            "BAD: 'This user mentions food quality and service in their reviews.'\n"
            'GOOD: \'This user leads with atmosphere descriptors ("loud", "cozy", "cramped") before '
            'food, and uses comparative framing ("better than X nearby"). Criticism is specific: '
            '"soggy fries", "server forgot our order twice" — never vague. Sentiment is critical '
            "even in 4-star reviews.'"
        ),
    },
}


# ---------------------------------------------------------------------------
# vLLM engine helpers
# ---------------------------------------------------------------------------


def _load_vllm(llm_model: str, **kwargs):
    """Load a vLLM ``LLM`` instance.

    Args:
        llm_model: HuggingFace model id or local path.
        **kwargs: Forwarded to ``vllm.LLM`` (``tensor_parallel_size``,
            ``gpu_memory_utilization``, ``dtype``, ``trust_remote_code``,
            ``max_model_len``).

    Returns:
        ``vllm.LLM`` instance.
    """
    try:
        from vllm import LLM
    except ImportError:
        raise ImportError("vllm is required for LLM-based text generation. Install it with: pip install vllm")

    vllm_kwargs = {
        k: kwargs[k]
        for k in (
            "tensor_parallel_size",
            "gpu_memory_utilization",
            "dtype",
            "trust_remote_code",
            "max_model_len",
        )
        if k in kwargs
    }
    logger.info("Loading vLLM model: %s", llm_model)
    return LLM(model=llm_model, **vllm_kwargs)


def _truncate_to_tokens(texts: List[str], tokenizer, max_tokens: int, truncation_side: str = "right") -> List[str]:
    """Truncate *text* to at most *max_tokens* tokens and decode back to a string.

    Only the token sequence is shortened; the surrounding prompt template is
    never touched, so instruction text is always preserved.

    Args:
        texts: Input strings to truncate.
        tokenizer: HuggingFace tokenizer (obtained from ``llm.get_tokenizer()``).
        max_tokens: Maximum number of tokens to keep.
        truncation_side: Side to truncate from (default "right").
    Returns:
        Truncated strings (original strings if they fit within *max_tokens*).
    """
    tokenizer.truncate_side = truncation_side
    ids = tokenizer(
        texts,
        add_special_tokens=False,
        truncation=True,
        max_length=max_tokens,
        padding=False,
        return_attention_mask=False,
    )
    return tokenizer.batch_decode(ids["input_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=False)


def _vllm_generate_batch(
    llm,
    prompts: List[str],
    max_output_tokens: int = 512,
    temperature: float = 0.3,
    stop: Optional[List[str]] = None,
) -> List[str]:
    """Run batched generation on a list of plain-text prompts.

    Prompts are formatted with the model's chat template (if available) and
    passed directly to vLLM without any truncation — history content is
    already bounded per-entry before the prompt is assembled.

    Args:
        llm: A ``vllm.LLM`` instance.
        prompts: List of user-turn strings.
        max_output_tokens: Maximum new tokens generated per prompt.
        temperature: Sampling temperature.
        stop: Optional stop strings.

    Returns:
        List of generated text strings (one per prompt).
    """
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        max_tokens=max_output_tokens,
        temperature=temperature,
        stop=stop or [],
    )

    tokenizer = llm.get_tokenizer()
    has_chat_template = getattr(tokenizer, "chat_template", None) is not None

    logging.info("Preparing %d prompts for vLLM generation", len(prompts))
    formatted: List[dict] = []
    if has_chat_template:
        for p in prompts:
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=True,
                add_generation_prompt=True,
            )
            formatted.append({"prompt_token_ids": ids})
    else:
        for p in prompts:
            ids = tokenizer.encode(p, add_special_tokens=False)
            formatted.append({"prompt_token_ids": ids})

    logging.info("Generated %d prompts for vLLM generation", len(formatted))
    outputs = llm.generate(formatted, sampling_params)
    logging.info("Generated %d outputs from vLLM", len(outputs))
    return [out.outputs[0].text.strip() for out in outputs]


# ---------------------------------------------------------------------------
# LLM generation functions (item)
# ---------------------------------------------------------------------------


def build_taste_prompt(items_text: str, dataset: str) -> str:
    examples = FEW_SHOT_EXAMPLES[dataset]["taste"]
    return (
        "You are analyzing a user's purchasing history listed chronologically "
        "from oldest to most recent. Each entry includes the product name, category, "
        "price, and the user's rating.\n\n"
        f"Products:\n- {items_text}\n\n"
        f"Example of bad vs good profile for reference:\n{examples}\n\n"
        "Write 2-3 sentences describing this user's product preferences. "
        "You MUST:\n"
        "- Reference specific product types or attributes they gravitate toward\n"
        "- Mention price sensitivity using actual values from the data\n"
        "- Use keywords that ONLY describe this specific user\n"
        "- Ground claims in ratings: only call something 'preferred' if rated highly\n"
        "User taste profile:"
    )


def build_technical_prompt(reviews_text: str, dataset: str) -> str:
    examples = FEW_SHOT_EXAMPLES[dataset]["technical"]
    return (
        "You are analyzing the reviewing style of a single user based on their product reviews.\n\n"
        f"Reviews:\n- {reviews_text}\n\n"
        f"Example of bad vs good profile for reference:\n{examples}\n\n"
        "Write 2-3 sentences describing this user's reviewing style. "
        "You MUST:\n"
        "- Use EXACT recurring words or phrases from the user's own reviews\n"
        "- Identify which aspects they criticize vs praise in their own vocabulary\n"
        "- Characterize their sentiment pattern specifically (e.g. 'qualifies praise with caveats')\n"
        "- Note user-specific quirks: comparisons, context mentions, conditions\n"
        "User technical profile:"
    )


# ---------------------------------------------------------------------------
# LLM generation functions (user)
# ---------------------------------------------------------------------------


def _generate_user_taste_profiles(
    user_item_meta: pd.Series,
    llm_model: str = "Qwen/Qwen2.5-7B-Instruct",
    max_items: int = 20,
    dataset_name: str = "beauty",
    **kwargs,
) -> pd.Series:
    """Produce a product-taste profile per user using a vLLM-hosted model.

    Each item-metadata entry is individually truncated to ``max_tokens_per_review``
    tokens before being included in the prompt, so the instruction text is never cut.

    Args:
        user_item_meta: Series (user_id → list[str]) of formatted item metadata.
        llm_model: HuggingFace model id or local path.
        max_items: Maximum items to include per user in the prompt.
        dataset_name: Dataset name (default "beauty").
        **kwargs:
            vllm_engine: Pre-loaded ``vllm.LLM`` instance.
            max_tokens_per_review (int): Max tokens per individual item entry (default 200).
            max_output_tokens (int): default 256.
            temperature (float): default 0.3.

    Returns:
        Series indexed by user-id, values are taste profile strings.
    """
    if (llm := kwargs.get("vllm_engine")) is None:
        llm = _load_vllm(llm_model, **kwargs)
    max_tokens_per_review: int = kwargs.get("max_tokens_per_review", 200)
    max_output_tokens: int = kwargs.get("max_output_tokens", 256)
    temperature: float = kwargs.get("temperature", 0.3)

    tokenizer = llm.get_tokenizer()
    user_ids = list(user_item_meta.index)
    prompts: List[str] = []
    fallbacks: Dict[int, str] = {}

    for uid in user_ids:
        items = user_item_meta.loc[uid][-max_items:]
        if not items:
            fallbacks[uid] = ""
            prompts.append("")
            continue
        truncated = _truncate_to_tokens(items, tokenizer, max_tokens_per_review)
        items_text = "\n- ".join(truncated)
        prompts.append(build_taste_prompt(items_text, dataset_name))

    valid_indices = [i for i, p in enumerate(prompts) if p]
    valid_prompts = [prompts[i] for i in valid_indices]

    logger.info(
        "Generating taste profiles for %d users with %s ...",
        len(valid_prompts),
        llm_model,
    )
    try:
        generated = _vllm_generate_batch(llm, valid_prompts, max_output_tokens, temperature)
    except Exception as e:
        logger.warning("vLLM taste profile generation failed: %s. Falling back to item list.", e)
        generated = [" | ".join(user_item_meta.loc[user_ids[i]][-max_items:]) for i in valid_indices]

    profiles: Dict[int, str] = dict(fallbacks)
    for gen_idx, data_idx in enumerate(valid_indices):
        profiles[user_ids[data_idx]] = generated[gen_idx]

    return pd.Series(profiles)


def _generate_user_technical_profiles(
    user_reviews: pd.Series,
    llm_model: str = "Qwen/Qwen2.5-7B-Instruct",
    max_reviews: int = 20,
    dataset_name: str = "beauty",
    **kwargs,
) -> pd.Series:
    """Produce a technical reviewing-style profile per user using a vLLM model.

    Each review is individually truncated to ``max_tokens_per_review`` tokens
    before being included in the prompt, so the instruction text is never cut.

    Args:
        user_reviews: Series (user_id → list[str]) of raw review texts.
        llm_model: HuggingFace model id or local path.
        max_reviews: Maximum reviews to include per user.
        dataset_name: Dataset name (default "beauty").
        **kwargs:
            vllm_engine: Pre-loaded ``vllm.LLM`` instance.
            max_tokens_per_review (int): Max tokens per individual review (default 200).
            max_output_tokens (int): default 256.
            temperature (float): default 0.3.

    Returns:
        Series indexed by user-id, values are technical profile strings.
    """
    if (llm := kwargs.get("vllm_engine")) is None:
        llm = _load_vllm(llm_model, **kwargs)
    max_tokens_per_review: int = kwargs.get("max_tokens_per_review", 200)
    max_output_tokens: int = kwargs.get("max_output_tokens", 256)
    temperature: float = kwargs.get("temperature", 0.3)

    tokenizer = llm.get_tokenizer()
    user_ids = list(user_reviews.index)
    prompts: List[str] = []
    fallbacks: Dict[int, str] = {}

    logger.info("Generating prompts...")
    for uid in user_ids:
        reviews = user_reviews.loc[uid]
        if isinstance(reviews, str):
            reviews = [reviews] if reviews else []
        reviews = reviews[:max_reviews]
        if not reviews:
            fallbacks[uid] = ""
            prompts.append("")
            continue
        truncated = _truncate_to_tokens(reviews, tokenizer, max_tokens_per_review)
        reviews_text = "\n- ".join(truncated)
        prompts.append(build_technical_prompt(reviews_text, dataset_name))

    valid_indices = [i for i, p in enumerate(prompts) if p]
    valid_prompts = [prompts[i] for i in valid_indices]

    logger.info(
        "Generating technical profiles for %d users with %s ...",
        len(valid_prompts),
        llm_model,
    )
    try:
        generated = _vllm_generate_batch(llm, valid_prompts, max_output_tokens, temperature)
    except Exception as e:
        logger.warning(
            "vLLM technical profile generation failed: %s. Falling back to review concatenation.",
            e,
        )
        generated = [" ".join(user_reviews.loc[user_ids[i]][:max_reviews]) for i in valid_indices]

    profiles: Dict[int, str] = dict(fallbacks)
    for gen_idx, data_idx in enumerate(valid_indices):
        profiles[user_ids[data_idx]] = generated[gen_idx]

    return pd.Series(profiles)


def _save_user_descriptions(
    series: pd.Series,
    output_path: Union[str, Path],
) -> None:
    """Persist a per-user description *series* to *output_path* as JSON.

    Args:
        series: Series with int user-id index and string values.
        output_path: Destination path (parent directories are created).
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({str(k): v for k, v in series.items()}, f, ensure_ascii=False)
    logger.info("Saved user descriptions to %s (%d users)", path, len(series))


# ---------------------------------------------------------------------------
# Dataset loading helper
# ---------------------------------------------------------------------------


def _load_mappers(load_cols: List[str], mapper_path: str, suffix: str = "") -> Dict[str, pd.Series]:
    mappers: Dict[str, pd.Series] = {}
    for c in set(load_cols).intersection([Key.USER, Key.ITEM, Key.EXP, Key.FEAT, Key.REV]):
        mapper_file = os.path.join(mapper_path, f"{c}{suffix}.parquet")
        if os.path.isfile(mapper_file):
            mapper = load_mapper_parquet(mapper_file)
        else:
            raise FileNotFoundError(f"Mapper for column {c} not found at {mapper_file}")
        mappers[c] = mapper

    return mappers


def _load_dataset(
    dataset_name: str, fold: str = "0", base_path: str = ".", suffix: str = ""
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.Series], Dict[str, List[int]]]:
    """Load interaction data, metadata, mappers, and split indices for a dataset.

    Args:
        dataset_name: Dataset identifier (e.g. ``'beauty'``, ``'yelp'``).
        fold: Partition fold identifier (default ``'0'``), already suffixed.
        base_path: Root containing the ``data/`` tree.
        suffix: File suffix of the dataset variant (``'_mini'`` for the test subset).

    Returns:
        Tuple of ``(data_df, metadata_df, mappers, split_ixs)``.
    """
    data_dir = os.path.join(base_path, "data", dataset_name, DATA_MODE)
    data_df = load_safe_parquet(os.path.join(data_dir, f"reviews{suffix}.parquet"))
    metadata_df = load_safe_parquet(os.path.join(data_dir, f"metadata{suffix}.parquet"))
    split_ixs = load_partition(os.path.join(data_dir, fold))
    mappers = _load_mappers(
        load_cols=[Key.REV, Key.FEAT],
        mapper_path=os.path.join(data_dir, "mappers"),
        suffix=suffix,
    )
    return data_df, metadata_df, mappers, split_ixs


# ---------------------------------------------------------------------------
# Primary entry-point
# ---------------------------------------------------------------------------


def generate_ui_descriptions(
    dataset_name: str,
    entity_type: str,
    desc_type: str,
    llm_model: str,
    max_reviews_per_entity: int,
    temperature: float = 0.3,
    max_tokens_per_review: int = 200,
    max_output_tokens: int = 512,
    fold: str = "0",
    base_path: Optional[str] = None,
    test: bool = False,
    max_users: Optional[int] = None,
    **vllm_kwargs,
) -> None:
    """Generate LLM user profiles and write them to *output_path*.

    This is the primary entry-point called by ``prepare_ui_llm_desc.py``.

    Args:
        dataset_name: Dataset identifier (e.g. ``'beauty'``, ``'yelp'``).
        entity_type: ``'user'``.
        desc_type: Description variant:

            * ``'taste'``      — product-preference profile from interaction history.
            * ``'technical'``  — reviewing-style profile from raw review texts.

        llm_model: HuggingFace model id or local path.
        max_reviews_per_entity: Maximum number of reviews (items) to include per entity
            in the LLM prompt. For ``'taste'`` profiles this caps the item
            history; for all other types it caps the review list.
        temperature: Sampling temperature (default ``0.3``).
        max_tokens_per_review: Maximum tokens allowed for each individual review or
            item-metadata entry in the prompt (default ``200``). Only the history
            content is truncated; the instruction text is always preserved in full.
        max_output_tokens: Maximum new tokens per output (default ``512``).
        fold: Partition fold identifier (default ``'0'``).
        base_path: Root containing ``data/`` (default: current working directory).
        test: Use the ``_mini`` subset (fold ``<fold>_mini`` and ``*_mini`` tables).
        max_users: If set, only profile the first ``max_users`` users by id.
        **vllm_kwargs: Runtime parameters forwarded to :func:`_load_vllm`:
            ``tensor_parallel_size``, ``gpu_memory_utilization``,
            ``dtype``, ``trust_remote_code``, ``max_model_len``.

    Raises:
        ValueError: If ``entity_type`` or ``desc_type`` is not recognised.
        FileNotFoundError: If required dataset files are missing.
    """
    if entity_type != "user":
        raise ValueError(f"Unknown entity_type '{entity_type}'. Expected 'user'.")
    valid_types = USER_DESC_TYPES
    if desc_type not in valid_types:
        raise ValueError(
            f"Unknown desc_type '{desc_type}' for entity_type='{entity_type}'. Expected one of: {valid_types}."
        )

    base_path = base_path or "."
    suffix = Suffix.TEST if test else ""
    fold = f"{fold}{suffix}"
    full_dataset = DATA_MAP.get(dataset_name.lower(), dataset_name)
    output_path = Path(base_path) / "data" / full_dataset / DATA_MODE / fold / "tokenizers" / "semantic_ids"
    output_path /= f"llm_{desc_type}__{entity_type}__{llm_model.replace('/', '--')}__{max_reviews_per_entity}__{str(temperature).replace('.', ',')}__{max_tokens_per_review}__{max_output_tokens}.json"

    logger.info("Generating %s %s descriptions → %s", entity_type, desc_type, output_path)

    data_df, metadata_df, mappers, split_ixs = _load_dataset(full_dataset, fold=fold, base_path=base_path, suffix=suffix)
    train_df = data_df.iloc[split_ixs["train"] + split_ixs["valid"]]
    if max_users is not None:
        keep = sorted(train_df[Key.USER].unique())[:max_users]
        train_df = train_df[train_df[Key.USER].isin(keep)]
        logger.info("Restricting generation to %d users", len(keep))

    llm = _load_vllm(llm_model, **vllm_kwargs)
    gen_kwargs: Dict[str, Any] = dict(
        vllm_engine=llm,
        max_tokens_per_review=max_tokens_per_review,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
    )

    if desc_type == "taste":
        logger.info("Preparing user item metadata for taste profiles...")
        user_item_meta = _get_user_item_metadata_texts(
            train_df,
            metadata_df,
            dataset_name,
            max_items=max_reviews_per_entity,
        )
        result = _generate_user_taste_profiles(
            user_item_meta,
            llm_model=llm_model,
            max_items=max_reviews_per_entity,
            dataset_name=dataset_name,
            **gen_kwargs,
        )
    else:
        logger.info("Preparing user reviews for technical profiles...")
        user_reviews = _get_entity_review_texts(train_df, mappers, entity="user")
        result = _generate_user_technical_profiles(
            user_reviews,
            llm_model=llm_model,
            max_reviews=max_reviews_per_entity,
            dataset_name=dataset_name,
            **gen_kwargs,
        )
    _save_user_descriptions(result, output_path)

    logger.info("Descriptions written to %s", output_path)
