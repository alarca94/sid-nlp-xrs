"""
Standalone script for pre-generating LLM user-profile caches
required before user-side semantic-ID training.

The profiles are ``taste`` and ``technical``. The output file is written to::

    <base-path>/data/<Dataset>/feat_only/<fold>/tokenizers/semantic_ids/
        llm_<desc-type>__user__<model>__<max-reviews>__<temperature>__<max-tokens-per-review>__<max-output-tokens>.json

The exact command to run is also printed in the ``FileNotFoundError`` raised by
``train_tokenizer.py`` when a required cache file is missing.

Arguments
---------
--entity-type {user}
    Generate descriptions for users.
--desc-type {taste,technical}
    Description variant:
      taste      product-preference profile from interaction history.
      technical  reviewing-style profile from raw review texts.
--llm-model MODEL
    HuggingFace model id or local path (e.g. Qwen/Qwen2.5-32B-Instruct-AWQ).
--max-reviews-per-entity N
    Maximum reviews (items) to include per user in the LLM prompt (default: 15).
    For taste profiles this caps the item history; for technical profiles it
    caps the review list.
--temperature F
    Sampling temperature (default: 0.7).
--max-tokens-per-review N
    Maximum tokens per individual review or item-metadata entry (default: 200).
    Only the history content is truncated; the instruction text is always preserved.
--max-output-tokens N
    Maximum new tokens per output (default: 512).
--dataset NAME
    Dataset identifier (e.g. beauty, yelp, cell).
--fold FOLD
    Partition fold identifier (default: 0).
--base-path PATH
    Root that contains ``data/`` (default: current directory).
--test
    Use the ``_mini`` subset (fold ``<fold>_mini``).
--max-users N
    Only profile the first N users by id (smoke testing).
--tensor-parallel-size N
    vLLM tensor parallelism degree (default: 1).
--gpu-memory-utilization F
    Fraction of GPU memory to allow vLLM to use (default: 0.9).
--max-model-len N
    Maximum context length for the vLLM engine (default: model maximum).
--dtype DTYPE
    Model dtype for vLLM (default: auto).
--trust-remote-code
    Pass trust_remote_code=True to the vLLM engine.

Example (paper setting)
-----------------------
::

    python prepare_ui_llm_desc.py \\
        --entity-type user \\
        --desc-type taste \\
        --llm-model Qwen/Qwen2.5-32B-Instruct-AWQ \\
        --dataset beauty
"""

import argparse
import logging
import sys


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="prepare_ui_llm_desc.py",
        description="Pre-generate LLM user profiles for semantic-ID training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- what to generate ---------------------------------------------------
    p.add_argument(
        "--entity-type",
        required=True,
        choices=["user"],
        metavar="{user}",
        help="Generate descriptions for users.",
    )
    p.add_argument(
        "--desc-type",
        required=True,
        choices=["taste", "technical"],
        metavar="{taste,technical}",
        help="User profile variant: taste or technical.",
    )
    p.add_argument(
        "--llm-model",
        required=True,
        metavar="MODEL",
        help="HuggingFace model id or local path.",
    )
    p.add_argument(
        "--max-reviews-per-entity",
        type=int,
        default=15,
        metavar="N",
        help="Maximum reviews (items) per entity in the LLM prompt. For user taste profiles this caps the item history; for all other types it caps the review list.",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        metavar="F",
        help="Sampling temperature (default: 0.7).",
    )
    p.add_argument(
        "--max-tokens-per-review",
        type=int,
        default=200,
        metavar="N",
        help="Maximum tokens per individual review or item-metadata entry (default: 200). Only history content is truncated; instruction text is always preserved.",
    )
    p.add_argument(
        "--max-output-tokens",
        type=int,
        default=512,
        metavar="N",
        help="Maximum new tokens per output (default: 512).",
    )

    # --- dataset / output ---------------------------------------------------
    p.add_argument(
        "--dataset",
        required=True,
        metavar="NAME",
        help="Dataset identifier (e.g. beauty, yelp, cell).",
    )
    p.add_argument(
        "--fold",
        default="0",
        metavar="FOLD",
        help="Partition fold identifier (default: 0).",
    )
    p.add_argument(
        "--base-path",
        default=None,
        metavar="PATH",
        help="Root that contains data/ (default: current directory).",
    )
    p.add_argument(
        "--test",
        action="store_true",
        help="Use the _mini subset of the dataset.",
    )
    p.add_argument(
        "--max-users",
        type=int,
        default=None,
        metavar="N",
        help="Only profile the first N users by id.",
    )

    # --- vLLM engine --------------------------------------------------------
    p.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        metavar="N",
        help="vLLM tensor parallelism degree (default: 1).",
    )
    p.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        metavar="F",
        help="Fraction of GPU memory for vLLM (default: 0.9).",
    )
    p.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        metavar="N",
        help="Maximum context length for the vLLM engine (default: model maximum).",
    )
    p.add_argument(
        "--dtype",
        default="auto",
        metavar="DTYPE",
        help="Model dtype for vLLM (default: auto).",
    )
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to the vLLM engine.",
    )
    return p


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    args = _build_parser().parse_args()

    vllm_kwargs = {
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": args.dtype,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.max_model_len is not None:
        vllm_kwargs["max_model_len"] = args.max_model_len

    from nlp_xrs.data_preparation.ui_desc_generation import generate_ui_descriptions

    generate_ui_descriptions(
        dataset_name=args.dataset,
        entity_type=args.entity_type,
        desc_type=args.desc_type,
        llm_model=args.llm_model,
        max_reviews_per_entity=args.max_reviews_per_entity,
        temperature=args.temperature,
        max_tokens_per_review=args.max_tokens_per_review,
        max_output_tokens=args.max_output_tokens,
        fold=args.fold,
        base_path=args.base_path,
        test=args.test,
        max_users=args.max_users,
        **vllm_kwargs,
    )


if __name__ == "__main__":
    main()
