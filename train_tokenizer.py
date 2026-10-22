"""
Script to train a semantic ID tokenizer (OPQ, Faiss RQ-KMeans, or RQ-VAE).

Each run is assigned a deterministic 16-char hex **run ID** (SHA256 of the
quantizer hyper-parameters).  All run-specific artefacts are placed under a
subdirectory named after the run ID so that different configurations never
overwrite each other.

Item embeddings (static, depend only on the embedding source/model/text-config)
are cached at the parent level and shared across runs::

    data/<dataset>/.../tokenizers/semantic_ids/<slug>/<emb_model>/
    ├── embeddings.pt               ← shared cache, no run ID
    └── <run_id>/                   ← per-run checkpoint
        ├── model_{type}.pt
        ├── mapping_{type}.pt
        └── run_info.json

TensorBoard logs::

    logs/tensorboard/tokenizers/<dataset>/<slug>/<run_id>/

Visualizations::

    results/tokenizers/<dataset>/<slug>/<run_id>/viz/

Registry (run ID → config + paths)::

    results/tokenizers/<dataset>/registry.json

Usage:
    python train_tokenizer.py --dataset yelp --config rqvae_qwen_3cb
    python train_tokenizer.py --dataset beauty --config opq_qwen_3cb
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch

from nlp_xrs.data.dataset import BaseDataset, get_default_args
from nlp_xrs.semantic_ids import SemanticIDTokenizer
from nlp_xrs.semantic_ids.utils import (
    _emb_config_slug,
    extract_item_embeddings,
    extract_user_embeddings,
    get_sae_dists,
    get_tok_emb_paths,
    save_run_registry,
)
from nlp_xrs.utils import funcs
from nlp_xrs.utils.constants import PACKAGE_NAME, RNG_SEED, Key, SeqMode
from nlp_xrs.utils.funcs import init_seed, namespace2dict, now_time
from nlp_xrs.utils.in_out import Colors, colored, get_acts_mapper, handle_dirs, load_auxiliary_config, load_config

DEFAULT_EMBEDDING_MODEL = "google/flan-t5-xl"

# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------


def load_dataset(
    dataset_name: str,
    requires_rating: bool = False,
    test: bool = False,
    load_metadata: bool = False,
    seed: int = RNG_SEED,
    base_path: Optional[str] = None,
):
    """Load dataset for tokenizer training (no history or explanation columns)."""
    args = get_default_args("peter", dataset=dataset_name, test=test, base_path=base_path)
    args.seed = seed
    args.paths = handle_dirs(args)

    logger = logging.getLogger(PACKAGE_NAME)
    cfg_parts = args.config_id.split("/")
    cfg_name = cfg_parts[-1]
    cfg_prefix = "/".join(cfg_parts[:-1])
    cfg = load_config(f"{cfg_name}.yaml", args.paths.config, args, prefix_path=cfg_prefix)

    cfg.data.requires_rating = requires_rating
    cfg.data.requires_feat = False
    cfg.data.requires_exp = False
    cfg.data.requires_context = False
    cfg.data.requires_nextitem = False
    cfg.data.seq_mode = SeqMode.NO_SEQ

    if load_metadata:
        cfg.data.load_metadata = True

    if not args.resume_training:
        init_seed(args.seed, reproducibility=True)
    else:
        logger.info(colored("-" * 40 + " RESUME TRAINING " + "-" * 40, Colors.BLUE))
        logger.warning(
            "--resume-training does not reset the RNG state to one of the last checkpoint, thus, results"
            " will not be reproducible"
        )

    device = torch.device("cpu")
    if torch.cuda.is_available():
        if not args.cuda:
            logger.warning(now_time() + "You have a CUDA device, so you should probably run with --cuda")
        else:
            device = torch.device("cuda")

    cfg.data.load_cols = [Key.REV]
    dataset = BaseDataset(cfg.data, cfg.model, device=device, seed=args.seed, logger=logger, paths=cfg.paths)
    return dataset


# ---------------------------------------------------------------------------
# Logging / output helpers
# ---------------------------------------------------------------------------


def update_logger(path: str, config: SimpleNamespace, args: argparse.Namespace, run_id: Optional[str] = None):
    if args.log_to_file:
        os.makedirs(path, exist_ok=True)
        method_tag = _emb_config_slug(config.emb_config)
        log_f = f"{args.dataset}_{getattr(config, 'tokenizer_type', 'rqvae')}_{method_tag}_{args.seed}"
        if run_id is not None:
            log_f = f"{log_f}_{run_id}"

        log_f = f"{log_f}.txt"
        handlers = [logging.FileHandler(filename=os.path.join(path, log_f), mode="w")]
    else:
        handlers = [logging.StreamHandler(stream=sys.stdout)]

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s - %(message)s", handlers=handlers, force=True
    )
    return logging.getLogger(PACKAGE_NAME)


# ---------------------------------------------------------------------------
# Embedding extraction (reused by explore_tokenizer.py)
# ---------------------------------------------------------------------------


def load_embeddings(
    args,
    config: SimpleNamespace,
    dataset: BaseDataset,
    emb_dir: Path,
    device: str,
    logger: logging.Logger,
    mode: str = "item",
) -> torch.Tensor:
    """Extract (or load cached) item or user embeddings for the given config.

    Args:
        args:    Training args (must have .dataset).
        config:  Tokenizer config with emb_config.source / emb_config.profiles,
                 emb_config.embedding_model, etc.
        dataset: Loaded BaseDataset.
        emb_dir: Directory used to cache the computed embeddings on disk.
        device:  Torch device string.
        logger:  Logger instance.
        mode:    ``'item'`` (default) or ``'user'``.

    Returns:
        Float tensor of shape ``(num_entities, embedding_dim)``.
    """
    emb_config = config.emb_config

    if mode == "user":
        logger.info("Extracting user embeddings (profiles=%s) ...\n", getattr(emb_config, "profiles", []))
        assert dataset.metadata is not None, (
            "User mode requires item metadata to build taste profiles; "
            "ensure load_metadata=True when calling load_dataset()."
        )
        all_embeddings = extract_user_embeddings(
            inter_df=dataset.data,
            metadata_df=dataset.metadata,
            mappers=dataset.mappers,
            train_ixs=dataset.split_ixs["train"],
            save_dir=emb_dir,
            emb_config=emb_config,
            dataset_name=args.dataset,
            device=device,
        )
        logger.info(f"User embeddings shape: {all_embeddings.shape}")
    else:
        logger.info(f"Extracting item embeddings (method={getattr(emb_config, 'source', 'text')}) ...\n")
        all_embeddings = extract_item_embeddings(
            inter_df=dataset.data,
            metadata_df=dataset.metadata,
            mappers=dataset.mappers,
            train_ixs=dataset.split_ixs["train"],
            save_dir=emb_dir,
            emb_config=emb_config,
            dataset_name=args.dataset,
            device=device,
            test=args.test,
        )
        logger.info(f"Item embeddings shape: {all_embeddings.shape}")

    return all_embeddings


# ---------------------------------------------------------------------------
# Core training routine (reused by explore_tokenizer.py)
# ---------------------------------------------------------------------------
def run_training(
    args,
    config: SimpleNamespace,
    run_id: str,
    output_dir: Path,
    logger: logging.Logger,
    dataset: BaseDataset,
    all_embeddings: torch.Tensor,
    tb_writer=None,
    verbose: bool = True,
    overwrite: bool = False,
    mode: str = "item",
) -> SemanticIDTokenizer:
    """Initialize, train, save, and visualize a tokenizer.

    Args:
        args:           Training args (dataset, use_train_only, paths, ...).
        config:         Tokenizer config (architecture, training hyper-params).
        run_id:         Unique run identifier (used for output paths).
        output_dir:     Root directory where tokenizer files are written.
        logger:         Logger instance.
        dataset:        Loaded BaseDataset.
        all_embeddings: Pre-computed entity embedding tensor (items or users).
        tb_writer:      Optional TensorBoard SummaryWriter.
        verbose:        Whether to print verbose logging.
        overwrite:      Whether to overwrite existing tokenizer.
        mode:           ``'item'`` (default) or ``'user'``.  When ``'user'``,
                        user IDs are placed in the ``Key.ITEM`` column of a
                        proxy DataFrame so the entity-agnostic tokenizer can
                        be trained without modification.
    Returns:
        The trained :class:`SemanticIDTokenizer`.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Config:\n{json.dumps(funcs.namespace2dict(config), indent=4)}")
    logger.info(f"Output directory: {output_dir}")

    logger.info("Initializing SemanticIDTokenizer...")
    tokenizer = SemanticIDTokenizer(config, dedup_items_on_load=False)
    logging.info(tokenizer.model)
    tokenizer.to(device)

    if tokenizer.exists(output_dir) and not overwrite:
        logger.info(
            "Tokenizer already exists. Use --overwrite to overwrite it. Otherwise, loading existing tokenizer..."
        )
        tokenizer.load(output_dir)
    else:
        logger.info("Training SemanticIDTokenizer...")
        train_data = dataset.data.loc[dataset.split_ixs["train"] + dataset.split_ixs["valid"]]

        tgt_key = Key.ITEM if mode == "item" else Key.USER
        if isinstance(train_data[tgt_key].iloc[0], (list, tuple)):
            train_data[tgt_key] = train_data[tgt_key].str[0]

        item_features = {"embeddings": all_embeddings}
        if hasattr(config, "sae_preset"):
            emb_model_alias = config.emb_config.embedding_model.replace("/", "--")
            acts_path = os.path.join(
                Path(dataset.paths.tokenizers).parent,
                "SAE",
                f"acts_{emb_model_alias}_{config.sae_preset}_{args.seed}.pt",
            )
            acts_mapper = get_acts_mapper(acts_path)

            sae_dists = torch.zeros(all_embeddings.shape[0], tokenizer.model.sae_codebook_size)
            train_g_rev = train_data[[tgt_key, Key.REV]]
            train_g_rev[tgt_key] = (
                train_g_rev[tgt_key].str[-1] if isinstance(train_g_rev[tgt_key].iloc[0], list) else train_g_rev[tgt_key]
            )
            train_dists = train_g_rev.groupby(tgt_key)[Key.REV].apply(
                get_sae_dists, acts_mapper=acts_mapper, sae_codebook_size=tokenizer.model.sae_codebook_size
            )
            sae_dists[train_dists.index.values] = torch.vstack(train_dists.values.tolist())
            # L1 normalize the concept targets
            sae_dists = sae_dists / sae_dists.norm(p=1, dim=1, keepdim=True).clamp(min=1e-10)
            item_features["concept_targets"] = sae_dists

        tokenizer.train(data=train_data, item_features=item_features, tb_writer=tb_writer, verbose=verbose, key=tgt_key)

        # Save tokenizer + metadata
        logger.info(f"Saving tokenizer to {output_dir}")
        tokenizer.save(output_dir)

    # -----------------------------------------------------------------------
    # Save run info
    # -----------------------------------------------------------------------
    emb_config = config.emb_config
    slug = _emb_config_slug(emb_config)
    results_dir = Path(args.paths.results, "tokenizers", args.dataset, slug, run_id)
    entity_label = "users" if mode == "user" else "items"
    run_info = {
        "dataset": args.dataset,
        "mode": mode,
        "config": namespace2dict(config),
        "use_train_only": getattr(args, "use_train_only", True),
        f"n_{entity_label}": int(tokenizer.num_items),
        "n_codebooks": int(tokenizer.get_token_length()),
        "codebook_size": int(config.codebook_size),
        "metrics": tokenizer.model.compute_codebook_metrics(all_embeddings, verbose=False),
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=4)

    # -----------------------------------------------------------------------
    # Sanity-check tokenization on a few sample entities
    # -----------------------------------------------------------------------
    entity_singular = "User" if mode == "user" else "Item"
    logger.info(f"\n--- Sample {entity_singular.lower()} tokenization ---")
    n_items = tokenizer.num_items
    sample_items = list(range(min(5, n_items)))
    tokens = tokenizer.tokenize(sample_items)
    for item_id, token_seq in zip(sample_items, tokens):
        logger.info(f"  {entity_singular} {item_id}: {token_seq}")

    reconstructed = tokenizer.detokenize(tokens)
    acc = sum(a == b for a, b in zip(sample_items, reconstructed)) / len(sample_items) * 100
    logger.info(f"Reconstruction accuracy on sample: {acc:.1f}%")

    return tokenizer


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Train a semantic ID tokenizer")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g. yelp, beauty)")
    parser.add_argument("--config", type=str, required=True, help="Tokenizer config name (e.g. rqvae_qwen_3cb)")
    parser.add_argument(
        "--base-path",
        type=str,
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Root path for data, logs, and checkpoints.",
    )
    parser.add_argument("--fold", type=str, default="0", help="Dataset fold number")
    parser.add_argument("--test", action="store_true", help="Run in test mode (small data)")
    parser.add_argument("--log-to-file", action="store_true", help="Write logs to a file")
    parser.add_argument("--only-embed", action="store_true", help="Only extract embeddings and exit")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["item", "user"],
        default="item",
        help=(
            "Entity mode: 'item' trains a tokenizer on item text embeddings "
            "(default, existing behaviour); 'user' trains a tokenizer on "
            "LLM-generated user profile embeddings (taste + technical profiles)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RNG_SEED,
        help=(
            "Random seed for the quantiser training RNG.  Different seeds "
            "produce isolated checkpoints under <run_id>/seed_<N>/ so they "
            "never overwrite each other."
        ),
    )

    args = parser.parse_args()

    args.paths = handle_dirs(args)
    config = load_auxiliary_config(Path(args.paths.config, "tokenizers", args.config).with_suffix(".yaml"))

    # Propagate entity to emb_config when the config omits it (backward compat).
    if not hasattr(config.emb_config, "entity"):
        config.emb_config.entity = args.mode

    config_stem = args.config

    log_path = os.path.join(args.paths.log, "tokenizers")

    # Resolve paths: run_id is the config hash; tok_dir includes a seed_<N>
    # subfolder when --seed differs from the default so that runs with
    # different seeds never overwrite each other.
    emb_dir, tok_dir, run_id = get_tok_emb_paths(args.paths, config, seed=args.seed)

    logger = update_logger(log_path, config, args, run_id=run_id)

    logger.info(f"Mode: {args.mode}")
    logger.info(f"Run ID: {run_id}  (config: {config_stem},  seed: {args.seed})")
    logger.info(f"Tokenizer path: {tok_dir}")

    # -----------------------------------------------------------------------
    # Resolve slug for TensorBoard / viz paths
    # -----------------------------------------------------------------------
    slug = _emb_config_slug(config.emb_config)

    # -----------------------------------------------------------------------
    # TensorBoard writer
    # -----------------------------------------------------------------------
    tb_writer = None
    tb_dir = None
    if not args.test:
        tb_dir = os.path.join(
            args.paths.log, "tensorboard", "tokenizers", args.dataset, slug, run_id, f"seed_{args.seed}"
        )
        try:
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(tb_dir, exist_ok=True)
            tb_writer = SummaryWriter(tb_dir)
            logger.info(f"TensorBoard logging to: {tb_dir}")
        except ImportError:
            logger.warning("TensorBoard not available — install tensorboard to enable logging.")

    # -----------------------------------------------------------------------
    # Load dataset
    # -----------------------------------------------------------------------
    logger.info(f"Loading dataset: {args.dataset}")
    emb_config = config.emb_config

    if args.mode == "user":
        # User mode: always need item metadata (for taste profile) and raw
        # reviews (for technical profile).  Ratings are required.
        dataset = load_dataset(
            args.dataset,
            requires_rating=True,
            test=args.test,
            load_metadata=True,
            seed=args.seed,
            base_path=args.base_path,
        )
    else:
        method = getattr(emb_config, "source", "text")
        dataset = load_dataset(
            args.dataset,
            requires_rating=(method in ("collaborative", "hybrid")),
            test=args.test,
            load_metadata=(method in ("text", "hybrid")),
            seed=args.seed,
            base_path=args.base_path,
        )

    logger.info(f"Dataset loaded: {dataset.data_info.n_items} items, {dataset.data_info.n_users} users")

    # -----------------------------------------------------------------------
    # Extract embeddings  (uses emb_dir, not tok_dir)
    # -----------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_embeddings = load_embeddings(args, config, dataset, emb_dir, device, logger, mode=args.mode)

    if args.only_embed:
        logger.info("Embeddings extracted. Exiting...")
        return

    # -----------------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------------
    # Re-seed immediately before training so that the quantiser RNG state is
    # fully determined by --seed, independent of any randomness consumed
    # during dataset loading or embedding extraction.
    init_seed(args.seed, reproducibility=True)
    run_training(args, config, run_id, tok_dir, logger, dataset, all_embeddings, tb_writer, mode=args.mode)

    if tb_writer is not None:
        tb_writer.close()
        logger.info(f"\nTensorBoard logs saved to: {tb_dir}")
        logger.info("To visualize: tensorboard --logdir logs/tensorboard")

    # -----------------------------------------------------------------------
    # Update registry
    # -----------------------------------------------------------------------
    registry_stem = "user_registry" if args.mode == "user" else "registry"
    viz_dir = Path(args.paths.results, "tokenizers", args.dataset, slug, run_id, "viz", f"seed_{args.seed}")
    save_run_registry(
        base_path=args.base_path,
        dataset=args.dataset,
        run_id=run_id,
        entry={
            "run_id": run_id,
            "mode": args.mode,
            "dataset": args.dataset,
            "config_stem": config_stem,
            "config": funcs.namespace2dict(config),
            "checkpoint_path": str(tok_dir),
            "tb_path": str(tb_dir) if tb_dir else None,
            "viz_path": str(viz_dir),
        },
        seed=args.seed,
        registry_stem=registry_stem,
    )

    logger.info("\nTokenizer training complete.")


if __name__ == "__main__":
    main()
