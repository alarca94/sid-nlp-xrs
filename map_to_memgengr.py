"""Transfer a trained SID mapping to the MemGenGR project cache.

Usage examples
--------------
# Transfer SID mapping only (skip dataset files if they already exist):
python map_to_memgengr.py \
    --dataset beauty \
    --config config/tokenizers/opq_qwen_4cb.yaml \
    --seed 1111

# Force re-transfer of dataset files (all_item_seqs, id_mapping, embeddings):
python map_to_memgengr.py \
    --dataset beauty \
    --config config/tokenizers/rqvae_qwen_4cb.yaml \
    --seed 24 \
    --force-dataset

# Override destination directory:
python map_to_memgengr.py \
    --dataset beauty \
    --config config/tokenizers/rqfaiss_qwen_4cb.yaml \
    --seed 1111 \
    --memgengr-path /path/to/MemGen-GR/cache/AmazonReviews2014/Beauty/processed

Output filename convention (aligned with the config file stem):
    {emb_model_short}_{n_codebooks}x{codebook_size}_{config_stem}_seed{seed}.sem_ids
e.g.:
    Qwen3-Embedding-0.6B_4x256_opq_qwen_4cb_seed1111.sem_ids
    Qwen3-Embedding-0.6B_4x256_opq_qwen_4cb_noPCA_seed1111.sem_ids
"""

import argparse
import json
import os
from pathlib import Path

import torch

from nlp_xrs.data.dataset import get_default_args
from nlp_xrs.semantic_ids.utils import get_tok_emb_paths
from nlp_xrs.utils.constants import DATA_MAP, Key
from nlp_xrs.utils.in_out import handle_dirs, load_auxiliary_config
from train_tokenizer import load_dataset

MEMGENGR_CACHE_ROOT = Path.home() / Path("RecSys/MemGen-GR/cache")


def _build_output_filename(config, seed: int, config_stem: str) -> str:
    emb_model = getattr(config.emb_config, "embedding_model", "unknown")
    emb_model_short = emb_model.split("/")[-1]
    n_cb = config.n_codebooks
    cb_size = config.codebook_size
    return f"{emb_model_short}_{n_cb}x{cb_size}_{config_stem}_seed{seed}.sem_ids"


def _memgengr_command(dataset: str, output_filename: str, n_codebooks: int) -> str:
    return (
        f"python main.py"
        f" --config=configs/rpg_{dataset}_{n_codebooks}cb.yaml"
        f" --model=RPG"
        f" --sem_ids_file={output_filename}"
    )


def transfer_dataset(dataset_name: str, emb_dir: Path, dest: Path, force: bool) -> None:
    """Write all_item_seqs.json, id_mapping.json, embeddings, and metadata.sentence.json
    to *dest*. Skipped if all files already exist and *force* is False."""

    seqs_file = dest / "all_item_seqs.json"
    mapping_file = dest / "id_mapping.json"
    meta_file = dest / "metadata.sentence.json"

    # Derive embedding file name from the embedding model
    # (re-load lazily so we only pay the cost when needed)
    def _emb_fname(config):
        emb_model_short = config.emb_config.embedding_model.split("/")[-1]
        return f"{emb_model_short}.sent_emb"

    # We need the dataset to build sequences/mappings.  Import here to avoid
    # circular deps and keep the top-level import light.
    if not force and seqs_file.exists() and mapping_file.exists() and meta_file.exists():
        print(f"  [dataset] Destination files already exist in {dest} — skipping (use --force-dataset to overwrite).")
        return

    print(f"  [dataset] Loading dataset '{dataset_name}' to build MemGenGR data files …")
    dataset = load_dataset(dataset_name, requires_rating=False, test=False, load_metadata=False)

    data = dataset.data.loc[dataset.split_ixs["train"] + dataset.split_ixs["valid"] + dataset.split_ixs["test"]]
    data = data.sort_values(by=Key.TIMESTEP)
    data = data[[Key.USER, Key.ITEM]].explode([Key.USER, Key.ITEM])

    user_map = dataset.mappers[Key.USER].idx2item["raw"]
    item_map = dataset.mappers[Key.ITEM].idx2item["raw"]

    data[Key.USER] = data[Key.USER].map(user_map)
    data[Key.ITEM] = data[Key.ITEM].map(item_map)

    dest.mkdir(parents=True, exist_ok=True)

    # all_item_seqs.json
    memgengr_data = data.groupby(Key.USER)[Key.ITEM].agg(list)
    memgengr_data.to_json(seqs_file)
    print(f"  [dataset] Written {seqs_file}")

    # id_mapping.json
    valid_items = set(data[Key.ITEM].tolist()).intersection(item_map.tolist())
    user2id: dict = {"[PAD]": 0, **{user: i + 1 for i, user in enumerate(user_map)}}
    item2id: dict = {"[PAD]": 0, **{item: i + 1 for i, item in enumerate(valid_items)}}
    id_mapping = {
        "user2id": user2id,
        "item2id": item2id,
        "id2item": list(item2id.keys()),
        "id2user": list(user2id.keys()),
    }
    with open(mapping_file, "w") as f:
        json.dump(id_mapping, f)
    print(f"  [dataset] Written {mapping_file}")

    # metadata.sentence.json  (dummy — sentences not used by RPG)
    item_to_sentence = {k: "" for k in id_mapping["item2id"] if k != "[PAD]"}
    with open(meta_file, "w") as f:
        json.dump(item_to_sentence, f)
    print(f"  [dataset] Written {meta_file}")


def transfer_embeddings(emb_dir: Path, dest: Path, force: bool, dataset_name: str) -> None:
    # .sent_emb  — raw numpy binary used by MemGenGR for semantic similarity
    emb_file = emb_dir / "embeddings.pt"
    if emb_file.exists():
        emb_model_short = emb_dir.name if "--" not in emb_dir.name else emb_dir.name.split("--")[-1]
        sent_emb_path = dest / f"{emb_model_short}.sent_emb"
        if force or not sent_emb_path.exists():
            print(f"  [dataset] Loading dataset '{dataset_name}' to build MemGenGR data files …")
            dataset = load_dataset(dataset_name, requires_rating=False, test=False, load_metadata=False)
            meta_file = dest / "metadata.sentence.json"
            with open(meta_file, "r") as f:
                item_to_sentence = json.load(f)

            embeddings = torch.load(emb_file, weights_only=True).numpy()
            idxs = dataset.mappers[Key.ITEM].item2idx.loc[list(item_to_sentence.keys())]
            embeddings[idxs].tofile(sent_emb_path)
            print(f"  [dataset] Written {sent_emb_path}")
        else:
            print(f"  [dataset] Embeddings file already exists — skipping {sent_emb_path}")
    else:
        print(f"  [dataset] WARNING: {emb_file} not found; skipping .sent_emb transfer.")


def transfer_sid_mapping(tok_dir: Path, config, dest: Path, output_filename: str) -> Path:
    """Load mapping_{tokenizer_type}.pt and write the SID JSON to *dest*."""
    tok_type = config.tokenizer_type
    mapping_pt = tok_dir / f"mapping_{tok_type}.pt"

    if not mapping_pt.exists():
        raise FileNotFoundError(
            f"Mapping file not found: {mapping_pt}\n"
            "Make sure the tokenizer has been trained for this dataset/config/seed combination."
        )

    checkpoint = torch.load(mapping_pt, weights_only=False)
    sid_map_raw = checkpoint["item_to_tokens"]

    dest.mkdir(parents=True, exist_ok=True)
    out_path = dest / output_filename
    # Convert int keys → string raw IDs is handled downstream; keep as-is for now.
    # sid_map_raw maps item_idx (int) -> token list.  MemGenGR expects raw item IDs.
    # We need the item_map to convert.  Load it lazily.
    from train_tokenizer import load_dataset

    dataset = load_dataset(config._dataset_name, requires_rating=False, test=False, load_metadata=False)
    item_map = dataset.mappers[Key.ITEM].idx2item["raw"]

    id_mapping_file = dest / "id_mapping.json"
    if id_mapping_file.exists():
        with open(id_mapping_file) as f:
            id_mapping = json.load(f)
        valid_items = set(id_mapping["item2id"].keys()) - {"[PAD]"}
    else:
        valid_items = set(item_map.tolist())

    sid_map = {item_map[item_id]: tokens for item_id, tokens in sid_map_raw.items() if item_map[item_id] in valid_items}

    with open(out_path, "w") as f:
        json.dump(sid_map, f)

    print(f"  [sid] Written {out_path}  ({len(sid_map)} items)")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Transfer a trained SID mapping (and optionally dataset files) to MemGenGR.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dataset", required=True, help="Dataset key (e.g. beauty, cell, yelp)")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to tokenizer YAML config (e.g. config/tokenizers/opq_qwen_4cb.yaml)",
    )
    parser.add_argument("--seed", required=True, type=int, help="Seed used for tokenizer training")
    parser.add_argument(
        "--memgengr-path",
        default=None,
        help=(
            "Override MemGenGR destination directory. "
            "Default: ~/RecSys/MemGen-GR/cache/AmazonReviews2014/<DATASET>/processed"
        ),
    )
    parser.add_argument(
        "--force-dataset",
        action="store_true",
        help="Re-transfer dataset files even if they already exist in the destination.",
    )
    parser.add_argument(
        "--base-path",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Project root (default: directory of this script).",
    )
    args = parser.parse_args()

    dummy_args = get_default_args("peter", dataset=args.dataset, test=False)
    dummy_args.base_path = args.base_path
    paths = handle_dirs(dummy_args)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(args.base_path) / config_path
    config = load_auxiliary_config(config_path)

    # Stash dataset name on config so transfer_sid_mapping can use it without
    # requiring an extra argument through the call chain.

    emb_dir, tok_dir, run_id = get_tok_emb_paths(paths, config, seed=args.seed)

    config._dataset_name = args.dataset

    print(f"Dataset   : {args.dataset}")
    print(f"Config    : {args.config}")
    print(f"Seed      : {args.seed}")
    print(f"Run ID    : {run_id}")
    print(f"tok_dir   : {tok_dir}")

    if args.memgengr_path:
        dest = Path(args.memgengr_path)
    else:
        if args.dataset == "yelp":
            dest = MEMGENGR_CACHE_ROOT / "Yelp" / "Yelp_2020" / "processed"
        else:
            dest = MEMGENGR_CACHE_ROOT / "AmazonReviews2014" / DATA_MAP.get(args.dataset, args.dataset) / "processed"

    print(f"Destination: {dest}\n")

    transfer_dataset(args.dataset, emb_dir, dest, force=args.force_dataset)
    transfer_embeddings(emb_dir, dest, force=args.force_dataset, dataset_name=args.dataset)

    config_stem = config_path.stem
    output_filename = _build_output_filename(config, args.seed, config_stem)
    transfer_sid_mapping(tok_dir, config, dest, output_filename)

    cmd = _memgengr_command(args.dataset, output_filename, config.n_codebooks)
    print(f"\nMemGenGR command:\n  {cmd}\n")


if __name__ == "__main__":
    main()
