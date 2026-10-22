"""Stage an isolated ``--base-path`` tree with the ``_mini`` subset of a processed dataset.

Everything is copied (never hard-linked) so runs inside the sandbox cannot write
through to the source tree. Used by the end-to-end unit tests and by
``scripts/smoke_repro.sh``.

CLI::

    python -m nlp_xrs.utils.mini_sandbox stage --source <repo> --dest <sandbox> [--with-derived] [--with-caches] [--profiles-from-fold 0]
    python -m nlp_xrs.utils.mini_sandbox materialize --dest <sandbox>
    python -m nlp_xrs.utils.mini_sandbox fill-profiles --dest <sandbox> --src-name <json> --dst-name <json>
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Optional, Union

from nlp_xrs.utils.constants import DATA_MAP, DATA_MODE, Suffix

PathLike = Union[str, Path]

PAPER_PROFILE_NAMES = (
    "llm_taste__user__Qwen--Qwen2.5-32B-Instruct-AWQ__15__0,7__200__512.json",
    "llm_technical__user__Qwen--Qwen2.5-32B-Instruct-AWQ__15__0,7__200__512.json",
)

_CORE_TABLES = ("reviews", "hist_reviews")
_DERIVED_TABLES = ("metadata", "user_neg_samples")
_SPLITS = ("train.npy", "validation.npy", "test.npy")


def feat_dir(base: PathLike, dataset: str = "beauty") -> Path:
    return Path(base) / "data" / DATA_MAP.get(dataset.lower(), dataset) / DATA_MODE


def mini_fold(base: PathLike, dataset: str = "beauty", fold: str = "0") -> Path:
    return feat_dir(base, dataset) / f"{fold}{Suffix.TEST}"


def mini_users(src_feat: Path) -> list:
    import pandas as pd

    reviews = pd.read_parquet(src_feat / f"reviews{Suffix.TEST}.parquet", columns=["user"])
    return sorted(reviews["user"].astype(int).unique())


def filter_profiles(
    src_dir: PathLike,
    dst_dir: PathLike,
    users: Iterable[int],
    names: Iterable[str] = PAPER_PROFILE_NAMES,
    fill: str = "unknown",
) -> None:
    """Copy user-profile JSON caches restricted to ``users``; absent users get ``fill``."""
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    users = list(users)
    for name in names:
        src = Path(src_dir) / name
        if not src.is_file():
            raise FileNotFoundError(f"Missing user-profile cache: {src}")
        with src.open() as f:
            raw = json.load(f)
        kept = {str(u): raw.get(str(u), fill) for u in users}
        with (dst_dir / name).open("w") as f:
            json.dump(kept, f)


def stage(
    source: PathLike,
    dest: PathLike,
    dataset: str = "beauty",
    fold: str = "0",
    with_derived: bool = False,
    with_caches: bool = False,
    profiles_from_fold: Optional[str] = None,
    config_source: Optional[PathLike] = None,
) -> Path:
    """Build ``dest`` as a self-contained base path for ``--test`` runs.

    Args:
        source: Base path holding the processed dataset (``<source>/data/...``).
        dest: Sandbox root; must not exist or be empty.
        dataset: Dataset key (e.g. ``'beauty'``).
        fold: Fold whose ``_mini`` split is staged.
        with_derived: Also copy ``metadata_mini`` and ``user_neg_samples_mini`` instead of
            regenerating them with ``prepare_data.py``.
        with_caches: Also copy the ``*_mini`` embedding caches under ``embeddings/``.
        profiles_from_fold: If set, filter the paper user-profile caches of this full fold onto
            the mini users.
        config_source: Config tree to copy (default ``<source>/config``).

    Returns:
        The sandbox root.
    """
    source, dest = Path(source), Path(dest)
    src_feat, dst_feat = feat_dir(source, dataset), feat_dir(dest, dataset)
    if not (src_feat / f"reviews{Suffix.TEST}.parquet").is_file():
        raise FileNotFoundError(f"Missing processed mini table under {src_feat}")
    dest.mkdir(parents=True, exist_ok=True)

    shutil.copytree(Path(config_source) if config_source else source / "config", dest / "config", dirs_exist_ok=True)

    (dst_feat / "mappers").mkdir(parents=True, exist_ok=True)
    dst_fold = mini_fold(dest, dataset, fold)
    dst_fold.mkdir(parents=True, exist_ok=True)

    src_orig = src_feat.parent / "orig"
    if src_orig.exists() and not (dst_feat.parent / "orig").exists():
        os.symlink(src_orig.resolve(), dst_feat.parent / "orig")

    tables = _CORE_TABLES + (_DERIVED_TABLES if with_derived else ())
    for name in tables:
        shutil.copy2(src_feat / f"{name}{Suffix.TEST}.parquet", dst_feat / f"{name}{Suffix.TEST}.parquet")
    for mapper in (src_feat / "mappers").glob(f"*{Suffix.TEST}.parquet"):
        shutil.copy2(mapper, dst_feat / "mappers" / mapper.name)
    src_fold = mini_fold(source, dataset, fold)
    for split in _SPLITS:
        shutil.copy2(src_fold / split, dst_fold / split)

    if with_caches:
        (dst_feat / "embeddings").mkdir(exist_ok=True)
        for cache in (src_feat / "embeddings").glob(f"*{Suffix.TEST}.*"):
            shutil.copy2(cache, dst_feat / "embeddings" / cache.name)

    if profiles_from_fold is not None:
        filter_profiles(
            src_feat / profiles_from_fold / "tokenizers" / "semantic_ids",
            dst_fold / "tokenizers" / "semantic_ids",
            mini_users(src_feat),
        )
    return dest


def materialize_full_fold(dest: PathLike, dataset: str = "beauty", fold: str = "0") -> None:
    """Copy the ``_mini`` tables and fold to unsuffixed names, for runs that must not use ``--test``."""
    dst_feat = feat_dir(dest, dataset)
    for table in dst_feat.glob(f"*{Suffix.TEST}.parquet"):
        shutil.copy2(table, dst_feat / table.name.replace(Suffix.TEST, ""))
    for mapper in (dst_feat / "mappers").glob(f"*{Suffix.TEST}.parquet"):
        shutil.copy2(mapper, dst_feat / "mappers" / mapper.name.replace(Suffix.TEST, ""))
    full = dst_feat / fold
    if full.exists():
        shutil.rmtree(full)
    shutil.copytree(mini_fold(dest, dataset, fold), full, symlinks=True)


def fill_profiles(dest: PathLike, src_name: str, dst_name: str, dataset: str = "beauty", fold: str = "0") -> None:
    """Rewrite a generated profile cache in the mini fold so every mini user has an entry."""
    sid_dir = mini_fold(dest, dataset, fold) / "tokenizers" / "semantic_ids"
    filter_profiles(sid_dir, sid_dir, mini_users(feat_dir(dest, dataset)), names=[src_name])
    if src_name != dst_name:
        os.replace(sid_dir / src_name, sid_dir / dst_name)


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m nlp_xrs.utils.mini_sandbox")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("stage")
    s.add_argument("--source", required=True)
    s.add_argument("--dest", required=True)
    s.add_argument("--dataset", default="beauty")
    s.add_argument("--config-source", default=None)
    s.add_argument("--with-derived", action="store_true")
    s.add_argument("--with-caches", action="store_true")
    s.add_argument("--profiles-from-fold", default=None)

    m = sub.add_parser("materialize")
    m.add_argument("--dest", required=True)
    m.add_argument("--dataset", default="beauty")

    f = sub.add_parser("fill-profiles")
    f.add_argument("--dest", required=True)
    f.add_argument("--dataset", default="beauty")
    f.add_argument("--src-name", required=True)
    f.add_argument("--dst-name", default=None)

    args = p.parse_args()
    if args.cmd == "stage":
        stage(
            args.source,
            args.dest,
            dataset=args.dataset,
            with_derived=args.with_derived,
            with_caches=args.with_caches,
            profiles_from_fold=args.profiles_from_fold,
            config_source=args.config_source,
        )
    elif args.cmd == "materialize":
        materialize_full_fold(args.dest, dataset=args.dataset)
    else:
        fill_profiles(args.dest, args.src_name, args.dst_name or args.src_name, dataset=args.dataset)


if __name__ == "__main__":
    main()
