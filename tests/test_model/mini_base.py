"""Shared temporary ``--base-path`` for the Beauty ``_mini`` end-to-end tests.

All checkpoints, logs, SID bundles, and SAE files are written under a temporary
directory that is removed after the module finishes. Set ``NLP_XRS_KEEP_SANDBOX=1``
to keep it for inspection, ``NLP_XRS_SANDBOX_DIR`` to choose its parent, and
``NLP_XRS_DATA_SOURCE`` to read the processed ``data/`` tree from another base path.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from nlp_xrs.data.dataset import get_default_args
from nlp_xrs.utils import mini_sandbox

REPO = Path(__file__).resolve().parents[2]
DATA_SOURCE = Path(os.environ.get("NLP_XRS_DATA_SOURCE", REPO))


class MiniSandbox:
    def __init__(self):
        self.root = None

    def setup(self) -> Path:
        mini = mini_sandbox.feat_dir(DATA_SOURCE, "beauty") / "reviews_mini.parquet"
        if not mini.is_file():
            raise unittest.SkipTest(f"processed Beauty mini data not found ({mini}); set NLP_XRS_DATA_SOURCE")
        self.root = Path(tempfile.mkdtemp(prefix="nlp_xrs_mini_", dir=os.environ.get("NLP_XRS_SANDBOX_DIR")))
        mini_sandbox.stage(
            DATA_SOURCE, self.root, dataset="beauty", with_derived=True, with_caches=True, config_source=REPO / "config"
        )
        return self.root

    def stage_paper_profiles(self) -> bool:
        """Filter the full-fold paper user profiles onto the mini users; ``False`` if they are not cached."""
        src = mini_sandbox.feat_dir(DATA_SOURCE, "beauty") / "0" / "tokenizers" / "semantic_ids"
        if not all((src / name).is_file() for name in mini_sandbox.PAPER_PROFILE_NAMES):
            return False
        users = mini_sandbox.mini_users(mini_sandbox.feat_dir(self.root, "beauty"))
        mini_sandbox.filter_profiles(src, self.mini_fold / "tokenizers" / "semantic_ids", users)
        return True

    def teardown(self) -> None:
        if self.root is not None and not os.environ.get("NLP_XRS_KEEP_SANDBOX"):
            shutil.rmtree(self.root, ignore_errors=True)

    @property
    def mini_fold(self) -> Path:
        return mini_sandbox.mini_fold(self.root, "beauty")

    def args(self, config_id: str):
        args = get_default_args(config_id, dataset="beauty", test=True, base_path=str(self.root))
        args.cuda = torch.cuda.is_available()
        return args
