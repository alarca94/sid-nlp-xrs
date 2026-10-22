"""End-to-end mini runs for SID and SAE-concept recommender configs.

Runs inside a temporary base path (see ``mini_base``). The first call for a
tokenizer config trains it and writes one bundle, ``<config>_mini.pt``, under
the ``_mini`` fold of the sandbox. Later tests reuse that file.
"""

import os
import subprocess
import sys
import unittest
from pathlib import Path

import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from nlp_xrs.run import run_experiment
from tests.test_model.mini_base import REPO, MiniSandbox

SANDBOX = MiniSandbox()
SAE_ACTS_NAME = "acts_Qwen--Qwen3-Embedding-0.6B_small_1111.pt"


def setUpModule():
    SANDBOX.setup()


def tearDownModule():
    SANDBOX.teardown()


def _sid_dir() -> Path:
    return SANDBOX.mini_fold / "tokenizers" / "semantic_ids"


def _run(config_id: str) -> None:
    args = SANDBOX.args(config_id)
    args.txt_len = 8
    run_experiment(args)


def _assert_one_sid_file(config_name: str) -> Path:
    sid_dir = _sid_dir()
    bundle = sid_dir / f"{config_name}_mini.pt"
    if not bundle.is_file():
        raise AssertionError(f"Expected one SID bundle at {bundle}")
    leftovers = [path for path in (sid_dir / config_name, sid_dir / "_emb_cache" / config_name) if path.exists()]
    if leftovers:
        raise AssertionError(f"{config_name} left extra artifacts: {leftovers}")
    payload = torch.load(bundle, weights_only=False, map_location="cpu")
    if "mapping" not in payload or "model" not in payload:
        raise AssertionError(f"{bundle} is not a single mapping+model bundle")
    return bundle


def _ensure_mini_sae() -> None:
    acts = SANDBOX.mini_fold / "SAE" / SAE_ACTS_NAME
    if acts.is_file():
        return
    subprocess.check_call(
        [
            sys.executable,
            str(REPO / "pretrain_sae.py"),
            "--dataset",
            "beauty",
            "--config",
            "small",
            "--test",
            "--seed",
            "1111",
            "--base-path",
            str(SANDBOX.root),
            "--emb_batch_size",
            "16",
            "--max-seq-length",
            "64",
            "--max-steps",
            "2",
        ],
        cwd=REPO,
    )
    if not acts.is_file():
        raise AssertionError(f"SAE activations were not written to {acts}")


class TestSidSaeEndToEnd(unittest.TestCase):
    def test_nrt_sid(self):
        _run("ipm26/sid_xrs/nrt-sid_agg__rqfaiss_qwen_3cb")
        _assert_one_sid_file("rqfaiss_qwen_3cb")

    def test_peter_sid(self):
        _run("ipm26/sid_xrs/peter-sid_agg__opq_qwen_3cb")
        _assert_one_sid_file("opq_qwen_3cb")

    def test_sequer_sid(self):
        _run("ipm26/sid_xrs/sequer-sid_agg__rqvae_qwen_3cb")
        _assert_one_sid_file("rqvae_qwen_3cb")

    def test_tfs_sid(self):
        _run("ipm26/sid_xrs/tfs-nofusion-tfidfmean-usetext-altmultiple-agg__rqfaiss_qwen_3cb")
        _assert_one_sid_file("rqfaiss_qwen_3cb")

    def test_peter_sae_conditioning(self):
        _ensure_mini_sae()
        _run("ipm26/sid_xrs/peter-sid_agg_sae_small_ext__rqfaiss_qwen_3cb")
        _assert_one_sid_file("rqfaiss_qwen_3cb")

    def test_sequer_sae_conditioning(self):
        _ensure_mini_sae()
        _run("ipm26/sid_xrs/sequer-sid_agg_sae_small_ext__rqfaiss_qwen_3cb")

    def test_tfs_sae_conditioning(self):
        _ensure_mini_sae()
        _run("ipm26/sid_xrs/sequertextfusion-sid_agg_sae_small_ext__rqfaiss_qwen_3cb")

    def test_peter_sae_supervision(self):
        if not SANDBOX.stage_paper_profiles():
            self.skipTest("cached paper user profiles for fold 0 are not available")
        _ensure_mini_sae()
        _run("ipm26/sid_xrs/peter-sid_agg_ui_sae_small")
        _assert_one_sid_file("rqvae_qwen_3cb")
        _assert_one_sid_file("rqvae_user_profiles")


if __name__ == "__main__":
    unittest.main()
