import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from nlp_xrs.utils.constants import PACKAGE_NAME
from nlp_xrs.utils.in_out import Colors, colored


class EarlyStopping:
    def __init__(
        self,
        monitor: str,
        ckpt_path: str,
        min_delta: float = 0.0,
        patience: int = 5,
        mode: str = "min",
        save_best: bool = True,
        syml_path: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.monitor = monitor
        self.patience = patience
        self.save_best = save_best
        self.sign = -1 if mode == "max" else 1
        self.min_delta = min_delta * self.sign
        self.ckpt_path = ckpt_path
        self.syml_path = syml_path
        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger(PACKAGE_NAME)

        if self.syml_path is None:
            ckpt_path = Path(self.ckpt_path)
            self.syml_path = ckpt_path.parent / "best" / ckpt_path.name

        if save_best:
            assert ckpt_path is not None

        self.patience_counter = 0
        self.best_epoch = 0

        self.best_trn_res = {}
        self.best_val_res = {self.monitor: np.inf}

    def reset(self, ckpts=()):
        self.patience_counter = 0
        self.best_epoch = 0

        self.best_trn_res = {}
        self.best_val_res = {self.monitor: np.inf}

        if ckpts:
            self.ckpt_path, self.syml_path = ckpts

    def setup_checkpoint(self, best_trn_res, best_val_res, best_epoch):
        self.best_trn_res.update(best_trn_res)
        self.best_val_res.update(best_val_res)
        self.best_epoch = best_epoch

    def save_checkpoint(self, epoch, model, optim, scheduler):
        if os.path.islink(self.syml_path):
            os.remove(self.syml_path)
        # Unwrap torch.compile's OptimizedModule so the saved state dict has
        # the same keys as the original model regardless of compile status.
        raw_model = getattr(model, "_orig_mod", model)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "monitor": self.monitor,
                "trn_res": self.best_trn_res,
                "val_res": self.best_val_res,
                # No need to store the random number generator states yet
                # 'torch_rng': torch.get_rng_state(),
                # 'torch_cuda_rng': torch.cuda.get_rng_state(),  # Defaults to current GPU. Use get_rng_state_all() if multi-GPU training.
                # 'numpy_rng': np.random.get_state(),
                # 'random_rng': random.getstate(),
                "model_stage": getattr(raw_model, "curr_stage", 0),
            },
            self.ckpt_path,
        )

    def get_best_results(self):
        return {
            self.monitor: self.best_val_res[self.monitor],
            "epoch": self.best_epoch,
            "trn": self.best_trn_res,
            "val": self.best_val_res,
        }

    def keys2text(self, d):
        if d:
            return {str(k): v for k, v in d.items()}
        return d

    def __call__(self, epoch, trn_res, val_res, model, optim, scheduler):
        trn_res, val_res = self.keys2text(trn_res), self.keys2text(val_res)
        # If there is no valid step, val_res will contain trn_res and trn_res will be dict()
        if (self.best_val_res[self.monitor] - val_res[self.monitor] - self.min_delta) * self.sign >= 0:
            self.patience_counter = 0
            self.best_epoch = epoch
            self.best_val_res.update(val_res)
            self.best_trn_res.update(trn_res)
            if self.save_best:
                # Save model params, epoch
                self.save_checkpoint(epoch, model, optim, scheduler)
        else:
            self.patience_counter += 1
            # print(f'Patience is running out... {self.patience - self.patience_counter} steps remaining...')
            self.logger.info(colored(f"Endured {self.patience_counter} time(s)", Colors.RED))
            if self.patience_counter >= self.patience:
                # self.logger.info(
                #     colored(funcs.now_time() + 'Cannot endure it anymore | Exiting from early stop', Colors.RED))
                os.symlink(self.ckpt_path, self.syml_path)
                self.logger.info(
                    f"\nLast checkpoint occurred at epoch {self.best_epoch} with validation score "
                    f"{self.best_val_res[self.monitor]:.4f}"
                )
                return True

        return False
