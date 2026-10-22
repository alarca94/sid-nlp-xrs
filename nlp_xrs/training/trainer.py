import gc
import importlib
import logging
import os
import shutil
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from nlp_xrs.evaluator import Evaluator
from nlp_xrs.training.callbacks import EarlyStopping
from nlp_xrs.training.scheduler import CosineScheduler, DummyScheduler
from nlp_xrs.utils import funcs
from nlp_xrs.utils.constants import CKPT_EXT, DATE_FORMAT, PACKAGE_NAME, InputType, Suffix, Task
from nlp_xrs.utils.in_out import Colors, colored, module_from_file, print_losses


def _cuda_sync(device):
    """Synchronize CUDA stream so perf_counter captures actual GPU completion."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def __call__(self, *args, **kwargs):
        if not self.shadow:
            self.register()
        self.update()
        self.apply_shadow()

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class BaseTrainer:
    def __init__(self, cfg, device, prediction_path, eval_only=False, resume_training=False, logger=None):
        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger(PACKAGE_NAME)
        self.device = device
        self.cfg = cfg
        self.max_epochs = self.cfg.train.epochs
        self._init_training()

        self.ckpt_path, self.ckpt_syml_path = self.get_ckpt_path()
        if not os.path.exists(self.ckpt_path):
            eval_only, resume_training = False, False
        self._init_tensorboard(eval_only, resume_training)

        self.start_epoch = 0
        self.early_stop = EarlyStopping(
            monitor="loss",
            ckpt_path=self.ckpt_path,
            syml_path=self.ckpt_syml_path,
            logger=self.logger,
            **vars(self.cfg.train.early_stop),
        )
        self.evaluator = Evaluator(
            self.model.TASKS,
            self.data,
            prediction_path,
            self.get_exp_metadata(),
            cfg.data.test_flag,
            cfg.gen_flag,
            cfg.save_results,
            self.model.MODEL_TYPE,
            self.cfg.paths.results,
            self.cfg.paths.ckpt,
            self.device,
            self.logger,
        )

    def _init_training(self):
        self.max_iters_per_epoch = getattr(self.cfg.train, "max_iters_per_epoch", float("inf"))

        self.model_class = self.get_model_class()
        self.data = self.load_data()
        # self.trn_data, self.val_data, self.tst_data = self.data.get_dataloaders()

        self.model = self.build_model()
        assert self.model is not None
        # Print the number of parameters in the model per module
        n_params = {
            name: sum(p.numel() for p in module.parameters())
            for name, module in self.model.named_modules()
            if "." not in name and name
        }
        total = sum(n_params.values())
        self.logger.info(f"Total number of parameters: {total:_}")
        for name, params in n_params.items():
            self.logger.info(f"\t{name}: {params:_} ({100 * params / total:.1f}%)")

        # Optional torch.compile (PyTorch 2.x) — enable via cfg.train.compile=true
        # or pass a mapping of kwargs: cfg.train.compile: {mode: reduce-overhead, backend: inductor}
        compile_cfg = getattr(self.cfg.train, "compile", False)
        if compile_cfg is not False and compile_cfg is not None:
            compile_kwargs = {} if isinstance(compile_cfg, bool) else vars(compile_cfg)
            self.logger.info(f"Compiling model with torch.compile({compile_kwargs})...")
            self.model = torch.compile(self.model, **compile_kwargs)

        self.set_optim_sched()

        # Mixed-precision (AMP) — enable via cfg.train.use_amp=true
        self.use_amp = getattr(self.cfg.train, "use_amp", False) and self.device.type == "cuda"
        self.amp_dtype = torch.bfloat16 if self.use_amp and torch.cuda.is_bf16_supported() else torch.float16
        self.scaler = torch.amp.GradScaler(self.device, enabled=self.use_amp)
        if self.use_amp:
            self.logger.info("Mixed-precision training (AMP - dtype: %s) enabled.", self.amp_dtype)

        # Initialize Exponential Moving Average (EMA) for teacher-student training
        if hasattr(self.cfg.train, "ema"):
            self.ema = EMA(self.model, getattr(self.cfg.train.ema, "decay", 0.95))

    def _init_tensorboard(self, eval_only, resume_training, tfb_folder=None):
        if not eval_only:
            if tfb_folder is None:
                tfb_folder = os.path.split(self.ckpt_path)[-1][: -len(CKPT_EXT)]
            if self.cfg.data.test_flag:
                tensorboard_path = os.path.join(self.cfg.paths.tmp_log, tfb_folder)
            else:
                tensorboard_path = os.path.join(self.cfg.paths.tensorboard, tfb_folder)
            # NOTE: Need to ignore_errors as it sometimes fails to remove the entire directory
            if os.path.isdir(tensorboard_path) and not resume_training:
                shutil.rmtree(tensorboard_path, ignore_errors=True)
                os.makedirs(tensorboard_path, exist_ok=True)
            self.tensorboard = SummaryWriter(tensorboard_path)

    def get_exp_metadata(self):
        cfg = self.cfg
        cfg_txt = funcs.dict2txt(funcs.flatten_dict(funcs.namespace2dict(cfg), key_sep="-"), entry_sep="|")
        return {
            "model_name": cfg.model.name + cfg.model.suffix,
            "dataset": cfg.data.dataset,
            "seed": cfg.seed,
            "split_ix": cfg.data.fold,
            "config": cfg_txt,
            "date": datetime.now(timezone.utc).strftime(DATE_FORMAT),
        }

    @staticmethod
    def get_optimizer(model_params, cfg):
        optim_name = cfg.__dict__.pop("name").lower()
        if optim_name == "sgd":
            cfg.lr = getattr(cfg, "lr", 1.0)
            return torch.optim.SGD(model_params, **vars(cfg))
        elif optim_name == "rmsprop":
            cfg.lr = getattr(cfg, "lr", 0.02)
            cfg.alpha = getattr(cfg, "alpha", 0.95)
            torch.optim.RMSprop(model_params, **vars(cfg))
        elif optim_name == "adam":
            cfg.lr = getattr(cfg, "lr", 1e-4)
            return torch.optim.Adam(model_params, **vars(cfg))
        elif optim_name == "adamw":
            cfg.lr = getattr(cfg, "lr", 1e-3)
            return torch.optim.AdamW(model_params, **vars(cfg))

    def get_scheduler(self, optim, cfg):
        if cfg.name.lower() == "steplr":
            return torch.optim.lr_scheduler.StepLR(
                optim, step_size=getattr(cfg, "step_size", 1), gamma=getattr(cfg, "gamma", 0.25)
            )
        elif cfg.name.lower() == "cosine":
            num_warmup_epochs = getattr(cfg, "num_warmup_epochs", self.max_epochs // 10)
            num_training_epochs = getattr(cfg, "num_training_epochs", self.max_epochs)
            return CosineScheduler(
                optim,
                warmup_epochs=num_warmup_epochs,
                max_epochs=num_training_epochs,
                min_factor=getattr(cfg, "min_factor", 0.1),
            )
        elif cfg.name.lower() == "reduceonplateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optim,
                mode=getattr(cfg, "mode", "min"),
                factor=getattr(cfg, "factor", 0.5),
                patience=getattr(cfg, "patience", 2),
                min_lr=getattr(cfg, "min_lr", 2e-5),
            )
        return DummyScheduler(optim)

    def set_optim_sched(self):
        self.optim = self.get_optimizer(self.model.parameters(), cfg=self.cfg.train.optim)
        self.sched = self.get_scheduler(self.optim, cfg=self.cfg.train.scheduler)

    def set_stage(self, stage):
        pass

    def load_best_model(self, ckpt=None, resume_training=False):
        self.logger.info(f"Loading best checkpoint ({self.ckpt_path})...")
        if ckpt is None:
            ckpt = torch.load(self.ckpt_path, map_location=self.device)
        self.start_epoch = ckpt["epoch"] + 1
        if resume_training:
            self.logger.info(f"Training will be resumed at epoch {self.start_epoch}...")
            self.optim.load_state_dict(ckpt["optimizer_state_dict"])
            self.sched.load_state_dict(ckpt["scheduler_state_dict"])
            self.early_stop.setup_checkpoint(ckpt.get("trn_res", {}), ckpt["val_res"], ckpt["epoch"])
        model = getattr(self.model, "_orig_mod", self.model)
        model.load_state_dict(ckpt["model_state_dict"])

    def get_ckpt_path(self):
        # NOTE: The symbolik link is most important in MultiStageTrainer to keep track of fully trained stages
        test_str = Suffix.TEST if self.cfg.data.test_flag else ""
        if getattr(self.cfg.model, "ckpt_ignore_suffix", False):
            model_id = self.cfg.model.name
        else:
            model_id = f"{self.cfg.model.name}{self.cfg.model.suffix}"

        if self.model_class.SEQ_MODE.requires_past_info():
            f_name = (
                f"{model_id}_{self.cfg.data.dataset}_{self.cfg.data.fold}_{self.cfg.data.hist_len}_"
                f"{self.cfg.seed}{test_str}"
            )
        else:
            f_name = f"{model_id}_{self.cfg.data.dataset}_{self.cfg.data.fold}_{self.cfg.seed}{test_str}"

        return os.path.join(self.cfg.paths.ckpt, f"{f_name}{CKPT_EXT}"), os.path.join(
            self.cfg.paths.best_ckpt, f"{f_name}{CKPT_EXT}"
        )

    def get_model_class(self):
        module_path = getattr(self.cfg.model, "path", f"{PACKAGE_NAME}.models")

        try:
            model_module = None
            if module_path == f"{PACKAGE_NAME}.models":
                for submodule in ["generative"]:
                    model_path = ".".join([module_path, submodule, self.cfg.model.name.lower()])
                    if importlib.util.find_spec(model_path, __name__):
                        model_module = importlib.import_module(model_path, __name__)
                        break
                if model_module is None:
                    raise NotImplementedError(f"{self.cfg.model.name} model was not found in any submodule.")
            else:
                # model_module = importlib.import_module(module_path)
                model_module = module_from_file(module_path, logger=self.logger)
            return getattr(model_module, self.cfg.model.name)
        except ModuleNotFoundError:
            raise ModuleNotFoundError(f'Model "{module_path}" could not be found.')

    def build_model(self):
        self.cfg.model.device = self.device
        return self.model_class(
            self.data.data_info, self.cfg.model, seed=self.cfg.seed, logger=self.logger, paths=self.cfg.paths
        ).to(self.device)

    def load_data(self):
        self.cfg.data.requires_exp = Task.EXPLANATION in self.model_class.TASKS or getattr(
            self.cfg.data, "requires_exp", False
        )
        self.cfg.data.requires_feat = Task.FEAT in self.model_class.TASKS or getattr(
            self.cfg.data, "requires_feat", False
        )
        self.cfg.data.requires_rating = Task.RATING in self.model_class.TASKS or getattr(
            self.cfg.data, "requires_rating", False
        )
        self.cfg.data.requires_context = Task.CONTEXT in self.model_class.TASKS or getattr(
            self.cfg.data, "requires_context", False
        )
        self.cfg.data.requires_nextitem = Task.NEXT_ITEM in self.model_class.TASKS or getattr(
            self.cfg.data, "requires_nextitem", False
        )
        self.cfg.data.seq_mode = self.model_class.SEQ_MODE

        data_path = f"{PACKAGE_NAME}.data.dataset"
        if self.model_class.INPUT_TYPE == InputType.REGULAR:
            data_class_name = "GenDataset"
        elif self.model_class.INPUT_TYPE == InputType.SEQUENTIAL:
            data_class_name = "SequentialDataset"
        elif self.model_class.INPUT_TYPE == InputType.CUSTOM:
            data_path = getattr(self.cfg.data, "path", data_path)
            data_class_name = getattr(self.cfg.data, "class_name", f"{self.cfg.model.name}Dataset")
        else:
            raise NotImplementedError("The selected model does not have an implemented input type")

        try:
            if data_path == f"{PACKAGE_NAME}.data.dataset":
                data_module = importlib.import_module(data_path)
            else:
                data_module = module_from_file(data_path, logger=self.logger)
            data_class = getattr(data_module, data_class_name)
        except Exception:
            raise ValueError(f'Unable to find the data class "{data_path}.{data_class_name}" could not be found.')

        return data_class(
            self.cfg.data,
            self.cfg.model,
            device=self.device,
            seed=self.cfg.seed,
            logger=self.logger,
            paths=self.cfg.paths,
        )

    def log_tensorboard(self, results, step):
        for desc, res in results.items():
            if isinstance(res, dict):
                for task, score in res.items():
                    self.tensorboard.add_scalar(f"{desc}/{task}", score, step)
            else:
                self.tensorboard.add_scalar(f"{desc}", res, step)

    def train_epoch(self, trn_data):
        # Turn on training mode which enables dropout.
        self.model.train()
        self.data.train()

        global_loss = defaultdict(float)  # {task: 0 for task in self.model.TASKS}
        total_samples = defaultdict(int)  # {task: 0 for task in self.model.TASKS}

        loader_iter = iter(trn_data)
        max_steps = min(len(trn_data), self.max_iters_per_epoch)

        for bix in range(max_steps):
            batch = next(loader_iter)

            batch_size = batch["size"]

            for k, v in batch.items():
                if hasattr(v, "is_cuda"):
                    batch[k] = v.to(self.device, non_blocking=True)

            self.optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp, dtype=self.amp_dtype):
                losses = self.model.compute_loss(batch)

            self.scaler.scale(losses["loss"]).backward()

            if self.cfg.train.clip_norm is not None:
                self.scaler.unscale_(self.optim)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.clip_norm)
            self.scaler.step(self.optim)
            self.scaler.update()

            for task in losses:
                global_loss[task] += batch_size * losses[task].item()
                total_samples[task] += batch_size

            step = bix + 1
            if step % self.cfg.train.log_interval == 0:
                print_losses(
                    {task: v / total_samples[task] for task, v in global_loss.items()},
                    self.device,
                    suffix=f"{step:5d}/{len(trn_data):5d} batches",
                    logger=self.logger,
                )

        global_loss = {task: v / total_samples[task] for task, v in global_loss.items()}
        print_losses(global_loss, self.device, suffix=f"{step:5d}/{len(trn_data):5d} batches", logger=self.logger)

        return global_loss

    def train_epoch_benchmark(self, trn_data):
        # Turn on training mode which enables dropout.
        self.model.train()
        self.data.train()

        benchmark_steps = getattr(self.cfg.train, "benchmark_steps", None)
        is_benchmarking = benchmark_steps is not None
        seg_times = defaultdict(float)  # accumulated seconds per segment
        seg_counts = 0

        # Enable fine-grained data benchmarking in the dataset (only works with num_workers=0)
        self.data._bench_enabled = is_benchmarking
        if is_benchmarking:
            self.data._bench.clear()

        global_loss = defaultdict(float)  # {task: 0 for task in self.model.TASKS}
        total_samples = defaultdict(int)  # {task: 0 for task in self.model.TASKS}

        loader_iter = iter(trn_data)
        max_steps = min(len(trn_data), self.max_iters_per_epoch)
        if is_benchmarking:
            max_steps = min(max_steps, benchmark_steps)

        for bix in range(max_steps):
            # ── Segment 1: data loading + collation ──────────────────────────
            _cuda_sync(self.device)
            t0 = time.perf_counter()
            batch = next(loader_iter)
            _cuda_sync(self.device)
            t1 = time.perf_counter()

            batch_size = batch["size"]

            # ── Segment 2: host-to-device transfer ───────────────────────────
            for k, v in batch.items():
                if hasattr(v, "is_cuda"):
                    batch[k] = v.to(self.device, non_blocking=True)
            _cuda_sync(self.device)
            t2 = time.perf_counter()

            # ── Segment 3: forward pass ───────────────────────────────────────
            self.optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp, dtype=self.amp_dtype):
                losses = self.model.compute_loss(batch)
            _cuda_sync(self.device)
            t3 = time.perf_counter()

            # ── Segment 4: backward pass ─────────────────────────────────────
            self.scaler.scale(losses["loss"]).backward()
            _cuda_sync(self.device)
            t4 = time.perf_counter()

            # ── Segment 5: optimizer step ─────────────────────────────────────
            # `clip_grad_norm` helps prevent the exploding gradient problem.
            if self.cfg.train.clip_norm is not None:
                self.scaler.unscale_(self.optim)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.clip_norm)
            self.scaler.step(self.optim)
            self.scaler.update()
            _cuda_sync(self.device)
            t5 = time.perf_counter()

            if is_benchmarking:
                seg_times["data_load"] += t1 - t0
                seg_times["host_to_device"] += t2 - t1
                seg_times["forward"] += t3 - t2
                seg_times["backward"] += t4 - t3
                seg_times["optim_step"] += t5 - t4
                seg_counts += 1

            for task in losses:
                global_loss[task] += batch_size * losses[task].item()
                total_samples[task] += batch_size

            step = bix + 1
            if step % self.cfg.train.log_interval == 0:
                print_losses(
                    {task: v / total_samples[task] for task, v in global_loss.items()},
                    self.device,
                    suffix=f"{step:5d}/{len(trn_data):5d} batches",
                    logger=self.logger,
                )
                if is_benchmarking:
                    self._log_benchmark(seg_times, seg_counts)

        if is_benchmarking and seg_counts > 0:
            self.logger.info("=== Benchmark Summary ===")
            self._log_benchmark(seg_times, seg_counts)

        global_loss = {task: v / total_samples[task] for task, v in global_loss.items()}
        print_losses(global_loss, self.device, suffix=f"{step:5d}/{len(trn_data):5d} batches", logger=self.logger)

        return global_loss

    def _log_benchmark(self, seg_times, n_steps):
        """Log mean per-step timings and percentage breakdown."""
        total = sum(seg_times.values())
        if total == 0:
            return
        labels = {
            "data_load": "Data load+collate",
            "host_to_device": "Host→Device     ",
            "forward": "Forward         ",
            "backward": "Backward        ",
            "optim_step": "Optim step      ",
        }
        lines = [f"[Benchmark] {n_steps} steps — total/step: {1000 * total / n_steps:.1f} ms"]
        for key, label in labels.items():
            ms = 1000 * seg_times[key] / n_steps
            pct = 100 * seg_times[key] / total
            lines.append(f"  {label}: {ms:6.2f} ms  ({pct:5.1f}%)")
        self.logger.info("\n".join(lines))
        self.data.log_data_benchmark(n_steps)

    def train(self, resume=False):
        if resume:
            if os.path.islink(self.ckpt_syml_path):
                self.logger.warning("Attempting to resume training on an already trained model! Exiting training...")
                return
            self.resume_training()

        epoch_times = {"Train": [], "Valid": []}
        val_task_res = None
        trn_data = self.data.get_dataloader("train")

        if hasattr(self.cfg.train, "epoch_log_steps"):
            epoch_steps = min(len(trn_data), self.max_iters_per_epoch)
            self.cfg.train.log_interval = max(int(np.ceil(epoch_steps / self.cfg.train.epoch_log_steps)), 1)

        if self.data.do_eval:
            val_data = self.data.get_dataloader("valid")

        for epoch in range(self.start_epoch, self.max_epochs):
            self.logger.info(f"Epoch {epoch}")

            st = time.time()
            trn_task_res = self.train_epoch(trn_data)
            epoch_times["Train"].append(time.time() - st)

            epoch_info = {"Train": trn_task_res, "LR": self.sched.get_last_lr()[0]}

            if self.data.do_eval:
                st = time.time()
                if hasattr(self.cfg.train, "ema"):
                    self.ema()

                val_task_res = self.evaluate(val_data, "valid")
                # Some models compute a validation loss that is different from the training aggregated loss
                if callable(getattr(self.model, "get_val_loss", None)):
                    val_task_res["agg_loss"] = val_task_res.pop("loss")
                    val_task_res["loss"] = self.model.get_val_loss(val_task_res)

                epoch_times["Valid"].append(time.time() - st)
                epoch_info["Valid"] = val_task_res

                print_losses(
                    val_task_res,
                    self.device,
                    suffix=f"valid loss {val_task_res['loss']:4.4f}",
                    color=Colors.GREEN,
                    logger=self.logger,
                )

                if hasattr(self.cfg.train, "ema"):
                    self.ema.restore()

            self.log_tensorboard(epoch_info, epoch)
            torch.cuda.empty_cache()
            if self.data.do_eval:
                if self.early_stop(epoch, trn_task_res, val_task_res, self.model, self.optim, self.sched):
                    break
            elif self.early_stop(epoch, {}, trn_task_res, self.model, self.optim, self.sched):
                break

            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            self.sched.step()
            self.logger.info(colored(f"Learning rate set to {self.sched.get_last_lr()[0]:2.8f}", Colors.YELLOW))

        self.logger.info(
            f"Avg. Train Time: {timedelta(seconds=np.mean(epoch_times['Train']))} ||| "
            f"Avg. Valid Time: {timedelta(seconds=np.mean(epoch_times['Valid']))}"
        )

        return self.early_stop.get_best_results()

    def test(self):
        self.logger.info("=" * 89)
        if not os.path.isfile(self.ckpt_syml_path):
            self.logger.warning("Attempting to evaluate a model that was not fully trained!")
        self.load_best_model()

        # Run on test data.
        tst_data = self.data.get_dataloader("test")
        test_task_losses = self.evaluate(tst_data, "test")
        print_losses(test_task_losses, self.device, suffix="End of training", logger=self.logger)

        self.logger.info("Generating text")
        self.generate(tst_data)

    @torch.no_grad()
    def evaluate(self, data, phase):
        # Turn on evaluation mode which disables dropout.
        self.model.eval()
        self.data.eval(phase)

        global_loss = defaultdict(float)  # {task: 0 for task in self.model.TASKS}
        global_loss["loss"] = 0
        total_sample = 0
        for bix, batch in enumerate(data):
            # if bix == 55:
            #     break
            batch_size = batch["size"]

            # Send all tensors in the batch to self.device
            for k, v in batch.items():
                if hasattr(v, "is_cuda"):
                    batch[k] = v.to(self.device, non_blocking=True)

            with torch.autocast(device_type=self.device.type, enabled=self.use_amp, dtype=self.amp_dtype):
                task_loss = self.model.compute_loss(batch)

            for task in task_loss:
                global_loss[task] += batch_size * task_loss[task].item()

            total_sample += batch_size

        return {task: v / total_sample for task, v in global_loss.items()}

    @torch.no_grad()
    def generate(self, data, gen_steps=0, phase="test"):
        # Turn on evaluation mode which disables dropout.
        self.model.eval()
        self.data.eval(phase)

        preds = defaultdict(list)
        labels = defaultdict(list)
        for bix, batch in enumerate(data):
            # Send all tensors in the batch to self.device
            for k, v in batch.items():
                if hasattr(v, "is_cuda"):
                    batch[k] = v.to(self.device, non_blocking=True)

            # NOTE: The results returned by mode.generate must be casted to list
            res, m_labels = self.model.generate(batch)

            for task in res:
                preds[task].extend(res[task])

            if gen_steps != 0 and (bix + 1) >= gen_steps:
                break

        labels.update(self.data.get_gen_labels(gen_steps, len(data), data.batch_size, preds.keys(), phase))

        # Free some memory
        del batch, data
        torch.cuda.empty_cache()
        gc.collect()

        if hasattr(self.model, "postprocess"):
            self.logger.info("Starting Post-processing step...")
            preds = self.model.postprocess(preds, labels)

        if gen_steps == 0:
            self.evaluator.evaluate(labels, preds)
        else:
            self.evaluator.print_generated(labels, preds)

    def resume_training(self):
        self.load_best_model(resume_training=True)


class MultiStageTrainer(BaseTrainer):
    def __init__(self, cfg, device, prediction_path, eval_only=False, resume_training=False):
        # NOTE: Multi-Stage Training does not currently handle HyperOpt search
        self.ms_cfg = cfg
        self.curr_stage = cfg.start_stage

        super().__init__(cfg.stages[self.curr_stage], device, prediction_path, eval_only, resume_training)

    # def _init_tensorboard(self, eval_only, resume_training, tfb_folder=None):
    #     if tfb_folder is None:
    #         tfb_folder = os.path.split(self.ckpt_path)[-1][:-len(CKPT_EXT)] + f'_{self.curr_stage}'
    #     super()._init_tensorboard(eval_only, resume_training, tfb_folder)

    def get_exp_metadata(self):
        cfg = self.ms_cfg.stages[0]
        cfg_txt = []
        for i, stage in enumerate(self.ms_cfg.stages):
            cfg_txt.append(f"{getattr(stage, 'name', str(i)).replace('-', '_')}_stage>")
            stage = SimpleNamespace(**vars(stage))
            delattr(stage, "name")
            cfg_txt[-1] += funcs.dict2txt(funcs.flatten_dict(funcs.namespace2dict(stage), key_sep="-"), entry_sep="|")
        cfg_txt = "\n".join(cfg_txt)
        return {
            "model_name": cfg.model.name + cfg.model.suffix,
            "dataset": cfg.data.dataset,
            "seed": cfg.seed,
            "split_ix": cfg.data.fold,
            "config": cfg_txt,
            "date": datetime.now(timezone.utc).strftime(DATE_FORMAT),
        }

    def set_config(self):
        if self.curr_stage == 0:
            self.cfg = self.ms_cfg.stages[0]
        else:
            funcs.nested_cfg_update(self.cfg, self.ms_cfg.stages[self.curr_stage], inplace=True)
        self.cfg.data.stage = self.curr_stage
        self.cfg.model.stage = self.curr_stage

    def get_ckpt_path(self, stage=None):
        if stage is None:
            stage = self.curr_stage

        ckpt_path, ckpt_syml_path = super().get_ckpt_path()
        # Add stage number to the checkpoint path (Tensorboard will create a folder per stage)
        return ckpt_path.replace(CKPT_EXT, f"_stage{stage}{CKPT_EXT}"), ckpt_syml_path.replace(
            CKPT_EXT, f"_stage{stage}{CKPT_EXT}"
        )

    def set_stage(self, stage, load_best=False, resume_training=False):
        self.curr_stage = stage
        # Update the config nested SimpleNamespace with the new stage configuration
        self.set_config()
        # Initialize Data, Model, Optim., Sched. with the new configurations (including the updated "stage" argument)
        self._init_training()
        # Load the best model so far (skip the load of Optim, Sched and Callbacks if resume_training)
        if load_best:
            self.load_best_model(resume_training=resume_training)

        if not resume_training:
            self.early_stop.reset(ckpts=(self.ckpt_path, self.ckpt_syml_path))

        # NOTE: By default, the max. number of epochs for the whole training is defined by cfg.train.epochs
        #  If cfg.train.epochs is specified for different stages, max_epochs will be extended from the last saved epoch.
        # NOTE: This configuration does not properly work when training is resumed at a later stage
        if self.curr_stage > 0 and hasattr(self.ms_cfg.stages[self.curr_stage].train, "epochs"):
            self.max_epochs = self.start_epoch + self.cfg.ms_cfg[self.curr_stage].train.epochs

    def resume_training(self, force=False):
        if os.path.isfile(self.ckpt_path):
            if os.path.islink(self.ckpt_syml_path) and not force:
                # Check which is the latest stage that was not fully trained
                stage = self.curr_stage + 1
                ckpt_syml_path = self.ckpt_syml_path.replace(f"_stage{self.curr_stage}", f"_stage{stage}")
                while stage < self.model.n_stages and os.path.islink(ckpt_syml_path):
                    stage += 1
                    ckpt_syml_path = ckpt_syml_path.replace(f"_stage{stage - 1}", f"_stage{stage}")

                if stage == self.model.n_stages:
                    # NOTE: "--force-training" is not yet implemented
                    self.logger.warning(
                        "Exiting training as this model was fully trained for all stages. If this behavior "
                        'needs to be overriden, pass the program argument "--force-training".'
                    )
                    return False
                else:
                    self.logger.warning(f"Training will be resumed at stage {stage}")
                    ckpt_path = self.ckpt_path.replace(f"_stage{self.curr_stage}", f"_stage{stage}")
                    if os.path.isfile(ckpt_path):
                        self.set_stage(stage, load_best=True, resume_training=True)
                    else:
                        self.set_stage(stage, load_best=True, resume_training=False)
            else:
                if force:
                    pass
                self.load_best_model(resume_training=True)
        else:
            stage = self.curr_stage - 1
            ckpt_syml_path = self.ckpt_syml_path.replace(f"_stage{self.curr_stage}", f"_stage{stage}")
            while stage >= 0 and not os.path.islink(ckpt_syml_path):
                stage -= 1
                ckpt_syml_path = ckpt_syml_path.replace(f"_stage{stage + 1}", f"_stage{stage}")
            ckpt_path = self.ckpt_path.replace(f"_stage{self.curr_stage}", f"_stage{stage + 1}")
            if stage < 0 and not os.path.isfile(ckpt_path):
                self.logger.warning("Training will start from scratch")
            else:
                if os.path.isfile(ckpt_path):
                    resume = True
                else:
                    resume = False

                if stage == self.curr_stage - 1:
                    self.load_best_model(resume_training=resume)
                else:
                    self.logger.warning("Start stage ")
                    self.set_stage(stage + 1, load_best=True, resume_training=resume)

        return True

    def load_best_model(self, ckpt=None, resume_training=False):
        self.logger.info(f"Loading best checkpoint ({self.ckpt_path})...")
        if ckpt is None:
            if not os.path.isfile(self.ckpt_path) and not resume_training:
                ckpt_path = self.ckpt_path.replace(f"_stage{self.curr_stage}", f"_stage{self.curr_stage - 1}")
                syml_path = self.ckpt_syml_path.replace(f"_stage{self.curr_stage}", f"_stage{self.curr_stage - 1}")
                if os.path.islink(syml_path) and os.path.isfile(ckpt_path):
                    ckpt = torch.load(ckpt_path, map_location=self.device)
                else:
                    raise FileNotFoundError(
                        'Either the checkpoint or the "best" symlink for this model stage does not exist!'
                    )
            else:
                ckpt = torch.load(self.ckpt_path, map_location=self.device)
        self.start_epoch = ckpt["epoch"] + 1
        if resume_training:
            msg = f"Training will be resumed at epoch {self.start_epoch}"
            if self.model.n_stages > 1:
                msg += f" (stage {self.model.curr_stage})"
            self.logger.info(f"{msg}...")
            self.optim.load_state_dict(ckpt["optimizer_state_dict"])
            self.sched.load_state_dict(ckpt["scheduler_state_dict"])
            self.early_stop.setup_checkpoint(ckpt.get("trn_res", {}), ckpt["val_res"], ckpt["epoch"])
        elif ckpt["model_stage"] == self.curr_stage - 1:
            self.start_epoch += self.early_stop.patience
        elif ckpt["model_stage"] < self.curr_stage - 1:
            self.logger.warning(
                f"Attempting to train model for stage {self.curr_stage} but last checkpoint is from "
                f"stage {ckpt['model_stage']}. Trainer will be set to stage {ckpt['model_stage'] + 1} "
                f"to continue training on the corresponding stage."
            )

        self.model.load_state_dict(ckpt["model_state_dict"])

    def train(self, resume=False):
        if resume:
            if not self.resume_training():
                return

        for stage in range(self.curr_stage, self.model.n_stages):
            stage_name = getattr(self.ms_cfg.stages[stage], "name", "")
            self.logger.info("Training stage {}{}".format(stage, f" ({stage_name.title()})" if stage_name else ""))
            # Only set new stage if training just entered that stage. If resuming training, setting the stage would
            #  just reset the optimizer and scheduler states.
            if not resume and stage > 0:
                self.logger.info("Configuring trainer for the new training stage...")
                self.set_stage(stage, load_best=True)  # Updates config and re-creates model and data with curr_stage
            else:
                resume = False

            super().train(resume=False)
