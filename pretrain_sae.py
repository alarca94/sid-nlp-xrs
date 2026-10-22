import argparse
import logging
import math
import os

import matplotlib
import numpy as np
import pandas as pd
import torch
import yaml
from sentence_transformers import SentenceTransformer
from torch import nn

from nlp_xrs.utils.constants import Key
from nlp_xrs.utils.funcs import init_seed
from train_tokenizer import load_dataset

matplotlib.use("Agg")
from collections import defaultdict
from collections.abc import Iterable
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from torch.utils.tensorboard.writer import SummaryWriter


def get_scheduler(scheduler_cfg: Dict, opt: torch.optim.Optimizer) -> Optional[LRScheduler]:
    """
    Initialize lr-scheduler based on configuration.
    """
    scheduler_type: Optional[str] = scheduler_cfg.pop("type", None)
    if scheduler_type is None:
        logging.info("no lr_scheduling is set")
        return None
    scheduler_type = scheduler_type.strip().lower()
    scheduler_dict = {"warmup_linear": WarmupLinearScheduler}

    if scheduler_type not in scheduler_dict:
        raise ValueError(f"Unknown scheduler_type {scheduler_type}")
    logging.info("lr_scheduling is set to %s", scheduler_type)

    return scheduler_dict[scheduler_type](optimizer=opt, **scheduler_cfg)


def autoencoder_loss(
    reconstruction: torch.Tensor,
    original_input: torch.Tensor,
    latent_activations: torch.Tensor,
    l1_weight: float,
) -> torch.Tensor:
    """
    :param reconstruction: output of Autoencoder.decode (shape: [batch, n_inputs])
    :param original_input: input of Autoencoder.encode (shape: [batch, n_inputs])
    :param latent_activations: output of Autoencoder.encode (shape: [batch, n_latents])
    :param l1_weight: weight of L1 loss
    :return: loss (shape: [1])
    """
    normed_mse = (((reconstruction - original_input) ** 2).mean(dim=1) / (original_input**2).mean(dim=1)).mean()
    normed_l1 = (latent_activations.abs().sum(dim=1) / original_input.norm(dim=1)).mean()
    return normed_mse + l1_weight * normed_l1


def normalize(x: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu = x.mean(dim=-1, keepdim=True)
    x = x - mu
    std = x.std(dim=-1, keepdim=True)
    x = x / (std + eps)
    return x, mu, std


def id_utilization_rate(ids: Union[np.ndarray, List[int]], n_ids: int) -> float:
    """
    Calculates the utilization rate for the given IDs given the ID space defined by the number of IDs.
    Utilization rate for non-empty lists is in the range [1/n_ids, 1] where utilization rate = 1/n_ids if only a single
    ID is used, and 1 if all IDs are used.

    :param ids: List of IDs.
    :param n_ids: Total number of possible IDs in the ID space.
    :return: The utilization rate.
    """
    n_unique = np.unique(ids, axis=0).shape[0]
    utilization_rate = n_unique / n_ids

    return utilization_rate


def id_perplexity_ratio(ids: Union[np.ndarray, List[int]], n_ids: int) -> float:
    """
    Calculates the perplexity ratio for the given IDs given the ID space.
    Perplexity ratio is in the range [1/n_ids, 1] where perplexity ratio = 1/n_ids if only a single ID is used, and 1 if
    samples are uniformly distributed across all IDs.

    :param ids: List of IDs.
    :param n_ids: Total number of possible IDs in the ID space.
    :return: Perplexity ratio.
    """
    _, counts = np.unique(ids, axis=0, return_counts=True)
    # Calculate empirical probabilities
    probabilities = counts / counts.sum()
    # Calculate Shannon entropy
    entropy = -(probabilities * np.log(probabilities)).sum()
    # Calculate perplexity
    perplexity = np.exp(entropy)
    # Calculate perplexity ratio
    perplexity_ratio = perplexity / n_ids

    return perplexity_ratio


def id_collision_rate(ids: Union[np.ndarray, List[int]]) -> float:
    """
    Calculates the collision rate for the given IDs.
    The collision rate is the number of ID collisions per sample. Collision rate = 0 if every ID appears only once and
    1 - 1 / n_samples if all IDs are identical (max collisions).

    :param ids: List of IDs.
    :return: Collision rate.
    """
    n_ids = len(ids)
    n_unique = len(np.unique(ids, axis=0))
    collisions = n_ids - n_unique
    collision_rate = collisions / n_ids

    return collision_rate


def id_gini_coefficient(ids: Union[np.ndarray, List[int]], n_ids: int) -> float:
    """
    Calculates the Gini coefficient for the given IDs given the ID space.

    Comparing to the conventional example for the Gini coefficient of income or wealth distribution within a population,
    the population consists of the theoretical IDs and the income or wealth of each individual is the number of times
    that ID appears in the list of IDs.

    :param ids: List of IDs.
    :param n_ids: Total number of possible IDs in the ID space.
    :return: Gini coefficient.
    """
    # Calculate the frequency of each ID
    _, counts = np.unique(ids, return_counts=True, axis=0)
    counts = np.concatenate([np.zeros(n_ids - len(counts), dtype=counts.dtype), counts])

    # Sort the counts in ascending order
    counts = np.sort(counts)

    n = n_ids
    total = counts.sum()

    if n_ids == 0 or total == 0:
        return 0.0

    indices = np.arange(1, n + 1)
    gini = np.dot(2.0 * indices - n - 1, counts) / (n * total)

    return float(gini)


def check_id_distribution(
    semantic_ids: torch.Tensor,
    n_codebooks: int,
    max_n_ids: List[int],
    metric_names: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Function for checking distribution of semantic IDs relative to the possible semantic ID space.

    :param semantic_ids: tensor of semantic IDs
    :param n_codebooks: number of codebooks
    :param max_n_ids: maximum number of IDs supported by this ID space
    :return: utilization rate, perplexity ratio, collision rate, Gini coefficient
    """
    metrics = {}
    if n_codebooks > 1:
        for i in range(n_codebooks):
            if verbose:
                logging.info(f"Checking distribution for ID {i + 1}")
            cb_metrics = check_id_distribution(
                semantic_ids[:, i], 1, [max_n_ids[i]], metric_names=metric_names, verbose=verbose
            )
            metrics |= {f"cb{i}_{k}": v for k, v in cb_metrics.items()}
        if verbose:
            logging.info("Checking distribution for whole-IDs")

    max_ids = math.prod(max_n_ids)

    numpy_ids = semantic_ids.cpu().numpy()
    if metric_names is None:
        metric_names = [Metric.UTILIZATION, Metric.COLLISION, Metric.PERPLEXITY, Metric.GINI]
    for metric_n in metric_names:
        match metric_n:
            case Metric.UTILIZATION:
                metrics[metric_n] = id_utilization_rate(numpy_ids, max_ids)
                optimal_v = 1.0
                vocab2code_ratio = semantic_ids.shape[0] / max_ids
                metrics[f"{metric_n}_ratio"] = metrics[metric_n] / vocab2code_ratio
                if verbose:
                    logging.info(f"\\tUtilization (item-level): {metrics[metric_n + '_ratio']:.4f} (Optim: 1.0)")
                    logging.info(f"\\tVocab2Space ratio: {vocab2code_ratio:.4f}")
            case Metric.COLLISION:
                metrics[metric_n] = id_collision_rate(numpy_ids)
                optimal_v = 0.0
            case Metric.PERPLEXITY:
                metrics[metric_n] = id_perplexity_ratio(numpy_ids, max_ids)
                optimal_v = 1.0
            case Metric.GINI:
                metrics[metric_n] = id_gini_coefficient(numpy_ids, max_ids)
                optimal_v = 0.0
            case _:
                raise NotImplementedError(f"Unsupported metric {metric_n}")

        if verbose:
            logging.info(f"\\t{metric_n.title()} rate: {metrics[metric_n]:.4f} (Optim: {optimal_v:1f})")

    return metrics


class Metric:
    UTILIZATION: str = "utilization"
    COLLISION: str = "collision"
    PERPLEXITY: str = "perplexity"
    GINI: str = "gini"


class WarmupLinearScheduler(LambdaLR):
    def __init__(self, optimizer, warmup_steps, scheduler_steps, min_ratio=0.01, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.scheduler_steps = scheduler_steps
        self.min_ratio = min_ratio
        super().__init__(optimizer, self.lr_lambda, last_epoch)

    def lr_lambda(self, step: int) -> float:
        if step < self.warmup_steps:
            # Linear warmup: 0 -> 1
            return float(step) / float(max(1, self.warmup_steps))
        if step <= self.scheduler_steps:
            # Linear decay: 1 -> min_ratio
            progress = float(step - self.warmup_steps) / float(max(1, self.scheduler_steps - self.warmup_steps))
            return max(self.min_ratio, 1.0 - (1.0 - self.min_ratio) * progress)

        return self.min_ratio


class TiedTranspose(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.linear = linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.linear.bias is None
        return nn.functional.linear(x, self.linear.weight.t(), None)

    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight.t()

    @property
    def bias(self) -> torch.Tensor:
        return self.linear.bias


class DimReducer(nn.Module):
    def __init__(self):
        super().__init__()

    def fit(self, train_x: torch.Tensor, return_projected: bool = False) -> Optional[torch.Tensor]:
        if return_projected:
            return train_x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.reconstruct(self.reduce(x))

    def reduce(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return x


class PCA(DimReducer):
    def __init__(self, cfg: Dict):
        super().__init__()

        self.input_dim = cfg["input_dim"]
        self.keep_variance = cfg.get("keep_variance", 0.9)
        self.whiten = cfg.get("whiten", False)

        k = torch.tensor(cfg.get("latent_dim")) if "latent_dim" in cfg else None
        self.register_buffer("k", k)
        self.register_buffer("mu", torch.zeros((1, self.input_dim)))
        self.register_buffer("V", torch.zeros((self.input_dim, self.input_dim)))
        self.register_buffer("S", torch.zeros((self.input_dim,)))
        self.register_buffer("eigvals", torch.zeros((self.input_dim,)))
        self.register_buffer("whiten_scale", torch.ones((self.input_dim,)))

    def fit(self, train_x: torch.Tensor, return_projected: bool = False) -> Optional[torch.Tensor]:
        self.mu = train_x.mean(dim=0, keepdim=True)
        x_centered = train_x - self.mu

        _, S, Vh = torch.linalg.svd(x_centered, full_matrices=False)
        self.V = Vh.T
        self.S = S
        self.eigvals = S**2 / (train_x.shape[0] - 1)

        explained_var = (self.eigvals / self.eigvals.sum()).cumsum(dim=0)
        k = torch.argmax((explained_var >= self.keep_variance).int()) + 1
        if self.k is None:
            self.k = k

        if self.k < k:
            logging.warning(
                f"Latent dimension is set to {self.k.item()}. The explained variance ({explained_var[self.k.item()]:.3f} is lower than "
                f"`keep_variance` ({self.keep_variance})!). Set `latent_dim` = {k.item()} to satisfy the requirement."
            )
        else:
            logging.info(
                f"The explained variance with {self.k.item()} components is {explained_var[self.k.item()]:.3f}"
            )

        if self.whiten:
            n = train_x.shape[0]
            self.whiten_scale = torch.sqrt(torch.tensor(n - 1, dtype=self.S.dtype, device=self.S.device)) / self.S

        if return_projected:
            return self.reduce(train_x)

    def reduce(self, x: torch.Tensor) -> torch.Tensor:
        projected = (x - self.mu) @ self.V[:, : self.k.item()]
        if self.whiten:
            projected = projected * self.whiten_scale[: self.k.item()]
        return projected

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        z = x / self.whiten_scale[: self.k.item()] if self.whiten else x
        return z @ self.V[:, : self.k.item()].T + self.mu


class SAE(nn.Module):
    """Residual Quantized Variational Autoencoder"""

    def __init__(
        self,
        tokenizer_cfg: Dict,
        device: Union[torch.device, str],
    ):
        """
        :param input_dim: Input dimension.
        :param latent_dim: Latent dimension, dimension of the quantization space.
        :param n_codebooks: Number of codebooks in the quantizer.
        :param codebook_size: Number of entries in each codebook.
        :param encoder_dims: List of encoder intermediate dimensions.
        :param decoder_dims: List of decoder intermediate dimensions.
        :param sample: Optional sample of data to initialize codebooks with residual k-means.

        In OpenAI's SAE paper (2024), they evaluate ReLU vs ProLU STE vs Gated vs TopK. Findings:
        - TopK leads to lowest MSE of reconstruction. Others require an L1 penalty (imperfect approximation of L_0).
        - Up to 90% of dead latents if careless. Solutions:
          - Tie encoder-decoder weights.
          - Auxiliary loss: reconstruction error using top-k_{aux} dead latents.
        - Dead latents are considered if it has not activated in N tokens (1e6 in the paper)
        """
        super().__init__()

        self.input_dim: int = tokenizer_cfg.get("input_dim")
        self.latent_dim: int = tokenizer_cfg.get("latent_dim")
        self.codebook_size: int = tokenizer_cfg.get("codebook_size")
        self.norm_input: bool = tokenizer_cfg.get("norm_input", False)
        # self.act_type: str = tokenizer_cfg.get("act_type", "topk").strip().lower
        self.top_k: int = tokenizer_cfg.get("top_k", 3)
        self.dead_after_n: int = int(tokenizer_cfg.get("dead_after_n", 1e3))
        self.aux_top_k: int = tokenizer_cfg.get("aux_top_k", 0)
        self.loss_weights = {"reconstruct": None, "aux": tokenizer_cfg.get("aux_lambda", 1 / 32)}

        if self.aux_top_k + self.top_k > self.codebook_size:
            raise ValueError("`aux_k` + `n_codebooks` (k) cannot exceed `codebook_size`")

        # Dim. Reduction: Semantic embeddings can be of greater dimension than codebook
        dim_reduction_cfg: Dict = tokenizer_cfg.get("dim_reduction", {})
        dim_reduction_type: str = dim_reduction_cfg.get("type", "pca").strip().lower()
        dim_reduction_cfg["latent_dim"] = self.latent_dim
        dim_reduction_cfg["input_dim"] = self.input_dim
        if dim_reduction_type == "pca":
            self.dim_reducer: DimReducer = PCA(dim_reduction_cfg)
        elif dim_reduction_type == "none":
            self.dim_reducer: DimReducer = DimReducer()  # Dummy
            assert self.input_dim == self.latent_dim
        else:
            raise NotImplementedError(f"{dim_reduction_type} is an invalid `type`. Please use one of [pca].")

        # SAE: Concept Extraction -- Equivalent to single codebook
        self.pre_bias = nn.Parameter(torch.zeros(self.latent_dim))
        self.latent_bias = nn.Parameter(torch.zeros(self.codebook_size))
        self.encoder = nn.Linear(self.latent_dim, self.codebook_size, bias=False)
        if tokenizer_cfg.get("tied", False):
            self.decoder = TiedTranspose(self.encoder)
        else:
            self.decoder = nn.Linear(self.codebook_size, self.latent_dim, bias=False)
        # self.codebook = torch.nn.Embedding(self.codebook_size, self.input_dim)

        self.stats_last_nonzero: torch.Tensor
        self.latents_activation_frequency: torch.Tensor
        self.latents_mean_square: torch.Tensor
        self.register_buffer("stats_last_nonzero", torch.zeros(self.codebook_size, dtype=torch.long))
        self.register_buffer("latents_activation_frequency", torch.ones(self.codebook_size, dtype=torch.float))
        self.register_buffer("latents_mean_square", torch.zeros(self.codebook_size, dtype=torch.float))

        self.device = device
        self.to(device)

    @property
    def codebook_sizes(self) -> List[int]:
        return [self.codebook_size]

    @property
    def num_codebooks(self) -> int:
        return self.top_k

    @staticmethod
    def _geometric_median(x: torch.Tensor, eps: float = 1e-6, max_iter: int = 200) -> torch.Tensor:
        median = x.mean(dim=0)  # [E]

        for _ in range(max_iter):
            # Distances from current estimate to each point: [B]
            dists = torch.norm(x - median.unsqueeze(0), dim=-1)  # [B]

            # Clamp to avoid division by zero for points coinciding with median
            dists = dists.clamp(min=eps)

            # Weiszfeld weights: inverse distances [B]
            weights = 1.0 / dists  # [B]

            # Weighted mean update: new median estimate [E]
            new_median = (weights.unsqueeze(1) * x).sum(dim=0) / weights.sum()

            # Check convergence
            if torch.norm(new_median - median) < eps:
                break

            median = new_median

        return median

    def pretrain_and_init(self, init_x: torch.Tensor):
        self.train()
        down_init_x = self.dim_reducer.fit(init_x, return_projected=True)

        self.pre_bias.data = self._geometric_median(down_init_x).float()

        # NOTE(aar): OpenAI did not really find evidence that this encoder init is useful
        with torch.no_grad():
            x = torch.randn(256, self.latent_dim, device=self.device, dtype=down_init_x.dtype)
            x /= x.norm(dim=-1, keepdim=True)
            x += self.pre_bias.data

            _, sparse_x, _, (mu, std) = self.encode_sparse(x)
            recons_x = self.decode_sparse(sparse_x, mu=mu, std=std)
            recons_norm = (recons_x - self.pre_bias.data).norm(dim=-1).mean()

            self.encoder.weight.data /= recons_norm.item()

        self.loss_weights["reconstruct"] = (
            1 / ((down_init_x.float().mean(dim=0, keepdim=True) - down_init_x.float()) ** 2).mean()
        )

    def topk_activation(
        self, x: torch.Tensor, top_k: int, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if mask is not None:
            x = x * mask

        values, indices = torch.topk(x, k=top_k, dim=-1)
        values = torch.relu(values)

        x_out = torch.zeros_like(x)
        x_out.scatter_(-1, indices, values)
        return x_out, values, indices

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Tuple[Optional[torch.Tensor], Optional[torch.Tensor]],
    ]:
        down_x, enc_x, sparse_x, _, (mu, std) = self.encode(x)

        # Stats to control for dead latents
        self.stats_last_nonzero *= (sparse_x == 0).all(dim=0).long()
        self.stats_last_nonzero += 1

        recons_x, up_x = self.decode(sparse_x, mu=mu, std=std)

        return down_x, enc_x, sparse_x, recons_x, up_x, (mu, std)

    def encode(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        down_x = self.dim_reducer.reduce(x)

        enc_x, sparse_x, indices, (mu, std) = self.encode_sparse(down_x)

        return down_x, enc_x, sparse_x, indices, (mu, std)

    def decode(
        self, sparse_x: torch.Tensor, mu: Optional[torch.Tensor], std: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        recons_x = self.decode_sparse(sparse_x, mu=mu, std=std)

        up_x = self.dim_reducer.reconstruct(recons_x)

        return recons_x, up_x

    def encode_sparse(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Normalize -- OpenAI preprocess
        norm_x, mu, std = x, torch.Tensor(), torch.Tensor()
        if self.norm_input:
            norm_x, mu, std = normalize(norm_x)

        # Upsample -- OpenAI encode_pre_act
        enc_x = self.encoder(norm_x - self.pre_bias) + self.latent_bias

        # Sparsify -- OpenAI encode
        sparse_x, _, indices = self.topk_activation(enc_x, top_k=self.top_k)

        return enc_x, sparse_x, indices, (mu, std)

    def decode_sparse(
        self, sparse_x: torch.Tensor, mu: Optional[torch.Tensor], std: Optional[torch.Tensor]
    ) -> torch.Tensor:
        # Reconstruct
        recons_x = self.decoder(sparse_x) + self.pre_bias

        # De-normalize -- OpenAI decode
        if self.norm_input:
            recons_x = recons_x * std + mu

        return recons_x

    def _mse(self, source: torch.Tensor, target: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        mse_loss = ((target - source) ** 2).mean()
        if normalize:
            mse_loss = self._mse(target.mean(dim=0).broadcast_to(target.shape[0]), target)
        return mse_loss

    def compute_losses(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        down_x, enc_x, _, recons_x, _, (mu, std) = self(x)

        losses = {}
        # Train on downsampled space
        losses["reconstruct"] = self._mse(recons_x, down_x)

        if self.aux_top_k > 0:
            aux_sparse_x, _, _ = self.topk_activation(
                enc_x, top_k=self.aux_top_k, mask=self.stats_last_nonzero > self.dead_after_n
            )
            sparse_recons_x = self.decode_sparse(aux_sparse_x, mu=mu, std=std)
            aux_norm_mse_loss = self._mse(sparse_recons_x, down_x - recons_x.detach() + self.pre_bias).nan_to_num(0)
            losses["aux"] = aux_norm_mse_loss

        losses["loss"] = sum([self.loss_weights[k] * v for k, v in losses.items()])
        return losses

    def fit(
        self,
        train_embs: torch.Tensor,
        train_cfg: Dict,
        tb_writer: Optional[SummaryWriter] = None,
    ):
        """Training function for SAE"""
        scheduler_cfg = train_cfg.get("scheduler_cfg")

        optim_cfg = train_cfg.get("optim", {})
        optim_cfg.pop("name", None)
        # PyYAML leaves bare scientific notation such as ``1e-4`` as a string.
        if "lr" in optim_cfg:
            optim_cfg["lr"] = float(optim_cfg["lr"])
        if "eps" in optim_cfg:
            optim_cfg["eps"] = float(optim_cfg["eps"])
        if "betas" in optim_cfg:
            optim_cfg["betas"] = tuple(float(beta) for beta in optim_cfg["betas"])
        optimizer = torch.optim.Adam(self.parameters(), **optim_cfg)

        num_epochs = train_cfg.get("epochs", 20_000)
        num_steps = train_cfg.get("steps", 15_000)
        batch_size = train_cfg.get("batch_size", 2_048)
        init_batch_size = train_cfg.get("init_batch_size", 3_072)

        # ema_multiplier = train_cfg.get("ema_multiplier")

        log_every_n = train_cfg.get("log_every_n", 10)
        report_metrics_every_n = train_cfg.get("report_metrics_every_n", 20)

        scheduler = None
        if scheduler_cfg is not None:
            scheduler = get_scheduler(scheduler_cfg, optimizer)

        # --- PRETRAIN/INIT PHASE ---
        init_batch = train_embs[torch.randperm(train_embs.shape[0])[:init_batch_size]].to(self.device)
        self.pretrain_and_init(init_batch)

        step = 0
        for epoch in range(num_epochs):
            self.train()
            total_losses = defaultdict(float)

            dataloader = torch.split(
                train_embs[torch.randperm(train_embs.shape[0])], split_size_or_sections=batch_size, dim=0
            )
            if epoch % report_metrics_every_n == 0:
                metrics = self.report_metrics(dataloader, verbose=(tb_writer is None))
                if tb_writer is not None:
                    for k, v in metrics.items():
                        tb_writer.add_scalar(f"Metric/{k}", v, epoch + 1)

            for _, data in enumerate(dataloader):
                data = data.to(self.device)

                optimizer.zero_grad()
                losses = self.compute_losses(data)

                # Backward pass
                losses["loss"].backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                for k, v in losses.items():
                    total_losses[k] += v.item()

                if (step := step + 1) == num_steps:
                    return

            if epoch % log_every_n == 0:
                log_str = f"Epoch {epoch}: "
                log_str += ", ".join([f"Total {k.title()}: {v / len(dataloader):.4f}" for k, v in total_losses.items()])
                if scheduler is not None:
                    log_str += f", LR: {scheduler.get_last_lr()[0]:.3e}"
                logging.info(log_str)

    @torch.no_grad()
    def generate(
        self,
        dataloader: Optional[Iterable] = None,
        data: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
        return_rmse: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        self.eval()
        sem_ids = []
        acts = []
        rmses = []
        if dataloader is None:
            if data is None:
                raise ValueError("Either data_loader or data must be provided to this method")
            if batch_size is None:
                batch_size = data.shape[0]

            dataloader = torch.split(data, split_size_or_sections=batch_size, dim=0)

        for batch in dataloader:
            batch = batch.to(self.device)
            down_x, _, batch_acts, batch_sids, (mu, std) = self.encode(batch)
            sem_ids.append(batch_sids)
            acts.append(batch_acts)
            if return_rmse:
                # Option 1: reconstruction MSE (Empirical range: 0.0014 - 0.0088)
                rmse = ((down_x - self.decode_sparse(batch_acts, mu=mu, std=std)) ** 2).mean(-1).sqrt()
                # # Option 2: ReLU of discarded concepts
                # disc_acts_sum = (torch.relu(enc_x) - batch_acts).sum(-1)
                # disc_acts_mean = (torch.relu(enc_x) - batch_acts).mean(-1)
                rmses.append(rmse)

        if return_rmse:
            return torch.cat(sem_ids, dim=0), torch.cat(acts, dim=0), torch.cat(rmses, dim=0)
        return torch.cat(sem_ids, dim=0), torch.cat(acts, dim=0)

    @torch.no_grad()
    def report_metrics(
        self, dataloader: Optional[Iterable] = None, semantic_ids: Optional[torch.Tensor] = None, verbose: bool = True
    ) -> Dict[str, float]:
        self.eval()
        if semantic_ids is None:
            if dataloader is None:
                raise ValueError("Either `dataloader` or `semantic_ids` must be provided to compute the metrics!")
            semantic_ids, _ = self.generate(dataloader)

        metrics = {}
        for level in range(self.num_codebooks):
            if verbose:
                logging.info(f"Metrics for Level {level}:")
            cb_metrics = check_id_distribution(semantic_ids[:, level], 1, [self.codebook_size], verbose=verbose)
            metrics |= {f"cb{level}_{k}": v for k, v in cb_metrics.items()}

        flattened_ids = semantic_ids.flatten()
        metrics["flat_utilization_rate"] = len(flattened_ids.unique()) / self.codebook_size
        if verbose:
            logging.info("Overall flattened Metrics:")
            logging.info(f"\\tUtilization rate: {metrics['flat_utilization_rate']:.4f}")
            logging.info("Overall Metrics (Orderless):")

        # max_n_ids = math.comb(self.codebook_size + self.num_codebooks - 1, self.num_codebooks)
        max_n_ids = math.prod([self.codebook_size - i for i in range(self.num_codebooks)])
        metrics_names = [Metric.UTILIZATION, Metric.COLLISION, Metric.PERPLEXITY]
        metrics |= check_id_distribution(
            semantic_ids.sort(-1).values, 1, [max_n_ids], metric_names=metrics_names, verbose=verbose
        )

        if verbose:
            logging.info("Overall Metrics (Ordered):")
        order_metrics = check_id_distribution(semantic_ids, 1, [max_n_ids], metric_names=metrics_names, verbose=verbose)
        metrics |= {f"ordered_{k}": v for k, v in order_metrics.items()}

        return metrics

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path, weights_only=False))


# ---------------------------------------------------------------------------
# SAE hyperparameter presets
# ---------------------------------------------------------------------------
# Scaled down from OpenAI's config (n_dirs=32768, bs=131072, dead_toks_threshold=10M)
# to suit a typical review dataset (~50k–300k reviews).
#
# Expansion ratio (codebook_size / latent_dim) drives monosemanticity:
#   OpenAI GPT SAE uses ~43× (32768/768).  For a domain-specific review corpus
#   the concept space is far smaller, but ratios below ~8× still produce polysemantic
#   directions.  We target 4–16× by compressing to latent_dim=128 (from a 1024-D
#   input embedding) and keeping codebook_sizes at 512 / 1024 / 2048.  PCA to 128
#   typically retains >95 % of variance on review embeddings.
#
#  small  – fast experimentation: 512 concepts,  4× expansion, top-k=8,  ~5k steps
#  medium – balanced:            1024 concepts,  8× expansion, top-k=16, ~10k steps
#  large  – rich concept space:  2048 concepts, 16× expansion, top-k=32, ~15k steps

# SAE_PRESETS: Dict[str, Dict] = {
#     "small": {
#         "tokenizer": {
#             "input_dim": None,  # filled from review_embs.shape[-1]
#             "latent_dim": 128,  # PCA target dim before SAE encoder (4× expansion)
#             "codebook_size": 512,  # number of concept directions (n_dirs)
#             "top_k": 8,  # active concepts per review (k); L0 ≈ 1.6 %
#             "aux_top_k": 64,  # auxk – dead-latent recovery (≈ 12.5 % of codebook)
#             "aux_lambda": 1 / 32,
#             "dead_after_n": 500,  # ~5 epochs × 100 batches
#             "norm_input": False,
#             "tied": False,
#             "dim_reduction": {"type": "pca"},
#         },
#         "train": {
#             "epochs": 200,
#             "steps": 5_000,
#             "batch_size": 1_024,
#             "init_batch_size": 2_048,
#             "log_every_n": 20,
#             "report_metrics_every_n": 50,
#             "optim": {"lr": 1e-4, "eps": 6.25e-10, "betas": (0.9, 0.999)},
#         },
#     },
#     "medium": {
#         "tokenizer": {
#             "input_dim": None,
#             "latent_dim": 128,  # 8× expansion
#             "codebook_size": 1_024,
#             "top_k": 16,  # L0 ≈ 1.6 %
#             "aux_top_k": 128,  # ≈ 12.5 % of codebook
#             "aux_lambda": 1 / 32,
#             "dead_after_n": 1_000,
#             "norm_input": False,
#             "tied": False,
#             "dim_reduction": {"type": "pca"},
#         },
#         "train": {
#             "epochs": 300,
#             "steps": 10_000,
#             "batch_size": 2_048,
#             "init_batch_size": 4_096,
#             "log_every_n": 20,
#             "report_metrics_every_n": 50,
#             "optim": {"lr": 1e-4, "eps": 6.25e-10, "betas": (0.9, 0.999)},
#         },
#     },
#     "large": {
#         "tokenizer": {
#             "input_dim": None,
#             "latent_dim": 128,  # 16× expansion
#             "codebook_size": 2_048,
#             "top_k": 32,  # L0 ≈ 1.6 %
#             "aux_top_k": 256,  # ≈ 12.5 % of codebook
#             "aux_lambda": 1 / 32,
#             "dead_after_n": 2_500,
#             "norm_input": False,
#             "tied": False,
#             "dim_reduction": {"type": "pca"},
#         },
#         "train": {
#             "epochs": 400,
#             "steps": 15_000,
#             "batch_size": 2_048,
#             "init_batch_size": 4_096,
#             "log_every_n": 20,
#             "report_metrics_every_n": 50,
#             "optim": {"lr": 1e-4, "eps": 6.25e-10, "betas": (0.9, 0.999)},
#         },
#     },
# }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_concept_distribution(
    acts: torch.Tensor,
    codebook_size: int,
    out_dir: str,
    config_name: str,
    n_train: int,
) -> None:
    """Four-panel concept distribution plot saved to *out_dir*."""
    acts_np = acts.cpu().float().numpy()  # [N, codebook_size]
    n_all = acts_np.shape[0]

    # Per-concept stats
    activation_counts = (acts_np > 0).sum(axis=0)  # how many reviews fire each concept
    sorted_indices = np.argsort(activation_counts)[::-1]
    sorted_counts = activation_counts[sorted_indices]
    mean_act = acts_np.mean(axis=0)
    sorted_mean_act = mean_act[sorted_indices]

    # Per-review stats
    concepts_per_review = (acts_np > 0).sum(axis=1)  # active concepts per review

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"SAE Concept Distribution  |  config={config_name}", fontsize=13)

    # (1) Concept frequency rank plot
    ax = axes[0, 0]
    ax.bar(np.arange(codebook_size), sorted_counts, width=1.0, color="steelblue", edgecolor="none")
    ax.axhline(n_all * 0.01, color="red", linestyle="--", linewidth=0.9, label="1 % of reviews")
    ax.set_xlabel("Concept rank (by frequency)")
    ax.set_ylabel("Reviews activating concept")
    ax.set_title("Concept activation frequency (sorted)")
    ax.legend(fontsize=8)

    # (2) Histogram of per-concept activation counts
    ax = axes[0, 1]
    dead_count = int((activation_counts == 0).sum())
    ax.hist(activation_counts, bins=50, color="steelblue", edgecolor="white")
    ax.set_xlabel("Reviews activating concept")
    ax.set_ylabel("Number of concepts")
    ax.set_title("Distribution of concept activation frequencies")
    ax.text(
        0.97,
        0.97,
        f"Dead: {dead_count}/{codebook_size} ({100 * dead_count / codebook_size:.1f} %)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="crimson",
    )

    # (3) Mean activation strength per concept (sorted)
    ax = axes[1, 0]
    ax.bar(np.arange(codebook_size), sorted_mean_act, width=1.0, color="darkorange", edgecolor="none")
    ax.set_xlabel("Concept rank (by frequency)")
    ax.set_ylabel("Mean activation value")
    ax.set_title("Mean concept activation strength (sorted)")

    # (4) Active-concepts-per-review distribution split by train vs test
    ax = axes[1, 1]
    split_mask = {"train+valid": np.arange(n_train), "test": np.arange(n_train, n_all)}
    colors_map = {"train+valid": "steelblue", "test": "darkorange"}
    bins = np.arange(0, concepts_per_review.max() + 2) - 0.5
    for split_label, mask_idx in split_mask.items():
        if len(mask_idx) == 0:
            continue
        ax.hist(
            concepts_per_review[mask_idx],
            bins=bins,
            alpha=0.6,
            color=colors_map[split_label],
            label=f"{split_label} (n={len(mask_idx):,})",
            density=True,
        )
    ax.set_xlabel("Active concepts per review")
    ax.set_ylabel("Density")
    ax.set_title("Active concepts per review (train+valid vs test)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"concept_distribution_{config_name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info("Concept distribution plot saved to %s", out_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _encode_and_save_activations(
    sae: SAE,
    review_embs: torch.Tensor,
    all_rev_ids: np.ndarray,
    acts_path: str,
) -> dict:
    """
    Run SAE inference on all review embeddings and save per-review activations
    together with aligned user_ids and item_ids so that downstream aggregation
    (``nlp_xrs.semantic_ids.concepts``) can group by item or user.

    Args:
        sae:          Trained SAE instance.
        review_embs:  Tensor (N_reviews, dim).
        all_rev_ids:  Review IDs corresponding to each row of review_embs.
        acts_path:    Output path for the activations file.

    Returns:
        Dictionary saved to ``acts_path``.
    """
    logging.info("Generating concept activations for all %d reviews...", len(review_embs))
    result = sae.generate(data=review_embs, batch_size=512)
    concept_ids, acts = result[0], result[1]
    logging.info("concept_ids shape: %s,  acts shape: %s", tuple(concept_ids.shape), tuple(acts.shape))

    payload = {
        "review_ids": all_rev_ids,
        "concept_ids": concept_ids.cpu().detach(),
        "acts": acts.gather(1, concept_ids).cpu().detach(),
    }
    torch.save(payload, acts_path)
    logging.info("Activations saved to %s", acts_path)
    return payload


def main():
    parser = argparse.ArgumentParser(description="Train a Sparse Autoencoder on review embeddings.")
    parser.add_argument("--dataset", type=str, default="beauty")
    parser.add_argument(
        "--config",
        type=str,
        default="medium",
        choices=["small", "medium", "large"],
        help="SAE hyperparameter preset (small / medium / large).",
    )
    parser.add_argument("--emb_model", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--emb_batch_size", type=int, default=128)
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=None,
        help="Truncate review text to this many tokens before encoding. Default: the model default.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Cap SAE training steps. Default: the value in the preset YAML.",
    )
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument(
        "--base-path",
        type=str,
        default=None,
        help="Root for data, logs, and results. Defaults to the repository root.",
    )
    parser.add_argument("--test", action="store_true", help="Load the _mini fold instead of the full dataset.")
    parser.add_argument(
        "--retrain",
        action="store_true",
        help=(
            "Retrain the SAE from scratch."
            "If not set, the SAE will be loaded from the checkpoint and the concept activations will be regenerated."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.info("Running SAE with the following arguments: %s", args)

    init_seed(args.seed, reproducibility=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Device: %s", device)

    logging.info("Loading dataset '%s'...", args.dataset)
    dataset = load_dataset(
        args.dataset,
        requires_rating=False,
        test=args.test,
        load_metadata=False,
        seed=args.seed,
        base_path=args.base_path,
    )

    out_path = os.path.join(dataset.data_path, dataset.cfg.fold, "SAE")
    os.makedirs(out_path, exist_ok=True)
    data = dataset.data
    revid2review = dataset.mappers[Key.REV].idx2item["raw"]

    train_ixs = dataset.split_ixs["train"] + dataset.split_ixs["valid"]
    test_ixs = dataset.split_ixs["test"]
    all_ixs = train_ixs + test_ixs

    # ------------------------------------------------------------------
    # Encode all reviews with the sentence embedder in one pass
    # ------------------------------------------------------------------
    emb_model_alias = args.emb_model.replace("/", "--")
    # The unsuffixed cache belongs to the full dataset id space. Mini runs must not read or replace it.
    review_embs_path = os.path.join(
        dataset.data_path, "embeddings", f"review_{emb_model_alias}{dataset.suffix}.pt"
    )
    os.makedirs(os.path.dirname(review_embs_path), exist_ok=True)
    if os.path.isfile(review_embs_path):
        logging.info("Review embeddings already exist in %s", review_embs_path)
        review_embs_ckpt = torch.load(review_embs_path, weights_only=True)
        review_embs = review_embs_ckpt["review_embs"]
        all_rev_ids = review_embs_ckpt["all_rev_ids"].numpy()
    else:
        all_rev_ids = data.loc[all_ixs, Key.REV].unique()
        all_reviews = revid2review.loc[all_rev_ids].tolist()

        logging.info("Encoding reviews with '%s'...", emb_model_alias)
        sent_embedder = SentenceTransformer(args.emb_model)
        if args.max_seq_length is not None:
            sent_embedder.max_seq_length = args.max_seq_length
            logging.info("Truncating review encodings to %d tokens", args.max_seq_length)
        review_embs_np = sent_embedder.encode(
            all_reviews,
            batch_size=args.emb_batch_size,
            show_progress_bar=True,
        )
        review_embs = torch.tensor(review_embs_np, dtype=torch.float32)
        torch.save({"review_embs": review_embs, "all_rev_ids": torch.tensor(all_rev_ids)}, review_embs_path)

    logging.info("Embedding shape: %s", tuple(review_embs.shape))

    revix_to_row = pd.Series({ix: i for i, ix in enumerate(all_rev_ids)})
    train_rows = revix_to_row.loc[data.loc[train_ixs, Key.REV].unique()]
    train_embs = review_embs[train_rows.values]
    logging.info(
        "Total reviews: %d  (train+valid: %d, test: %d)",
        len(all_rev_ids),
        len(train_rows),
        review_embs.shape[0] - len(train_rows),
    )

    # ------------------------------------------------------------------
    # Build SAE config (input_dim taken from embedding shape)
    # ------------------------------------------------------------------
    with open(os.path.join(dataset.paths.config, "sae", f"{args.config}.yaml"), "r") as f:
        preset = yaml.load(f, Loader=yaml.FullLoader)
    # preset = SAE_PRESETS[args.config]
    tokenizer_cfg = dict(preset["tokenizer"])
    tokenizer_cfg["input_dim"] = review_embs.shape[-1]
    train_cfg = dict(preset["train"])
    if args.max_steps is not None:
        train_cfg["steps"] = args.max_steps
        train_cfg["epochs"] = 1

    logging.info("SAE preset '%s': %s", args.config, tokenizer_cfg)

    model_path = os.path.join(out_path, f"model_{emb_model_alias}_{args.config}_{args.seed}.pt")
    acts_path = os.path.join(out_path, f"acts_{emb_model_alias}_{args.config}_{args.seed}.pt")

    # ------------------------------------------------------------------
    # Train SAE (skipped when --retrain is not set)
    # ------------------------------------------------------------------
    sae = SAE(tokenizer_cfg, device=device)

    if args.retrain or not os.path.isfile(model_path):
        tb_dir = os.path.join(
            dataset.paths.tensorboard, "SAE", f"{emb_model_alias}_{args.config}_{args.seed}{dataset.suffix}"
        )
        os.makedirs(tb_dir, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=tb_dir)

        logging.info("Training SAE on %d embeddings...", len(train_rows))
        sae.fit(train_embs, train_cfg, tb_writer=tb_writer)

        if tb_writer is not None:
            tb_writer.close()

        sae.save(model_path)
        logging.info("Model saved to %s", model_path)
    else:
        logging.info("Loading SAE checkpoint from %s", model_path)
        sae.load(model_path)

    # ------------------------------------------------------------------
    # Encode *all* reviews (including test) with the trained SAE
    # ------------------------------------------------------------------
    payload = _encode_and_save_activations(
        sae=sae,
        review_embs=review_embs,
        all_rev_ids=all_rev_ids,
        acts_path=acts_path,
    )
    concept_ids, acts = payload["concept_ids"], payload["acts"]

    # ------------------------------------------------------------------
    # Plot concept distribution
    # ------------------------------------------------------------------
    plot_dir = os.path.join(
        dataset.paths.results, "SAE", args.dataset, f"{emb_model_alias}_{args.config}_{args.seed}{dataset.suffix}"
    )
    os.makedirs(plot_dir, exist_ok=True)
    R = concept_ids.shape[0]
    C = sae.codebook_size
    expanded_acts = torch.zeros(R, C, dtype=acts.dtype).scatter(1, concept_ids, acts)
    plot_concept_distribution(
        acts=expanded_acts,
        codebook_size=tokenizer_cfg["codebook_size"],
        out_dir=plot_dir,
        config_name=args.config,
        n_train=len(train_rows),
    )


if __name__ == "__main__":
    main()
