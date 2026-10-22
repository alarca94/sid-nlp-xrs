import logging
import re
from typing import List, Set

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

from nlp_xrs.data.utils import postprocessing
from nlp_xrs.utils.constants import (
    PACKAGE_NAME,
    PAD_TOK,
    SPECIAL_TOKENS,
    TOP_KS,
    Key,
    ModelType,
    Task,
)
from nlp_xrs.utils.in_out import save_results

from .metrics import (
    bleu_score,
    concept_activation_correlation,
    concept_coverage,
    concept_kl_divergence,
    concept_mrr,
    concept_ndcg,
    concept_popularity_discrepancy,
    concept_sample_popularity_discrepancy,
    concept_set_overlap,
    feat_miss_ratio,
    feature_coverage_ratio,
    feature_diversity,
    feature_prf,
    hr,
    mrr,
    ndcg,
    rouge_score,
    token_diversity,
    unique_sentence_ratio,
)
from .utils import feature_detect


class Evaluator:
    def __init__(
        self,
        tasks,
        data,
        prediction_path,
        exp_metadata,
        test_flag,
        gen_flag,
        save_flag,
        model_type,
        results_path,
        ckpt_path,
        device="cpu",
        logger=None,
        do_lower_case=False,
    ):
        """
        Initialize the Evaluator for computing metrics across multiple recommendation tasks.

        Args:
            tasks: List of Task enums defining which evaluation tasks to perform
            data: Data object with tokenizers, mappers, feature sets, and dataset info
            prediction_path: Path to save generated predictions (used when gen_flag is True)
            exp_metadata: Dictionary of experiment metadata to include in results and identify experiments
            test_flag: Not used currently
            gen_flag: Whether to save generated text predictions to file
            save_flag: Whether to save evaluation results to file
            model_type: ModelType.GENERATIVE
            results_path: Path to save evaluation results (used when save_flag is True)
            ckpt_path: Path to model checkpoint file
            device: Device for computation ("cpu" or "cuda")
            logger: Optional logger instance; creates default if None
            do_lower_case: Whether to make text evaluation case-insensitive
        """
        self.tasks = [t if t != Task.MLM else Task.NEXT_ITEM for t in tasks]
        self.results = exp_metadata.copy()
        self.rating_range = (data.data_info.min_rating, data.data_info.max_rating)
        self.w_tok = data.tok  # [Key.EXP]
        self.mappers = data.mappers
        self.i_pad_tok = data.data_info.special_items[PAD_TOK]
        self.feature_set = data.feature_set
        self.i2f = data.i2f
        self.model_type = model_type
        self.data_info = data.data_info

        if getattr(self.data_info, "sae_n_concepts", 0) > 0:
            concept_ids = np.arange(self.data_info.sae_n_concepts)
            self.sae_concept_stats = pd.DataFrame(
                {
                    "concept_id": concept_ids,
                    "popularity": 0,
                    "mean_act": 0,
                    "std_act": 0,
                    "total_act": 0,
                }
            ).set_index("concept_id")
            train_ixs = data.split_ixs["train"] + data.split_ixs["valid"]
            n_train = len(train_ixs)
            trn_acts = data.acts_mapper.loc[data.data.loc[train_ixs, Key.REV]].explode(["concept_id", "act"])
            trn_acts = trn_acts.groupby("concept_id").agg({"act": ["count", "mean", "std", "sum"]})
            self.sae_concept_stats["popularity"] = trn_acts["act"]["count"] / n_train
            self.sae_concept_stats["mean_act"] = trn_acts["act"]["mean"]
            self.sae_concept_stats["std_act"] = trn_acts["act"]["std"]
            self.sae_concept_stats["total_act"] = trn_acts["act"]["sum"]
            self.sae_concept_stats.fillna(0, inplace=True)
            self.sae_concept_stats.sort_values(by="popularity", ascending=False, inplace=True)

        # self.i_tok = toks[Key.ITEM]
        # self.u_tok = toks[Key.USER]
        self.prediction_path = prediction_path
        self.do_lower_case = do_lower_case
        self.device = device
        if self.do_lower_case:
            self.feature_set = list(set([f.lower() for f in self.feature_set]))
        self.test_flag = test_flag
        self.gen_flag = gen_flag
        self.save_flag = save_flag
        self.results_path = results_path
        self.ckpt_path = ckpt_path
        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger(PACKAGE_NAME)

        # NOTE: ngram_idf is only computed for word unigrams (same as CompExp)
        self.ngram_idf = self.get_ngram_idf()

        task_metric_map = {
            Task.RATING: self.compute_rating,
            Task.NEXT_ITEM: self.compute_next_item,
            Task.TOPN: self.compute_topn,
            Task.EXPLANATION: self.compute_exp,
        }

        if hasattr(self, "sae_concept_stats"):
            task_metric_map[Task.CONTEXT] = self.compute_concept

        self.metric_fns = {task: task_metric_map[task] for task in self.tasks if task in task_metric_map}

    def get_ngram_idf(self):
        unique_sent_words = self.mappers[Key.EXP].idx2item["raw"].apply(postprocessing).str.split().apply(set)
        return np.log(len(unique_sent_words) / unique_sent_words.explode().value_counts()) + 1

    def decode_skip(self, ids, skip_decode=False):
        if skip_decode:
            text = ids
        else:
            text = self.w_tok.decode(ids)

        # Make sure special tokens are separated by a space
        for tok in [getattr(self.w_tok, k) for k in list(SPECIAL_TOKENS.keys())]:
            text = re.sub(rf"\s*{tok}\s*", f" {tok} ", text)

        # Make NLG evaluation case-insensitive
        if self.do_lower_case:
            text = text.lower()

        text = postprocessing(text)  # PETER processes punctuations: "good!" -> "good !"

        tokens = []
        for i, token in enumerate(text.split()):
            if token in [self.w_tok.eos_token, self.w_tok.pad_token]:  # or i == TXT_LEN:
                break
            tokens.append(token)
        return tokens

    def compute_rating(self, labels, preds):
        # Compute RMSE and MAE metrics
        pred = np.clip(preds[Task.RATING], *self.rating_range)
        self.results["RMSE"] = np.sqrt(mean_squared_error(labels[Task.RATING], pred))
        self.logger.info("RMSE {:7.4f}".format(self.results["RMSE"]))
        self.results["MAE"] = mean_absolute_error(labels[Task.RATING], pred)
        self.logger.info("MAE {:7.4f}".format(self.results["MAE"]))

    def full_rec_eval(self, labels, preds, ks, pad_tok, prefix=""):
        preds = np.array(preds)

        y_true = np.array(labels)
        mask = y_true != pad_tok
        y_true = y_true[mask]

        for k in ks:
            if k > preds.shape[-1]:
                continue

            y_pred = preds[mask, :k]
            self.results[f"{prefix}HR@{k}"] = hr(y_true, y_pred)
            self.logger.info("{}HR@{} {:7.4f} %".format(prefix, k, self.results[f"{prefix}HR@{k}"] * 100))
            self.results[f"{prefix}MRR@{k}"] = mrr(y_true, y_pred)
            self.logger.info("{}MRR@{} {:7.4f} %".format(prefix, k, self.results[f"{prefix}MRR@{k}"] * 100))
            self.results[f"{prefix}NDCG@{k}"] = ndcg(y_true, y_pred)
            self.logger.info("{}NDCG@{} {:7.4f} %".format(prefix, k, self.results[f"{prefix}NDCG@{k}"] * 100))

    def compute_next_item(self, labels, pred):
        """Logit version of the sequential recommendation evaluation.
        Task.NEXT_ITEM considers all items in the vocabulary
        Task.NEXT_ITEM_SAMPLE considers the positive item (first) and n_neg negative items (last)
        """

        def get_sample_metrics():
            topk_mask = pred[Task.NEXT_ITEM_SAMPLE] <= k
            ndcg_v = (1.0 / np.log2(pred[Task.NEXT_ITEM_SAMPLE][topk_mask] + 1.0)).sum() / len(topk_mask)
            hr_v = topk_mask.mean()
            return hr_v, ndcg_v

        # Compute HR@k, MRR@k and NDCG@k with k in (5, 10, 20). Ignore index will ignore cold_start users in the test
        self.full_rec_eval(labels[Task.NEXT_ITEM], pred[Task.NEXT_ITEM], TOP_KS, self.i_pad_tok)

        if Task.NEXT_ITEM_SAMPLE in pred:
            pred[Task.NEXT_ITEM_SAMPLE] = np.array(pred[Task.NEXT_ITEM_SAMPLE])
            pred[Task.NEXT_ITEM_SAMPLE] = (-pred[Task.NEXT_ITEM_SAMPLE]).argsort().argsort()[:, 0] + 1
            self.results["S-MRR"] = (1.0 / pred[Task.NEXT_ITEM_SAMPLE]).mean()
            self.logger.info("S-MRR {:7.4f} %".format(self.results["S-MRR"] * 100))

            for k in TOP_KS:
                self.results[f"S-HR@{k}"], self.results[f"S-NDCG@{k}"] = get_sample_metrics()
                self.logger.info("S-HR@{} {:7.4f} %".format(k, self.results[f"S-HR@{k}"] * 100))
                self.logger.info("S-NDCG@{} {:7.4f} %".format(k, self.results[f"S-NDCG@{k}"] * 100))

    def compute_topn(self, labels, pred):
        self.full_rec_eval(labels[Task.TOPN], pred[Task.TOPN], TOP_KS, self.i_pad_tok, prefix="TN-")

    # ------------------------------------------------------------------
    # SAE concept prediction evaluation (Task.CONTEXT)
    # ------------------------------------------------------------------

    def _concept_stat_arrays(self):
        """Return (concept_popularity, concept_mean_act) as numpy arrays indexed by concept_id."""
        n_concepts = self.data_info.sae_n_concepts
        idx = range(n_concepts)
        concept_popularity: np.ndarray = (
            self.sae_concept_stats["popularity"].reindex(idx, fill_value=0.0).to_numpy(dtype=float)
        )
        concept_mean_act: np.ndarray = (
            self.sae_concept_stats["mean_act"].reindex(idx, fill_value=0.0).to_numpy(dtype=float)
        )
        return concept_popularity, concept_mean_act

    def _concept_per_recall(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Per-concept recall: fraction of test samples where concept c is predicted given it is a label."""
        n_concepts = self.data_info.sae_n_concepts
        label_count = np.zeros(n_concepts)
        hit_count = np.zeros(n_concepts)
        for i in range(len(y_pred)):
            pred_set = set(y_pred[i].tolist())
            for c in y_true[i]:
                label_count[c] += 1
                if c in pred_set:
                    hit_count[c] += 1
        with np.errstate(invalid="ignore"):
            recall = np.where(label_count > 0, hit_count / label_count, np.nan)
        return recall

    def _concept_act_hit_miss(self, y_pred: np.ndarray, y_true: np.ndarray, concept_mean_act: np.ndarray):
        """Mean training activation of correctly predicted (hit) vs missed label concepts."""
        hit_acts, miss_acts = [], []
        for i in range(len(y_pred)):
            pred_set = set(y_pred[i].tolist())
            for c in y_true[i]:
                (hit_acts if c in pred_set else miss_acts).append(concept_mean_act[c])
        mean_hit = float(np.mean(hit_acts)) if hit_acts else float("nan")
        mean_miss = float(np.mean(miss_acts)) if miss_acts else float("nan")
        return mean_hit, mean_miss

    def _log_concept_split_shift(
        self,
        y_true: np.ndarray,
        concept_popularity: np.ndarray,
        concept_mean_act: np.ndarray,
    ):
        """Log mean/std of concept popularity and activation for train vs test labels."""
        test_concepts = y_true.ravel()
        test_pop = concept_popularity[test_concepts]
        test_act = concept_mean_act[test_concepts]
        train_pop: np.ndarray = self.sae_concept_stats["popularity"].to_numpy(dtype=float)
        train_act: np.ndarray = self.sae_concept_stats["mean_act"].to_numpy(dtype=float)
        self.logger.info(
            "ConceptShift/popularity — "
            f"train: mean={train_pop.mean():.4f} std={train_pop.std():.4f} | "
            f"test-labels: mean={test_pop.mean():.4f} std={test_pop.std():.4f}"
        )
        self.logger.info(
            "ConceptShift/mean_act — "
            f"train: mean={train_act.mean():.4f} std={train_act.std():.4f} | "
            f"test-labels: mean={test_act.mean():.4f} std={test_act.std():.4f}"
        )

    def compute_concept(self, labels, preds):
        """
        Evaluate SAE concept prediction (Task.CONTEXT).

        Expects preds[Task.CONTEXT] and labels[Task.CONTEXT] to be
        List[List[int]] convertible to a [n_test, top_k] numpy array of
        concept indices in descending order of predicted / ground-truth
        activation magnitude respectively.

        Writes metrics under the ``"Cpt/"`` prefix into ``self.results`` and
        logs all values.  Requires ``self.sae_concept_stats`` to be populated.
        """
        y_pred = np.array(preds[Task.CONTEXT])  # [n_test, top_k]
        y_true = np.array(labels[Task.CONTEXT])  # [n_test, top_k]
        n_concepts = self.data_info.sae_n_concepts
        pfx = "Cpt/"

        concept_popularity, concept_mean_act = self._concept_stat_arrays()

        # --- Set-level overlap ---
        overlap = concept_set_overlap(y_pred, y_true)
        for name, val in overlap.items():
            key = f"{pfx}{name}"
            self.results[key] = val
            self.logger.info(f"{key} {val:7.4f}")

        # --- Binary NDCG@K ---
        ndcg_v = concept_ndcg(y_pred, y_true)
        self.results[f"{pfx}NDCG"] = ndcg_v
        self.logger.info(f"{pfx}NDCG {ndcg_v:7.4f}")

        # --- Activation-weighted NDCG@K (uses training mean_act as relevance proxy) ---
        act_ndcg_v = concept_ndcg(y_pred, y_true, concept_rel=concept_mean_act)
        self.results[f"{pfx}ActNDCG"] = act_ndcg_v
        self.logger.info(f"{pfx}ActNDCG {act_ndcg_v:7.4f}")

        # --- MRR (rank of first correct concept) ---
        mrr_v = concept_mrr(y_pred, y_true)
        self.results[f"{pfx}MRR"] = mrr_v
        self.logger.info(f"{pfx}MRR {mrr_v:7.4f}")

        # --- Concept vocabulary coverage ---
        cov_v = concept_coverage(y_pred, n_concepts)
        cov_v_true = concept_coverage(y_true, n_concepts)
        self.results[f"{pfx}Cov"] = cov_v
        self.results[f"{pfx}CovTrue"] = cov_v_true
        self.logger.info(f"{pfx}Cov {cov_v:7.4f} | {pfx}CovTrue {cov_v_true:7.4f}")

        # --- Popularity discrepancy: vocabulary-level (global head/tail collapse) ---
        pop_disc = concept_popularity_discrepancy(y_pred, concept_popularity, n_concepts)
        self.results[f"{pfx}PopDisc"] = pop_disc
        self.logger.info(f"{pfx}PopDisc {pop_disc:7.4f}")

        # --- Popularity discrepancy: sample-level (per-prediction bias direction) ---
        spop = concept_sample_popularity_discrepancy(y_pred, y_true, concept_popularity)
        for name, val in spop.items():
            key = f"{pfx}{name}"
            self.results[key] = val
            self.logger.info(f"{key} {val:7.4f}")

        # --- KL divergence of concept frequency distributions ---
        kl_div = concept_kl_divergence(y_pred, y_true, n_concepts)
        self.results[f"{pfx}KL"] = kl_div
        self.logger.info(f"{pfx}KL {kl_div:7.4f}")

        # --- Activation magnitude correlation with predicted rank ---
        pearson_r, spearman_r = concept_activation_correlation(y_pred, concept_mean_act)
        self.results[f"{pfx}PearsonR"] = pearson_r
        self.results[f"{pfx}SpearmanR"] = spearman_r
        self.logger.info(f"{pfx}PearsonR {pearson_r:7.4f} | {pfx}SpearmanR {spearman_r:7.4f}")

        # --- Activation magnitude: hit vs missed concepts (saved + logged) ---
        mean_act_hit, mean_act_miss = self._concept_act_hit_miss(y_pred, y_true, concept_mean_act)
        self.results[f"{pfx}MeanActHit"] = mean_act_hit
        self.results[f"{pfx}MeanActMiss"] = mean_act_miss
        self.logger.info(f"{pfx}MeanActHit {mean_act_hit:7.4f} | {pfx}MeanActMiss {mean_act_miss:7.4f}")

        # --- Per-concept recall: top/bottom summary (logged only) ---
        per_recall = self._concept_per_recall(y_pred, y_true)
        valid_mask = (per_recall > 0) & ~np.isnan(per_recall)
        recall_valid: pd.Series = pd.Series(
            per_recall[valid_mask], index=np.where(valid_mask)[0], name="recall"
        ).sort_values(ascending=False)
        if len(recall_valid):
            self.logger.info("Per-concept recall — top-5 (always hit):\n" + recall_valid.head(5).to_string())
            self.logger.info("Per-concept recall — bottom-5 (often missed):\n" + recall_valid.tail(5).to_string())

        # --- Split distribution shift (logged only) ---
        self._log_concept_split_shift(y_true, concept_popularity, concept_mean_act)

    def decode_labels(self, labels, mapper=None, sep=None, as_tokens=True, skip_decode=True):
        if mapper is not None:
            labels = [mapper.get_items(idxs) for idxs in labels]

        if sep is not None:
            if as_tokens:
                return [self.decode_skip(sep.join(idxs), skip_decode=skip_decode) for idxs in labels]
            else:
                return [sep.join(self.decode_skip(sep.join(ids), skip_decode=skip_decode)) for ids in labels]

        return [self.decode_skip(ids) for ids in labels]

    @staticmethod
    def labels2toks(labels, mapper, split=True):
        gt = []
        for exps in labels:
            exps = mapper.get_items(exps)
            exps = [postprocessing(exp) for exp in exps]
            if split:
                exps = [exp.split() for exp in exps]
            gt.append(exps)
        return gt

    def decode_preds(self, preds, mapper=None):
        #  based on tokenizer
        exps = []
        for exp in preds:
            if mapper is not None:
                text = mapper.get_item(exp)
            else:
                text = re.sub(rf"^(\s?{self.w_tok.pad_token}\s?)+", "", self.w_tok.decode(exp))
                for token in [getattr(self.w_tok, k) for k in list(SPECIAL_TOKENS.keys())]:
                    text = re.sub(rf"\s*{token}\s*", f" {token} ", text)

            if self.do_lower_case:
                text = text.lower()

            text = postprocessing(text)  # PETER processes punctuations: "good!" -> "good !"

            if mapper is not None:
                exps.append(text.split())
            else:
                tokens = []
                for i, token in enumerate(text.split()):
                    if token in [self.w_tok.eos_token, self.w_tok.pad_token]:  # or i == TXT_LEN:
                        break
                    tokens.append(token)
                exps.append(tokens)
        return exps

    def compute_exp_metrics(
        self,
        gt_tokens: List[List[List[int]]],
        pred_tokens: List[List[int]],
        gt_text: List[List[str]],
        pred_text: List[str],
        gt_feats: List[Set[str]],
        items: List[int],
    ):
        # ngram_idf = self.get_ngram_idf()

        for n in [1, 2, 4]:
            self.results[f"BLEU-{n}"] = bleu_score(gt_tokens, pred_tokens, n_gram=n, smooth=False)
            self.logger.info("BLEU-{} {:7.4f}".format(n, self.results[f"BLEU-{n}"]))
        self.results["USR"], self.results["USN"] = unique_sentence_ratio(pred_text)
        self.logger.info("USR {:7.4f} | USN {:7}".format(self.results["USR"], self.results["USN"]))
        pred_feats = feature_detect(pred_text, self.feature_set)
        self.results["DIV"] = feature_diversity(pred_feats)  # time-consuming
        self.logger.info("DIV {:7.4f}".format(self.results["DIV"]))
        self.results["FCR"] = feature_coverage_ratio(pred_feats, self.feature_set)
        self.logger.info("FCR {:7.4f}".format(self.results["FCR"]))
        self.results["FP"], self.results["FR"], self.results["FF"] = feature_prf(
            pred_feats, gt_feats, ignore=self.w_tok.unk_token
        )
        self.logger.info("FP {:7.4f} | FR {:7.4f}".format(self.results["FP"], self.results["FR"]))
        # NOTE (2024-01-14): ROUGE changed to ROUGE_Multi as we now have multiple references per hypothesis
        #  (https://github.com/google-research/google-research/blob/master/rouge/rouge_scorer.py)
        ROUGE = rouge_score(gt_text, pred_text)  # a dictionary
        for k, v in ROUGE.items():
            self.results[k] = v
            self.logger.info(f"{k} {v:7.4f}")

        # CompExp new metrics: IDF/Word, Avg. Exp. Length, Exp. 2-gram Repetition and Word IDF
        self.results["rep/l"], self.results["seq_rep_2"], self.results["UTN"] = token_diversity(pred_tokens)
        self.logger.info(
            f"rep/l {self.results['rep/l']:7.4f} | seq_rep_2 {self.results['seq_rep_2']:7.4f} | "
            f"UTN {self.results['UTN']:7}"
        )

        self.results["avg_len"] = np.mean(list(map(len, pred_tokens)))
        self.results["w_idf"] = np.mean([np.mean([self.ngram_idf.get(w, 1) for w in pred]) for pred in pred_tokens])
        self.logger.info(f"avg_len {self.results['avg_len']:7.4f} | w_idf {self.results['w_idf']:7.4f}")

        # Hallucination metric
        pred_feat_enc = [set(self.mappers[Key.FEAT].get_idxs(list(feats))) for feats in pred_feats]
        self.results["C-FHR"], self.results["D-FHR"], self.results["FHR"] = feat_miss_ratio(
            pred_feat_enc, items, self.i2f
        )
        self.logger.info(
            f"C-FHR {self.results['C-FHR']:.4f} | D-FHR {self.results['D-FHR']:.4f} | FHR {self.results['FHR']:.4f}"
        )

    def compute_exp(self, labels, pred):
        gt_tokens = self.labels2toks(labels[Task.EXPLANATION], self.mappers[Key.EXP])
        gt_feats = self.labels2toks(labels[Task.FEAT], self.mappers[Key.FEAT], split=False)
        gt_feats = list(map(set, gt_feats))
        if self.model_type != ModelType.GENERATIVE:
            raise ValueError("Invalid model type!")
        pred_tokens = self.decode_preds(pred[Task.EXPLANATION])

        gt_text = [list(map(" ".join, toks)) for toks in gt_tokens]
        pred_text = list(map(" ".join, pred_tokens))

        self.compute_exp_metrics(gt_tokens, pred_tokens, gt_text, pred_text, gt_feats, labels[f"metadata-{Key.ITEM}"])

        if self.device == torch.device("cuda"):
            torch.cuda.empty_cache()

        if self.gen_flag:
            self.print_generated(labels, pred, gt_text, pred_text, self.prediction_path)
            self.logger.info(f"Generated text saved to ({self.prediction_path})")

    def print_generated(self, labels, pred, gt_text=None, pred_text=None, prediction_path=None):
        if gt_text is None:
            gt_tokens = self.labels2toks(labels[Task.EXPLANATION], self.mappers[Key.EXP])
            gt_text = [list(map(" ".join, toks)) for toks in gt_tokens]

        gt_text = [str(ts) for ts in gt_text]

        if pred_text is None:
            pred_tokens = self.decode_preds(pred[Task.EXPLANATION])
            pred_text = list(map(" ".join, pred_tokens))

        df = {"True": gt_text, "Pred": pred_text}
        if Task.CONTEXT in pred:
            if getattr(self.data_info, "sae_n_concepts", 0) > 0:
                df["Cntx"] = list(map(str, pred[Task.CONTEXT]))
                df["Cntx-GT"] = list(map(str, labels[Task.CONTEXT]))
            else:
                df["Cntx"] = self.w_tok.batch_decode(pred[Task.CONTEXT])

        if Task.RATING in pred:
            df["Rating"] = pred[Task.RATING]

        df["Items"] = self.mappers[Key.ITEM].get_items(labels[f"metadata-{Key.ITEM}"])
        df["User"] = self.mappers[Key.USER].get_items(labels[f"metadata-{Key.USER}"])
        # df['Items'] = self.i_tok.batch_decode(labels[f'metadata-{Key.ITEM}'].T.tolist(), skip_special_tokens=True)
        # if len(labels[f'metadata-{Key.ITEM}'].shape) == 1:
        #     df['Items'] = [', '.join(seq) for seq in df['Items']]
        #
        # df['User'] = self.u_tok.batch_decode(labels[f'metadata-{Key.USER}'])
        df = pd.DataFrame.from_dict(df, orient="columns")
        if prediction_path is not None:
            df.to_json(prediction_path, orient="records", indent=2)
        else:
            self.logger.debug(df.to_json(orient="records", indent=2))

    def evaluate(self, labels, pred):
        self.logger.info("Computing performance metrics")
        # results.update({m: 0 for m in })
        for task, metric_fn in self.metric_fns.items():
            if task in labels:
                metric_fn(labels, pred)
            else:
                self.logger.warning(f"Task {task} appears as a model task but no labels are given for it!")

        if self.save_flag:
            save_results(self.results, self.results_path)
