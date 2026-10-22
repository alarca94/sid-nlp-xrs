from typing import Dict, List, Optional, Tuple

import numpy as np
from nltk import ngrams
from scipy.stats import pearsonr, spearmanr

from nlp_xrs.evaluator.bleu import compute_bleu
from nlp_xrs.evaluator.rouge import rouge_multi

# ------------------------------------------ #
# ----------------- RATING ----------------- #
# ------------------------------------------ #
# Note: RMSE and MAE are directly computed using sklearn implementations

# -------------------------------------------------------- #
# ----------------- TOP-K RECOMMENDATION ----------------- #
# -------------------------------------------------------- #


def hr(y_true, y_pred):
    """
    Compute Hit Ratio (HR@K) for top-K recommendation.

    Measures the proportion of test cases where the true item appears
    in the top-K recommended items.

    Metric name in results: 'HR@K' (e.g., 'HR@5', 'HR@10', 'HR@20')

    Args:
        y_true: Array of true item IDs, shape [batch_size]
        y_pred: Array of predicted top-K item IDs, shape [batch_size, K]

    Returns:
        float: Hit ratio value between 0 and 1 (reported as percentage in logs)
    """
    # y_true :: [BSZ, ]  |||  y_pred :: [BSZ, K]
    return np.equal(y_true.reshape(-1, 1), y_pred).any(axis=1).mean()


def mrr(y_true, y_pred):
    """
    Compute Mean Reciprocal Rank (MRR@K) for top-K recommendation.

    Measures the average of reciprocal ranks of the first relevant item
    in the recommendation list. Higher values indicate better performance.

    Metric name in results: 'MRR@K' (e.g., 'MRR@5', 'MRR@10', 'MRR@20')

    Args:
        y_true: Array of true item IDs, shape [batch_size]
        y_pred: Array of predicted top-K item IDs, shape [batch_size, K]

    Returns:
        float: MRR value between 0 and 1 (reported as percentage in logs)
    """
    # y_true :: [BSZ, ]  |||  y_pred :: [BSZ, K]
    rank_i = np.argwhere(np.equal(y_true.reshape(-1, 1), y_pred))[:, 1] + 1
    return 1 / y_pred.shape[0] * (1 / rank_i).sum()


def ndcg(y_true, y_pred, rels=None):
    """
    Compute Normalized Discounted Cumulative Gain (NDCG@K) for top-K recommendation.

    Measures ranking quality by considering both the relevance of items and their positions.
    Implementation for next item prediction where only one item is relevant at a given time.

    Metric name in results: 'NDCG@K' (e.g., 'NDCG@5', 'NDCG@10', 'NDCG@20')

    Args:
        y_true: Array of true item IDs, shape [batch_size]
        y_pred: Array of predicted top-K item IDs, shape [batch_size, K]
        rels: Optional array of relevance scores for each item. Defaults to all ones.

    Returns:
        float: NDCG value between 0 and 1 (reported as percentage in logs)
    """

    # y_true :: [BSZ, ]  |||  y_pred :: [BSZ, K].
    # Implementation for next item prediction (Only one item is relevant at a given time)
    def dcg(rel, rank):
        return ((2**rel - 1) / np.log2(rank + 1)).sum()

    if rels is None:
        rels = np.ones(y_true.max() + 1)

    ranks = np.argwhere(np.equal(y_true.reshape(-1, 1), y_pred))
    rel_i = rels[y_true[ranks[:, 0]]]
    rank_i = ranks[:, 1] + 1
    return dcg(rel_i, rank_i) / dcg(np.ones_like(y_true), np.ones_like(y_true))


# -------------------------------------------------- #
# ----------------- EXPLAINABILITY ----------------- #
# -------------------------------------------------- #


def token_diversity(samples):
    """
    Compute token-level diversity metrics for generated text sequences.

    Calculates three metrics:
    1. rep/l: Ratio of repeated tokens to total tokens
    2. seq_rep_2: Average 2-gram repetition rate across sequences
    3. UTN: Number of unique tokens across all sequences

    Metric names in results: 'rep/l', 'seq_rep_2', 'UTN'

    Args:
        samples: List of token lists, where each inner list represents a generated sequence

    Returns:
        tuple: (rep_l, seq_rep_2_mean, unique_token_count)
            - rep_l (float): Repetition per length, NaN if total_count is 0
            - seq_rep_2_mean (float): Mean 2-gram repetition rate
            - unique_token_count (int): Total number of unique tokens
    """
    rep_count, total_count = 0, 0
    seq_rep_2_list, uniq_tokens = [], set()
    for tokens in samples:
        # Compute rep/l
        uniq = set(tokens)
        rep_count += len(tokens) - len(uniq)
        total_count += len(tokens)

        # Compute seq_rep_2
        grams = list(ngrams(tokens, 2))
        if grams:
            seq_rep_2 = 1 - len(set(grams)) / len(grams)
            seq_rep_2_list.append(seq_rep_2)

        uniq_tokens |= set(uniq)

    rep_l = np.nan if total_count == 0 else rep_count / total_count
    return rep_l, np.mean(seq_rep_2_list), len(uniq_tokens)


def feature_prf(pred_feats, gt_feats, ignore=None):
    """
    Compute Precision, Recall, and F1-score for feature detection in explanations.

    Evaluates how well the predicted features match the ground truth features
    by computing standard information retrieval metrics.

    Metric names in results: 'FP' (Feature Precision), 'FR' (Feature Recall), 'FF' (Feature F1)

    Args:
        pred_feats: List of sets containing predicted features for each sample
        gt_feats: List of sets containing ground truth features for each sample
        ignore: Optional feature token to ignore (e.g., unknown token)

    Returns:
        tuple: (precision, recall, f1_score)
            - precision (float): Average feature precision across samples
            - recall (float): Average feature recall across samples
            - f1_score (float): Average feature F1-score across samples
    """
    p_sum, r_sum, f_sum = 0, 0, 0

    length = 0
    for pred, gt in zip(pred_feats, gt_feats):
        if not gt:
            continue

        if ignore is not None:
            pred.discard(ignore)
        matches = pred.intersection(gt)

        precision = len(matches) / len(pred) if pred else 0
        recall = len(matches) / len(gt)
        p_sum += precision
        r_sum += recall
        if (denominator := precision + recall) != 0:
            f_sum += 2 * precision * recall / denominator

        length += 1

    return p_sum / length, r_sum / length, f_sum / length


def feature_coverage_ratio(feature_batch, feature_set):
    """
    Compute Feature Coverage Ratio (FCR) - ratio of unique features used to total features available.

    Measures the diversity of features mentioned across all generated explanations
    compared to the complete feature vocabulary.

    Metric name in results: 'FCR'

    Args:
        feature_batch: List of sets, each containing features from one generated explanation
        feature_set: Complete set of all possible features in the vocabulary

    Returns:
        float: Ratio of covered features to total features (between 0 and 1)
    """
    features = set()
    for fb in feature_batch:
        features = features | fb

    return len(features) / len(feature_set)


def feature_diversity(feature_batch):
    """
    Compute Feature Diversity (DIV) - average pairwise feature overlap across explanations.

    Measures how similar the generated explanations are in terms of their features
    by computing the average intersection size between all pairs of feature sets.
    Higher values indicate more similar explanations (less diversity).

    Metric name in results: 'DIV'

    Args:
        feature_batch: List of sets, each containing features from one generated explanation

    Returns:
        float: Average number of shared features across all pairs of explanations

    Note: This computation is time-consuming for large batches as it compares all pairs.
    """
    list_len = len(feature_batch)

    total_count = 0
    for i, x in enumerate(feature_batch):
        for j in range(i + 1, list_len):
            y = feature_batch[j]
            total_count += len(x & y)

    denominator = list_len * (list_len - 1) / 2
    return total_count / denominator


def feat_miss_ratio(pred_feats: List[set], items: List[int], i2f: Dict[int, set], eps: float = 1e-8):
    """
    Compute Feature Hallucination Rate (FHR) metrics for explanations.

    Measures how often the model generates features that don't actually belong to the
    recommended item, indicating hallucinated or incorrect explanations.

    Metric names in results:
        - 'C-FHR': Corpus-level Feature Hallucination Rate (ratio of all hallucinated features)
        - 'D-FHR': Document-level Feature Hallucination Rate (ratio of docs with any hallucination)
        - 'FHR': Average per-document Feature Hallucination Rate

    Args:
        pred_feats: List of sets containing predicted feature IDs for each sample
        items: List of item IDs for which explanations were generated
        i2f: Dictionary mapping item IDs to their true feature sets
        eps: Small epsilon value to avoid division by zero

    Returns:
        tuple: (C-FHR, D-FHR, FHR)
            - C-FHR (float): Corpus-level hallucination rate (NaN if no features predicted)
            - D-FHR (float): Document-level hallucination rate (proportion of docs with errors)
            - FHR (float): Average per-document hallucination rate
    """
    num, den, fhr, missed_docs = 0, 0, 0, 0
    for i, f in zip(items, pred_feats):
        n_missed = len(f.difference(i2f[i]))
        den += len(f)
        num += n_missed
        fhr += n_missed / (len(f) + eps)
        missed_docs += int(n_missed > 0)

    # If C-FHR returns NaN, it means the model did not predict any feature in the explanations
    cfhr = np.nan if den == 0 else num / den
    return cfhr, missed_docs / len(pred_feats), fhr / len(pred_feats)


# ------------------------------------------------ #
# ----------------- TEXT QUALITY ----------------- #
# ------------------------------------------------ #


def rouge_score(references, generated):
    """
    Compute ROUGE scores for generated explanations against multiple references.

    ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures n-gram overlap
    between generated text and reference texts. Supports multiple references per hypothesis.

    Metric names in results:
        - 'rouge_1/f_score', 'rouge_1/r_score', 'rouge_1/p_score': ROUGE-1 (unigrams)
        - 'rouge_2/f_score', 'rouge_2/r_score', 'rouge_2/p_score': ROUGE-2 (bigrams)
        - 'rouge_l/f_score', 'rouge_l/r_score', 'rouge_l/p_score': ROUGE-L (longest common subsequence)

    Args:
        references: List of lists of strings, where each inner list contains reference
                   explanations for one sample
        generated: List of strings, each being a generated explanation

    Returns:
        dict: Dictionary with ROUGE metric names as keys and scores (scaled to 0-100) as values
    """
    score = rouge_multi(generated, references)
    rouge_s = {k: (v * 100) for (k, v) in score.items()}
    """
    "rouge_1/f_score": rouge_1_f,
    "rouge_1/r_score": rouge_1_r,
    "rouge_1/p_score": rouge_1_p,
    "rouge_2/f_score": rouge_2_f,
    "rouge_2/r_score": rouge_2_r,
    "rouge_2/p_score": rouge_2_p,
    "rouge_l/f_score": rouge_l_f,
    "rouge_l/r_score": rouge_l_r,
    "rouge_l/p_score": rouge_l_p,
    """
    return rouge_s


def bleu_score(references, generated, n_gram=4, smooth=False):
    """
    Compute BLEU score for generated explanations against multiple references.

    BLEU (Bilingual Evaluation Understudy) measures n-gram precision between
    generated text and reference texts. Supports multiple references per hypothesis.

    Metric names in results: 'BLEU-1', 'BLEU-2', 'BLEU-4'

    Args:
        references: List of lists of token lists, where each outer list contains
                   reference explanations (as token lists) for one sample
        generated: List of token lists, each being a generated explanation
        n_gram: Maximum n-gram order to consider (default: 4)
        smooth: Whether to apply smoothing for zero counts (default: False)

    Returns:
        float: BLEU score scaled to 0-100
    """
    # formatted_ref = [[ref] for ref in references]
    bleu_s, _, _, _, _, _ = compute_bleu(references, generated, n_gram, smooth)
    return bleu_s * 100


def unique_sentence_ratio(sentences):
    """
    Compute Unique Sentence Ratio (USR) and count (USN) for generated explanations.

    Measures text diversity by counting how many unique explanations were generated
    compared to the total number of explanations.

    Metric names in results: 'USR' (ratio), 'USN' (count)

    Args:
        sentences: List of strings, each being a generated explanation

    Returns:
        tuple: (unique_ratio, unique_count)
            - unique_ratio (float): Ratio of unique sentences to total sentences
            - unique_count (int): Number of unique sentences
    """
    unique_sents = set(sentences)
    return len(unique_sents) / len(sentences), len(unique_sents)


# ------------------------------------------------------------------ #
# ------------- SAE CONCEPT PREDICTION METRICS -------------------- #
# ------------------------------------------------------------------ #


def concept_set_overlap(
    preds: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """
    Set-level overlap metrics between predicted and ground-truth concept sets.

    Both ``preds`` and ``labels`` are ``[n_test, top_k]`` integer arrays of
    concept indices ranked in descending order.  Since both sets have the same
    fixed size K, Precision@K = Recall@K, so F1@K = P@K = R@K per sample;
    Jaccard is the independently informative set metric.

    Metric names (returned dict keys): 'Jaccard', 'Precision', 'Recall', 'F1'.

    Args:
        preds:  ``[n_test, top_k]`` predicted concept indices.
        labels: ``[n_test, top_k]`` ground-truth concept indices.

    Returns:
        Dict with macro-averaged 'Jaccard', 'Precision', 'Recall', 'F1'.
    """
    n = preds.shape[0]
    jaccard_vals = np.empty(n)
    # Since preds and labels contain same size, precision, recall, and f1 are the same
    prf_vals = np.empty(n)

    for i in range(n):
        pred_set = set(preds[i].tolist())
        label_set = set(labels[i].tolist())
        intersection = len(pred_set & label_set)
        union = len(pred_set | label_set)
        p = intersection / len(pred_set) if pred_set else 0.0
        jaccard_vals[i] = intersection / union if union > 0 else 0.0
        prf_vals[i] = p

    return {
        "Jaccard": float(np.mean(jaccard_vals)),
        "P-R-F1": float(np.mean(prf_vals)),
    }


def concept_ndcg(
    preds: np.ndarray,
    labels: np.ndarray,
    decay_concept_rel: bool = False,
    concept_rel: Optional[np.ndarray] = None,
) -> float:
    """
    NDCG@K for concept set prediction.

    Evaluates whether the model's concept ranking places true label concepts
    early in the list.  Optionally weights relevance by activation magnitude.

    Metric names in results: 'Cpt/NDCG' (binary) and 'Cpt/ActNDCG'
    (activation-weighted when ``concept_rel`` is provided).

    Args:
        preds:       ``[n_test, top_k]`` predicted concept indices (ranked).
        labels:      ``[n_test, top_k]`` ground-truth concept indices (ranked
                     by descending activation magnitude).
        decay_concept_rel: Whether to decay the relevance score by the position of the concept in the list.
        concept_rel: Optional ``[n_concepts]`` array of relevance scores
                     (e.g. training mean activation per concept).  When
                     ``None``, binary relevance is used (1 if in label set).

    Returns:
        Float NDCG@K, macro-averaged over test samples.
    """
    n, k = preds.shape
    # discount factors: 1/log2(j+1) for j = 1..K
    discounts = 1.0 / np.log2(np.arange(2, k + 2, dtype=float))
    ndcg_vals = np.empty(n)

    for i in range(n):
        label_set = set(labels[i].tolist())

        if decay_concept_rel:
            rel_pred = np.array([discounts[i] if c in label_set else 0.0 for i, c in enumerate(preds[i])])
            ideal_rels = discounts
        elif concept_rel is not None:
            # Activation-weighted relevance; zero for non-label concepts
            rel_pred = np.array([concept_rel[c] if c in label_set else 0.0 for c in preds[i]])
            # Ideal: label concepts sorted by descending activation magnitude
            ideal_rels = np.sort(concept_rel[labels[i]])[::-1]
        else:
            rel_pred = np.array([1.0 if c in label_set else 0.0 for c in preds[i]])
            # All K label concepts are equally relevant → IDCG uses all ones
            ideal_rels = np.ones(k)

        dcg = float(np.dot(rel_pred, discounts))
        idcg = float(np.dot(ideal_rels, discounts))
        ndcg_vals[i] = dcg / idcg if idcg > 0 else 0.0

    return float(np.mean(ndcg_vals))


def concept_mrr(
    preds: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Mean Reciprocal Rank for concept set prediction.

    The reciprocal rank for sample ``i`` is ``1/r`` where ``r`` is the
    1-indexed position of the first predicted concept that belongs to the
    ground-truth label set.  Zero if no hit within top-K.

    Metric name in results: 'Cpt/MRR'.

    Args:
        preds:  ``[n_test, top_k]`` predicted concept indices.
        labels: ``[n_test, top_k]`` ground-truth concept indices.

    Returns:
        Float MRR, averaged over test samples.
    """
    n = preds.shape[0]
    rr_vals = np.zeros(n)
    for i in range(n):
        label_set = set(labels[i].tolist())
        for rank, c in enumerate(preds[i], start=1):
            if c in label_set:
                rr_vals[i] = 1.0 / rank
                break
    return float(np.mean(rr_vals))


def concept_coverage(
    preds: np.ndarray,
    n_concepts: int,
) -> float:
    """
    Fraction of the full concept vocabulary that appears in any prediction.

    Low coverage indicates the model collapses to a small set of popular
    concepts (head bias).

    Metric name in results: 'Cpt/Coverage'.

    Args:
        preds:      ``[n_test, top_k]`` predicted concept indices.
        n_concepts: Total number of concepts in the vocabulary.

    Returns:
        Float in ``[0, 1]``.
    """
    return len(np.unique(preds.ravel())) / max(1, n_concepts)


def concept_popularity_discrepancy(
    preds: np.ndarray,
    concept_popularity: np.ndarray,
    n_concepts: int,
) -> float:
    """
    Average absolute difference between training popularity and prediction
    frequency, macro-averaged over all concepts.

    ``popularity_discrepancy = (1/C) * sum_c |pop_train(c) - freq_pred(c)|``

    where ``pop_train(c)`` is the fraction of training samples containing
    concept ``c`` and ``freq_pred(c)`` is the fraction of test predictions
    containing ``c``.  A high value signals popularity bias.

    Metric name in results: 'Cpt/PopDisc'.

    Args:
        preds:              ``[n_test, top_k]`` predicted concept indices.
        concept_popularity: ``[n_concepts]`` training popularity per concept.
        n_concepts:         Vocabulary size.

    Returns:
        Float average popularity discrepancy.
    """
    n = preds.shape[0]
    freq_pred = np.zeros(n_concepts)
    unique, counts = np.unique(preds.ravel(), return_counts=True)
    freq_pred[unique] = counts / n
    return float(np.mean(np.abs(concept_popularity - freq_pred)))


def concept_sample_popularity_discrepancy(
    preds: np.ndarray,
    labels: np.ndarray,
    concept_popularity: np.ndarray,
) -> Dict[str, float]:
    """
    Per-sample popularity discrepancy between predicted and label concept sets,
    aggregated over test samples.

    For each sample ``i``:

    .. code-block:: text

        pop_pred(i)   = mean popularity of preds[i]  concepts
        pop_label(i)  = mean popularity of labels[i] concepts
        disc(i)       = |pop_pred(i) − pop_label(i)|

    Returns the mean of ``disc(i)`` together with the raw mean popularities so
    the *direction* of bias (predictions skewing more/less popular than labels)
    is immediately visible.

    Metric names in results:
        - ``'Cpt/SPopBias'``    – mean (pop_pred − pop_label) per sample; positive
          means the model systematically predicts more popular concepts than the
          labels, negative means the opposite.
        - ``'Cpt/AvgPopPred'``  – mean predicted-set popularity across samples
        - ``'Cpt/AvgPopLabel'`` – mean label-set popularity across samples

    Args:
        preds:              ``[n_test, top_k]`` predicted concept indices.
        labels:             ``[n_test, top_k]`` ground-truth concept indices.
        concept_popularity: ``[n_concepts]`` training popularity per concept.

    Returns:
        Dict with keys ``'SPopBias'``, ``'AvgPopPred'``, ``'AvgPopLabel'``.
    """
    pop_pred = concept_popularity[preds].mean(axis=1)  # [n_test]
    pop_label = concept_popularity[labels].mean(axis=1)  # [n_test]
    return {
        "SPopBias": float(np.mean(pop_pred - pop_label)),
        "AvgPopPred": float(np.mean(pop_pred)),
        "AvgPopLabel": float(np.mean(pop_label)),
    }


def concept_kl_divergence(
    preds: np.ndarray,
    labels: np.ndarray,
    n_concepts: int,
    eps: float = 1e-10,
) -> float:
    """
    KL divergence KL(P_label ‖ P_pred) between empirical concept frequency
    distributions over the test set.

    Captures global distributional alignment: a large KL means the model's
    concept distribution differs substantially from the ground-truth one.

    Metric name in results: 'Cpt/KL'.

    Args:
        preds:      ``[n_test, top_k]`` predicted concept indices.
        labels:     ``[n_test, top_k]`` ground-truth concept indices.
        n_concepts: Vocabulary size.
        eps:        Smoothing constant to avoid log(0).

    Returns:
        Float KL divergence (nats).
    """
    label_counts = np.zeros(n_concepts)
    pred_counts = np.zeros(n_concepts)
    unique, counts = np.unique(labels.ravel(), return_counts=True)
    label_counts[unique] = counts / max(counts.sum(), 1)
    unique, counts = np.unique(preds.ravel(), return_counts=True)
    pred_counts[unique] = counts / max(counts.sum(), 1)

    p_label = label_counts
    p_pred = pred_counts

    mask = p_label > 0
    kl = float(np.sum(p_label[mask] * np.log((p_label[mask] + eps) / (p_pred[mask] + eps))))
    return kl


def concept_activation_correlation(
    preds: np.ndarray,
    concept_mean_act: np.ndarray,
) -> Tuple[float, float]:
    """
    Correlation between predicted concept rank (1-indexed position in top-K)
    and the concept's ground-truth activation magnitude from training stats.

    A good model should place high-activation concepts at early positions,
    yielding a *negative* Pearson/Spearman correlation (rank ↑ → activation ↓).
    Flatness or positive correlation signals rank-insensitivity.

    Metric names in results: 'Cpt/PearsonR', 'Cpt/SpearmanR'.

    Args:
        preds:            ``[n_test, top_k]`` predicted concept indices.
        concept_mean_act: ``[n_concepts]`` mean activation per concept
                          (from training statistics).

    Returns:
        Tuple ``(pearson_r, spearman_r)``, each in ``[-1, 1]``.
    """
    n, k = preds.shape
    ranks = np.tile(np.arange(1, k + 1, dtype=float), n)
    acts = concept_mean_act[preds.ravel()].astype(float)
    return float(pearsonr(ranks, acts).statistic), float(spearmanr(ranks, acts).statistic)  # type: ignore[union-attr]
