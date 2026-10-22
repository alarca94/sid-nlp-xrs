import json
import logging
import os
import re
import sys
import time
from collections.abc import Callable
from types import SimpleNamespace
from typing import Optional, Tuple

import nltk
import numpy as np
import pandas as pd
import stanza
import torch
import yaml
from tqdm import tqdm

from download_datasets import AMAZON_DOMAINS
from nlp_xrs.utils.constants import DATA_MAP, Key
from nlp_xrs.utils.in_out import get_paths, save_safe_parquet




def read_yelp(data_orig_path: str, return_features: bool = True) -> Tuple[pd.DataFrame, Optional[set]]:
    data = pd.read_pickle(os.path.join(data_orig_path, "review_splits.pkl"))
    data = data["train"] + data["val"] + data["test"]
    data = pd.DataFrame.from_records(data)

    col_map = {
        "reviewerID": Key.USER,
        "asin": Key.ITEM,
        "reviewText": Key.REV,
        "unixReviewTime": Key.TIMESTAMP,
        "overall": Key.RATING,
    }
    data.rename(col_map, axis=1, inplace=True)
    data.dropna(subset=[Key.REV], inplace=True, axis=0)
    data.reset_index(drop=True, inplace=True)

    data = data[list(col_map.values())]

    feats = None
    if return_features:
        # feats = data['feature'].unique().tolist()
        with open(os.path.join(data_orig_path, "feature", "features.json"), "r") as f:
            feats = set(json.loads(f.read()))

    return data, feats




def read_amazon(data_orig_path: str, return_features: bool = True) -> Tuple[pd.DataFrame, Optional[set]]:
    data = pd.read_json(os.path.join(data_orig_path, "reviews.json.gz"), orient="records", lines=True)

    c_map = {
        "asin": Key.ITEM,
        "overall": Key.RATING,
        "reviewerID": Key.USER,
        "reviewText": Key.REV,
        "unixReviewTime": Key.TIMESTAMP,
    }

    data = data[c_map.keys()]
    data = data.rename(c_map, axis=1)

    data[Key.RATING] = data[Key.RATING].astype(int)

    feats = None
    if return_features:
        if os.path.isfile(os.path.join(data_orig_path, "feature", "features.json")):
            # Pre-filtered features
            with open(os.path.join(data_orig_path, "feature", "features.json"), "r") as f:
                feats = set(json.loads(f.read()).keys())
        else:
            # Sentires features
            rev_df = pd.DataFrame.from_records(
                pd.read_pickle(os.path.join(data_orig_path, "feature", "reviews.pickle"))
            )
            feats = set(rev_df["sentence"].dropna().explode().str[0].unique().tolist())

    return data, feats


def read_orig_data(
    dataset: str, data_orig_path: str, return_features: bool = True
) -> Tuple[pd.DataFrame, Optional[set]]:
    st = time.time()

    logging.info(f"Loading {DATA_MAP.get(dataset.lower(), dataset)} dataset...")
    feats = None
    if DATA_MAP.get(dataset, dataset).lower() == "yelp":
        data, feats = read_yelp(data_orig_path, return_features)

    elif DATA_MAP.get(dataset, dataset).lower() in {d.lower() for d in AMAZON_DOMAINS}:
        data, feats = read_amazon(data_orig_path, return_features)

    else:
        raise NotImplementedError(f"read_data() does not currently supports {dataset} dataset.")

    data = data[data[Key.REV].str.len() > 0]
    # data.dropna(subset=[Key.REV], inplace=True)
    logging.info(f"Ellapsed time {time.time() - st:.4f}s")
    return data, feats


def clean_sentence(s):
    s = re.sub(r"https?://.*", " ", s)
    s = re.sub(r"\.\.+", "...", s)
    s = re.sub(r"`", "'", s)
    s = re.sub(r"^\s*-", "", s).replace("*", " ").replace("-", " - ").replace("/", " / ").replace("~", " ")

    s = re.sub(r"\s\s+", " ", s).strip().lower()
    return s


def preprocess_reviews(data, verbose=False):
    revs = data[Key.REV]
    sents = revs.apply(nltk.sent_tokenize).explode()
    sents2 = sents.str.split("\n").explode()
    if verbose:
        if len(sents) != len(sents2):
            diff = sents.value_counts(sort=False) != sents2.value_counts(sort=False)
            print(f"List of sentences where we needed to split_lines: {diff[diff].index.tolist()}")
        else:
            assert (sents.value_counts(sort=False) == sents2.value_counts(sort=False)).all()

    # Clean sentences
    sents = sents2.apply(clean_sentence)

    # Filter too long and short sentences
    lens = sents.str.len()
    mask = np.logical_and(lens > 3, lens < 400)
    sents2 = sents[mask]

    # Aggregate back review sentences
    return sents2.groupby(level=0).agg("\n\n".join)


def nlp_exp_rule(doc, feats: set):
    """nlp_exp_rule borrowed from https://github.com/HCDM/XRec/tree/main/SAER"""
    out = {Key.FEAT: [], Key.EXP: []}
    for sen in doc.sentences:
        feats_in_exp = set()
        is_exp = False
        for w in sen.words:
            if w.text in feats:
                feats_in_exp.add(w.text)
                while w.deprel == "compound":
                    w = sen.words[w.head - 1]

                if w.deprel == "nsubj" or w.deprel == "nsubj:pass":
                    head = sen.words[w.head - 1]
                    if head.xpos in {"JJ", "JJR", "JJS", "RB", "VBN", "NN"}:
                        is_exp = True
            elif w.xpos in {"JJ", "JJR", "JJS"}:
                head = sen.words[w.head - 1]
                if head.deprel == "root" and head.text in feats:
                    is_exp = True
        if is_exp:
            out[Key.EXP].append(" ".join(w.text for w in sen.words))
            out[Key.FEAT].append(list(feats_in_exp))
    return out


def naive_exp_rule(doc, feats: set):
    out = {Key.FEAT: [], Key.EXP: []}

    for sen in doc.sentences:
        feats_in_exp = set()
        is_exp = False

        words = sen.words
        for word in words:
            text = word.text

            if text in feats:
                feats_in_exp.add(text)
                is_exp = True

        if is_exp:
            out[Key.EXP].append(" ".join(w.text for w in words))
            out[Key.FEAT].append(list(feats_in_exp))

    return out


@torch.no_grad()
def extract_exps_stanza(
    dataset: str, paths: SimpleNamespace, nlp_processors: str, rule_fn: Callable, batch_size: int = 64
):
    handlers = [logging.StreamHandler(stream=sys.stdout)]

    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s - %(message)s", handlers=handlers)

    if not os.path.isdir(paths.data):
        os.makedirs(paths.data)

    logging.info("Reading data...")
    data, feats = read_orig_data(dataset, paths.data_orig)

    logging.info("Preprocessing reviews...")
    sentences = preprocess_reviews(data)
    data["sents"] = ""
    data.loc[sentences.index.values, "sents"] = sentences
    # NOTE - Yelp: ~1k samples are removed due to either too long or too short reviews
    data = data[data["sents"] != ""].reset_index(drop=True)
    sentences = data["sents"].tolist()
    data.drop("sents", axis=1, inplace=True)
    # data = data.iloc[:(batch_size * 10)]

    logging.info("Converting to tokenized documents...")
    nlp = stanza.Pipeline(
        "en",
        processors=nlp_processors,
        tokenize_no_ssplit=True,
        use_gpu=(nlp_processors != "tokenize"),
        logging_level="ERROR",
    )

    # n_batches = int(np.ceil(len(sentences)/batch_size))
    feat_exps = [rule_fn(d, feats) for d in tqdm(nlp.stream(sentences, batch_size=batch_size), total=len(sentences))]

    logging.info("Extracting Feats-Exps from documents...")
    # exps = pd.DataFrame.from_records(list(map(partial(rule_fn, feats=feats), docs)))
    feat_exps = pd.DataFrame.from_records(feat_exps)

    data = pd.concat((data, feat_exps), axis=1)

    # NOTE - Yelp: ~65k samples are removed due to empty explanations
    empty_mask = data[Key.EXP].apply(len) == 0
    if sum(empty_mask) > 0:
        logging.info(f"Removing {sum(empty_mask)} / {data.shape[0]} records due to empty explanation")
        save_safe_parquet(data[empty_mask].reset_index(drop=True), os.path.join(paths.data, "empty_exps.parquet"))
        data = data[~empty_mask].reset_index(drop=True)

    logging.info("Saving processed data...")
    save_safe_parquet(data, os.path.join(paths.data, "reviews.parquet"))


def run(dataset, base_path):
    paths = get_paths(dataset=dataset, fold="0", base_path=base_path, hyperopt=False)

    with open(os.path.join(paths.config, "data_preparation", f"{dataset.lower()}.yaml"), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    rule_fn = {"nlp_exp_rule": nlp_exp_rule, "naive_exp_rule": naive_exp_rule}[config["exp_rule"]]

    extract_exps_stanza(dataset, paths, config["stanza_tools"], rule_fn)
