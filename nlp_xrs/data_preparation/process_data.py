import os
from types import SimpleNamespace
from typing import List

import yaml

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import logging
import random
import re
import shutil
from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm

from nlp_xrs.data.utils import Mapper, get_transform_mapping
from nlp_xrs.utils.constants import (
    AMAZON_DOMAINS,
    DATA_MAP,
    MAX_HIST_LEN,
    RNG_SEED,
    SPECIAL_TOKENS,
    TXT_LEN,
    Key,
    Prefix,
    SeqMode,
    Suffix,
    Task,
)
from nlp_xrs.utils.funcs import init_seed
from nlp_xrs.utils.in_out import get_paths, load_mapper_parquet, load_partition, load_safe_parquet, save_safe_parquet

tqdm.pandas()


# Download Amazon datasets: wget https://jmcauley.ucsd.edu/data/amazon_v2/categoryFiles/{Category}.json.gz
# Download Amazon metadata: https://jmcauley.ucsd.edu/data/amazon_v2/metaFiles2/meta_Movies_and_TV.json.gz


def iterative_kcore_filter(df, kcore, verbose=1):
    def kcore_filter(d, col):
        return d[d[col].map(d[col].value_counts()).ge(kcore)]

    copy_df = df.copy()
    prev_sz = -1
    if verbose == 1:
        logging.info(f"Starting Iterative K-Core item and user filtering with K = {kcore}...")
        logging.info(f"Initial number of interactions: {df.shape[0]}")
        logging.info(f"Initial number of users: {df[Key.USER].nunique()}")
        logging.info(f"Initial number of items: {df[Key.ITEM].nunique()}")
    while prev_sz != copy_df.shape[0]:
        # Filter by user profile size
        prev_sz = copy_df.shape[0]
        copy_df = kcore_filter(copy_df, Key.USER)
        copy_df = kcore_filter(copy_df, Key.ITEM)

    if verbose == 1:
        logging.info(f"Final number of interactions: {copy_df.shape[0]}")
        logging.info(f"Final number of users: {copy_df[Key.USER].nunique()}")
        logging.info(f"Final number of items: {copy_df[Key.ITEM].nunique()}")
    return copy_df


def split_data(datasets, base_path, test_ratio=0.1, f_name="reviews.parquet", split="0"):
    val_ratio = test_ratio

    for dataset in datasets:
        data_path = get_paths(dataset, split, base_path).data
        logging.info(f"Splitting dataset {dataset}")
        data = load_safe_parquet(os.path.join(data_path, f_name))
        if isinstance(data, (list, tuple)):
            data = pd.DataFrame.from_records(data)

        data = data.sort_values(by=[Key.USER, Key.TIMESTEP])
        assert (data.index.values == list(range(data.shape[0]))).all()

        ucounts = data.groupby(Key.USER).size().values
        logging.info(f"Min. user count: {ucounts.min()}. Max. user count: {ucounts.max()}")
        # Users with 2 interactions or less go to the training set
        mask = (ucounts > 2).astype(int)
        uoffsets = ucounts.cumsum()
        split_ixs = np.zeros((data.shape[0],), dtype=int)
        if isinstance(test_ratio, float):
            assert isinstance(val_ratio, float)
            assert test_ratio < 1.0
            tst_start_ixs = uoffsets - np.maximum(ucounts * test_ratio, mask).astype(int)
            val_start_ixs = tst_start_ixs - np.maximum(ucounts * val_ratio, mask).astype(int)
        elif isinstance(test_ratio, int):
            assert isinstance(val_ratio, int)
            assert all(ucounts > (test_ratio + val_ratio))
            tst_start_ixs = uoffsets - test_ratio
            val_start_ixs = tst_start_ixs - val_ratio
        else:
            raise TypeError("test_ratio is neither int nor float")
        for vix, tix, offset in zip(val_start_ixs, tst_start_ixs, uoffsets):
            split_ixs[tix:offset] = 2
            split_ixs[vix:tix] = 1

        if not os.path.exists(os.path.join(data_path, split)):
            os.makedirs(os.path.join(data_path, split))
        else:
            shutil.rmtree(os.path.join(data_path, split))
            os.mkdir(os.path.join(data_path, split))

        trn_ixs = data.iloc[np.argwhere(split_ixs == 0).squeeze()].index.values
        val_ixs = data.iloc[np.argwhere(split_ixs == 1).squeeze()].index.values
        tst_ixs = data.iloc[np.argwhere(split_ixs == 2).squeeze()].index.values
        assert (trn_ixs == np.argwhere(split_ixs == 0).squeeze()).all()
        assert (val_ixs == np.argwhere(split_ixs == 1).squeeze()).all()
        assert (tst_ixs == np.argwhere(split_ixs == 2).squeeze()).all()

        trn = data.loc[trn_ixs]
        val = data.loc[val_ixs]
        tst = data.loc[tst_ixs]
        trn_u = set(trn.user.unique())
        val_u = set(val.user.unique())
        tst_u = set(tst.user.unique())
        assert len(val_u.difference(trn_u)) == 0
        assert len(tst_u.difference(val_u)) == 0
        assert len(tst_u.difference(trn_u)) == 0
        # if mode != 'feat_only':
        #     assert len(trn_u.difference(val_u)) == 0
        #     assert len(trn_u.difference(tst_u)) == 0
        trn_i = trn.item.unique()
        val_i = val.item.unique()
        tst_i = tst.item.unique()
        val_i_diff = set(val_i).difference(trn_i)
        tst_i_diff = set(tst_i).difference(trn_i)
        logging.info(f"Number of items in valid but not in train: {len(val_i_diff)}")
        logging.info(f"Number of items in test but not in train: {len(tst_i_diff)}")
        logging.info(f"Percentage of missing item samples (Val): {sum(val.item.isin(val_i_diff)) / val.shape[0]:.4f}")
        logging.info(f"Percentage of missing item samples (Test): {sum(tst.item.isin(tst_i_diff)) / tst.shape[0]:.4f}")
        logging.info(
            f"Ratio of non-empty feature in training: {trn[Key.FEAT].explode().explode().isna().sum() / trn[Key.FEAT].explode().shape[0]:.4f}"
        )
        logging.info(
            f"Ratio of non-empty feature in valid: {val[Key.FEAT].explode().explode().isna().sum() / val[Key.FEAT].explode().shape[0]:.4f}"
        )
        logging.info(
            f"Ratio of non-empty feature in test: {tst[Key.FEAT].explode().explode().isna().sum() / tst[Key.FEAT].explode().shape[0]:.4f}"
        )
        logging.info(
            f"Final ratios for dataset {dataset} are: {np.unique(split_ixs, return_counts=True)[1] / len(split_ixs)}"
        )

        # Assert that train, valid, and test sets were properly created based on timestep
        trn_latest_steps = trn[[Key.USER, Key.TIMESTEP]].groupby(Key.USER)[Key.TIMESTEP].agg("max")
        val_earliest_steps = val[[Key.USER, Key.TIMESTEP]].groupby(Key.USER)[Key.TIMESTEP].agg("min")
        val_latest_steps = val[[Key.USER, Key.TIMESTEP]].groupby(Key.USER)[Key.TIMESTEP].agg("max")
        tst_earliest_steps = tst[[Key.USER, Key.TIMESTEP]].groupby(Key.USER)[Key.TIMESTEP].agg("min")

        phase_steps = trn_latest_steps.to_frame("train_latest")
        phase_steps["valid_earliest"] = phase_steps["train_latest"].tolist()
        phase_steps.loc[val_earliest_steps.index.values, "valid_earliest"] = val_earliest_steps.tolist()
        phase_steps["valid_latest"] = phase_steps["valid_earliest"].tolist()
        phase_steps.loc[val_latest_steps.index.values, "valid_latest"] = val_latest_steps.tolist()
        phase_steps["test_earliest"] = phase_steps["valid_latest"].tolist()
        phase_steps.loc[tst_earliest_steps.index.values, "test_earliest"] = tst_earliest_steps.tolist()

        assert ((phase_steps["valid_earliest"] - phase_steps["train_latest"]) > 0).all()
        assert ((phase_steps["valid_latest"] - phase_steps["valid_earliest"]) >= 0).all()
        assert ((phase_steps["test_earliest"] - phase_steps["valid_latest"]) > 0).all()

        np.save(os.path.join(data_path, split, "train"), trn_ixs)
        np.save(os.path.join(data_path, split, "validation"), val_ixs)
        np.save(os.path.join(data_path, split, "test"), tst_ixs)

    logging.info("Finished!")


def encode(data_path, data, cols, suffix="", save=True):
    new_data = data.copy()

    if not os.path.isdir(os.path.join(data_path, "mappers")):
        os.makedirs(os.path.join(data_path, "mappers"))

    for c in cols:
        if os.path.isfile(os.path.join(data_path, "mappers", f"{c}{suffix}.parquet")):
            # Check that the mapping is correct
            logging.info(f"Mapping for column {c} already exists, checking its consistency...")
            mapper = load_mapper_parquet(os.path.join(data_path, "mappers", f"{c}{suffix}.parquet"))
            vals = new_data[c]
            while isinstance(vals.iloc[0], list):
                vals = vals.explode()
            assert set(vals.unique().tolist()).issubset(mapper.values.tolist())
        else:
            logging.info(f"Encoding column: {c}")
            vals = new_data[c].explode()
            if isinstance(vals.iloc[0], list):
                logging.info(f"Column: {c} contains nested lists")
                vals.index = pd.MultiIndex.from_tuples(list(zip(vals.index.tolist(), range(len(vals)))))
                vals = vals.explode()
            if isinstance(vals.iloc[0], str):
                vals = vals.apply(lambda t: re.sub("\\x00", "", t))
            unique_vals = vals.unique().tolist()
            rev_mapping = pd.Series(range(len(unique_vals)), index=unique_vals)
            vals = vals.map(rev_mapping, na_action="ignore")
            assert not vals.isna().any()
            # if isinstance(new_data[c].iloc[0], list):
            if vals.index.get_level_values(0).value_counts().nunique() > 1:
                levels = list(range(vals.index.nlevels))
                logging.info(f"Column: {c} has {len(levels)}-level multi-index")
                while levels:
                    vals = vals.groupby(level=levels).agg(list)
                    levels = levels[:-1]
            new_data[c] = vals
            if save:
                mapping = pd.Series(rev_mapping.index.values, index=rev_mapping.values, name="raw")
                save_safe_parquet(mapping.reset_index(), os.path.join(data_path, "mappers", f"{c}{suffix}.parquet"))

    return new_data


def iterative_kcore_filter_revs(df, min_revs=5, kcore=5, subset=[Key.USER], verbose=1):
    """Filter users with less than min_revs unique reviews"""

    def kcore_rev_filter(d, col):
        dedup_data = d.drop_duplicates(subset=[c, Key.REV])
        dedup_counts = dedup_data[c].value_counts()
        return d[d[col].map(dedup_counts) >= min_revs]

    # logging.info(colored(f'Filtering out {subset} with less than {kcore} unique reviews', Colors.BLUE))
    prev_sz = -1
    copy_df = df.copy()

    while prev_sz != copy_df.shape[0]:
        prev_sz = copy_df.shape[0]
        for c in subset:
            copy_df = kcore_rev_filter(copy_df, c)
        copy_df = iterative_kcore_filter(copy_df, kcore, verbose=0)

    if verbose:
        logging.info(f"Started with {df.shape[0]} samples and ended with {copy_df.shape[0]} samples")
        logging.info(f"Started with {df[Key.USER].nunique()} users and ended with {copy_df[Key.USER].nunique()} users")
        logging.info(f"Started with {df[Key.ITEM].nunique()} items and ended with {copy_df[Key.ITEM].nunique()} items")
        logging.info(
            f"Started with {df[Key.REV].nunique()} unique revs and ended with {copy_df[Key.REV].nunique()} unique revs"
        )

    return copy_df


def sliding_w_hist(g: pd.DataFrame, max_l: int = MAX_HIST_LEN):
    return [g.index.values[max(i - max_l, 0) : i].tolist() for i in range(g.shape[0])]


def process_xrec_datasets(base_path: str, datasets: list = ("beauty", "cell", "yelp"), test: bool = False):
    from nlp_xrs.utils.in_out import Colors, colored

    suffix = Suffix.TEST if test else ""
    f_name = f"reviews{suffix}"
    split = f"0{suffix}"
    for dataset in datasets:
        paths = get_paths(dataset, split, base_path)
        data_path = paths.data

        with open(os.path.join(paths.config, "data_preparation", f"{dataset.lower()}.yaml"), "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        min_revs = config["min_revs"]
        kcore = config["kcore"]
        dedup_uir = config["dedup_uir"]
        min_exp_toks = config["min_exp_toks"]
        max_u_samples = config["max_u_samples"]

        if not os.path.exists(data_path):
            os.makedirs(data_path)
            logging.info(
                colored(
                    f"{dataset} dataset does not contain the reviews{suffix}.parquet file needed under {data_path}",
                    Colors.RED,
                )
            )
            continue

        logging.info(f"Reading {dataset}{suffix} dataset...")
        if os.path.isfile(os.path.join(data_path, f"old_{f_name}.parquet")):
            data = load_safe_parquet(os.path.join(data_path, f"old_{f_name}.parquet"))
            logging.info(colored(f"WARNING: Emptying {split}/ folder as the dataset is being recreated", Colors.YELLOW))
        else:
            data = load_safe_parquet(os.path.join(data_path, f"{f_name}.parquet"))
            save_safe_parquet(data, os.path.join(data_path, f"old_{f_name}.parquet"))
        # data = pd.DataFrame.from_records(pd.read_pickle(os.path.join(DATA_PATHS[dataset], mode, f_name)))

        # Fix column format
        rat_cols = [c for c in data.columns if Key.RATING in c]
        data[rat_cols] = data[rat_cols].astype(float)
        data[Key.TIMESTAMP] = pd.to_datetime(data[Key.TIMESTAMP], unit=config.get("time_unit", "ns"))

        if dedup_uir:
            logging.info("Deduplicating user-item interactions and user reviews...")
            data = data.sort_values(Key.TIMESTAMP).reset_index(drop=True)
            data = data.drop_duplicates(subset=[Key.USER, Key.ITEM], keep="first")
            data = data.drop_duplicates(subset=[Key.USER, Key.REV], keep="first")

        logging.info("Iterative k-core filtering...")
        data = iterative_kcore_filter_revs(data, min_revs=min_revs, kcore=kcore, subset=[Key.USER, Key.ITEM])

        if min_exp_toks is not None:
            logging.info(f"Filtering out explanation sentences with less than {min_exp_toks} tokens...")
            data = data.explode([Key.FEAT, Key.EXP])
            data = data[data[Key.EXP].str.split().str.len() >= min_exp_toks]
            groupby_cols = {c: list for c in [Key.FEAT, Key.EXP]}
            dedup_cols = {c: "first" for c in data.columns if c not in groupby_cols}
            data = data.groupby(level=0).agg({**dedup_cols, **groupby_cols})
            data = iterative_kcore_filter_revs(data, min_revs=min_revs, kcore=kcore, subset=[Key.USER, Key.ITEM])

        if max_u_samples is not None:
            logging.info(f"Filtering out users exceeding {max_u_samples} interactions...")
            u_cnts = data.value_counts(Key.USER)
            valid_us = u_cnts[u_cnts <= max_u_samples].index.values
            data = iterative_kcore_filter_revs(
                data[data[Key.USER].isin(valid_us)], min_revs=min_revs, kcore=kcore, subset=[Key.USER, Key.ITEM]
            )

        # Extract encoding mappers and save to separate file
        logging.info("Encoding...")
        data = encode(data_path, data, [Key.USER, Key.ITEM, Key.FEAT, Key.REV, Key.EXP], suffix)

        logging.info("Adding Time Step order")
        data.sort_values(Key.TIMESTAMP, inplace=True)
        data[Key.TIMESTEP] = list(range(data.shape[0]))

        logging.info("Optimizing memory by downcasting")
        data[[Key.USER, Key.ITEM, Key.TIMESTEP]] = data[[Key.USER, Key.ITEM, Key.TIMESTEP]].apply(
            pd.to_numeric, downcast="integer"
        )
        # data[Key.EXP] = pd.to_numeric(data[Key.EXP].explode(), downcast="integer").groupby(level=0).agg(list)
        rat_cols = [Key.RATING] + [c for c in data.columns if c.startswith(f"{Key.RATING}_")]
        data[rat_cols] = data[rat_cols].apply(pd.to_numeric, downcast="float")
        # data[Key.FEAT] = pd.to_numeric(
        #     data[Key.FEAT].explode().reset_index().set_index(['index'], append=True)[Key.FEAT].explode(),
        #     downcast="integer").groupby(level=[0, 1]).agg(list).groupby(level=1).agg(list)
        data[Key.REV] = pd.to_numeric(data[Key.REV], downcast="integer")

        # Extract sequential features grouped by Key.USER and Key.ITEM
        # NOTE 1: There is no Key.ADJ in XRec datasets
        # NOTE 2: max_seq_length is selected through EDA (Users surpassing 200-500 reviews are aggregated users in TripAdvisor)
        logging.info("Extracting sequential features...")
        tqdm.pandas()
        data = data.sort_values(by=[Key.USER, Key.TIMESTEP]).reset_index(drop=True)

        orig_index = data.index.values
        assert (orig_index == list(range(data.shape[0]))).all()
        old_cols = list(data.columns)
        hist_cols = []
        for groupby in [Key.USER, Key.ITEM]:
            logging.info(f"Extracting history for {groupby} column")
            hist_cols.append(f"{groupby}_{Prefix.HIST[:-1]}")
            data.sort_values([groupby, Key.TIMESTEP], inplace=True)
            histories = data[[groupby, Key.TIMESTEP]].groupby(groupby).progress_apply(sliding_w_hist)
            data[hist_cols[-1]] = sum(histories.tolist(), [])
        data = data.loc[orig_index]

        data[old_cols].to_parquet(os.path.join(data_path, f"{f_name}.parquet"))
        save_safe_parquet(data[hist_cols], os.path.join(data_path, f"{Prefix.HIST}{f_name}.parquet"))

    split_data(datasets=datasets, base_path=base_path, f_name=f"{f_name}.parquet", split=split)


def read_data(path: str, fold: str = "0", data_file: str = "reviews.parquet", test: bool = False):
    suffix = Suffix.TEST if test else ""
    fold += suffix
    data_file = f"{suffix}.".join(data_file.split("."))

    ixs_dir = os.path.join(path, str(fold))
    data_path = os.path.join(path, data_file)
    assert os.path.exists(data_path)
    data = load_safe_parquet(data_path)
    split_ixs = load_partition(ixs_dir)

    # Assert uniqueness of indexes in dataset and partitions
    assert data.index.is_unique
    for _, v in split_ixs.items():
        assert len(np.unique(v)) == len(v)

    if not pd.api.types.is_float_dtype(data[Key.RATING].dtype):
        data[Key.RATING] = pd.to_numeric(data[Key.RATING], downcast="float")

    # Ensure monotonic index by steps of 1
    monotonic_ix = (data.sort_index().index.values == np.arange(stop=data.shape[0], dtype=int)).all()
    if not monotonic_ix:
        data.reset_index(inplace=True)
        mapping = pd.Series(data.index.values, index=data["index"])
        for c, v in split_ixs.items():
            split_ixs[c] = mapping.loc[split_ixs[c]].tolist()
        data.drop("index", inplace=True, axis=1)

    # NOTE: P5 does subword tokenization of user and item encoded IDs.
    mapper_path = os.path.join(path, "mappers")
    mappers = {}
    for c in (Key.USER, Key.ITEM, Key.EXP, Key.FEAT, Key.REV):
        if os.path.isfile(os.path.join(mapper_path, f"{c}{suffix}.parquet")):
            mapper = load_mapper_parquet(os.path.join(mapper_path, f"{c}{suffix}.parquet"))
        else:
            data[c], mapper = get_transform_mapping(data[c])
            save_safe_parquet(mapper.reset_index(), os.path.join(mapper_path, f"{c}{suffix}.parquet"))

        mappers[c] = Mapper(mapper)

    mappers[Key.ITEM].add_items(SPECIAL_TOKENS.values())  # mappers[Key.ITEM].get_idx(UNK_TOK)
    mappers[Key.EXP].add_items([""])

    return data, split_ixs, mappers


def flatten_cols(data, filter_unique=False, preserve_order=False):
    data[Key.FEAT] = data[Key.FEAT].apply(partial(sum, start=[]))

    if preserve_order:
        # NOTE 1: ~x3 slower but preserves temporal order of sentences (unique starts from most recent sentences)
        # NOTE 2: pandas .unique() states "Uniques are returned in order of appearance"
        unique_fn = lambda l: pd.Series(l[::-1]).unique().tolist()[::-1]
    else:
        unique_fn = lambda l: list(set(l))

    if filter_unique:
        for c in [Key.FEAT, Key.EXP]:
            data[c] = data[c].apply(unique_fn)

    for c in [col for col in data.columns if Prefix.HIST in col]:
        mask = data[c].str.len() > 0
        if Key.FEAT in c:
            data.loc[mask, c] = data.loc[mask, c].parallel_apply(lambda ls: sum(map(partial(sum, start=[]), ls), []))
        elif Key.EXP in c:
            data.loc[mask, c] = data.loc[mask, c].apply(partial(sum, start=[]))
        if filter_unique:
            if preserve_order:
                data.loc[mask, c] = data.loc[mask, c].parallel_apply(unique_fn)
            else:
                data.loc[mask, c] = data.loc[mask, c].apply(unique_fn)


def get_hist_data(data, hist_data_path, cols, hist_len, test):
    def sliding_w_hist(g: pd.DataFrame, max_l: int = MAX_HIST_LEN):
        return [g.index.values[max(i - max_l, 0) : i].tolist() for i in range(g.shape[0])]

    def getter(ixs, select):
        return data[select].values[ixs].tolist()

    if os.path.isfile(hist_data_path) and not test:
        data = pd.concat([data, pd.read_pickle(hist_data_path)], axis=1)
    else:
        # tqdm.pandas()
        orig_index = data.index.values
        hist_cols = []
        for groupby in [Key.USER, Key.ITEM]:
            logging.info(f"Extracting history for {groupby} column")
            hist_cols.append(f"{groupby}_{Prefix.HIST[:-1]}")
            data.sort_values([groupby, Key.TIMESTEP], inplace=True)
            histories = data[[groupby]].groupby(groupby)[groupby].apply(sliding_w_hist)
            data[hist_cols[-1]] = sum(histories.tolist(), [])
        data = data.loc[orig_index]
        # data[hist_cols].to_pickle(hist_data_path)

    data.sort_index(inplace=True)
    for groupby, select_cols in cols.items():
        max_l = hist_len[groupby]
        idxr_col = f"{groupby}_{Prefix.HIST[:-1]}"
        has_hist_mask = data[idxr_col].str.len() > 0
        idxr = data.loc[has_hist_mask, idxr_col]
        if max_l < MAX_HIST_LEN:
            idxr = idxr.str[-max_l:]
        for c in select_cols:
            new_c = f"{idxr_col}_{c}"
            data[new_c] = np.empty((len(data), 0)).tolist()
            data.loc[has_hist_mask, new_c] = idxr.map(partial(getter, select=c))

        data.drop(idxr_col, axis=1, inplace=True)
    return data


def complete_cfg(
    cfg: SimpleNamespace,
    dataset: str = "beauty",
    fold: str = "0",
    seq_mode: SeqMode = SeqMode.NO_SEQ,
    test_flag: bool = True,
    batch_size: int = 128,
    model_tasks: List[Task] = [],
    seed: int = RNG_SEED,
):
    if hasattr(cfg, "stages"):
        mod_cfg = cfg.stages[0]
    else:
        mod_cfg = cfg
    mod_cfg.data.requires_exp = Task.EXPLANATION in model_tasks or getattr(mod_cfg.data, "requires_exp", False)
    mod_cfg.data.requires_feat = Task.FEAT in model_tasks or getattr(mod_cfg.data, "requires_feat", False)
    mod_cfg.data.requires_rating = Task.RATING in model_tasks or getattr(mod_cfg.data, "requires_rating", False)
    mod_cfg.data.requires_context = Task.CONTEXT in model_tasks or getattr(mod_cfg.data, "requires_context", False)
    mod_cfg.data.requires_nextitem = Task.NEXT_ITEM in model_tasks or getattr(mod_cfg.data, "requires_nextitem", False)
    mod_cfg.data.txt_len = TXT_LEN.get(dataset, 35)
    mod_cfg.data.seq_mode = seq_mode
    mod_cfg.data.fold = fold
    mod_cfg.data.dataset = dataset
    mod_cfg.data.test_flag = test_flag
    cfg.data.batch_size = batch_size
    mod_cfg.data.txt_len = TXT_LEN.get(dataset, 35)
    mod_cfg.model.txt_len = TXT_LEN.get(dataset, 35)
    mod_cfg.seed = seed
    mod_cfg.gen_flag = False
    mod_cfg.save_results = False
    mod_cfg.data.eval_workers = 0


def get_uneg_samples(dataset, fold, base_path, n_neg=99, test=False):
    def neg_sample(item_seq: set, replace: bool = False):
        # neg_items = self.item_set.difference(self.user_item.loc[user])
        # assert item_seq.issubset(self.item_set)
        neg_items = list(item_set.difference(item_seq))
        if n_neg >= len(neg_items):
            random.shuffle(neg_items)
        else:
            neg_items = np.random.choice(neg_items, n_neg, replace=replace).tolist()
        return neg_items

    suffix = Suffix.TEST if test else ""
    logging.info(f"Sampling User Negative items for the {dataset}{suffix} dataset...")
    init_seed(RNG_SEED)

    logging.info("Reading the data, split and mappers...")
    data_path = get_paths(dataset, fold + suffix, base_path).data
    neg_sample_path = os.path.join(data_path, f"user_neg_samples{suffix}.parquet")
    if not os.path.exists(neg_sample_path):
        data, split_ixs, mappers = read_data(data_path, fold, test=test)

        keep_cols = [Key.USER, Key.ITEM, Key.FEAT, Key.EXP, Key.TIMESTEP]
        data = data[keep_cols]

        item_set = set(mappers[Key.ITEM].item2idx.tolist())

        logging.info("Sampling user-negatives...")
        u2i = data[[Key.USER, Key.ITEM]].groupby(Key.USER)[Key.ITEM].agg(set)
        user_neg_samples = {uid: neg_sample(iset) for uid, iset in u2i.items()}

        if test:
            user_neg_samples = {uid: negs + [negs[-1]] * (n_neg - len(negs)) for uid, negs in user_neg_samples.items()}

        # Make sure all users have n_neg items
        uneg_n = list(map(len, user_neg_samples.values()))

        assert (min(uneg_n) == n_neg) and (max(uneg_n) == n_neg)

        logging.info("Saving negative samples...")
        save_safe_parquet(pd.Series(user_neg_samples).rename_axis(Key.USER).reset_index(name=Key.ITEM), neg_sample_path)
    else:
        logging.warning(f"Negative sample file already exists! To re-create it, remove it first ({neg_sample_path})")


def process_metadata(dataset: str, base_path: str, fold: str = "0", test: bool = False):
    suffix = Suffix.TEST if test else ""
    paths = get_paths(dataset, fold + suffix, base_path)
    metadata_path = os.path.join(paths.data, f"metadata{suffix}.parquet")

    if os.path.exists(metadata_path):
        logging.warning(f"Metadata file already exists! To re-create it, remove it first ({metadata_path})")
        return

    init_seed(RNG_SEED)
    if DATA_MAP.get(dataset, dataset).lower() in set([d.lower() for d in AMAZON_DOMAINS]):
        logging.info(f"Processing metadata for the {dataset}{suffix} dataset...")

        meta_path = os.path.join(paths.data_orig, "meta.json.gz")
        metadata = pd.read_json(meta_path, orient="records", lines=True)

        logging.info("Reading the data, split and mappers...")
        _, _, mappers = read_data(paths.data, fold, test=test)

        logging.info("Filtering items that do not appear in any interaction...")
        metadata.rename(columns={"asin": Key.ITEM}, inplace=True)
        metadata = metadata[metadata[Key.ITEM].isin(mappers[Key.ITEM].item2idx.index)]

        logging.info("Mapping item IDs to encoded values...")
        metadata[Key.ITEM] = metadata[Key.ITEM].map(mappers[Key.ITEM].item2idx)

        logging.info("Saving metadata...")
        save_safe_parquet(metadata, metadata_path)
    elif dataset.lower() == "yelp":
        logging.info(f"Processing metadata for the {dataset}{suffix} dataset...")

        meta_path = os.path.join(paths.data_orig, "yelp_academic_dataset_business.json")
        metadata = pd.read_json(meta_path, orient="records", lines=True)

        logging.info("Reading the data, split and mappers...")
        _, _, mappers = read_data(paths.data, fold, test=test)

        logging.info("Filtering items that do not appear in any interaction...")
        metadata.rename(columns={"business_id": Key.ITEM}, inplace=True)
        metadata = metadata[metadata[Key.ITEM].isin(mappers[Key.ITEM].item2idx.index)].reset_index(drop=True)

        logging.info("Mapping item IDs to encoded values...")
        metadata[Key.ITEM] = metadata[Key.ITEM].map(mappers[Key.ITEM].item2idx)

        logging.info("Saving metadata...")
        save_safe_parquet(metadata, metadata_path)

    logging.info("Finished!")


def get_aux_data(model_name, **kwargs):
    dataset = kwargs["dataset"]
    fold = kwargs["fold"]
    base_path = kwargs["base_path"]
    test = kwargs.get("test", False)
    if "sequer" in model_name.lower():
        get_uneg_samples(dataset, fold, base_path, n_neg=kwargs.get("n_neg", 99), test=test)

    logging.info("Finished!")


def run(base_path, mode="preprocess_split", **kwargs):
    if mode == "preprocess_split":
        # kwargs include "datasets": list (to specify which datasets need to be processed), and "test": bool (If
        # the dataset to preprocess is a subsample, denoted by the filename reviews_mini.parquet). Processed datasets
        # are saved to parquet format.
        process_xrec_datasets(base_path=base_path, **kwargs)
    elif mode == "process_metadata":
        process_metadata(base_path=base_path, **kwargs)
    elif mode.startswith("get_aux_data_"):
        # kwargs are model-specific
        model_name = "_".join(mode.split("_")[3:])
        get_aux_data(model_name=model_name, base_path=base_path, **kwargs)
    else:
        raise ValueError(f"Invalid value for process mode ({mode})")
