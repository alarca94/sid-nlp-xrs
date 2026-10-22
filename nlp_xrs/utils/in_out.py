import importlib.util
import json
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml

from . import funcs
from .constants import DATA_MAP, DATA_MODE, PACKAGE_NAME, PROJECT_FOLDER, TXT_LEN, USER_PATH, Suffix

DICT_COLS_META_KEY = "dict_columns"
SET_COLS_META_KEY = "set_columns"


def find_config_hash(config_hash: str, sid_tok_path: str) -> Dict:
    sid_tok_path = Path(sid_tok_path)
    for slug_dir in sid_tok_path.iterdir():
        if not slug_dir.is_dir():
            continue
        for emb_dir in slug_dir.iterdir():
            if not emb_dir.is_dir():
                continue
            for run_dir in emb_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                if run_dir.name == config_hash:
                    for seed_dir in run_dir.iterdir():
                        if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                            continue
                        config_path = seed_dir / "run_info.json"
                        if not config_path.exists():
                            continue
                        with open(config_path, "r") as f:
                            return json.load(f)["config"]
                    raise FileNotFoundError(f"Config hash found at {run_dir} but no run_info.json exists!")
    raise FileNotFoundError(f"Config hash {config_hash} not found!")


def load_auxiliary_config(
    config_path: Union[Path, str],
    file_prefix: Optional[str] = None,
    file_extension: Optional[str] = None,
    config_hash: Optional[str] = None,
    strict_matching: bool = True,
    sid_tok_path: Optional[str] = None,
) -> SimpleNamespace:
    config_path = Path(config_path)
    # Config file can come from multiple machines, so we need to join configs
    if file_prefix is None and file_extension is None:
        file_paths = [config_path]
    else:
        assert file_prefix is not None and file_extension is not None
        if strict_matching:
            file_paths = [config_path / f"{file_prefix}{file_extension}"]
        else:
            file_paths = list(config_path.glob(f"{file_prefix}*{file_extension}"))

    config_dict = {}
    for file_path in file_paths:
        if config_hash is not None:
            if not file_path.exists():
                assert sid_tok_path is not None
                config_dict = find_config_hash(config_hash, sid_tok_path)
                break
            with open(file_path, "r") as f:
                config_dict.update(json.load(f)[config_hash]["config"])
        else:
            """Load configuration from YAML file."""
            with open(file_path, "r") as f:
                config_dict.update(yaml.safe_load(f))

    return funcs.dict2namespace(config_dict)


def load_config(config_file, config_path, args=None, prefix_path="", as_dict=False, ho_space=False):
    logger = logging.getLogger(PACKAGE_NAME)

    # Load and update model specific config
    with open(os.path.join(config_path, prefix_path, config_file), "r") as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)

    model_config = {} if model_config is None else model_config

    stages_cfg = {}
    if "stages" in model_config:
        stages_cfg = model_config.copy()
        model_config = stages_cfg["stages"][0]

    # Load default config files
    config = model_config.copy()
    if not ho_space:
        for cfg_f in os.listdir(os.path.join(config_path, "default")):
            if os.path.isfile(os.path.join(config_path, "default", cfg_f)) and cfg_f.endswith(".yaml"):
                key = cfg_f.split(".")[0]
                with open(os.path.join(config_path, "default", cfg_f), "r") as f:
                    def_cfg = yaml.load(f, Loader=yaml.FullLoader)
                def_cfg = {} if def_cfg is None else def_cfg
                config[key] = funcs.update_dict(def_cfg, model_config.get(key, {}))

        for k in model_config:
            if k not in config:
                config[k] = model_config[k]

        if args:
            config["data"]["dataset"] = args.dataset
            config["data"]["fold"] = args.fold
            config["data"]["test_flag"] = args.test
            txt_len = TXT_LEN.get(args.dataset, 35)
            if getattr(args, "txt_len", None) is not None:
                txt_len = args.txt_len
            config["data"]["txt_len"] = txt_len
            config["model"]["txt_len"] = txt_len
            config["gen_flag"] = not args.no_generate
            config["save_results"] = args.save_results
            config["seed"] = args.seed
            config["device"] = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
            config["paths"] = vars(args.paths)

            cfg_model_name = (config["model"]["name"] + config["model"]["suffix"]).lower()
            if cfg_model_name != config_file.split(".")[0].lower():
                logger.warning(
                    f"Config. file name ({config_file.split('.')[0].lower()}) does not match model name+suffix "
                    f"specified inside ({cfg_model_name})"
                )

            if args.test:
                config["train"]["epochs"] = 1
                config["gen_flag"] = False
                config["save_results"] = False

        if stages_cfg:
            config = {"stages": [config] + stages_cfg["stages"][1:]}

        if config["model"].get("use_pretrained_lm", False) and "pretrained_model_name" in config["model"]:
            lm_model = config["model"]["pretrained_model_name"]
            config["data"]["word_tok"] = lm_model
            config["model"]["suffix"] = (
                "-" + config["model"].get("pretrained_alias", lm_model.replace("/", "--")) + config["model"]["suffix"]
            )

    if as_dict:
        return config

    return funcs.dict2namespace(config)


def load_partition(ixs_dir: str) -> Dict[str, List[int]]:
    assert os.path.exists(ixs_dir)
    train_ixs: List[int] = sorted(np.load(os.path.join(ixs_dir, "train.npy")).tolist())
    valid_ixs: List[int] = sorted(np.load(os.path.join(ixs_dir, "validation.npy")).tolist())
    test_ixs: List[int] = sorted(np.load(os.path.join(ixs_dir, "test.npy")).tolist())
    return {"train": train_ixs, "valid": valid_ixs, "test": test_ixs}


def print_exp_analysis(analysis):
    def dict2str(results, color=None):
        # return f"{{{', '.join([f'{colored(k, color)}: {v:.2f}'for k, v in results])}}}"
        out_str = f"{{{', '.join([f'{k}: {v}' for k, v in results.items()])}}}"
        if color is not None:
            out_str = colored(out_str, color)
        return out_str

    config_key = "config"
    results_key = "val"
    for tid, trial in analysis.results.items():
        if results_key in trial:
            if tid == analysis.best_trial.trial_id:
                print(
                    f"{colored('Results:', Colors.BLUE)} {dict2str(trial[results_key], Colors.GREEN)}, "
                    f"{colored('Config:', Colors.BLUE)} {dict2str(trial[config_key], Colors.GREEN)}"
                )
            else:
                print(
                    f"{colored('Results:', Colors.BLUE)} {dict2str(trial[results_key])}, "
                    f"{colored('Config:', Colors.BLUE)} {dict2str(trial[config_key])}"
                )
        else:
            print(f"Trial {tid} does not contain {results_key} argument")


def read_results(results_path, data_mode: str = DATA_MODE):
    file_paths = list(Path(results_path).glob(f"{data_mode}_results*.csv"))
    all_res = []
    for file_path in file_paths:
        if not os.path.isfile(file_path):
            continue
        res = pd.read_csv(file_path)
        name_components = file_path.stem.split("_")
        source = name_components[-1]
        if source == "results":
            source = None
        res["source"] = source
        all_res.append(res)
    return pd.concat(all_res, axis=0, ignore_index=True)


def save_results(curr_res, results_path, data_mode: str = DATA_MODE):
    machine_suffix = f"_{os.environ.get('MACHINE_ID', '')}" if "MACHINE_ID" in os.environ else ""
    filename = f"{data_mode}_results{machine_suffix}.csv"
    is_new_file = not os.path.isfile(os.path.join(results_path, filename))
    if not is_new_file:
        results = pd.read_csv(os.path.join(results_path, filename))
    else:
        results = pd.DataFrame(columns=curr_res.keys())
    missing_cols = set(curr_res.keys()).difference(results.columns)
    for c in missing_cols:
        results[c] = np.nan
    missing_cols = set(results.columns).difference(curr_res.keys())
    for c in missing_cols:
        curr_res[c] = np.nan
    # results = results.append(pd.DataFrame().from_records([curr_res]))
    results = pd.concat((results, pd.DataFrame().from_records([curr_res])))
    results.to_csv(os.path.join(results_path, filename), index=False)


def plot_mask(mask, save_to_file: bool = True, save_path: str = "images/mask.png"):
    plt.imshow(mask, cmap="Greys", interpolation="nearest")
    plt.xticks(np.arange(0, mask.shape[1], 1))
    plt.yticks(np.arange(0, mask.shape[0], 1))
    plt.grid()
    if save_to_file:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()


class Colors:
    BLACK = "\033[1;30m"
    RED = "\033[1;31m"
    GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
    BLUE = "\033[1;34m"
    PURPLE = "\033[1;35m"
    CYAN = "\033[1;36m"
    WHITE = "\033[1;37m"
    LIGHT_GRAY = "\033[0;37m"
    ENDC = "\033[0m"


def colored(text, color):
    return color + text + Colors.ENDC


def print_colored_dict(d):
    if isinstance(d, list):
        for subd in d:
            print_colored_dict(subd)
    else:
        print()
        for k, v in d.items():
            print(f"{k}: {v}")


def print_losses(loss, device, suffix=None, color=None, logger=None):
    losses = loss.copy()
    # for task in [Task.CONTEXT, Task.EXPLANATION]:
    #     if task in losses:
    #         losses[task] = math.exp(losses[task])

    out_str = " | ".join([f"{task} {v:4.4f}" for task, v in losses.items()])
    if suffix is not None:
        out_str += f" | {suffix}"
    if device.type == "cuda":
        out_str += f" | {funcs.get_gpu_usage(device)}"
    if color is not None:
        out_str = colored(out_str, color)

    if logger is None:
        logger = logging.getLogger(PACKAGE_NAME)
    logger.info(out_str)


def print_namespace(ns, skipkeys=None, logger=None):
    d = funcs.namespace2dict(ns, skipkeys)
    if logger is None:
        logger = logging.getLogger(PACKAGE_NAME)
    logger.info(json.dumps(d, indent=4))


def update_logger(args):
    if args.log_to_file:
        os.makedirs(args.paths.raw_log, exist_ok=True)
        mode = "a" if args.eval_only or args.resume_training else "w"
        log_f = f"{args.dataset}_{args.fold}_{args.config_id.replace('/', '--')}_{args.seed}_{DATA_MODE}.txt"
        handlers = [logging.FileHandler(filename=os.path.join(args.paths.raw_log, log_f), mode=mode)]
    else:
        handlers = [logging.StreamHandler(stream=sys.stdout)]

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s - %(message)s", handlers=handlers)
    return logging.getLogger(PACKAGE_NAME)


def get_paths(dataset: str, fold: str, base_path: Optional[str] = None, hyperopt=False):
    if base_path is None:
        base_path = f"{USER_PATH}/{PROJECT_FOLDER}/"
    elif base_path.startswith("~/"):
        base_path = f"{USER_PATH}/{base_path[2:]}"

    # Root paths
    paths = {
        "base": os.path.join(base_path),
        "data": os.path.join(base_path, "data", DATA_MAP.get(dataset.lower(), dataset), DATA_MODE),
        "data_orig": os.path.join(base_path, "data", DATA_MAP.get(dataset.lower(), dataset), "orig"),
        "results": os.path.join(base_path, "results"),
        "ckpt": os.path.join(base_path, "checkpoints", DATA_MAP.get(dataset.lower(), dataset), DATA_MODE),
        "log": os.path.join(base_path, "logs"),
        "config": os.path.join(base_path, "config"),
    }
    # Data-specific paths
    paths["mappers"] = os.path.join(paths["data"], "mappers")
    paths["embeddings"] = os.path.join(paths["data"], "embeddings")
    paths["we"] = os.path.join(paths["data"], fold, "word_embeddings")
    paths["tokenizers"] = os.path.join(paths["data"], fold, "tokenizers")
    paths["aux_data"] = os.path.join(paths["data"], fold, "aux")

    # Log-specific paths
    paths["raw_log"] = os.path.join(paths["log"], "raw")
    paths["tmp_log"] = os.path.join(paths["log"], "tmp")
    paths["pred_log"] = os.path.join(paths["log"], "pred")
    paths["tensorboard"] = os.path.join(paths["log"], "tensorboard")
    if hyperopt:
        paths["ray_log"] = os.path.join(paths["log"], "hyperopt")

    # Checkpoint-specific paths
    paths["best_ckpt"] = os.path.join(paths["ckpt"], "best")

    return SimpleNamespace(**paths)


def handle_dirs(args, hyperopt=False):
    test = getattr(args, "test", False)
    fold = getattr(args, "fold", "0")

    paths = get_paths(args.dataset, fold + (Suffix.TEST if test else ""), args.base_path, hyperopt)

    for k, path in vars(paths).items():
        os.makedirs(path, exist_ok=True)

    return paths


def module_from_file(file_path: str, module_name: Optional[str] = None, logger: Optional[logging.Logger] = None):
    if logger is None:
        logger = logging.getLogger(PACKAGE_NAME)

    if not file_path.endswith(".py"):
        logger.warning("File path to load module does not point to a python script (.py extension)")
        if module_name is None:
            module_name = os.path.split(file_path)[-1]

    if file_path.startswith("~/"):
        file_path = f"{USER_PATH}/{file_path[2:]}"

    if module_name is None:
        module_name = ".".join(os.path.split(file_path)[-1].split(".")[:-1])

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from file {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_safe_parquet(df: pd.DataFrame, path: str):
    df = df.copy()
    dict_cols = [c for c in df.columns if df[c].dropna().apply(lambda x: isinstance(x, dict)).any()]
    set_cols = [c for c in df.columns if df[c].dropna().apply(lambda x: isinstance(x, set)).any()]
    for col in dict_cols:
        df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, dict) else x)
    for col in set_cols:
        df[col] = df[col].apply(lambda x: list(x) if isinstance(x, set) else x)
    # Store dict column names in parquet metadata
    table = pa.Table.from_pandas(df)
    meta = table.schema.metadata or {}
    meta[DICT_COLS_META_KEY.encode()] = json.dumps(dict_cols).encode()
    meta[SET_COLS_META_KEY.encode()] = json.dumps(set_cols).encode()
    table = table.replace_schema_metadata(meta)
    pq.write_table(table, path)


def load_safe_parquet(path: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
    table = pq.read_table(path, columns=columns)
    meta = table.schema.metadata or {}
    dict_cols = json.loads(meta.get(DICT_COLS_META_KEY.encode(), b"[]"))
    set_cols = json.loads(meta.get(SET_COLS_META_KEY.encode(), b"[]"))

    list_col_names = [
        field.name for field in table.schema if pa.types.is_list(field.type) and field.name not in dict_cols
    ]
    new_cols = {}
    for col_name in list_col_names:
        py_col = table.column(col_name).to_pylist()  # C++ nested conversion
        if col_name in set_cols:
            py_col = [set(x) if x is not None else x for x in py_col]
        new_cols[col_name] = py_col

    df = table.to_pandas()

    for col_name, values in new_cols.items():
        df[col_name] = values

    for col in dict_cols:
        df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) else x)

    # Fix Arrow-backed scalar dtypes → numpy
    for col in df.columns:
        if hasattr(df[col].dtype, "pyarrow_dtype"):
            df[col] = df[col].astype(df[col].dtype.numpy_dtype)

    return df


def load_mapper_parquet(path: str) -> pd.Series:
    df = pd.read_parquet(path, dtype_backend="pyarrow")
    return df.set_index(df.columns[0])[df.columns[1]]


def get_acts_mapper(acts_path: str) -> pd.DataFrame:
    # Path example: data/Beauty/feat_only/0/SAE/acts_Qwen--Qwen3-Embedding-0.6B_small_1111.pt
    if not os.path.exists(acts_path):
        raise FileNotFoundError(f"Activations file not found: {acts_path}")
    acts_dict = torch.load(acts_path, weights_only=False, map_location="cpu")  # type: ignore
    acts_mapper = pd.DataFrame(
        {
            "review_id": acts_dict["review_ids"],
            "concept_id": acts_dict["concept_ids"].tolist(),
            "act": acts_dict["acts"].tolist(),
        }
    ).set_index("review_id")
    return acts_mapper
