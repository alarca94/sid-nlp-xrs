import os
from enum import Enum, auto
from typing import Optional

DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
PACKAGE_NAME = "nlp_xrs"
USER_PATH = os.path.expanduser("~")
PROJECT_FOLDER = "nlp-xrs-cikm24"

AMAZON_DOMAINS = (
    "Amazon_Instant_Video",
    "Apps_for_Android",
    "Automotive",
    "Baby",
    "Beauty",
    "Books",
    "CDs_and_Vinyl",
    "Cell_Phones_and_Accessories",
    "Clothing_Shoes_and_Jewelry",
    "Digital_Music",
    "Electronics",
    "Health_and_Personal_Care",
    "Grocery_and_Gourmet_Food",
    "Home_and_Kitchen",
    "Kindle_Store",
    "Movies_and_TV",
    "Musical_Instruments",
    "Office_Products",
    "Patio_Lawn_and_Garden",
    "Pet_Supplies",
    "Sports_and_Outdoors",
    "Tools_and_Home_Improvement",
    "Toys_and_Games",
    "Video_Games",
)
DATA_MODE = "feat_only"
DATA_MAP = {
    "yelp": "Yelp",
    "beauty": "Beauty",
    "digital_music": "Digital_Music",
    "cell": "Cell_Phones_and_Accessories",
    "toys": "Toys_and_Games",
    "sports": "Sports_and_Outdoors",
    "cd": "CDs_and_Vinyl",
}
HYPEROPT_SUBDIR = "hyper"

CKPT_EXT = ".ckpt"

UNK_TOK = "<unk>"
PAD_TOK = "<pad>"
BOS_TOK = "<bos>"
EOS_TOK = "<eos>"
SEP_TOK = "<sep>"
MASK_TOK = "<mask>"
CLS_TOKEN = "<cls>"
SPECIAL_TOKENS = {
    "pad_token": PAD_TOK,
    "bos_token": BOS_TOK,
    "eos_token": EOS_TOK,
    "unk_token": UNK_TOK,
    "sep_token": SEP_TOK,
    "mask_token": MASK_TOK,
    "cls_token": CLS_TOKEN,
}


class Prefix:
    ITEM = "item_"
    USER = "user_"
    HIST = "hist_"
    EMB = "emb_"
    GEN = "gen_"


class Suffix:
    TEST = "_mini"


class Key:
    FEAT = "feature"
    ADJ = "adj"
    ASP = "aspect"
    SCO = "sco"
    REV = "review"
    CONCEPT_ID = "concept_id"
    CONCEPT_ACT = "concept_act"
    EXP = "text"
    USER = "user"
    ITEM = "item"
    ITEM_RAW = "item_raw"
    SEQ_LEN = "seq_length"
    RATING = "rating"
    TIMESTAMP = "timestamp"
    TIMESTEP = "timestep"
    SEGMENT_I = "item_segment_ids"
    SEGMENT_EXP = "text_segment_ids"
    REV_LEN = "text_length"
    CONTEXT = "context"
    MASK = "mask"
    POS_OPT = "pos_answer"
    NEG_OPT = "neg_answer"
    NEG_EVAL = "neg_samples"


def HistKey(key: str, group_key: Optional[str] = None) -> str:
    if group_key is None:
        return f"{Prefix.HIST}{key}"
    return f"{group_key}_{Prefix.HIST}{key}"


BASE_COLS = [Key.ITEM, Key.USER, Key.TIMESTEP, Key.EXP, Key.FEAT, Key.RATING]

RNG_SEED = 1111
ALL_SEEDS = [1111, 24, 53, 126, 675]

HIST_I_MODE = 1
HIST_REV_MODE = 3

DEFAULT_HIST_LEN = 10
MAX_HIST_LEN = 400

# NOTE: To get the text length per dataset, run check_text_length() from eda.py
TXT_LEN = {"beauty": 35, "cell": 35, "yelp": 35}
TOP_KS = [1, 5, 10, 20]
VOCAB_SIZE = 5000

ENC_DEC_MODELS = ["t5"]

METRICS = [
    "RMSE",
    "MAE",
    # "HR@5",
    # "MRR@5",
    # "NDCG@5",
    # "HR@10",
    # "MRR@10",
    # "NDCG@10",
    "HR@20",
    "MRR@20",
    "NDCG@20",
    "DIV",
    "FCR",
    "FP",
    "FR",
    "FHR",
    "D-FHR",
    "USR",
    "USN",
    "avg_len",
    "w_idf",
    "rep/l",
    "seq_rep_2",
    "BLEU-1",
    "BLEU-2",
    "BLEU-4",
    "rouge_1/f_score",
    "rouge_1/p_score",
    "rouge_1/r_score",
    "rouge_2/f_score",
    "rouge_2/p_score",
    "rouge_2/r_score",
    "rouge_l/f_score",
    "rouge_l/p_score",
    "rouge_l/r_score",
]
NEXTITEM_METRICS = (
    ["S-MRR"]
    + [f"{m}@{k}" for k in TOP_KS for m in ["S-HR", "S-NDCG"]]
    + [f"{m}@{k}" for k in TOP_KS for m in ["HR", "MRR", "NDCG"]]
)
DIRECT_RECSYS_METRICS = ["RMSE", "MAE"]
TEXT_Q_METRICS = [
    "USR",
    "USN",
    "avg_len",
    "w_idf",
    "rep/l",
    "seq_rep_2",
    "BLEU-1",
    "BLEU-2",
    "BLEU-4",
    "ROUGE-1",
    "ROUGE-2",
    "ROUGE-L",
]
EXP_METRICS = ["DIV", "FCR", "FP", "FR", "FHR", "D-FHR"]

REDUNDANT_METRICS = []  # ["USN", "seq_rep_2", "BLEU-2", "BLEU-1", "ROUGE-2", "avg_len"]

METRIC_TYPES = {
    "explainability": EXP_METRICS,
    "text_quality": TEXT_Q_METRICS,
    "next_item": NEXTITEM_METRICS,
    "direct_recsys": DIRECT_RECSYS_METRICS,
}

MIN_METRICS = [
    "RMSE",
    "MAE",
    "DIV",
    "FHR",
    "D-FHR",
    "rep/l",
    "seq_rep_2",
]

IGNORE_COLORING = ["# runs", "seeds", "avg_len"]


class SeqMode(Enum):
    NO_SEQ = auto()
    # Get Item interaction history
    HIST_ITEM = auto()
    # Get Item interaction history but rating prediction only occurs for the cand.item
    HIST_ITEM_RATING_LAST = auto()
    # Get Item interaction history + User Review History
    HIST_ITEM_U_EXP = auto()
    # Get Item interaction history + User Review History (in the form of sentence embeddings)
    HIST_ITEM_U_EXP_EMB = auto()
    # Get Item interaction history (in the form of corresponding review sentence embeddings)
    HIST_REV2ITEM_EMB = auto()
    # Get User Review History + Item Review History
    HIST_UI_EXP = auto()
    # Other custom sequence modes
    CUSTOM = auto()

    # # Get User Review History + Item Review History (in the form of sentence embeddings)
    # HIST_ITEM_UI_EXP_EMB = 7

    def requires_past_info(self) -> bool:
        return self.value >= self.HIST_ITEM.value

    def requires_item_seq(self) -> bool:
        return self.requires_past_info() and self.HIST_UI_EXP.value <= self.value < self.CUSTOM.value

    def requires_past_user_exp(self) -> bool:
        return self.HIST_ITEM_U_EXP.value <= self.value < self.CUSTOM.value

    def requires_past_ui_exp(self) -> bool:
        return self.HIST_UI_EXP.value <= self.value < self.CUSTOM.value


class Task(Enum):
    RATING = "rating"
    CONTEXT = "context"
    EXTRACTION = "extraction"
    EXPLANATION = "explanation"
    # GENERATION = 'generation'
    NEXT_ITEM = "next_item"
    NEXT_ITEM_SAMPLE = "next_item_sample"
    FEAT = "feature"
    TOPN = "top-n"
    ASPECT = "aspect"
    NSP = "next_sentence_prediction"
    L2 = "l2"
    MLM = "masked_language_modelling"

    def __str__(self):
        return self.value

    # def __get__(self, instance, owner):
    #     return self.value


class InputType(Enum):
    REGULAR = auto()
    SEQUENTIAL = auto()
    TEMPLATE = auto()
    CUSTOM = auto()


class ModelType(Enum):
    GENERATIVE = auto()
    EXTRACTIVE = auto()
    HYBRID = auto()
    RECOMMENDER = auto()


CFG_ENTRY_TYPES = ("data", "model", "train")

# MODEL SPECIFIC CONSTANTS:
