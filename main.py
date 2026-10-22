import argparse
import os

from nlp_xrs.run import run_experiment
from nlp_xrs.utils.constants import RNG_SEED

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NLP-XRS (Text-based Explainable Recommendation)")
    parser.add_argument(
        "--config-id",
        type=str,
        default="peter",
        help="config id to load the config file -- path prefix without .yaml extension (peter, aux/sequer, etc)",
    )
    parser.add_argument("--dataset", type=str, default=None, help="dataset name (beauty, cell, yelp)")
    parser.add_argument(
        "--base-path",
        type=str,
        default=os.path.dirname(os.path.abspath(__file__)),
        help="path to data, logs, checkpoints, etc.",
    )
    parser.add_argument("--fold", type=str, default="0", help="data partition index")
    parser.add_argument("--cuda", action="store_true", help="use CUDA")
    parser.add_argument("--test", action="store_true", help="use a small fraction of the data")
    parser.add_argument(
        "--no-save-results",
        dest="save_results",
        default=True,
        action="store_false",
        help="whether results will be saved or not",
    )
    parser.add_argument(
        "--no-generate", action="store_true", help="whether generated text will be saved to log file or not"
    )
    parser.add_argument(
        "--train-only", action="store_true", help="Whether to train a model and skip evaluation on the test set"
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Whether to load an existing checkpoint and evaluate it (skipping training)",
    )
    parser.add_argument("--resume-training", action="store_true", help="Whether to resume training")
    parser.add_argument(
        "--stage", type=int, default=0, help="Stage from which to start training (load latest saved checkpoint)"
    )
    parser.add_argument(
        "--log-to-file", action="store_true", help="Whether to load an existing checkpoint and evaluate it"
    )
    parser.add_argument("--seed", type=int, default=RNG_SEED, help="seed for reproducibility")
    parser.add_argument(
        "--txt-len",
        type=int,
        default=None,
        help="Override the explanation length. Default: the dataset constant.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Whether to overwrite an existing fully trained model")
    args = parser.parse_args()
    if args.dataset is None:
        parser.error("--dataset should be provided for loading data")

    if args.resume_training and args.eval_only:
        parser.error("Set one of --resume-training or --eval-only to True.")

    run_experiment(args)
