import argparse
import logging
import os
import sys

from nlp_xrs.data_preparation import extract_exps, process_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NLP-XRS (Text-based Explainable Recommendation)")
    parser.add_argument(
        "--base-path",
        type=str,
        default=os.path.abspath(os.path.dirname(__file__)),
        help="path to data, logs, checkpoints, etc.",
    )
    parser.add_argument(
        "--action",
        type=str,
        default="preprocess_split",
        help='data preparation action to perform ("extract_exps", "preprocess_split", "process_metadata", or '
        '"get_aux_data_sequer"). Defaults to "preprocess_split"',
    )
    parser.add_argument(
        "--datasets",
        type=lambda x: x.split(","),
        default="beauty,cell,yelp",
        help="comma-separated list of datasets to process (beauty, cell, yelp)",
    )
    parser.add_argument("--test", action="store_true", help="use a small fraction of the data")
    args = parser.parse_args()

    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s - %(message)s", handlers=handlers)

    if args.action == "extract_exps":
        for dataset in args.datasets:
            # Extract explanations for the specified dataset (beauty, cell, yelp)
            extract_exps.run(dataset=dataset, base_path=args.base_path)
    elif args.action == "preprocess_split":
        # NOTE: Test dataset was obtained from the already processed dataset in a jupyter notebook.
        process_data.run(base_path=args.base_path, mode=args.action, datasets=args.datasets, test=args.test)
    elif args.action == "process_metadata":
        for dataset in args.datasets:
            process_data.run(base_path=args.base_path, mode=args.action, dataset=dataset, test=args.test)
    elif args.action.startswith("get_aux_data_"):
        for dataset in args.datasets:
            # Get auxiliary data for the specified dataset, split and model
            data_params = {"dataset": dataset, "fold": "0", "test": args.test}
            model_params = {
                "sequer": {"n_neg": 99},
            }.get(args.action[len("get_aux_data_") :], None)
            if model_params is None:
                raise ValueError(f"Invalid action in prepare_data argument: {args.action}")

            process_data.run(base_path=args.base_path, mode=args.action, **data_params, **model_params)
    else:
        raise ValueError(f"Invalid action in prepare_data argument: {args.action}")
