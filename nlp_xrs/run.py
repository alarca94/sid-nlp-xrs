import os

import torch

from nlp_xrs.training.trainer import BaseTrainer, MultiStageTrainer
from nlp_xrs.utils.constants import DATA_MODE
from nlp_xrs.utils.funcs import init_seed, now_time
from nlp_xrs.utils.in_out import (
    Colors,
    colored,
    handle_dirs,
    load_config,
    print_namespace,
    update_logger,
)


def run_experiment(args):
    args.paths = handle_dirs(args)

    logger = update_logger(args)

    cfg_parts = args.config_id.split("/")
    cfg_name = cfg_parts[-1]
    cfg_prefix = "/".join(cfg_parts[:-1])
    cfg = load_config(f"{cfg_name}.yaml", args.paths.config, args, prefix_path=cfg_prefix)

    if not args.resume_training:
        # Set the random seed manually for reproducibility.
        init_seed(args.seed, reproducibility=True)
    else:
        logger.info(colored("-" * 40 + " RESUME TRAINING " + "-" * 40, Colors.BLUE))
        logger.warning(
            "--resume-training does not reset the RNG state to one of the last checkpoint, thus, results"
            " will not be reproducible"
        )

    device = torch.device("cpu")
    if torch.cuda.is_available():
        if not args.cuda:
            logger.warning(now_time() + "You have a CUDA device, so you should probably run with --cuda")
        else:
            device = torch.device("cuda")

    # NOTE: the pred_file name does not contain the RNG_SEED as we only need a single output for qualitative analysis
    if hasattr(cfg, "stages"):
        cfg_model_name = (cfg.stages[0].model.name + cfg.stages[0].model.suffix).lower()
    else:
        cfg_model_name = (cfg.model.name + cfg.model.suffix).lower()
    pred_file = f"{args.dataset}_{args.fold}_{cfg_model_name}_{args.seed}_{DATA_MODE}.txt"
    prediction_path = os.path.join(args.paths.log, "pred", pred_file)

    logger.info("-" * 40 + " ARGUMENTS " + "-" * 40)
    print_namespace(cfg, skipkeys=["paths"], logger=logger)
    logger.info("-" * 40 + " ARGUMENTS " + "-" * 40)

    if hasattr(cfg, "stages"):
        cfg.start_stage = args.stage
        trainer_class = MultiStageTrainer
    else:
        trainer_class = BaseTrainer

    trainer = trainer_class(cfg, device, prediction_path, args.eval_only, args.resume_training)

    if not os.path.exists(trainer.ckpt_path) and (args.eval_only or args.resume_training):
        logger.warning(f"Pretrained model checkpoint not found with path: {trainer.ckpt_path}. Training from scratch")
        args.eval_only = False
        args.resume_training = False
    elif os.path.exists(trainer.ckpt_path) and os.path.exists(trainer.ckpt_syml_path):
        if args.overwrite:
            args.eval_only = False
            args.resume_training = False
            logger.warning(f"Overwriting existing fully trained model at {trainer.ckpt_path}. Training from scratch...")
        else:
            args.eval_only = True
            logger.warning(
                f"Existing fully trained model found at {trainer.ckpt_path}. Set --overwrite to True to train from scratch. Evaluating..."
            )

    if not args.eval_only:
        trainer.train(resume=args.resume_training)
    if not args.train_only:
        trainer.test()
