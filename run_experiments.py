import argparse
import logging
import os
import shlex
import subprocess
import sys
import time
from datetime import timedelta
from multiprocessing import Pool, Queue, current_process

from nlp_xrs.utils.in_out import Colors, colored

MAIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py")


def exp_gen(datasets, config_ids, seeds, eval_only=False, resume_training=False, base_path=""):
    command = f"{shlex.quote(sys.executable)} {shlex.quote(MAIN)} --config-id={{}} --dataset={{}} --fold=0 --cuda --seed={{}} --log-to-file"
    if eval_only:
        command += " --eval-only"
    if resume_training:
        command += " --resume-training"
    if base_path:
        command += f' --base-path="{base_path}"'
    for seed in seeds:
        for dataset in datasets:
            for config_id in config_ids:
                yield command.format(config_id, dataset, seed)


def run(command):
    gpu_id = gpu_queue.get()
    try:
        command = f"CUDA_VISIBLE_DEVICES={gpu_id} " + command
        ident = current_process().ident
        logging.info(f"Running {ident} on GPU {gpu_id} -- Command: {command.split()[3:]}")
        start = time.time()
        return_code = subprocess.call(command, shell=True)
        msg = (
            f"Finished {ident} in {timedelta(seconds=round(time.time() - start))} "
            f"with return_code {return_code}. Releasing GPU {gpu_id}"
        )
        logging.info(colored(msg, Colors.GREEN if return_code == 0 else Colors.RED))
        return command, return_code
    finally:
        gpu_queue.put(gpu_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run explainable recommendation experiments across GPUs.")
    parser.add_argument("--datasets", nargs="+", default=["beauty", "cell", "yelp"])
    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="Config ids without the .yaml suffix, e.g. nrt or ipm26/sid_xrs/nrt-sid_agg__opq_qwen_3cb",
    )
    parser.add_argument("--seeds", nargs="+", default=["1111", "24", "53", "91", "126"])
    parser.add_argument("--gpus", nargs="+", type=int, default=[0])
    parser.add_argument("--jobs-per-gpu", type=int, default=1)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--resume-training", action="store_true")
    parser.add_argument("--base-path", default="")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[logging.StreamHandler(sys.stdout)])

    commands = exp_gen(
        args.datasets,
        args.configs,
        args.seeds,
        eval_only=args.eval_only,
        resume_training=args.resume_training,
        base_path=args.base_path,
    )
    gpu_list = [gpu_id for gpu_id in args.gpus for _ in range(args.jobs_per_gpu)]
    gpu_queue = Queue()
    for gpu in gpu_list:
        gpu_queue.put(gpu)

    pool = Pool(processes=len(gpu_list))
    failed = [command for command, code in pool.imap_unordered(run, commands) if code != 0]
    pool.close()
    pool.join()
    if failed:
        logging.info(colored(f"{len(failed)} run(s) failed:\n" + "\n".join(failed), Colors.RED))
        sys.exit(1)
