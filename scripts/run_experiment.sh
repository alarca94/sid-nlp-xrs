#!/usr/bin/env bash
# Run one step of the IPM-26 pipeline.
#
#   scripts/run_experiment.sh embed --dataset beauty --config opq_qwen_3cb [--mode item|user] [--seed 1111]
#   scripts/run_experiment.sh sae   --dataset beauty --config small|medium|large [--seed 1111]
#   scripts/run_experiment.sh sid   --dataset beauty --config rqfaiss_qwen_3cb [--mode item|user] [--seed 1111]
#   scripts/run_experiment.sh xrs   --dataset beauty --config nrt [--seed 1111]
#
# Config names are file stems under config/tokenizers/ (embed, sid), config/sae/ (sae),
# or config/ (xrs). Nested ids are allowed for xrs, e.g. ipm26/sid_xrs/peter-sid_agg__rqfaiss_qwen_3cb.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 {embed|sae|sid|xrs} --dataset NAME --config NAME [extra args...]" >&2
    exit 1
fi

step="$1"
shift

dataset=""
config=""
mode="item"
seed="1111"
extra=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) dataset="$2"; shift 2 ;;
        --config) config="$2"; shift 2 ;;
        --mode) mode="$2"; shift 2 ;;
        --seed) seed="$2"; shift 2 ;;
        *) extra+=("$1"); shift ;;
    esac
done

if [[ -z "$dataset" || -z "$config" ]]; then
    echo "Both --dataset and --config are required." >&2
    exit 1
fi

cd "$(dirname "$0")/.."

case "$step" in
    embed)
        python train_tokenizer.py --dataset "$dataset" --config "$config" --mode "$mode" --seed "$seed" --only-embed "${extra[@]}"
        ;;
    sae)
        python pretrain_sae.py --dataset "$dataset" --config "$config" --seed "$seed" "${extra[@]}"
        ;;
    sid)
        python train_tokenizer.py --dataset "$dataset" --config "$config" --mode "$mode" --seed "$seed" --log-to-file "${extra[@]}"
        ;;
    xrs)
        python main.py --config-id "$config" --dataset "$dataset" --fold 0 --cuda --seed "$seed" --log-to-file "${extra[@]}"
        ;;
    *)
        echo "Unknown step: $step" >&2
        exit 1
        ;;
esac
