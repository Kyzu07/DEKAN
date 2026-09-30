#!/usr/bin/env bash
# authors' per-horizon configs (scripts/ETTh1.sh, ETTm1.sh, ECL.sh) with --seq_len 336
# usage: bash scripts/l_336/authors_L336.sh <etth1|ettm1|weather|ecl> [pred_len ...]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."
mkdir -p logs/l336
GROUP="$1"; shift

run() {  # run <ds_tag> <data> <root> <csv> <enc_in> <pred_len> <hidden> <SCI> <bs> <lr>
  python -u run.py --is_training 1 --root_path "$3" --data_path "$4" \
    --model_id "$1_336_$6" --model Amplifier --data "$2" --features M \
    --seq_len 336 --label_len 48 --pred_len "$6" --enc_in "$5" \
    --hidden_size "$7" --SCI "$8" --batch_size "$9" --learning_rate "${10}" \
    --des Exp --itr 1 2>&1 | tee "logs/l336/$1_336_$6_authors.log"
}

case $GROUP in
  etth1)
    for T in ${@:-96 192 336 720}; do
      case $T in
        96)  run ETTh1 ETTh1 ./dataset/ETT-small/ ETTh1.csv 7 96  64  0 256 0.02 ;;
        192) run ETTh1 ETTh1 ./dataset/ETT-small/ ETTh1.csv 7 192 512 0 256 0.02 ;;
        336) run ETTh1 ETTh1 ./dataset/ETT-small/ ETTh1.csv 7 336 512 0 256 0.03 ;;
        720) run ETTh1 ETTh1 ./dataset/ETT-small/ ETTh1.csv 7 720 512 0 256 0.03 ;;
      esac
    done ;;
  ettm1)
    for T in ${@:-336 720}; do
      case $T in
        720) run ETTm1 ETTm1 ./dataset/ETT-small/ ETTm1.csv 7 720 128 0 256 0.005 ;;
        *)   PRED_LENS="$T" bash "${SCRIPT_DIR}/ETTm1_L336.sh" ;;
      esac
    done ;;
  weather)
    PRED_LENS="${*:-96 192}" bash "${SCRIPT_DIR}/Weather_L336.sh" ;;
  ecl)
    for T in ${@:-96 192 336 720}; do
      case $T in
        96)  run Electricity custom ./dataset/electricity/ electricity.csv 321 96  512  1 16 0.005 ;;
        192) run Electricity custom ./dataset/electricity/ electricity.csv 321 192 512  1 16 0.002 ;;
        336) run Electricity custom ./dataset/electricity/ electricity.csv 321 336 1024 1 16 0.0005 ;;
        720) run Electricity custom ./dataset/electricity/ electricity.csv 321 720 1024 1 16 0.0005 ;;
      esac
    done ;;
  *) echo "unknown group $GROUP" >&2; exit 2 ;;
esac
