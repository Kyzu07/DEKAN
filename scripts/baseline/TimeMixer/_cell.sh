# sourced by TimeMixer_<dataset>_L336.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
PYTHON="${PYTHON:-python}"
mkdir -p logs/l336

run_cell() {
  local ds=$1 T=$2; shift 2
  local log="logs/l336/${ds}_${T}.log"
  if [ -f "$log" ] && grep -q '^mse:' "$log"; then
    echo "skip ${ds} T=${T} (already in ${log})"; return 0
  fi
  "$PYTHON" -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --model TimeMixer \
    --model_id "${ds}_336_${T}" \
    --features M \
    --seq_len 336 \
    --label_len 0 \
    --pred_len "$T" \
    --e_layers 2 \
    --down_sampling_layers 3 \
    --down_sampling_window 2 \
    --down_sampling_method avg \
    --decomp_method moving_avg \
    --moving_avg 25 \
    --channel_independence 1 \
    --use_norm 1 \
    --learning_rate 0.01 \
    --batch_size 128 \
    --loss MSE \
    --itr 1 \
    --des L336 \
    "$@" 2>&1 | tee "$log"
}
