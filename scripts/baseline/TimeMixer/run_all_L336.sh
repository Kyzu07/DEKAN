#!/usr/bin/env bash
# bash scripts/l_336/run_all_L336.sh [dataset ...]
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASETS=("$@"); [ ${#DATASETS[@]} -eq 0 ] && DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 Weather Electricity Traffic)
for ds in "${DATASETS[@]}"; do bash "$DIR/TimeMixer_${ds}_L336.sh"; done
