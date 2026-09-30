#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for script in ETTh2_L336.sh ETTm2_L336.sh Weather_L336.sh Traffic_L336.sh; do
  echo "========== Running ${script} =========="
  bash "${SCRIPT_DIR}/${script}"
done
