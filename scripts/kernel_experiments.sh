#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON:-python}"
experiment=("$python_bin" ts_variance_decay_experiment.py)
compare=("$python_bin" compare.py)
common=(
  --seed 1987
  --teacher-seed 2466153720824162363
  --initial-seed 5947395936097286929
  --noise-seed 3067050576792127651
  --teacher-initialization-scale 8
  --input-dim 1 --grid-size 41 --lower -1 --upper 1
  --hidden 32 --layers 3 --teacher-noise 0.5 --initial-size 1
  --steps 250 --learning-rate 0.03 --optimizer adam
  --schedule log --schedule-q 2 --lambda-base 0.25
  --initialization lecun_normal --initialization-scale 8
  --regularizer-matern-nu 2.5 --regularizer-rq-alpha 1
  --gp-lengthscale 0.05 --gp-nu 2.5 --gp-jitter 1e-6
  --ensemble-members 5 --ensemble-frequency 1 --snapshot-frequency 5
  --device cpu
)

for kernel in matern rbf rq; do
  if [[ "$kernel" == "rq" ]]; then
    regularizer_kernel=rational_quadratic
  else
    regularizer_kernel="$kernel"
  fi
  "${experiment[@]}" --output "runs/pilot-$kernel" --plot "${common[@]}" \
    --num-runs 5 --rounds 100 --regularizer-kernel "$regularizer_kernel"
done

mkdir -p comparisons/paper-kernel
"${compare[@]}" runs/pilot-matern runs/pilot-rbf runs/pilot-rq \
  --metric cumulative_regret --out comparisons/paper-kernel
"${compare[@]}" runs/pilot-matern runs/pilot-rbf runs/pilot-rq \
  --metric gp_variance --out comparisons/paper-kernel
