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
  --schedule log --lambda-base 0.25 --theta-lengthscale-c 1
  --regularizer-kernel rbf
  --gp-lengthscale 0.05 --gp-nu 2.5 --gp-jitter 1e-6
  --ensemble-members 5 --ensemble-frequency 1 --snapshot-frequency 5
  --device cpu
)

for q in 1 2 3 4; do
  "${experiment[@]}" --output "runs/pilot-log-q-$q" --plot "${common[@]}" \
    --num-runs 10 --rounds 300 --initialization lecun_normal \
    --initialization-scale 8 --schedule-q "$q"
done

mkdir -p comparisons/paper-schedule
"${compare[@]}" runs/pilot-log-q-1 runs/pilot-log-q-2 runs/pilot-log-q-3 runs/pilot-log-q-4 \
  --metric cumulative_regret --out comparisons/paper-schedule
"${compare[@]}" runs/pilot-log-q-1 runs/pilot-log-q-2 runs/pilot-log-q-3 runs/pilot-log-q-4 \
  --metric gp_variance --out comparisons/paper-schedule
