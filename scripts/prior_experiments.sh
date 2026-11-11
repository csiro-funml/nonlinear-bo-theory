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
  --regularizer-kernel rbf
  --gp-lengthscale 0.05 --gp-nu 2.5 --gp-jitter 1e-6
  --ensemble-members 5 --ensemble-frequency 1 --snapshot-frequency 5
  --device cpu
)

for scale in 1 4 8 16; do
  "${experiment[@]}" --output "runs/pilot-gaussian-$scale" --plot "${common[@]}" \
    --num-runs 5 --rounds 100 --initialization lecun_normal \
    --initialization-scale "$scale"
done

"${experiment[@]}" --output runs/pilot-uniform --plot "${common[@]}" \
  --num-runs 5 --rounds 100 --initialization xavier_uniform \
  --initialization-scale 1

mkdir -p comparisons/paper-prior
"${compare[@]}" runs/pilot-gaussian-1 runs/pilot-gaussian-4 runs/pilot-gaussian-8 runs/pilot-gaussian-16 runs/pilot-uniform \
  --metric cumulative_regret --out comparisons/paper-prior
"${compare[@]}" runs/pilot-gaussian-1 runs/pilot-gaussian-4 runs/pilot-gaussian-8 runs/pilot-gaussian-16 runs/pilot-uniform \
  --metric gp_variance --out comparisons/paper-prior
"${compare[@]}" runs/pilot-gaussian-1 runs/pilot-gaussian-4 runs/pilot-gaussian-8 runs/pilot-gaussian-16 runs/pilot-uniform \
  --metric optimum_visits --out comparisons/paper-prior
