#!/bin/bash
# One SLURM array task = one run of an experiment config.
# Submit with scripts/submit.sh, which sets the array size:
#
#   scripts/submit.sh configs/my_experiment.yaml [--final]
#
# Adjust the resources below to the cluster. Runs replay stored runtimes, so a
# run needs one core; memory and time depend on the selector (random forests
# with 500 trees are the slowest).
#SBATCH --job-name=oas
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail
CONFIG="$1"
shift

# Activate the environment (default: .venv in the repository).
source "${OAS_VENV:-.venv}/bin/activate"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# A run whose result exists exits at once, so resubmitting a whole
# experiment only redoes the runs that are missing.
python -m oas_experiments.run "$CONFIG" --index "$SLURM_ARRAY_TASK_ID" "$@"
