#!/bin/bash
# Submit every run of an experiment config as one SLURM job array.
#
#   scripts/submit.sh configs/my_experiment.yaml            # pilot runs
#   scripts/submit.sh configs/my_experiment.yaml --final    # thesis results
#
# Further arguments go to `python -m oas_experiments.run`. Set MAX_PARALLEL to
# limit the number of tasks running at once (default 100).
set -euo pipefail
CONFIG="$1"
source "${OAS_VENV:-.venv}/bin/activate"
N=$(python -m oas_experiments.list "$CONFIG" --count)
MISSING=$(python -m oas_experiments.list "$CONFIG" --missing --count)
echo "$CONFIG: $N runs, $MISSING without a result"
if [ "$MISSING" -eq 0 ]; then
    exit 0
fi
mkdir -p logs
sbatch --array="0-$((N - 1))%${MAX_PARALLEL:-100}" "$(dirname "$0")/slurm_array.sh" "$@"
