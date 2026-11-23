#!/bin/bash
#SBATCH --job-name=conf_rank
#SBATCH --output=logs_conf_rank/rank_%A_%a.out
#SBATCH --error=logs_conf_rank/rank_%A_%a.err
#SBATCH --array=0-3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=10G
#SBATCH --time=0-01:00:00


module purge
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate rl_env

set -euo pipefail

DATASETS=(bpic2017_accepted bpic2012_accepted traffic_fines_1 sepsis_cases_1)
OUTPUT_DIR=results/confounder_ranking
LIST_ONLY=${LIST_ONLY:-false}

TOP_K=15
N_JOBS=${SLURM_CPUS_PER_TASK:-8}

declare -A EXCLUDE=(
  [bpic2017_accepted]=""
  [bpic2012_accepted]=""
  [traffic_fines_1]=""
  [sepsis_cases_1]=""
)

if (( SLURM_ARRAY_TASK_ID >= ${#DATASETS[@]} )); then
    echo "Array task ${SLURM_ARRAY_TASK_ID} has no dataset (only ${#DATASETS[@]}) -- check --array." >&2
    exit 1
fi
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
mkdir -p logs_conf_rank "$OUTPUT_DIR"
echo "Task ${SLURM_ARRAY_TASK_ID}: confounder ranking for ${DATASET}"

if [[ "${LIST_ONLY}" == true ]]; then
    python -u src/select_exclude_col_single.py --dataset "${DATASET}" --list_activities
    exit 0
fi

EXCL=${EXCLUDE[$DATASET]:-}
python -u src/select_exclude_col_single.py \
    --dataset "${DATASET}" \
    ${EXCL:+--exclude_from_candidates ${EXCL}} \
    --n_jobs "${N_JOBS}" \
    --top_k "${TOP_K}" \
    --output_dir "${OUTPUT_DIR}"

echo "Task ${SLURM_ARRAY_TASK_ID} (${DATASET}) finished."
my-job-stats -a -n -s
