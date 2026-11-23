#!/bin/bash
#SBATCH --job-name=real_proxies
#SBATCH --output=logs_real_proxies/proxies_%A_%a.out
#SBATCH --error=logs_real_proxies/proxies_%A_%a.err
#SBATCH --array=0-1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0-02:00:00


module purge
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate rl_env

set -euo pipefail

DATASETS=(traffic_fines_1 sepsis_cases_1)

declare -A CONFOUNDERS=(
  [bpic2017_accepted]="CreditScore"
  [bpic2012_accepted]="AMOUNT_REQ"
  [traffic_fines_1]="lastSent"
  [sepsis_cases_1]="Infusion"
)

GAUSSIAN_NOISES=0.05,0.1,0.2,0.3,0.4,0.5
TWINS_NOISES=0.05,0.1,0.2,0.3,0.4,0.5

if (( SLURM_ARRAY_TASK_ID >= ${#DATASETS[@]} )); then
    echo "Array task ${SLURM_ARRAY_TASK_ID} has no dataset (only ${#DATASETS[@]}) -- check --array." >&2
    exit 1
fi
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
CONF=${CONFOUNDERS[$DATASET]}
if [[ -z "${CONF}" || "${CONF}" == TODO ]]; then
    echo "No confounder set for ${DATASET} -- fill in CONFOUNDERS." >&2
    exit 1
fi
mkdir -p logs_real_proxies
echo "Task ${SLURM_ARRAY_TASK_ID}: proxies of [${CONF}] for ${DATASET}"

python -u src/real_log_proxies.py --dataset "${DATASET}" --confounders "${CONF}" --proxy_type gaussian \
    --proxy_noise "${GAUSSIAN_NOISES}"

python -u src/real_log_proxies.py --dataset "${DATASET}" --confounders "${CONF}" --proxy_type twins \
    --proxy_noise "${TWINS_NOISES}"

echo "Task ${SLURM_ARRAY_TASK_ID} (${DATASET}) finished."
my-job-stats -a -n -s
