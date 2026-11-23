#!/bin/bash
#SBATCH --job-name=bpic12SeqCEVAE
#SBATCH --output=logs_seqCEVAE_real/bpic12_%A_%a.out
#SBATCH --error=logs_seqCEVAE_real/bpic12_%A_%a.err
#SBATCH --array=0-12
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0-08:00:00


module purge
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate rl_env

set -euo pipefail

BASE_DATASET=bpic2012_accepted
CONFOUNDER="AMOUNT_REQ"
SEEDS=(0)
OUTPUT_ROOT_BASE=results_realworld_seqCEVAE

GAUSSIAN_NOISES=(0.05 0.1 0.2 0.3 0.4 0.5)
TWINS_NOISES=(0.05 0.1 0.2 0.3 0.4 0.5)
PROXY_SETTINGS=("none")
for n in "${GAUSSIAN_NOISES[@]}"; do PROXY_SETTINGS+=("gaussian:${n}"); done
for n in "${TWINS_NOISES[@]}";    do PROXY_SETTINGS+=("twins:${n}");    done

EPOCHS=1000
PATIENCE=50
N_PER_CASE=10

if [[ "${CONFOUNDER}" == TODO || -z "${CONFOUNDER}" ]]; then
    echo "Set CONFOUNDER for ${BASE_DATASET}." >&2; exit 1
fi
N_PROXY=${#PROXY_SETTINGS[@]}
N_TASKS=$(( ${#SEEDS[@]} * N_PROXY ))
if (( SLURM_ARRAY_TASK_ID >= N_TASKS )); then
    echo "Array task ${SLURM_ARRAY_TASK_ID} has no combination (only ${N_TASKS}) -- check --array." >&2; exit 1
fi
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID / N_PROXY ))]}
SETTING=${PROXY_SETTINGS[$(( SLURM_ARRAY_TASK_ID % N_PROXY ))]}
COLS_TAG=$(echo "${CONFOUNDER}" | sed 's/[^0-9A-Za-z_.,]/_/g; s/,/-/g')
if [[ "${SETTING}" == none ]]; then
    DATASET=${BASE_DATASET}
else
    DATASET=${BASE_DATASET}_proxy-${COLS_TAG}-${SETTING%%:*}${SETTING#*:}
fi
RUN_NAME=${CONFOUNDER//,/+}
OUTPUT_DIR=${OUTPUT_ROOT_BASE}/seed${SEED}/${BASE_DATASET}/${DATASET}
mkdir -p logs_seqCEVAE_real "$OUTPUT_DIR"
echo "Task ${SLURM_ARRAY_TASK_ID}: seed ${SEED}, ${DATASET}, hidden=[${CONFOUNDER}]"

python -u src/seq_cevae_log.py \
    --dataset "${DATASET}" \
    --exclude_cols "${CONFOUNDER}" \
    --run_name "${RUN_NAME}" \
    --output_dir "${OUTPUT_DIR}" \
    --seed "${SEED}" \
    --epochs "${EPOCHS}" \
    --patience "${PATIENCE}" \
    --n_per_case "${N_PER_CASE}"

echo "Task ${SLURM_ARRAY_TASK_ID} (seed ${SEED}, ${DATASET}) finished."
my-job-stats -a -n -s
