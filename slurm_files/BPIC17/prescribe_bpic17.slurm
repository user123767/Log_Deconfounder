#!/bin/bash
#SBATCH --job-name=BPIC17
#SBATCH --output=BPIC17_log/Real_%A_%a.out
#SBATCH --error=BPIC17_log/Real_%A_%a.err
#SBATCH --array=0-5
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0-12:00:00



module purge
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate rl_env

set -euo pipefail

DATASET_BASE=${DATASET_BASE:-bpic2017_accepted}
HIDDEN=${HIDDEN:-CreditScore}
PROXY_TYPE=${PROXY_TYPE:-gaussian}
PROXY_NOISES=(${PROXY_NOISES:-0.3})

SEEDS=(0)
TRAIN_DATAS=(logged cevae)
AGENTS=(${AGENTS:-cql seq_vi causal_rl})
OUTPUT_ROOT_BASE=${OUTPUT_ROOT_BASE:-results_realworld_prescribe}
CEVAE_ROOT_BASE=${CEVAE_ROOT_BASE:-results_realworld_seqCEVAE}

CQL_UPDATES=100000
CRL_PPO_ITERS=100000

N_AGENTS=${#AGENTS[@]}
N_NOISES=${#PROXY_NOISES[@]}
N_DATA=${#TRAIN_DATAS[@]}
N_TASKS=$(( ${#SEEDS[@]} * N_DATA * N_NOISES * N_AGENTS ))
if (( SLURM_ARRAY_TASK_ID >= N_TASKS )); then
    echo "Array task ${SLURM_ARRAY_TASK_ID} has no combination (only ${N_TASKS}) -- check --array." >&2
    exit 1
fi
AGENT=${AGENTS[$(( SLURM_ARRAY_TASK_ID % N_AGENTS ))]}
REST=$(( SLURM_ARRAY_TASK_ID / N_AGENTS ))
NOISE=${PROXY_NOISES[$(( REST % N_NOISES ))]}
REST=$(( REST / N_NOISES ))
TRAIN_DATA=${TRAIN_DATAS[$(( REST % N_DATA ))]}
SEED=${SEEDS[$(( REST / N_DATA ))]}
RUN_NAME=${HIDDEN//,/+}

DATASET=${DATASET_BASE}_proxy-${HIDDEN//,/-}-${PROXY_TYPE}${NOISE}
OUTPUT_DIR=${OUTPUT_ROOT_BASE}/seed${SEED}/${DATASET}
CEVAE_DIR=${CEVAE_ROOT_BASE}/seed${SEED}/${DATASET_BASE}/${DATASET}
if [[ "${TRAIN_DATA}" == cevae && ! -f "${CEVAE_DIR}/cevae_static_${RUN_NAME}.csv" ]]; then
    echo "Missing ${CEVAE_DIR}/cevae_static_${RUN_NAME}.csv -- run the seq_cevae_log slurm first (same seed / log / hidden column)." >&2
    exit 1
fi
mkdir -p BPIC17_log "$OUTPUT_DIR"
echo "Task ${SLURM_ARRAY_TASK_ID}: seed=${SEED}, agent=${AGENT}, train_data=${TRAIN_DATA}, dataset=${DATASET}, hidden=[${HIDDEN}], output=${OUTPUT_DIR}"

python -u src/prescribe_real_log.py \
    --dataset "${DATASET}" \
    --exclude_cols "${HIDDEN}" \
    --run_name "${RUN_NAME}" \
    --agent "${AGENT}" \
    --train_data "${TRAIN_DATA}" \
    --cevae_dir "${CEVAE_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --seed "${SEED}" \
    --cql_updates "${CQL_UPDATES}" \
    --crl_ppo_iters "${CRL_PPO_ITERS}"

echo "Task ${SLURM_ARRAY_TASK_ID} (seed ${SEED}, ${AGENT}, ${TRAIN_DATA}, ${DATASET}) finished."
my-job-stats -a -n -s
