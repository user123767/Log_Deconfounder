#!/bin/bash
#SBATCH --job-name=sb_procedure_conf
#SBATCH --output=logs_simbank/procedure_conf_%A_%a.out
#SBATCH --error=logs_simbank/procedure_conf_%A_%a.err
#SBATCH --array=0-539
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0-24:59:00


set -euo pipefail

STEP=${STEP:-agents}
TRAIN_DATA=${TRAIN_DATA:-cevae}

SEEDS=(0 1 2 3 4)
INTERVENTION=choose_procedure
DELTAS=(0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.99)
CONFOUNDER_SETS=("amount" "est_quality" "amount,est_quality")
PROXY_COLS=amount,est_quality
GAUSSIAN_NOISES=(0.3)
TWINS_NOISES=()
AGENTS=(${AGENTS:-cql seq_vi causal_rl})

FILE_PREFIX=cevae
Z_DIM=3
EPOCHS=1000
PATIENCE=50
N_PER_CASE=10

CQL_UPDATES=50000
EVAL_BASELINES=true

PROXY_SETTINGS=()
for n in ${GAUSSIAN_NOISES[@]+"${GAUSSIAN_NOISES[@]}"}; do PROXY_SETTINGS+=("gaussian:${n}"); done
for n in ${TWINS_NOISES[@]+"${TWINS_NOISES[@]}"}; do PROXY_SETTINGS+=("twins:${n}");    done
N_AGENTS=${#AGENTS[@]}
N_SETS=${#CONFOUNDER_SETS[@]}
N_PROXY=${#PROXY_SETTINGS[@]}
N_COMBOS=$(( ${#DELTAS[@]} * N_SETS * N_PROXY ))

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "STEP=cevae:  --array=0-$(( ${#SEEDS[@]} * N_COMBOS - 1 ))"
    echo "STEP=agents: --array=0-$(( ${#SEEDS[@]} * N_COMBOS * N_AGENTS - 1 ))"
    exit 0
fi

module purge
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate rl_env

case "${STEP}" in
    cevae)  N_PER_SEED=${N_COMBOS} ;;
    agents) N_PER_SEED=$(( N_COMBOS * N_AGENTS )) ;;
    *) echo "STEP must be cevae or agents, got '${STEP}'." >&2; exit 1 ;;
esac
N_TASKS=$(( ${#SEEDS[@]} * N_PER_SEED ))
if (( SLURM_ARRAY_TASK_ID >= N_TASKS )); then
    [[ "${STEP}" == cevae ]] && exit 0
    echo "Array task ${SLURM_ARRAY_TASK_ID} has no combination (only ${N_TASKS}) -- check --array." >&2
    exit 1
fi
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID / N_PER_SEED ))]}
TASK=$(( SLURM_ARRAY_TASK_ID % N_PER_SEED ))
if [[ "${STEP}" == agents ]]; then
    AGENT_IDX=$(( TASK % N_AGENTS ))
    AGENT=${AGENTS[$AGENT_IDX]}
    TASK=$(( TASK / N_AGENTS ))
fi
SETTING=${PROXY_SETTINGS[$(( TASK % N_PROXY ))]}
PROXY_TYPE=${SETTING%%:*}
NOISE=${SETTING#*:}
COMBO=$(( TASK / N_PROXY ))
DELTA=${DELTAS[$(( COMBO / N_SETS ))]}
EXCLUDE_COLS=${CONFOUNDER_SETS[$(( COMBO % N_SETS ))]}
RUN_NAME=${EXCLUDE_COLS//,/+}

DATASET=SimBank_${INTERVENTION}_delta${DELTA}_proxy-${PROXY_COLS//,/-}-${PROXY_TYPE}${NOISE}
DATASET_PARAMS=src/dataset_params_${INTERVENTION}.pkl
if [[ "${STEP}" == agents && "${TRAIN_DATA}" == logged ]]; then
    OUTPUT_ROOT_BASE=results_baselines
else
    OUTPUT_ROOT_BASE=results_treatment_seqCEVAE_final
fi
OUTPUT_DIR=${OUTPUT_ROOT_BASE}/seed${SEED}/SimBank/${DATASET}
if [[ ! -f "${DATASET_PARAMS}" ]]; then
    echo "Missing ${DATASET_PARAMS} -- run simbank_data_generation.py --intervention ${INTERVENTION} first." >&2
    exit 1
fi
mkdir -p "$OUTPUT_DIR"

if [[ "${STEP}" == cevae ]]; then
    echo "Task ${SLURM_ARRAY_TASK_ID}: seed ${SEED}, sequence CEVAE on ${DATASET}, hidden=[${EXCLUDE_COLS}]"
    python -u src/seq_cevae_simbank.py \
        --dataset "${DATASET}" \
        --dataset_params "${DATASET_PARAMS}" \
        --exclude_cols "${EXCLUDE_COLS}" \
        --run_name "${RUN_NAME}" \
        --output_dir "${OUTPUT_DIR}" \
        --file_prefix "${FILE_PREFIX}" \
        --seed "${SEED}" \
        --z_dim "${Z_DIM}" \
        --epochs "${EPOCHS}" \
        --patience "${PATIENCE}" \
        --n_per_case "${N_PER_CASE}"
else
    if [[ "${TRAIN_DATA}" == cevae && ! -f "${OUTPUT_DIR}/cevae_dataset_${RUN_NAME}.npz" ]]; then
        echo "Missing ${OUTPUT_DIR}/cevae_dataset_${RUN_NAME}.npz -- run STEP=cevae first." >&2
        exit 1
    fi
    EVAL_BL=false
    (( AGENT_IDX == 0 )) && EVAL_BL=${EVAL_BASELINES}
    echo "Task ${SLURM_ARRAY_TASK_ID}: seed=${SEED}, agent=${AGENT}, train_data=${TRAIN_DATA}, dataset=${DATASET}, hidden=[${EXCLUDE_COLS}]"
    python -u src/Log_deconfounder.py \
        --agent "${AGENT}" \
        --train_data "${TRAIN_DATA}" \
        --dataset "${DATASET}" \
        --dataset_params "${DATASET_PARAMS}" \
        --exclude_cols "${EXCLUDE_COLS}" \
        --run_name "${RUN_NAME}" \
        --output_dir "${OUTPUT_DIR}" \
        --seed "${SEED}" \
        --cql_updates "${CQL_UPDATES}" \
        --eval_baselines "${EVAL_BL}"
fi

echo "Task ${SLURM_ARRAY_TASK_ID} (${STEP}, seed ${SEED}, ${DATASET}, ${RUN_NAME}) finished."
my-job-stats -a -n -s
