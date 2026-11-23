#!/bin/bash
#SBATCH --job-name=simbank_gen
#SBATCH --output=logs/simbank_gen_%A_%a.out
#SBATCH --error=logs/simbank_gen_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --array=0-1

module purge
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate rl_env

set -euo pipefail

INTERVENTIONS=(choose_procedure set_ir_3_levels)
DELTAS="0.9,0.95,0.99"
PROXY_COLS=amount,est_quality
PROXY_TYPE=gaussian
PROXY_NOISES="0.6,0.7,0.8,0.9,1,10"

INTERVENTION=${INTERVENTIONS[$SLURM_ARRAY_TASK_ID]}
mkdir -p logs

python src/simbank_data_generation.py \
    --intervention "$INTERVENTION" \
    --deltas "$DELTAS" \
    --proxy_cols "$PROXY_COLS" \
    --proxy_type "$PROXY_TYPE" \
    --proxy_noise "$PROXY_NOISES" \
    --params_dir src \
    --save_val
