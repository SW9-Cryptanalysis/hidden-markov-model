#!/bin/bash
#SBATCH --job-name=hmm_job           # Name of the job
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
#SBATCH --time=12:00:00               # Time limit (HH:MM:SS)
#SBATCH --partition=l4          # Partition/queue name
#SBATCH --cpus-per-task=8             # Number of CPU cores per task
#SBATCH --mem=24G                     # Memory per node

mkdir -p logs

# Run your command
echo "Job started on $(hostname) at $(date)" | tee logs/train_live_${SLURM_JOB_ID}.log
echo "Running with: -c ${C_ARG} -r ${R_ARG} -b ${B_ARG}" | tee -a logs/train_live_${SLURM_JOB_ID}.log

uv run python -u main.py -c "$C_ARG" -r "$R_ARG" -b "$B_ARG" 2>&1 | tee -a logs/train_live_${SLURM_JOB_ID}.log

echo "Job finished at $(date)" | tee -a logs/train_live_${SLURM_JOB_ID}.log