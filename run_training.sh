#!/bin/bash
#SBATCH --job-name=hmm_job           # Name of the job
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
#SBATCH --time=12:00:00               # Time limit (HH:MM:SS)
#SBATCH --partition=l4          # Partition/queue name
#SBATCH --cpus-per-task=8             # Number of CPU cores per task
#SBATCH --mem=24G                     # Memory per node

# Optional: Load your environment or module
# module load python/3.11              # or any relevant module
# If you use uv (like `uv run`), ensure it's available on PATH
# module load uv                     # if uv is a module
# or if it's installed via pipx, activate that env
# source ~/.local/bin/uv             # example path if installed manually

# Run your command
echo "Job started on $(hostname) at $(date)" | tee logs/train_live_${SLURM_JOB_ID}.log

uv run python -u main.py -c z408 -r 1000 -b 100 2>&1 | tee -a logs/train_live_${SLURM_JOB_ID}.log

echo "Job finished at $(date)" | tee -a logs/train_live_${SLURM_JOB_ID}.log