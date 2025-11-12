#!/bin/bash
#SBATCH --job-name=cryptanalysis_train
#SBATCH --output=slurm-%j.out              # temporary log file at job start
#SBATCH --error=slurm-%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --partition=l4

# --- ENVIRONMENT SETUP ---
cd ~/hidden-markov-model
mkdir -p logs

# If you use a virtual environment:
#source ~/my_venv/bin/activate

# Or, if you use modules on AI-LAB:
# module load python/3.10
# module load cuda/12.1

# --- RUN TRAINING ---
echo "Job started on $(hostname) at $(date)" | tee logs/train_live_${SLURM_JOB_ID}.log
uv run python -u main.py 2>&1 | tee -a logs/train_live_${SLURM_JOB_ID}.log
echo "Job finished at $(date)" | tee -a logs/train_live_${SLURM_JOB_ID}.log