#!/bin/bash
#SBATCH --job-name=z408_job           # Name of the job
#SBATCH --output=logs/%x_%j.out       # Output log file (%x=job name, %j=job ID)
#SBATCH --error=logs/%x_%j.err        # Error log file
#SBATCH --time=12:00:00               # Time limit (HH:MM:SS)
#SBATCH --partition=l4          # Partition/queue name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks=1                    # Number of tasks
#SBATCH --cpus-per-task=8             # Number of CPU cores per task
#SBATCH --mem=24G                     # Memory per node

# Optional: Load your environment or module
module load python/3.11              # or any relevant module
# If you use uv (like `uv run`), ensure it's available on PATH
# module load uv                     # if uv is a module
# or if it's installed via pipx, activate that env
# source ~/.local/bin/uv             # example path if installed manually

# Run your command
uv run main.py -c z408 -r 1000 -b 100