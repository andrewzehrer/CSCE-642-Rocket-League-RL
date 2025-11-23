#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --job-name=phase_01
#SBATCH --output=%x_%j.log

# Activate venv
source /$SCRATCH/CSCE-642-Rocket-League-RL/venv/bin/activate

# Run your program
python -m scripts.1v0_plr