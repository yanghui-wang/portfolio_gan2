#!/bin/bash
#SBATCH --job-name=port_gan       # Name for your job
#SBATCH --ntasks=1               # Number of tasks
#SBATCH --cpus-per-task=8        # CPUs per task
#SBATCH --gres=gpu:1             # Request 1 GPU per node
#SBATCH --nodelist=gpu114       # Specify the computing node 
#SBATCH --time=2-00:00:00        # Request 2 days runtime (D-HH:MM:SS)
#SBATCH --output=%j_output.log   # Save output to a log file
#SBATCH --mem=256G   # Save output to a log file


echo "JOB STARTED"
date
sleep 10

echo "PYTHON ABOUT TO START"
echo "===== SHM SIZE ====="
df -h /dev/shm
echo "===================="

# Your commands below
 
python run_pipeline.py --stage train --training-mode full-train --epochs 30 --batch-size 32
python run_pipeline.py --stage evaluate


echo "JOB END"