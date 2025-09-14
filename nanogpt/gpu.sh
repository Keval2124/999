#!/bin/bash -l

#SBATCH --job-name=dialoge_gpt_gpu_single
#SBATCH --output=dialoge_gpt_gpu_output_%j.log
#SBATCH --error=dialoge_gpt_gpu_error_%j.log
#SBATCH --partition=gpu-l40s-low   
#SBATCH --nodes=1                   # Request 1 nodes
#SBATCH --ntasks=1                  # Request 1 task (single process)
#SBATCH --cpus-per-task=48          # Request 48 CPUs for the task (adjust as needed)
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --mem=300G                  # Request 300 GB RAM
#SBATCH --time=1-00:00:00           # 3-day time limit

module load cuda/12.8.0-gcc14.2.0 
module load openmpi/5.0.8/gcc-14.2.0
module load ffmpeg/7.0.2-gcc14.2.0
module load gcc/14.2.0-gcc11.5.0 
module purge


# Set environment variables
export GLOO_SOCKET_IFNAME=lo
export PYTHONPATH=$PWD:$PYTHONPATH
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Debug information
echo "=== SLURM Environment ==="
echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
echo "SLURM_PROCID: $SLURM_PROCID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID" # Will be empty/not set
echo "SLURM_JOB_GPUS: $SLURM_JOB_GPUS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "Number of Nodes: $SLURM_JOB_NUM_NODES"
echo "Number of Tasks: $SLURM_NTASKS"
echo "CPUs per Task: $SLURM_CPUS_PER_TASK"
echo "========================"

python train.py 