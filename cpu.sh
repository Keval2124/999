#!/bin/bash -l

#SBATCH --job-name=transcribe_cpu_array
#SBATCH --output=transcribe_cpu_output_%j_%a.log  
#SBATCH --error=transcribe_cpu_error_%j_%a.log
#SBATCH --partition=nodes
#SBATCH --nodes=1                    # 2 nodes per array job
#SBATCH --ntasks-per-node=1          # 1 task per node
#SBATCH --cpus-per-task=40           # Use all 40 cores per task
#SBATCH --mem=300G                   # Memory per node
#SBATCH --time=3-00:00:00
#SBATCH --array=0-1                  # 2 array jobs

# Load required modules (NO purge in between)
module load ffmpeg/7.0.2-gcc14.2.0
module load cuda/12.8.0-gcc14.2.0
module load openmpi/5.0.8/gcc-14.2.0

# Install packages

# Environment variables
export GLOO_SOCKET_IFNAME=lo
export PYTHONPATH=$PWD:$PYTHONPATH
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Debug info
echo "Job ID: $SLURM_JOB_ID"
echo "Job Node List: $SLURM_JOB_NODELIST"
echo "Number of Nodes: $SLURM_JOB_NUM_NODES"
echo "Number of Tasks: $SLURM_NTASKS"
echo "CPUs per Task: $SLURM_CPUS_PER_TASK"

du -sh ~/.cache

# Run script - $SLURM_ARRAY_TASK_ID will be 0, 1,
# python transcribe_distributed.py $SLURM_ARRAY_TASK_ID 2
python transcribe.py 

du -sh ~/.cache