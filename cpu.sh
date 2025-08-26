#!/bin/bash -l

#SBATCH --job-name=transcribe_cpu_array
#SBATCH --output=transcribe_cpu_output_%j.log  
#SBATCH --error=transcribe_cpu_error_%j.log
#SBATCH --partition=nodes
#SBATCH --nodes=1                    # 2 nodes per array job
#SBATCH --ntasks-per-node=1          # 1 task per node
#SBATCH --cpus-per-task=120           # Use all 40 cores per task
#SBATCH --mem=300G                   # Memory per node
#SBATCH --time=3-00:00:00

# Load required modules (NO purge in between)


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

ulimit -u 740

# Run script - $SLURM_ARRAY_TASK_ID will be 0, 1,
# python transcribe_distributed.py $SLURM_ARRAY_TASK_ID 2
python transcript.py 

du -sh ~/.cache
