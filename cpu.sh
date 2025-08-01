#!/bin/bash -l

#SBATCH --job-name=transcribe_cpu_array
#SBATCH --output=transcribe_cpu_output_%j_%a.log  # %a for array task ID
#SBATCH --error=transcribe_cpu_error_%j_%a.log
#SBATCH --partition=nodes
#SBATCH --nodes=3                    # 1 node per array job
#SBATCH --ntasks=3                   # 1 task per array job
#SBATCH --cpus-per-task=40           # Use all 40 cores for the single task
#SBATCH --mem=300G                   # Adjust memory per node (standard nodes have ~380G)
#SBATCH --time=3-00:00:00
#SBATCH --array=0-2                  # Run 3 jobs (0, 1, 2)

# Load required modules (NO purge in between)
module load ffmpeg/7.0.2-gcc14.2.0
module load cuda/12.8.0-gcc14.2.0
module load openmpi/5.0.8/gcc-14.2.0

# Install packages
pip install torch torchaudio transformers huggingface_hub networkx transformers[torch] 'accelerate>=0.26.0' peft importlib_metadata

# Environment variables
export GLOO_SOCKET_IFNAME=lo
export PYTHONPATH=$PWD:$PYTHONPATH
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Debug info
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Job Node List: $SLURM_JOB_NODELIST"

du -sh ~/.cache

# Run script - $SLURM_ARRAY_TASK_ID will be 0, 1, or 2
python transcribe_distributed.py $SLURM_ARRAY_TASK_ID 3 

du -sh ~/.cache