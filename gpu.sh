#!/bin/bash -l

#SBATCH --job-name=nanogpt_gpu_single
#SBATCH --output=nanogpt_gpu_output_%j.log
#SBATCH --error=nanogpt_gpu_error_%j.log
#SBATCH --partition=gpu-a100-lowbig        # Target L40S GPU partition (ensure this exists on your cluster)
#SBATCH --nodes=2                   # Request 2 nodes
#SBATCH --ntasks=2                  # Request 1 task (single process)
#SBATCH --cpus-per-task=48           # Request 4 CPUs for the task (adjust as needed)
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --mem=150G                  # Request 150 GB RAM
#SBATCH --time=1-00:00:00           # 3-day time limit

module load cuda/12.8.0-gcc14.2.0 
module load openmpi/5.0.8/gcc-14.2.0
module load ffmpeg/7.0.2-gcc14.2.0
module load gcc/14.2.0-gcc11.5.0 
module purge

# Load Anaconda and activate the environment
pip install torch torchaudio transformers huggingface_hub networkx transformers[torch] 'accelerate>=0.26.0' peft importlib_metadata

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
echo "========================"

du -sh ~/.cache
#python finetune_bert.py '[1, 2, 3]'
# torchrun --nproc_per_node=40 control.py 
python transcribe_distributed.py 
# Check cache size
du -sh ~/.cache