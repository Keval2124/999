#!/usr/bin/env python3
import os
import sys
import math
import subprocess

# Set environment variables at the very beginning
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

def initialize_environment():
    """Initialize environment for worker processes"""
    try:
        # Try to load ffmpeg module
        subprocess.run("module load ffmpeg/7.0.2-gcc14.2.0", shell=True, check=True, capture_output=True)
    except subprocess.CalledProcessError:
        # If module command fails, set the direct path to ffmpeg
        pass

def get_ffmpeg_path():
    """Get the correct path to ffmpeg"""
    try:
        # Try to find ffmpeg in PATH first
        result = subprocess.run(['which', 'ffmpeg'], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        # Fallback to hardcoded path on Barkla
        return "/opt/apps/pkg/applications/spack_apps/v0231_apps/linux-rocky9-x86_64_v3/gcc-14.2.0/ffmpeg-7.0.2-b46utrz7nybdkunpltzlw3q3w3fouwjv/bin/ffmpeg"

# Initialize environment
initialize_environment()

# Now import transcript after setting environment
import transcript

def main():
    # Get node info from command line arguments
    if len(sys.argv) >= 3:
        node_id = int(sys.argv[1])        # SLURM_ARRAY_TASK_ID
        total_nodes = int(sys.argv[2])  
        print(f"Using command line arguments: node_id={node_id}, total_nodes={total_nodes}")
    else:
        # Fallback to SLURM env vars
        total_nodes = int(os.getenv("SLURM_JOB_NUM_NODES", "1"))
        node_id = int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))
        print(f"Using SLURM environment variables: node_id={node_id}, total_nodes={total_nodes}")

    # Debug: Print SLURM environment variables
    print("=== SLURM Environment ===")
    print(f"SLURM_JOB_NUM_NODES: {os.getenv('SLURM_JOB_NUM_NODES')}")
    print(f"SLURM_PROCID: {os.getenv('SLURM_PROCID')}")
    print(f"SLURM_ARRAY_TASK_ID: {os.getenv('SLURM_ARRAY_TASK_ID')}")
    print(f"SLURM_NTASKS: {os.getenv('SLURM_NTASKS')}")
    print("========================")
    
    # Verify ffmpeg path
    ffmpeg_path = get_ffmpeg_path()
    print(f"Using ffmpeg path: {ffmpeg_path}")
    
    # Check if ffmpeg exists
    if not os.path.exists(ffmpeg_path):
        print(f"WARNING: ffmpeg not found at {ffmpeg_path}")
    else:
        print(f"SUCCESS: ffmpeg found at {ffmpeg_path}")

    # Build the list of audio indices
    wav_dir = transcript.Config.AUDIO_FOLDER
    print(f"Looking for WAV files in: {wav_dir}")
    
    if not os.path.exists(wav_dir):
        print(f"ERROR: Directory {wav_dir} does not exist!")
        return
    
    wavs = [f for f in os.listdir(wav_dir) if f.endswith('.wav')]
    print(f"Found {len(wavs)} WAV files")
    
    indices = sorted(int(f.split('.')[0]) for f in wavs if f.split('.')[0].isdigit())
    print(f"Found {len(indices)} valid numbered WAV files")
    
    if len(indices) == 0:
        print("ERROR: No valid numbered WAV files found!")
        return

    # Slice the list
    per_node = math.ceil(len(indices) / total_nodes)
    start = node_id * per_node
    end = min(start + per_node, len(indices))
    my_files = indices[start:end]

    print(f"[NODE {node_id}] processing files {start}–{end-1}  ({len(my_files)} files)")
    
    # Process the files
    transcript.transcribe_batch(my_files)

if __name__ == '__main__':
    main()