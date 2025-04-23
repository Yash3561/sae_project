#!/bin/bash -l
#SBATCH --job-name=Llama_ActivationGen     # Job name
#SBATCH --output=%x.%j.out                 # Standard output and error log (%x=jobname, %j=jobID)
#SBATCH --error=%x.%j.err
#SBATCH --nodes=1                          # Run on a single node
#SBATCH --ntasks-per-node=1                         # Run a single task
#SBATCH --cpus-per-task=4                  # Request 4 CPUs (adjust if needed)
#SBATCH --mem=64G                          # Request memory (e.g., 64GB; adjust if Llama needs more)
#SBATCH --partition=gpu                    # <<< VERIFY/CHANGE: Wulver GPU partition name (e.g., dgx, gpu, tesla)
#SBATCH --gres=gpu:a100:1                       # <<< VERIFY/CHANGE: Wulver GPU request syntax (e.g., gpu:1, gpu:T4:1, gpu:A100:1)
#SBATCH --account=2025-spring-ds-680-md748-ygc2 # Your specific course account
#SBATCH --qos=standard                     # <<< VERIFY/CHANGE: QOS for the partition (check Wulver docs)
#SBATCH --time=48:00:00                    # <<< Adjust time: e.g., 24 hours for train split (~25k examples), more for train+test

# --- Wulver Environment Setup ---
echo "Starting job on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running in directory: $(pwd)"
start_time=$(date +%s)

echo "Loading modules..."
module purge # Start with a clean environment
module load easybuild
module load Anaconda3/2023.09-0    # <<< VERIFY/CHANGE: Use the correct Anaconda/Miniconda module name on Wulver
module load CUDA/12.4.0          # <<< VERIFY/CHANGE: Use CUDA version matching your torch install in requirements.txt

echo "Activating Conda environment..."
conda activate ./envs/nlp_sae_env # Activate the environment created previously

# --- Set Caches (Recommended) ---
# Cache HuggingFace models/datasets in your course directory
export HF_HOME="/course/2025/spring/ds/680/md748/ygc2/.cache/huggingface"
export HF_DATASETS_CACHE="/course/2025/spring/ds/680/md748/ygc2/.cache/huggingface/datasets"
# Cache PyTorch Hub models if needed
export TORCH_HOME="/course/2025/spring/ds/680/md748/ygc2/.cache/torch"
mkdir -p $HF_HOME
mkdir -p $HF_DATASETS_CACHE
mkdir -p $TORCH_HOME
echo "Cache directories set/created:"
echo "HF_HOME: $HF_HOME"
echo "HF_DATASETS_CACHE: $HF_DATASETS_CACHE"
echo "TORCH_HOME: $TORCH_HOME"

# --- Run Script ---
echo "Running generate_activations.py..."
# Run the script with desired arguments
# Process the full training split in this example:
# Pass your HF token here
python scripts/generate_activations.py \
    --dataset_split train \
    --num_examples -1 \
    --hf_token "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" \
    --save_dir "./llama_activations_train" # Save to a specific directory for train split

# Optional: Generate for test split as well (if time/storage allows)
# echo "Running generate_activations.py for test split..."
python scripts/generate_activations.py \
     --dataset_split test \
     --num_examples -1 \
     --hf_token "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" \
     --save_dir "./llama_activations_test"

# --- Cleanup ---
conda deactivate
echo "Conda environment deactivated."

end_time=$(date +%s)
runtime=$((end_time - start_time))
echo "Job finished at $(date)"
echo "Total runtime: ${runtime} seconds"
# --- End of Script ---
