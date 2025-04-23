#!/bin/bash -l
#SBATCH --job-name=SAE_Training            # Job name
#SBATCH --output=%x.%j.out                 # Standard output log
#SBATCH --error=%x.%j.err                  # Standard error log
#SBATCH --nodes=1                          # Run on a single node
#SBATCH --ntasks-per-node=1                # Run a single task
#SBATCH --cpus-per-task=4                  # Request 4 CPUs (adjust if needed for DataLoader)
#SBATCH --mem=64G                         # <<< Adjust: SAE training might need more memory (start with 128G)
#SBATCH --partition=gpu                    # <<< From your previous script - VERIFY THIS is correct for Wulver
#SBATCH --gres=gpu:a100:1                  # <<< From your previous script - VERIFY THIS requests an A100 on the 'gpu' partition
#SBATCH --account=2025-spring-ds-680-md748-ygc2 # Your specific course account (Looks Correct)
#SBATCH --qos=standard                     # <<< From your previous script - VERIFY THIS is correct QOS for 'gpu' partition
#SBATCH --time=24:00:00                    # <<< Adjust time: Training can take days (e.g., 72 hours = 3 days) - Check Wulver limits

# --- Wulver Environment Setup ---
echo "Starting SAE Training job on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running in directory: $(pwd)"
start_time=$(date +%s)

echo "Loading modules..."
module purge # Start with a clean environment
module load easybuild
module load Anaconda3/2023.09-0    # <<< VERIFY/CHANGE: Use the correct Anaconda/Miniconda module name used for setup
module load CUDA/12.4.0          # <<< VERIFY/CHANGE: Use CUDA version matching your torch install in requirements.txt

echo "Activating Conda environment..."
# Use the environment path relative to your submission directory
conda activate ./envs/nlp_sae_env

# --- Set Caches (Good Practice) ---
# Cache HuggingFace models/datasets in your course directory
export HF_HOME="/course/2025/spring/ds/680/md748/ygc2/.cache/huggingface"
export TORCH_HOME="/course/2025/spring/ds/680/md748/ygc2/.cache/torch"
mkdir -p $HF_HOME
mkdir -p $TORCH_HOME
echo "Cache directories set/created:"
echo "HF_HOME: $HF_HOME"
echo "TORCH_HOME: $TORCH_HOME"


# --- Run Training Script ---
echo "Running sae_train.py..."

# MODIFY ARGUMENTS AS NEEDED (Especially --l1_coeff!)
python scripts/sae_train.py \
    --activation_dir "./llama_activations_train" \
    --save_dir "./sae_models_layer24" \
    --l1_coeff 1e-3 \
    --epochs 50 \
    --batch_size_sae 16 \
    --lr 1e-4 \
    --checkpoint_freq 5 \
    --log_to_wandb \
    --wandb_project "llama3-imdb-sae-layer24" \
    # --wandb_entity "your_wandb_username" # Optional: Add your W&B username if using wandb

echo "Python script finished with exit code $?."

# --- Cleanup ---
conda deactivate
echo "Conda environment deactivated."

end_time=$(date +%s)
runtime=$((end_time - start_time))
echo "Job finished at $(date)"
echo "Total runtime: ${runtime} seconds ($(($runtime / 3600)) hours)"
# --- End of Script ---
