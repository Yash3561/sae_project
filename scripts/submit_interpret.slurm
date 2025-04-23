#!/bin/bash -l
#SBATCH --job-name=SAE_Interpret_v2
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G                  # Memory to load Llama3 + SAE
#SBATCH --partition=gpu            # <<< VERIFY/CHANGE: Wulver GPU partition
#SBATCH --gres=gpu:a100:1          # <<< VERIFY/CHANGE: Wulver GPU request
#SBATCH --account=2025-spring-ds-680-md748-ygc2 # <<< YOUR Account
#SBATCH --qos=standard             # <<< VERIFY/CHANGE: QOS for partition
#SBATCH --time=00:30:00            # <<< Adjust time: 30 mins should be enough for a few examples

# --- Wulver Environment Setup ---
echo "Starting Interpretation job on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running in directory: $(pwd)" # Should be sae_project
start_time=$(date +%s)

echo "Loading modules..."
module purge
module load easybuild              # Load easybuild first
module load Anaconda3/2023.09-0    # <<< VERIFY/CHANGE: Use correct Anaconda module
module load CUDA/12.4.0          # <<< VERIFY/CHANGE: Use CUDA version matching torch install

echo "Activating Conda environment..."
conda activate ./envs/nlp_sae_env # Relative to submission dir (sae_project)

# --- Set Caches (Good Practice) ---
export HF_HOME="/course/2025/spring/ds/680/md748/ygc2/.cache/huggingface"
export TORCH_HOME="/course/2025/spring/ds/680/md748/ygc2/.cache/torch"
mkdir -p $HF_HOME
mkdir -p $TORCH_HOME

# --- Run Interpretation Script ---
echo "Running interpret_sae.py..."

# --- CHOOSE CHECKPOINT TO ANALYZE ---
# Specify the path to the checkpoint you want to load (e.g., epoch 10 or 15)
CHECKPOINT_FILE="./sae_models_layer24/sae_checkpoint_epoch_010.pt" # <<< MODIFY AS NEEDED

python scripts/interpret_sae.py \
    --sae_checkpoint_path "${CHECKPOINT_FILE}" \
    --num_examples 5 \
    --top_k_features 10 \
    --analysis_mode compare_examples

echo "Python script finished with exit code $?."

# --- Cleanup ---
conda deactivate
echo "Conda environment deactivated."

end_time=$(date +%s)
runtime=$((end_time - start_time))
echo "Job finished at $(date)"
echo "Total runtime: ${runtime} seconds"
# --- End of Script ---
