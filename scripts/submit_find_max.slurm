#!/bin/bash -l
#SBATCH --job-name=FindMaxAct         # Job name base
#SBATCH --output=FindMaxAct_F%a_%A.out # Output file pattern: FindMaxAct_F<TASK_ID>_<ARRAY_JOB_ID>.out
#SBATCH --error=FindMaxAct_F%a_%A.err  # Error file pattern: FindMaxAct_F<TASK_ID>_<ARRAY_JOB_ID>.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4           # Request CPUs (adjust if data loading is heavy)
#SBATCH --mem=64G                   # Memory needed to load activations + SAE model
#SBATCH --partition=gpu             # <<< VERIFY/CHANGE: Wulver GPU partition
#SBATCH --gres=gpu:a100:1           # <<< VERIFY/CHANGE: Request GPU for faster SAE inference
#SBATCH --account=2025-spring-ds-680-md748-ygc2 # <<< YOUR Account
#SBATCH --qos=standard              # <<< VERIFY/CHANGE: QOS for partition
#SBATCH --time=02:00:00             # <<< Adjust time: 2 hours per feature? Monitor & adjust. Max per job task.
#SBATCH --array=9988,11257,11310,1986,9157,16082,9407,15710,13655,15417,10410,8377,14269,10350,8178,13536,8435,6225,9164,9333 # <<< EDIT THIS: List of feature indices from Epoch 10 output

# --- Wulver Environment Setup ---
echo "Starting Max Activation job on $(hostname)"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID, Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running for Feature Index: ${SLURM_ARRAY_TASK_ID}" # Slurm uses task ID as the feature index here
echo "Running in directory: $(pwd)" # Should be sae_project
start_time=$(date +%s)

echo "Loading modules..."
module purge
module load easybuild
module load Anaconda3/2023.09-0    # <<< VERIFY/CHANGE: Use correct Anaconda module
module load CUDA/12.4.0          # <<< VERIFY/CHANGE: Use CUDA version matching torch

echo "Activating Conda environment..."
conda activate ./envs/nlp_sae_env # Relative to submission dir (sae_project)

# --- Set Caches ---
export HF_HOME="/course/2025/spring/ds/680/md748/ygc2/.cache/huggingface"
export TORCH_HOME="/course/2025/spring/ds/680/md748/ygc2/.cache/torch"
mkdir -p $HF_HOME
mkdir -p $TORCH_HOME

# --- Run Interpretation Script ---
echo "Running find_max_activations.py for Feature ${SLURM_ARRAY_TASK_ID}..."

# --- CHOOSE CHECKPOINT TO ANALYZE ---
CHECKPOINT_FILE="./sae_models_layer24/sae_checkpoint_epoch_010.pt" # <<< USING EPOCH 10

# Define output directory and file based on feature index
RESULTS_DIR="./results"
mkdir -p ${RESULTS_DIR} # Ensure results directory exists
OUTPUT_FILENAME="${RESULTS_DIR}/feature_${SLURM_ARRAY_TASK_ID}_max_activating.txt"

python scripts/find_max_activations.py \
    --activation_dir "./llama_activations_train" \
    --sae_checkpoint_path "${CHECKPOINT_FILE}" \
    --feature_index ${SLURM_ARRAY_TASK_ID} \
    --activation_dim 4096 \
    --sae_expansion_factor 4 \
    --batch_size_llama 4 \
    --max_seq_len 512 \
    --dataset_id "imdb" \
    --dataset_split "train" \
    --tokenizer_id "meta-llama/Llama-3.1-8B-Instruct" \
    --top_n 100 \
    --context_tokens 15 \
    --device "cuda" \
    --output_file "${OUTPUT_FILENAME}"

echo "Python script finished with exit code $?."

# --- Cleanup ---
conda deactivate
echo "Conda environment deactivated."

end_time=$(date +%s)
runtime=$((end_time - start_time))
echo "Job finished at $(date)"
echo "Total runtime: ${runtime} seconds"
# --- End of Script ---
