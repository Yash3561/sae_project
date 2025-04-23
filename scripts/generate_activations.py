# generate_activations.py (with DEBUG prints and CORRECTED saving logic)
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from huggingface_hub import login
import gc
import warnings
import os
import glob
from tqdm import tqdm # Use standard tqdm for scripts
import argparse
import traceback

print("--- generate_activations.py starting ---")

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Generate Llama 3 activations for SAE training.")
parser.add_argument('--model_id', type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Hugging Face model ID")
parser.add_argument('--dataset_id', type=str, default="imdb", help="Dataset ID")
parser.add_argument('--dataset_split', type=str, default="train", help="Dataset split to process (e.g., 'train', 'test', 'train+test')")
parser.add_argument('--layer_num', type=int, default=24, help="Target Llama layer number")
parser.add_argument('--hook_activation_dim', type=int, default=4096, help="Expected activation dimension from the hook")
parser.add_argument('--num_examples', type=int, default=-1, help="Number of examples to process (-1 for all in split)")
parser.add_argument('--batch_size_llama', type=int, default=4, help="Batch size for Llama 3 inference")
parser.add_argument('--save_dir', type=str, default="./llama_activations", help="Directory to save activation files")
parser.add_argument('--max_seq_len', type=int, default=512, help="Max sequence length for tokenizer")
parser.add_argument('--hf_token', type=str, default="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", help="Hugging Face Access Token")
args = parser.parse_args()

# --- Derived Configuration ---
TARGET_LLAMA_LAYER_PATH = f"model.layers[{args.layer_num}].mlp"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(args.save_dir, exist_ok=True)

print("\n--- Configuration ---")
print(f"Model ID: {args.model_id}")
print(f"Dataset ID: {args.dataset_id}")
print(f"Dataset Split: {args.dataset_split}")
print(f"Target Layer: {args.layer_num} (Path: {TARGET_LLAMA_LAYER_PATH})")
print(f"Hook Activation Dim: {args.hook_activation_dim}")
print(f"Number of Examples: {'All' if args.num_examples == -1 else args.num_examples}")
print(f"Llama Batch Size: {args.batch_size_llama}")
print(f"Max Sequence Length: {args.max_seq_len}")
print(f"Activation Save Dir: {args.save_dir}")
print(f"Device: {DEVICE}")
print(f"Using HF Token: {'Yes' if args.hf_token else 'No'}")
print("-" * 30)

if DEVICE == torch.device("cpu"):
    warnings.warn("Running on CPU is extremely slow!")

# --- Hugging Face Login ---
if args.hf_token:
    print("Attempting Hugging Face login...")
    try:
        login(token=args.hf_token)
        print("Hugging Face login successful.")
    except Exception as e:
        print(f"Hugging Face login failed: {e}")
else:
    print("Skipping Hugging Face login.")

# --- Load Dataset ---
print(f"\nLoading dataset {args.dataset_id}, split '{args.dataset_split}'...")
try:
    if "+" in args.dataset_split: # Handle combined splits like "train+test"
        split_parts = args.dataset_split.split('+')
        print(f"Loading and concatenating splits: {split_parts}")
        loaded_datasets = [load_dataset(args.dataset_id, split=part) for part in split_parts]
        # Use concatenate_datasets if available, requires datasets>=1.6.0
        from datasets import concatenate_datasets
        full_dataset = concatenate_datasets(loaded_datasets)
    else:
        full_dataset = load_dataset(args.dataset_id, split=args.dataset_split)
    print(f"Dataset loaded successfully. Number of examples: {len(full_dataset)}")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# Determine number of examples to process
num_to_process = len(full_dataset) if args.num_examples == -1 else min(args.num_examples, len(full_dataset))
print(f"Will process {num_to_process} examples.")

# --- Load Model & Tokenizer ---
print("\nLoading Llama 3 model and tokenizer (4-bit quantized)...")
model = None
tokenizer = None
gc.collect()
torch.cuda.empty_cache()
try:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Tokenizer pad_token set to eos_token.")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval() # Set to evaluation mode
    print("Llama 3 Model and Tokenizer Loaded Successfully.")
except Exception as e:
    print(f"Error loading Llama 3 model or tokenizer: {e}")
    exit(1)

# --- Prompt Function ---
def create_instruct_prompt(review_text):
    # Simplified prompt for efficiency, assuming model understands sentiment task
    messages = [
        {"role": "system", "content": "Analyze the sentiment of the following movie review."},
        {"role": "user", "content": f"Review:\n{review_text}"}
    ]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Trim prompt slightly if it risks exceeding max_seq_len just due to template overhead
    # This is a heuristic, might need adjustment
    estimated_overhead = 50 # Estimate tokens for template
    max_review_len = args.max_seq_len - estimated_overhead
    if len(tokenizer.encode(review_text)) > max_review_len:
         # Simple truncation (better methods exist, e.g., keep start/end)
         truncated_review = tokenizer.decode(tokenizer.encode(review_text)[:max_review_len])
         messages[1]['content'] = f"Review:\n{truncated_review}"
         prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return prompt_text

# --- Hook Setup ---
print("\n--- Setting up Activation Hook ---")
activation_store = []
hook_handle = None

# <<< Includes DEBUG prints inside the hook >>>
def get_activation_hook(module, input, output):
    """Hook function to capture layer output activations."""
    # --- ADD PRINT ---
    print(f"--- DEBUG: Hook fired for module {module.__class__.__name__}! ---")
    activation_data = output[0] if isinstance(output, tuple) else output
    # Ensure it's a tensor before detaching
    if isinstance(activation_data, torch.Tensor):
        activation_data = activation_data.detach().to("cpu", dtype=torch.float32)
        if activation_data.shape[-1] != args.hook_activation_dim:
            warnings.warn(f"Captured activation dimension ({activation_data.shape[-1]}) != expected ({args.hook_activation_dim})!")
        activation_store.append(activation_data)
        # --- ADD PRINT ---
        print(f"--- DEBUG: Appended activation. activation_store size now: {len(activation_store)} ---")
    else:
        print(f"--- DEBUG: Hook output was not a tensor, type: {type(activation_data)}. Skipping append. ---")


target_module = None
hook_registered = False
try:
    current_module = model
    # Corrected path navigation assuming model object structure
    for part in TARGET_LLAMA_LAYER_PATH.split('.'):
        if part.isdigit():
             current_module = current_module[int(part)] # Should not happen with corrected path
        elif '[' in part and ']' in part:
             module_name = part[:part.find('[')]
             index = int(part[part.find('[')+1:part.find(']')])
             current_module = getattr(current_module, module_name)[index]
        else:
             current_module = getattr(current_module, part)
    target_module = current_module

    assert target_module is not None, "Target module not found!"
    hook_handle = target_module.register_forward_hook(get_activation_hook)
    hook_registered = True
    print(f"Successfully found and hooked module: {target_module.__class__.__name__} at {TARGET_LLAMA_LAYER_PATH}")

except Exception as e:
    print(f"!!! Error finding/hooking target module at path '{TARGET_LLAMA_LAYER_PATH}': {e}")
    print("Check layer path and model structure.")
    traceback.print_exc()
    exit(1)


# --- Process Data and Extract Activations ---
if hook_registered and model is not None:
    print(f"\nStarting activation generation loop for {num_to_process} examples...")
    print(f"Saving activations to: {args.save_dir}")

    # Clear existing activation files if any (optional, prevents mixing runs)
    existing_files = glob.glob(os.path.join(args.save_dir, "activations_batch_*.pt"))
    if existing_files:
        print(f"Deleting {len(existing_files)} existing activation files from previous runs...")
        for f in existing_files:
            try:
                os.remove(f)
            except OSError as e:
                print(f"Warning: Could not delete file {f}: {e}")

    processed_count = 0
    batch_num = 0
    pbar = tqdm(total=num_to_process, desc="Generating Activations")
    try:
        for i in range(0, num_to_process, args.batch_size_llama):
            batch_indices = range(i, min(i + args.batch_size_llama, num_to_process))
            if not batch_indices: break

            # Prepare batch prompts
            current_batch_size = len(batch_indices)
            texts = [create_instruct_prompt(full_dataset[j]['text']) for j in batch_indices]

            # Tokenize batch
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_seq_len).to(DEVICE)

            activation_store = [] # Clear store for THIS specific batch
            with torch.no_grad():
                # Run model forward pass
                try:
                    # <<< Includes DEBUG prints before/after model run >>>
                    print(f"\n--- DEBUG: Running model(**inputs) for batch starting at {i}, current_batch_size {current_batch_size} ---")
                    output = model(**inputs) # Run inference
                    print(f"--- DEBUG: Finished model(**inputs) for batch starting at {i} ---")

                except Exception as e:
                    print(f"\nError during model inference for batch starting at index {i}: {e}")
                    print("Skipping batch.")
                    pbar.update(current_batch_size)
                    continue # Skip to next batch

            # <<< Includes DEBUG print after model run >>>
            print(f"--- DEBUG: After model run for batch {batch_num}, activation_store size is: {len(activation_store)} ---")

            # Process and save captured activations for this batch
            # <<< Using the **CORRECTED** saving logic >>>
            if activation_store:
                # --- CORRECTED SAVING LOGIC ---
                try:
                    # activation_store should contain ONE tensor of shape [batch_size, seq_len, dim]
                    if len(activation_store) == 1:
                        batch_activations = activation_store[0] # Get the single tensor from the list

                        # Check the dimension we care about
                        if batch_activations.shape[-1] != args.hook_activation_dim:
                            print(f"Warning: Dimension mismatch in captured activation for batch {batch_num}! Got shape {batch_activations.shape}, expected dim {args.hook_activation_dim}. Skipping save.")
                        elif batch_activations.nelement() == 0:
                            print(f"Warning: Captured empty tensor for batch {batch_num}. Skipping save.")
                        else:
                            # Save the activation tensor (should be shape [~4, seq_len, 4096])
                            save_path = os.path.join(args.save_dir, f"activations_batch_{batch_num:05d}.pt")
                            torch.save(batch_activations, save_path)
                            # Log the actual shape saved
                            print(f"--- SAVED: batch {batch_num} activations to {save_path} with shape {batch_activations.shape} ---") # Print actual shape
                            batch_num += 1
                    else:
                        # This case should ideally not happen now, but good to log
                        print(f"Warning: activation_store size is {len(activation_store)} (expected 1) for batch {batch_num}. Skipping save.")

                except Exception as e:
                     print(f"Error retrieving or saving activations for batch {batch_num}: {e}")
                     traceback.print_exc()
                # --- END OF CORRECTED SAVING LOGIC ---

            else:
                 # This warning is still relevant if the hook truly captures nothing
                 print(f"Warning: No activations captured for batch starting at index {i} (activation_store empty). Hook might not be working correctly.")
            # <<< END OF section with corrected saving logic >>>

            processed_count += current_batch_size
            pbar.update(current_batch_size)

            # Optional: Clear CUDA cache less frequently to potentially speed up loop
            if batch_num % 50 == 0 and batch_num > 0: # Changed frequency
                 gc.collect()
                 if DEVICE == torch.device('cuda'): torch.cuda.empty_cache()

        pbar.close()
        print(f"\nFinished processing {processed_count} examples.")
        print(f"Saved {batch_num} activation files to {args.save_dir}")

    except Exception as e:
        print(f"\nError during activation generation loop: {e}")
        traceback.print_exc()
    finally:
        # --- Remove Hook ---
        if hook_handle:
            hook_handle.remove()
            print("Forward hook removed.")
else:
    print("\nActivation generation skipped as hook was not registered or model not loaded.")

print("\n--- generate_activations.py finished ---")