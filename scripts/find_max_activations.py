# scripts/find_max_activations.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer # Only need tokenizer here
from datasets import load_dataset
import gc
import os
import glob
from tqdm import tqdm
import argparse
import traceback
import heapq # For efficient top-N tracking

print("--- find_max_activations.py starting ---")

# --- Define SAE Class (Must match training) ---
# This is needed to load the state_dict correctly
class SimpleSAE(nn.Module):
    def __init__(self, d_in, d_sae, device=None): # No need for l1_coeff at inference
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Ensure bias=True matches how it was trained (nn.Linear default is True)
        self.encoder = nn.Linear(d_in, d_sae, bias=True, device=self.device)
        self.decoder = nn.Linear(d_sae, d_in, bias=True, device=self.device)
        self.to(self.device)

    def forward(self, x):
        with torch.no_grad():
            # Move input to device and ensure float32
            x_float = x.to(device=self.device, dtype=torch.float32)
            encoded_features = self.encoder(x_float)
            feature_acts = F.relu(encoded_features)
        return feature_acts
print("SimpleSAE class defined.")

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Find Max Activating Examples for an SAE feature.")
# Required Args
parser.add_argument('--activation_dir', type=str, required=True, help="Directory containing TRAIN activation files (*.pt)")
parser.add_argument('--sae_checkpoint_path', type=str, required=True, help="Path to the trained SAE checkpoint (.pt file)")
parser.add_argument('--feature_index', type=int, required=True, help="Index of the SAE feature to analyze")
# Config Args (must match SAE training and activation generation)
parser.add_argument('--activation_dim', type=int, default=4096, help="Input dimension of activations")
parser.add_argument('--sae_expansion_factor', type=int, default=4, help="Expansion factor used for SAE training")
parser.add_argument('--batch_size_llama', type=int, default=4, help="Batch size used during Llama 3 activation generation")
parser.add_argument('--max_seq_len', type=int, default=512, help="Max sequence length used during activation generation")
# Analysis Args
parser.add_argument('--top_n', type=int, default=20, help="Number of top activating examples to find")
parser.add_argument('--context_tokens', type=int, default=10, help="Number of tokens before/after the max activating token to show")
# Other Args
parser.add_argument('--dataset_id', type=str, default="imdb", help="Dataset ID (must match activation source)")
parser.add_argument('--dataset_split', type=str, default="train", help="Dataset split activations were generated from")
parser.add_argument('--tokenizer_id', type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Tokenizer ID")
parser.add_argument('--device', type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use ('cuda' or 'cpu')")
parser.add_argument('--output_file', type=str, default=None, help="Optional file to save results (defaults to printing to standard out)")


args = parser.parse_args()

# --- Derived Configuration ---
SAE_INPUT_DIM = args.activation_dim
SAE_DIM = SAE_INPUT_DIM * args.sae_expansion_factor
DEVICE = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
TARGET_FEATURE_INDEX = args.feature_index

print("\n--- Configuration ---")
print(f"Activation Dir: {args.activation_dir}")
print(f"SAE Checkpoint: {args.sae_checkpoint_path}")
print(f"Feature Index to Analyze: {args.feature_index}")
print(f"Activation Dim: {SAE_INPUT_DIM}")
print(f"SAE Feature Dim (d_sae): {SAE_DIM}")
print(f"Batch Size (Llama): {args.batch_size_llama}")
print(f"Max Sequence Length: {args.max_seq_len}")
print(f"Top N Examples: {args.top_n}")
print(f"Context Tokens: {args.context_tokens}")
print(f"Device: {DEVICE}")
print("-" * 30)

# Validate feature index
if not (0 <= TARGET_FEATURE_INDEX < SAE_DIM):
    print(f"!!! ERROR: feature_index {TARGET_FEATURE_INDEX} is out of bounds for SAE dimension {SAE_DIM}")
    exit(1)

# --- Load SAE Checkpoint ---
print(f"\nLoading SAE checkpoint from: {args.sae_checkpoint_path}")
sae_model = SimpleSAE(d_in=SAE_INPUT_DIM, d_sae=SAE_DIM, device=DEVICE)
try:
    checkpoint = torch.load(args.sae_checkpoint_path, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        sae_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded SAE model state_dict from epoch {checkpoint.get('epoch', 'N/A')}")
    elif 'state_dict' in checkpoint: # Handle other common checkpoint formats
         sae_model.load_state_dict(checkpoint['state_dict'])
         print(f"Loaded SAE model state_dict from epoch {checkpoint.get('epoch', 'N/A')}")
    else:
        sae_model.load_state_dict(checkpoint)
        print("Loaded SAE state_dict directly from file.")
    sae_model.eval()
    print("SAE model loaded and set to eval mode.")
except FileNotFoundError:
    print(f"!!! ERROR: Checkpoint file not found at {args.sae_checkpoint_path}")
    exit(1)
except Exception as e:
    print(f"!!! ERROR: Failed to load SAE checkpoint: {e}")
    traceback.print_exc()
    exit(1)

# --- Load Tokenizer & Dataset Text ---
print("\nLoading tokenizer and dataset text...")
try:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    full_dataset_text = load_dataset(args.dataset_id, split=args.dataset_split)['text']
    print(f"Loaded text for {len(full_dataset_text)} examples from split '{args.dataset_split}'.")
except Exception as e:
    print(f"Error loading tokenizer or dataset text: {e}")
    exit(1)

# --- Process Activations and Find Top Examples ---
print(f"\nSearching for Top {args.top_n} activations for Feature {TARGET_FEATURE_INDEX}...")
activation_file_pattern = os.path.join(args.activation_dir, "activations_batch_*.pt")
activation_files = sorted(glob.glob(activation_file_pattern))

if not activation_files:
    print(f"!!! ERROR: No activation files found in {args.activation_dir}")
    exit(1)

print(f"Found {len(activation_files)} activation files to process.")

# Use a min-heap to keep track of the top N activations efficiently
# Store tuples: (activation_value, global_token_idx, filename, local_token_idx_in_file)
top_activations_heap = []

global_token_idx_offset = 0 # Keep track of cumulative tokens processed

pbar = tqdm(enumerate(activation_files), total=len(activation_files), desc="Processing Files")
for batch_file_idx, file_path in pbar:
    try:
        # Load activations directly to target device if enough RAM, else CPU then move
        # Loading directly to CPU is safer for large datasets if node RAM > GPU VRAM
        batch_activations = torch.load(file_path, map_location="cpu") # Shape [batch, seq, dim]
        batch_size, seq_len, act_dim = batch_activations.shape

        if act_dim != args.activation_dim:
            print(f"Warning: Skipping file {file_path}, activation dim {act_dim} != expected {args.activation_dim}")
            num_tokens_in_file = batch_size * seq_len
            global_token_idx_offset += num_tokens_in_file
            continue

        # Reshape and move to device for SAE inference
        num_tokens_in_file = batch_size * seq_len
        flattened_activations = batch_activations.view(num_tokens_in_file, act_dim).to(DEVICE)
        del batch_activations; # Free CPU memory

        # Get SAE feature activations
        feature_acts = sae_model(flattened_activations) # Shape [tokens, sae_dim]
        target_feature_acts = feature_acts[:, TARGET_FEATURE_INDEX].to("cpu") # Move results back to CPU
        del flattened_activations, feature_acts; # Free GPU memory

        # Find max activation *within this batch* for the target feature
        max_val_in_batch, local_token_idx_in_batch = torch.max(target_feature_acts, dim=0)
        max_val_item = max_val_in_batch.item()
        local_token_idx_item = local_token_idx_in_batch.item()

        # Convert local batch token index to global token index
        global_token_idx = global_token_idx_offset + local_token_idx_item

        # Update the min-heap
        heap_item = (max_val_item, global_token_idx, file_path, local_token_idx_item)
        if len(top_activations_heap) < args.top_n:
            heapq.heappush(top_activations_heap, heap_item)
        elif max_val_item > top_activations_heap[0][0]:
            heapq.heapreplace(top_activations_heap, heap_item)

        # Update the global token offset for the next file
        global_token_idx_offset += num_tokens_in_file

        if (batch_file_idx + 1) % 100 == 0: # Print progress
             pbar.set_postfix({"Min Top Act": f"{top_activations_heap[0][0]:.4f}" if top_activations_heap else "N/A"})
             gc.collect() # Collect garbage periodically
             if DEVICE == torch.device('cuda'): torch.cuda.empty_cache()


    except Exception as e:
        print(f"\nError processing file {file_path}: {e}")
        traceback.print_exc()

pbar.close()
print(f"\nFinished processing activation files. Found {len(top_activations_heap)} potential top activations.")

# Sort the heap to get top N in descending order
top_activations_sorted = sorted(top_activations_heap, key=lambda x: x[0], reverse=True)

# --- Retrieve Text Context for Top N ---
print(f"\nRetrieving text context for Top {len(top_activations_sorted)} Activations...")

results_output = []
results_output.append(f"--- Max Activating Examples for Feature {TARGET_FEATURE_INDEX} ---")
results_output.append(f"SAE Checkpoint: {args.sae_checkpoint_path}")
results_output.append("-" * 60)

# Helper function to map global token index back to example and token index within example
# This is complex because activation files store flattened batches.
def map_global_token_to_example(global_token_idx, dataset_len, act_files_list, llama_gen_batch_size, max_seq_len_gen, tokenizer_for_len):
    """Maps a global token index back to its original example index and token index within that example."""
    tokens_processed = 0
    for file_idx, file_path in enumerate(act_files_list):
        # Calculate how many tokens this file represents WITHOUT loading it if possible
        # Need to know how many examples were in the batch that created this file
        start_example_idx_for_file = file_idx * llama_gen_batch_size
        end_example_idx_for_file = min(start_example_idx_for_file + llama_gen_batch_size, dataset_len)
        num_examples_in_file_batch = end_example_idx_for_file - start_example_idx_for_file

        if num_examples_in_file_batch <= 0: continue

        # Calculate (approx) tokens in this file batch by re-tokenizing (or load shape)
        # Re-tokenizing is safer if padding varied, but slower. Loading shape is faster.
        try:
            # OPTIMIZATION: Could try loading just the shape?
            # temp_acts = torch.load(file_path, map_location='cpu')
            # num_tokens_in_file = temp_acts.shape[0] * temp_acts.shape[1] # batch * seq_len
            # del temp_acts

            # Re-tokenize to get precise count including padding/truncation for this specific batch
            batch_texts = full_dataset_text[start_example_idx_for_file:end_example_idx_for_file]
            inputs = tokenizer_for_len(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_len_gen)
            num_tokens_in_file = inputs['input_ids'].numel() # Total tokens including padding

        except Exception as e:
            print(f"Warning: Error getting token count for file {file_path}: {e}. Skipping.")
            continue # Cannot reliably map if token count is unknown

        if tokens_processed <= global_token_idx < tokens_processed + num_tokens_in_file:
            # The token is in this file's batch
            local_flat_idx = global_token_idx - tokens_processed
            padded_seq_len = inputs['input_ids'].shape[1] # The actual seq length used for this batch (with padding)

            example_idx_in_batch = local_flat_idx // padded_seq_len
            token_idx_in_example_seq = local_flat_idx % padded_seq_len

            original_example_idx = start_example_idx_for_file + example_idx_in_batch
            return original_example_idx, token_idx_in_example_seq

        tokens_processed += num_tokens_in_file

    return None, None # Not found

# Iterate through sorted top activations
for rank, (act_value, global_token_idx, source_file, _) in enumerate(top_activations_sorted):
    print(f"\nProcessing Rank {rank+1}/{len(top_activations_sorted)} | Activation: {act_value:.4f} | Global Index: {global_token_idx}")

    example_idx, token_idx_in_seq = map_global_token_to_example(
        global_token_idx, len(full_dataset_text), activation_files,
        args.batch_size_llama, args.max_seq_len, tokenizer
    )

    if example_idx is not None:
        print(f"  Mapped to Example Index: {example_idx}, Token Index in Sequence: {token_idx_in_seq}")
        try:
            # Get original text and re-tokenize *just this example*
            target_text = full_dataset_text[example_idx]
            target_input_ids = tokenizer(target_text, return_tensors="pt", truncation=True, max_length=args.max_seq_len)['input_ids'][0]

            if token_idx_in_seq >= len(target_input_ids):
                 print(f"  Warning: Mapped token index ({token_idx_in_seq}) is out of bounds for re-tokenized example length ({len(target_input_ids)}). Might be padding.")
                 # Still record basic info
                 results_output.append(f"Rank {rank+1}/{len(top_activations_sorted)} | Activation: {act_value:.4f} | Example Index: {example_idx} | Token Index: {token_idx_in_seq} (Likely Padding)")
                 results_output.append("-" * 20)
                 continue

            # Extract context
            start_ctx = max(0, token_idx_in_seq - args.context_tokens)
            end_ctx = min(len(target_input_ids), token_idx_in_seq + args.context_tokens + 1)
            context_ids = target_input_ids[start_ctx:end_ctx]
            activating_token_id = target_input_ids[token_idx_in_seq]

            # Decode for display
            activating_token_text = tokenizer.decode(activating_token_id)
            # Careful decoding context to handle partial words / special tokens
            context_text_before = tokenizer.decode(target_input_ids[start_ctx:token_idx_in_seq])
            context_text_after = tokenizer.decode(target_input_ids[token_idx_in_seq+1:end_ctx])

            # Represent clearly
            context_str = f"...{context_text_before} >>>{activating_token_text}<<< {context_text_after}..."
            # Clean up potential extra spaces around the highlight
            context_str = context_str.replace("  >>>", " >>>").replace("<<<  ", "<<< ")

            print(f"  Activating Token: '{activating_token_text}' (ID: {activating_token_id})")
            print(f"  Context: {context_str}")

            results_output.append(f"Rank {rank+1}/{len(top_activations_sorted)} | Activation: {act_value:.4f} | Example Index: {example_idx} | Token Index: {token_idx_in_seq}")
            results_output.append(f"Token: '{activating_token_text}'")
            results_output.append(f"Context: {context_str}")
            results_output.append("-" * 20)

        except Exception as e:
            print(f"  Error retrieving/decoding context for example {example_idx}: {e}")
            results_output.append(f"Rank {rank+1}/{len(top_activations_sorted)} | Activation: {act_value:.4f} | Example Index: {example_idx} | Token Index: {token_idx_in_seq} | Error decoding context")
            results_output.append("-" * 20)
    else:
         print(f"  Warning: Could not map global token index {global_token_idx} back to an example.")
         results_output.append(f"Rank {rank+1}/{len(top_activations_sorted)} | Activation: {act_value:.4f} | Global Index: {global_token_idx} | Error: Could not map to example")
         results_output.append("-" * 20)

# --- Save or Print Results ---
if args.output_file:
    try:
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for line in results_output:
                f.write(line + '\n')
        print(f"\nResults saved to {args.output_file}")
    except Exception as e:
        print(f"\nError saving results to file '{args.output_file}': {e}")
        print("\n--- Results (Fallback Print) ---")
        for line in results_output: print(line)
else:
    # Default to printing if no output file specified
    print("\n--- Results ---")
    for line in results_output: print(line)


print("\n--- find_max_activations.py finished ---")
