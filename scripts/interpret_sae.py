# scripts/interpret_sae.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
import gc
import os
import traceback
import warnings
import argparse
import random

print("--- interpret_sae.py starting ---")

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Interpret SAE features for specific examples.")
# Model/Data Args
parser.add_argument('--model_id', type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Hugging Face model ID")
parser.add_argument('--dataset_id', type=str, default="imdb", help="Dataset ID for examples")
parser.add_argument('--layer_num', type=int, default=24, help="Target Llama layer number (must match SAE training)")
parser.add_argument('--hook_activation_dim', type=int, default=4096, help="Expected activation dimension")
parser.add_argument('--max_seq_len', type=int, default=512, help="Max sequence length for tokenizer")
# SAE Args
parser.add_argument('--sae_checkpoint_path', type=str, required=True, help="Path to the trained SAE checkpoint (.pt file)")
parser.add_argument('--sae_expansion_factor', type=int, default=4, help="Expansion factor used when training the SAE")
# Interpretation Args
parser.add_argument('--num_examples', type=int, default=5, help="Number of random positive/negative examples to analyze")
parser.add_argument('--top_k_features', type=int, default=15, help="Number of top activating features to display per example")
parser.add_argument('--analysis_mode', type=str, default="compare_examples", choices=["compare_examples", "max_activating"], help="Type of analysis to perform")
# TODO: Add args for max_activating mode (feature_index, num_search_examples etc.)

args = parser.parse_args()

# --- Derived Configuration ---
TARGET_LLAMA_LAYER_PATH = f"model.layers[{args.layer_num}].mlp"
SAE_INPUT_DIM = args.hook_activation_dim
SAE_DIM = SAE_INPUT_DIM * args.sae_expansion_factor
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\n--- Configuration ---")
print(f"Model ID: {args.model_id}")
print(f"Dataset ID: {args.dataset_id}")
print(f"Target Layer: {args.layer_num} (Path: {TARGET_LLAMA_LAYER_PATH})")
print(f"Hook Activation Dim: {args.hook_activation_dim}")
print(f"SAE Checkpoint: {args.sae_checkpoint_path}")
print(f"SAE Expansion Factor: {args.sae_expansion_factor}")
print(f"SAE Feature Dim (d_sae): {SAE_DIM}")
print(f"Num Examples to Analyze: {args.num_examples}")
print(f"Top K Features: {args.top_k_features}")
print(f"Analysis Mode: {args.analysis_mode}")
print(f"Device: {DEVICE}")
print("-" * 30)

# --- Define SAE Class ---
class SimpleSAE(nn.Module):
    def __init__(self, d_in, d_sae, device=None): # No need for l1_coeff at inference
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = nn.Linear(d_in, d_sae, device=self.device)
        self.decoder = nn.Linear(d_sae, d_in, device=self.device)
        self.to(self.device)

    def forward(self, x):
        with torch.no_grad():
            x_float = x.to(torch.float32).to(self.device)
            encoded_features = self.encoder(x_float)
            feature_acts = F.relu(encoded_features)
        return feature_acts
print("SimpleSAE class defined.")

# --- Load SAE Checkpoint ---
print(f"\nLoading SAE checkpoint from: {args.sae_checkpoint_path}")
sae_model = SimpleSAE(d_in=SAE_INPUT_DIM, d_sae=SAE_DIM, device=DEVICE)
try:
    checkpoint = torch.load(args.sae_checkpoint_path, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        sae_model.load_state_dict(checkpoint['model_state_dict'])
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

# --- Load Base Model & Tokenizer ---
# Note: Loading this large model takes time and memory
print("\nLoading base Llama 3 model and tokenizer (4-bit)...")
model = None
tokenizer = None
gc.collect(); torch.cuda.empty_cache()
try:
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, quantization_config=quantization_config,
        torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    print("Base Llama 3 Model and Tokenizer Loaded Successfully.")
except Exception as e:
    print(f"!!! Error loading Llama 3 model or tokenizer: {e}")
    traceback.print_exc()
    exit(1)

# --- Hook Setup ---
print("\nSetting up hook for interpretation...")
activation_store = [] # Global store for the hook
hook_handle = None
def get_activation_hook_interpret(module, input, output):
    activation_data = output[0] if isinstance(output, tuple) else output
    activation_store.append(activation_data.detach()) # Keep on GPU

try:
    current_module = model
    for part in TARGET_LLAMA_LAYER_PATH.split('.'):
        if part.isdigit(): raise ValueError("Digits should not be in path parts")
        elif '[' in part and ']' in part:
             module_name = part[:part.find('[')]
             index = int(part[part.find('[')+1:part.find(']')])
             current_module = getattr(current_module, module_name)[index]
        else: current_module = getattr(current_module, part)
    target_module = current_module
    assert target_module is not None, "Target module not found!"
    hook_handle = target_module.register_forward_hook(get_activation_hook_interpret)
    print(f"Hook registered on {target_module.__class__.__name__} at {TARGET_LLAMA_LAYER_PATH}")
except Exception as e:
    print(f"!!! Error registering hook: {e}")
    if hook_handle: hook_handle.remove()
    exit(1)

# --- Prompt Function ---
def create_instruct_prompt(review_text):
    messages = [{"role": "system", "content": "Analyze sentiment."}, {"role": "user", "content": f"Review:\n{review_text}"}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# --- Helper to get SAE Activations ---
def get_sae_activations(prompt_text, llama_model, sae_model_trained, tokenizer_used):
    global activation_store
    activation_store = []
    inputs = tokenizer_used(prompt_text, return_tensors="pt", truncation=True, max_length=args.max_seq_len).to(DEVICE)
    seq_len = inputs['input_ids'].shape[1]

    with torch.no_grad():
        try: outputs = llama_model(**inputs)
        except Exception as e: print(f"Inference error: {e}"); return None
    if not activation_store: print("Error: No activations captured!"); return None

    llama_activations = activation_store[0] # Shape [1, seq_len, act_dim]
    reshaped_activations = llama_activations.view(1 * seq_len, args.hook_activation_dim)
    feature_acts = sae_model_trained(reshaped_activations) # Shape [tokens, sae_dim]
    return feature_acts # Return flat version [tokens, sae_dim]

# --- Load Dataset for Examples ---
print(f"\nLoading dataset {args.dataset_id} for examples...")
try:
    # Load both train and test to sample from
    train_dataset = load_dataset(args.dataset_id, split="train")
    test_dataset = load_dataset(args.dataset_id, split="test")
    # Separate positive and negative examples
    pos_examples = [ex for ex in test_dataset if ex['label'] == 1]
    neg_examples = [ex for ex in test_dataset if ex['label'] == 0]
    print(f"Found {len(pos_examples)} positive and {len(neg_examples)} negative examples in test set.")
except Exception as e:
    print(f"Error loading dataset examples: {e}")
    exit(1)

# --- Analysis ---
if args.analysis_mode == "compare_examples":
    print(f"\n--- Analyzing {args.num_examples} Positive vs Negative Examples ---")

    if not pos_examples or not neg_examples:
        print("Not enough positive/negative examples found in test set.")
        exit(1)

    # Sample random examples
    random.shuffle(pos_examples)
    random.shuffle(neg_examples)
    examples_to_analyze = pos_examples[:args.num_examples] + neg_examples[:args.num_examples]

    results = {'positive': [], 'negative': []}

    for i, example in enumerate(examples_to_analyze):
        label_str = "Positive" if example['label'] == 1 else "Negative"
        print(f"\nProcessing {label_str} Example {i // 2 + 1}/{args.num_examples}...")
        prompt = create_instruct_prompt(example['text'])
        feature_acts = get_sae_activations(prompt, model, sae_model, tokenizer)

        if feature_acts is not None:
            # Calculate summary stats for this example
            avg_l0 = (feature_acts.abs() > 1e-6).float().sum(dim=-1).mean().item()
            sum_acts = feature_acts.sum(dim=0) # Sum activation for each feature across all tokens
            max_acts = feature_acts.max(dim=0)[0] # Max activation for each feature across all tokens

            top_k_sum_vals, top_k_sum_indices = torch.topk(sum_acts, k=args.top_k_features)
            top_k_max_vals, top_k_max_indices = torch.topk(max_acts, k=args.top_k_features)

            print(f"  Avg L0: {avg_l0:.2f}")
            print(f"  Top {args.top_k_features} Features (Sum Activation):")
            for idx, val in zip(top_k_sum_indices.tolist(), top_k_sum_vals.tolist()):
                 print(f"    Feature {idx}: {val:.4f}")
            print(f"  Top {args.top_k_features} Features (Max Activation):")
            for idx, val in zip(top_k_max_indices.tolist(), top_k_max_vals.tolist()):
                 print(f"    Feature {idx}: {val:.4f}")

            # Store results (optional, can just print)
            results[label_str.lower()].append({
                'index_in_split': test_dataset.select([i for i, ex in enumerate(test_dataset) if ex['text'] == example['text']])[0], # Find original index if needed
                'avg_l0': avg_l0,
                'top_sum_features': top_k_sum_indices.tolist(),
                'top_max_features': top_k_max_indices.tolist()
            })
        else:
            print("  Failed to get activations for this example.")

elif args.analysis_mode == "max_activating":
    print("\n--- Max Activating Example Analysis (Placeholder) ---")
    print("This mode requires searching many examples for a specific feature index.")
    print("Please implement this analysis separately or add specific arguments.")
    # TODO: Implement logic to load many activations and find max activating examples for a given feature index.

# --- Clean up hook ---
if hook_handle:
    hook_handle.remove()
    print("\nForward hook removed.")

print("\n--- interpret_sae.py finished ---")
