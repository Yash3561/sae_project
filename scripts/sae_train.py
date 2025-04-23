# train_sae.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gc
import warnings
import os
import glob
from tqdm import tqdm
import argparse
import traceback
import time

# Optional: Import wandb if using it
try:
    import wandb
    wandb_available = True
except ImportError:
    wandb_available = False
    print("wandb not installed, skipping wandb logging.")

print("--- train_sae.py starting ---")

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Train a Sparse Autoencoder (SAE) on Llama activations.")
# Data Args
parser.add_argument('--activation_dir', type=str, required=True, help="Directory containing saved activation files (*.pt)")
parser.add_argument('--save_dir', type=str, default="./sae_checkpoints", help="Directory to save SAE model checkpoints and final model")
# Model Args
parser.add_argument('--activation_dim', type=int, default=4096, help="Input dimension of activations (must match generated files)")
parser.add_argument('--expansion_factor', type=int, default=4, help="SAE dictionary size = activation_dim * expansion_factor")
parser.add_argument('--l1_coeff', type=float, default=1e-3, help="Coefficient for the L1 sparsity penalty (NEEDS TUNING!)")
# Training Args
parser.add_argument('--epochs', type=int, default=20, help="Number of training epochs")
parser.add_argument('--batch_size_sae', type=int, default=8, help="Number of activation files to load per SAE training step (adjust based on GPU VRAM)")
parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate for AdamW optimizer")
parser.add_argument('--checkpoint_freq', type=int, default=1, help="Save checkpoint every N epochs (set to 0 to disable intermediate saves)")
# Logging Args
parser.add_argument('--log_to_wandb', action='store_true', help="Enable logging to Weights & Biases")
parser.add_argument('--wandb_project', type=str, default="llama3_sae_imdb", help="W&B project name")
parser.add_argument('--wandb_entity', type=str, default=None, help="W&B entity (username or team)") # Set if needed

args = parser.parse_args()

# --- Derived Configuration ---
SAE_INPUT_DIM = args.activation_dim
SAE_DIM = SAE_INPUT_DIM * args.expansion_factor
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(args.save_dir, exist_ok=True)

print("\n--- Configuration ---")
print(f"Activation Dir: {args.activation_dir}")
print(f"Save Dir: {args.save_dir}")
print(f"Activation Dim: {SAE_INPUT_DIM}")
print(f"SAE Expansion Factor: {args.expansion_factor}")
print(f"SAE Feature Dim (d_sae): {SAE_DIM}")
print(f"L1 Coefficient: {args.l1_coeff}")
print(f"Epochs: {args.epochs}")
print(f"SAE Batch Size (Files): {args.batch_size_sae}")
print(f"Learning Rate: {args.lr}")
print(f"Checkpoint Freq: {args.checkpoint_freq} epochs")
print(f"Device: {DEVICE}")
print(f"Log to W&B: {args.log_to_wandb and wandb_available}")
if args.log_to_wandb and wandb_available:
    print(f"  W&B Project: {args.wandb_project}")
    print(f"  W&B Entity: {args.wandb_entity or 'default'}")
print("-" * 30)

# --- Weights & Biases Setup (Optional) ---
if args.log_to_wandb and wandb_available:
    try:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity, # Optional: your wandb username or team name
            config=vars(args) # Log hyperparameters
        )
        print("Weights & Biases initialized.")
    except Exception as e:
        print(f"Error initializing wandb: {e}. Disabling wandb logging.")
        args.log_to_wandb = False # Disable if init fails
else:
    args.log_to_wandb = False # Ensure disabled if not requested or available

# --- SAE Model Definition ---
class SimpleSAE(nn.Module):
    # Using the same definition as in Colab Cell 7
    def __init__(self, d_in, d_sae, l1_coeff=0.001, device=None):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.l1_coeff = l1_coeff
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = nn.Linear(d_in, d_sae, device=self.device)
        self.decoder = nn.Linear(d_sae, d_in, device=self.device)
        nn.init.kaiming_uniform_(self.encoder.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.decoder.weight, nonlinearity='relu')
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)
        self.to(self.device)

    def forward(self, x):
        x_float = x.to(torch.float32)
        encoded_features = self.encoder(x_float)
        feature_acts = F.relu(encoded_features)
        sae_out = self.decoder(feature_acts)
        mse_loss = F.mse_loss(sae_out, x_float)
        l1_loss = feature_acts.abs().sum(dim=-1).mean() # Raw L1 loss
        total_loss = mse_loss + self.l1_coeff * l1_loss
        return {
            "sae_out": sae_out, "feature_acts": feature_acts, "loss": total_loss,
            "mse_loss": mse_loss, "l1_loss": l1_loss
        }
print("SimpleSAE class defined.")

# --- Activation Dataset & Collate ---
# Using the same definitions as in Colab Cell 9
class ActivationDataset(Dataset):
    def __init__(self, file_pattern, activation_dim):
        self.file_paths = sorted(glob.glob(file_pattern))
        self.activation_dim = activation_dim
        if not self.file_paths:
            warnings.warn(f"No activation files found matching pattern: {file_pattern}")
        else:
            print(f"Found {len(self.file_paths)} activation files matching {file_pattern}")
            try:
                first_acts = torch.load(self.file_paths[0], map_location='cpu')
                if first_acts.shape[-1] != self.activation_dim:
                     warnings.warn(f"DIM MISMATCH! Files have dim {first_acts.shape[-1]} but expected {self.activation_dim}.")
                del first_acts
            except Exception as e: warnings.warn(f"Could not check first file: {e}")
    def __len__(self): return len(self.file_paths)
    def __getitem__(self, idx):
        try:
            activations = torch.load(self.file_paths[idx], map_location='cpu')
            if activations.shape[-1] != self.activation_dim: return torch.tensor([])
            num_tokens = activations.shape[0] * activations.shape[1]
            flattened_activations = activations.view(num_tokens, self.activation_dim)
            return flattened_activations.to(torch.float32)
        except Exception as e: print(f"Err loading {self.file_paths[idx]}: {e}"); return torch.tensor([])

def custom_collate(batch_list):
    batch_list = [t for t in batch_list if isinstance(t, torch.Tensor) and t.nelement() > 0]
    return torch.cat(batch_list, dim=0) if batch_list else torch.tensor([])
print("ActivationDataset and custom_collate defined.")

# --- DataLoader ---
print("\nInitializing DataLoader...")
activation_file_pattern = os.path.join(args.activation_dir, "activations_batch_*.pt")
activation_dataset = ActivationDataset(activation_file_pattern, SAE_INPUT_DIM)
train_loader = None
if len(activation_dataset) > 0:
    train_loader = DataLoader(
        activation_dataset, batch_size=args.batch_size_sae, shuffle=True,
        num_workers=2, # Can use > 0 workers on cluster
        collate_fn=custom_collate, pin_memory=True # pin_memory=True if loading to GPU
    )
    print(f"DataLoader initialized with {len(train_loader)} batches per epoch.")
else:
    print("!!! Error: No activation files found. Cannot train SAE. Exiting. !!!")
    exit(1)

# --- Instantiate Model and Optimizer ---
print(f"\nInstantiating SAE with d_in={SAE_INPUT_DIM}, d_sae={SAE_DIM}, L1={args.l1_coeff}")
sae_model_to_train = SimpleSAE(d_in=SAE_INPUT_DIM, d_sae=SAE_DIM, l1_coeff=args.l1_coeff, device=DEVICE)
optimizer = optim.AdamW(sae_model_to_train.parameters(), lr=args.lr)
print("SAE model and optimizer instantiated.")

# --- Training Loop ---
print(f"\n--- Starting SAE Training for {args.epochs} epochs ---")
global_step = 0
start_train_time = time.time()

for epoch in range(args.epochs):
    epoch_start_time = time.time()
    sae_model_to_train.train() # Set model to training mode

    total_loss_epoch = 0
    total_mse_loss_epoch = 0
    total_l1_loss_epoch = 0
    total_l0_norm_epoch = 0
    total_tokens_processed_epoch = 0
    num_batches_processed = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
    for batch_tokens in pbar:
        if not isinstance(batch_tokens, torch.Tensor) or batch_tokens.nelement() == 0:
            # print("Warning: Skipping empty batch.") # Can be noisy
            continue

        batch_tokens = batch_tokens.to(DEVICE)
        current_batch_token_count = batch_tokens.shape[0]
        if current_batch_token_count == 0: continue

        try:
            optimizer.zero_grad()
            sae_output_dict = sae_model_to_train(batch_tokens)
            loss = sae_output_dict["loss"]
            mse_loss = sae_output_dict["mse_loss"]
            l1_loss = sae_output_dict["l1_loss"] # Raw L1

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\nWarning: NaN/Inf loss detected! Skipping batch {num_batches_processed}. MSE={mse_loss.item()}, L1={l1_loss.item()}")
                optimizer.zero_grad() # Clear potentially bad gradients
                continue

            loss.backward()
            # Optional: Gradient clipping
            # torch.nn.utils.clip_grad_norm_(sae_model_to_train.parameters(), max_norm=1.0)
            optimizer.step()

            # --- Logging (Batch Level) ---
            loss_item = loss.detach().item()
            mse_loss_item = mse_loss.detach().item()
            l1_loss_item = l1_loss.detach().item()

            total_loss_epoch += loss_item * current_batch_token_count
            total_mse_loss_epoch += mse_loss_item * current_batch_token_count
            total_l1_loss_epoch += l1_loss_item * current_batch_token_count

            # Calculate L0 norm (more efficient under no_grad)
            with torch.no_grad():
                feature_acts = sae_output_dict["feature_acts"]
                l0_norm = (feature_acts.abs() > 1e-6).float().sum(dim=-1).mean().item()
            total_l0_norm_epoch += l0_norm * current_batch_token_count

            num_batches_processed += 1
            total_tokens_processed_epoch += current_batch_token_count
            global_step += 1

            # Update tqdm progress bar
            pbar.set_postfix({
                'Loss': f"{loss_item:.4f}", 'MSE': f"{mse_loss_item:.4f}",
                'L1r': f"{l1_loss_item:.2f}", 'L0': f"{l0_norm:.2f}" })

            # Optional: Log to W&B more frequently (e.g., every N steps)
            if args.log_to_wandb and global_step % 100 == 0:
                wandb.log({
                    "batch_loss": loss_item, "batch_mse_loss": mse_loss_item,
                    "batch_l1_raw": l1_loss_item, "batch_l0_norm": l0_norm,
                    "global_step": global_step, "epoch": epoch + 1
                })

            # Clean up (important for long runs)
            del batch_tokens, sae_output_dict, loss, mse_loss, l1_loss, feature_acts
            if DEVICE == torch.device('cuda'): torch.cuda.empty_cache()


        except torch.cuda.OutOfMemoryError:
            print("\nCUDA OutOfMemoryError! Try reducing --batch_size_sae or --expansion_factor.")
            raise # Stop execution
        except Exception as e:
            print(f"\nError during training batch {num_batches_processed}: {e}")
            traceback.print_exc()
            raise # Stop execution

    # --- Epoch End ---
    epoch_duration = time.time() - epoch_start_time
    if total_tokens_processed_epoch > 0:
        avg_loss = total_loss_epoch / total_tokens_processed_epoch
        avg_mse = total_mse_loss_epoch / total_tokens_processed_epoch
        avg_l1_raw = total_l1_loss_epoch / total_tokens_processed_epoch
        avg_l0 = total_l0_norm_epoch / total_tokens_processed_epoch

        print(f"\n--- Epoch {epoch+1}/{args.epochs} Complete ---")
        print(f"  Time: {epoch_duration:.2f}s")
        print(f"  Processed {total_tokens_processed_epoch} tokens across {num_batches_processed} batches.")
        print(f"  Avg Loss: {avg_loss:.6f}")
        print(f"  Avg MSE : {avg_mse:.6f}")
        print(f"  Avg Raw L1: {avg_l1_raw:.4f} (Coeff: {args.l1_coeff})")
        print(f"  Avg L0 Norm: {avg_l0:.2f} features")

        # Log epoch metrics to W&B
        if args.log_to_wandb:
            wandb.log({
                "epoch_loss": avg_loss, "epoch_mse_loss": avg_mse,
                "epoch_l1_raw": avg_l1_raw, "epoch_l0_norm": avg_l0,
                "epoch": epoch + 1
            })
    else:
        print(f"\nEpoch {epoch+1} had no valid tokens processed. Time: {epoch_duration:.2f}s")

    # --- Save Checkpoint ---
    if args.checkpoint_freq > 0 and (epoch + 1) % args.checkpoint_freq == 0:
        checkpoint_name = f"sae_checkpoint_epoch_{epoch+1:03d}.pt"
        checkpoint_path = os.path.join(args.save_dir, checkpoint_name)
        try:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': sae_model_to_train.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': vars(args) # Save config with checkpoint
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
            # Optional: Save to W&B as artifact
            if args.log_to_wandb:
                wandb.save(checkpoint_path) # Saves file associated with the run
        except Exception as e:
            print(f"Error saving checkpoint: {e}")

    gc.collect() # Force garbage collection at end of epoch

# --- End of Training ---
total_train_time = time.time() - start_train_time
print(f"\n--- Training Finished ---")
print(f"Total training time: {total_train_time / 3600:.2f} hours")

# --- Save Final Model ---
final_model_name = f"sae_final_model_e{args.epochs}_l1_{args.l1_coeff:.0e}.pt"
final_model_path = os.path.join(args.save_dir, final_model_name)
try:
    torch.save({
        'model_state_dict': sae_model_to_train.state_dict(),
        'config': vars(args)
        }, final_model_path)
    print(f"Final SAE model saved to {final_model_path}")
    # Optional: Save final model to W&B
    if args.log_to_wandb:
        wandb.save(final_model_path)
        wandb.finish() # End the W&B run
except Exception as e:
    print(f"Error saving final model: {e}")

print("\n--- train_sae.py finished ---")
