import os
import platform
import argparse
import time
import math
import warnings
import torch
import torch.distributed as dist
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext

from transformers import AutoTokenizer

from model.model import MiniMindLM
from model.LMConfig import LMConfig
from model.dataset import PretrainDataset

# Import Transformer Engine components with fallback
try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import Format, DelayedScaling
    has_te = True
    print("Transformer Engine found. FP8 training is available.")
except ImportError:
    te = None
    # Define dummy classes/functions if TE not available
    Format = type('Format', (), {'HYBRID': None, 'E4M3': None}) # Dummy Format
    DelayedScaling = type('DelayedScaling', (), {}) # Dummy Recipe
    has_te = False
    print("Transformer Engine not found. FP8 training will be disabled.")

warnings.filterwarnings('ignore')


def Logger(content):
    # Print only from rank 0 in DDP mode
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    # Simple cosine decay, can be replaced with more complex schedules
    # Add warmup if needed (currently warmup_iters is unused)
    # if current_step < args.warmup_iters:
    #     return lr * current_step / args.warmup_iters
    # decay_ratio = (current_step - args.warmup_iters) / (total_steps - args.warmup_iters)
    decay_ratio = current_step / total_steps
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return lr * coeff # Simple cosine decay, adjust minimum LR if needed


def train_epoch(epoch, wandb, fp8_recipe):
    """
    Trains the model for one epoch.

    Args:
        epoch (int): The current epoch number (0-indexed).
        wandb: The WandB run object, or None if not used.
        fp8_recipe: The Transformer Engine FP8 recipe instance, or None.
    """
    model.train() # Ensure model is in training mode
    loss_fct = nn.CrossEntropyLoss(reduction='none') # Calculate loss per element
    start_time = time.time()
    total_loss = 0.0 # Accumulates *unscaled* loss over the epoch
    log_loss = 0.0   # Accumulates *unscaled* loss for logging interval

    # Calculate total number of batches correctly
    num_batches_per_epoch = len(train_loader)

    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        # --- Update Learning Rate ---
        # Calculate based on optimizer steps, not batch steps
        current_optimizer_step = (epoch * iter_per_epoch) + ((step + 1) // args.accumulation_steps)
        total_optimizer_steps = args.epochs * iter_per_epoch
        lr = get_lr(current_optimizer_step, total_optimizer_steps, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # --- Determine FP8 Autocast Context *INSIDE* the loop ---
        fp8_autocast_context = nullcontext() # Default to nullcontext (no-op)
        if args.fp8 and has_te and fp8_recipe is not None:
            # Let TE infer the default group by passing None.
            fp8_group_for_amax = None
            # if ddp: fp8_group_for_amax = dist.group.WORLD # Alternative if needed

            # Create a NEW fp8_autocast context manager instance for this step
            fp8_autocast_context = te.fp8_autocast(
                enabled=True,
                fp8_recipe=fp8_recipe, # Use the recipe passed to the function
                fp8_group=fp8_group_for_amax
            )
        elif args.fp8 and (not has_te or fp8_recipe is None):
            # This warning should ideally be logged once during setup, but good to be safe.
            # Logger("Warning: FP8 requested but Transformer Engine not found or FP8 recipe missing. FP8 disabled for this step.")
            pass # fp8_autocast_context remains nullcontext()

        # This flag seems unused based on the model code provided earlier.
        # If the model's TE layers *do* use it, it should be set appropriately.
        is_first_microbatch_flag = None

        # --- Combined AMP and FP8 Forward Pass ---
        with ctx: # AMP autocast (e.g., torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16))
            with fp8_autocast_context: # FP8 autocast (active only if args.fp8, has_te, and fp8_recipe exist)
                # Model forward pass
                res = model(X) # Pass is_first_microbatch if TE layers require it

                # Ensure the model output includes logits. Adapt if the output structure is different.
                if hasattr(res, 'logits'):
                    logits = res.logits
                else:
                    # Assuming the first element is logits if not a dict/object with .logits
                    # Adjust this based on your MiniMindLM's actual return type
                    logits = res[0]

                # Calculate main cross-entropy loss
                step_loss = loss_fct(
                    logits.view(-1, logits.size(-1)), # Shape: [batch*seq_len, vocab_size]
                    Y.view(-1)                        # Shape: [batch*seq_len]
                ).view(Y.size())                      # Reshape back to: [batch, seq_len]

                # Apply the loss mask and calculate the mean loss for this step
                step_loss = (step_loss * loss_mask).sum() / loss_mask.sum()

                # --- Add Auxiliary Loss (e.g., for MoE) ---
                # Retrieve aux_loss from the model/block if applicable
                aux_loss_val = torch.tensor(0.0, device=step_loss.device, dtype=step_loss.dtype)
                if lm_config.use_moe:
                     try:
                         # 1. Check if returned in the main model output (preferred)
                         if hasattr(res, 'aux_loss') and res.aux_loss is not None:
                             aux_loss_val = res.aux_loss
                         # 2. Fallback: Sum from layers (less ideal, depends on internal structure)
                         #    Requires MiniMindBlock to store aux_loss after its forward pass.
                         else:
                             model_ref = model.module if ddp else model # Get underlying model if using DDP
                             layer_aux_losses = [
                                 getattr(l, 'aux_loss', torch.tensor(0.0, device=step_loss.device))
                                 for l in model_ref.layers
                             ]
                             aux_loss_val = sum(layer_aux_losses)

                         # Add aux loss to step loss if it's a valid tensor/float
                         if isinstance(aux_loss_val, torch.Tensor) and aux_loss_val.requires_grad:
                              step_loss += aux_loss_val
                         elif isinstance(aux_loss_val, float) and aux_loss_val != 0.0:
                              # Handle case where aux_loss might be a float
                              step_loss += torch.tensor(aux_loss_val, device=step_loss.device, dtype=step_loss.dtype)

                     except AttributeError as e:
                         # Logger(f"Debug: Could not retrieve aux_loss: {e}") # Uncomment for debugging
                         pass # Continue without aux loss if retrieval fails

                # Normalize loss for gradient accumulation BEFORE scaling
                loss_for_backward = step_loss / args.accumulation_steps

        # --- Backward Pass and Gradient Scaling ---
        # scaler.scale() multiplies the loss, .backward() computes gradients on scaled loss
        scaler.scale(loss_for_backward).backward()

        # Accumulate the *unscaled* step loss for logging purposes
        total_loss += step_loss.item()
        log_loss += step_loss.item()

        # --- Optimizer Step (after accumulating gradients) ---
        if (step + 1) % args.accumulation_steps == 0:
            # 1. Unscale gradients: Divide gradients by the scale factor
            scaler.unscale_(optimizer)
            # 2. Clip gradients (applied to unscaled gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            # 3. Optimizer step: Updates weights using unscaled gradients
            scaler.step(optimizer)
            # 4. Update scale factor for next iteration
            scaler.update()
            # 5. Zero gradients for the next accumulation cycle
            optimizer.zero_grad(set_to_none=True) # More memory efficient

            # --- Logging ---
            # Log based on optimizer steps
            current_opt_step = (step + 1) // args.accumulation_steps
            if current_opt_step % args.log_interval == 0 :
                # Average loss over the logging interval (log_interval * accumulation_steps batches)
                avg_loss = log_loss / (args.log_interval * args.accumulation_steps)
                elapsed_time = time.time() - start_time
                # Estimate remaining time based on optimizer steps processed so far in the epoch
                estimated_total_epoch_time = elapsed_time / current_opt_step * iter_per_epoch if current_opt_step > 0 else 0
                remaining_epoch_time = estimated_total_epoch_time - elapsed_time

                Logger(
                    f'Epoch:[{epoch + 1}/{args.epochs}] OptStep:[{current_opt_step}/{iter_per_epoch}] | '
                    f'Loss:{avg_loss:.4f} | LR:{optimizer.param_groups[-1]["lr"]:.6f} | '
                    f'Elapsed: {elapsed_time // 60:.0f}min | Est. Epoch Remain: {remaining_epoch_time // 60:.0f}min'
                )
                # Log to wandb if enabled (only on rank 0)
                if wandb is not None and (not ddp or dist.get_rank() == 0):
                    # Log total optimizer steps across all epochs
                    global_optimizer_step = epoch * iter_per_epoch + current_opt_step
                    wandb.log({
                        "step": global_optimizer_step,
                        "epoch_frac": epoch + current_opt_step / iter_per_epoch, # Fractional epoch
                        "loss": avg_loss,
                        "lr": optimizer.param_groups[-1]['lr'],
                        "scale_factor": scaler.get_scale() # Log grad scaler factor
                    })
                log_loss = 0.0 # Reset log loss accumulator

            # --- Checkpointing ---
            # Save based on optimizer steps
            if current_opt_step % args.save_interval == 0 :
                 if not ddp or dist.get_rank() == 0: # Save only on rank 0
                    model.eval() # Set model to evaluation mode for saving state_dict
                    moe_path = '_moe' if lm_config.use_moe else ''
                    fp8_path = '_fp8' if args.fp8 and has_te else '' # Only add fp8 if actually used
                    # Include epoch and optimizer step in checkpoint name
                    ckp_name = f'pretrain_dim{lm_config.dim}_ep{epoch+1}_optstep{current_opt_step}{moe_path}{fp8_path}.pth'
                    ckp_path = os.path.join(args.save_dir, ckp_name)

                    # Get state dict correctly whether using DDP or not
                    state_dict_to_save = model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()

                    # Save checkpoint (consider saving optimizer and scaler state too for resuming)
                    # checkpoint = {
                    #     'model': state_dict_to_save,
                    #     'optimizer': optimizer.state_dict(),
                    #     'scaler': scaler.state_dict(),
                    #     'epoch': epoch,
                    #     'optimizer_step': current_opt_step,
                    #     'args': args # Save args for reference
                    # }
                    # torch.save(checkpoint, ckp_path)
                    torch.save(state_dict_to_save, ckp_path) # Save only model state for now
                    Logger(f"Saved checkpoint to {ckp_path}")
                    model.train() # Set model back to training mode

    # --- End of Epoch Logging ---
    # Average loss over all batches in the epoch
    avg_epoch_loss = total_loss / num_batches_per_epoch
    Logger(f"Epoch {epoch + 1} finished. Average Loss: {avg_epoch_loss:.4f}")
    if wandb is not None and (not ddp or dist.get_rank() == 0):
        wandb.log({"epoch_loss": avg_epoch_loss, "epoch": epoch + 1})

def init_model(lm_config, fp8_enabled: bool = False):
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
    # Pass fp8_enabled flag to the model constructor
    model = MiniMindLM(lm_config, fp8_enabled=fp8_enabled).to(args.device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    Logger(f'LLM total parameters: {num_params / 1e6:.3f} Million')
    return model, tokenizer


def init_distributed_mode():
    # Initializes DDP environment
    if not ddp: return 0, "cuda:0" # Return dummy values if not DDP

    # RANK, LOCAL_RANK, WORLD_SIZE are expected to be set by torchrun/slurm
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])

    # NCCL is the standard backend for GPU training
    dist.init_process_group(backend="nccl")
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    Logger(f"Initialized DDP on rank {ddp_rank}/{ddp_world_size}, local rank {ddp_local_rank}, device {device}")
    return ddp_local_rank, device


# Main execution block
# Example run command:
# torchrun --nproc_per_node <num_gpus> train_pretrain.py --ddp --fp8 --batch_size <bs> --accumulation_steps <steps>
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining with Transformer Engine")
    # Directories and Paths
    parser.add_argument("--out_dir", type=str, default="out", help="Output directory for checkpoints and logs.")
    parser.add_argument("--save_dir", type=str, default=None, help="Specific directory for saving checkpoints (defaults to out_dir).")
    parser.add_argument("--data_path", type=str, default="./dataset/pretrain_hq.jsonl", help="Path to the pretraining data.")
    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU.") # Reduced default for potential memory increase with FP8/MoE
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Peak learning rate.")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="Number of steps to accumulate gradients.")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value.")
    parser.add_argument("--warmup_iters", type=int, default=0, help="Number of warmup iterations (currently unused in get_lr).")
    # Model Configuration
    parser.add_argument('--dim', default=512, type=int, help="Model dimension.")
    parser.add_argument('--n_layers', default=8, type=int, help="Number of transformer layers.")
    parser.add_argument('--max_seq_len', default=512, type=int, help="Maximum sequence length.")
    parser.add_argument('--use_moe', action='store_true', help="Enable Mixture of Experts layers.") # Use action='store_true'
    # Hardware and Precision
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to use (ignored in DDP).")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16", "float16"], help="AMP dtype.")
    parser.add_argument("--fp8", action="store_true", help="Enable FP8 training with Transformer Engine.")
    # Distributed Training
    parser.add_argument("--ddp", action="store_true", help="Enable Distributed Data Parallel training.")
    # Logging and Saving
    parser.add_argument("--use_wandb", action="store_true", help="Enable logging with Weights & Biases.")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain-TE", help="WandB project name.")
    parser.add_argument("--log_interval", type=int, default=10, help="Log frequency (in optimizer steps).")
    parser.add_argument("--save_interval", type=int, default=100, help="Checkpoint save frequency (in optimizer steps).")
    # Other
    parser.add_argument("--num_workers", type=int, default=1, help="Number of dataloader workers.")
    parser.add_argument('--local_rank', type=int, default=-1, help="Local rank (set by torchrun, used internally).") # Keep for compatibility if needed
    args = parser.parse_args()

    # --- Setup ---
    ddp = args.ddp or int(os.environ.get("RANK", -1)) != -1 # Check DDP via arg or env var
    ddp_local_rank, device = 0, args.device

    if ddp:
        ddp_local_rank, device = init_distributed_mode()
        args.device = torch.device(device) # Ensure args.device reflects DDP device

    # Set save directory
    if args.save_dir is None:
        args.save_dir = args.out_dir
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True) # Ensure out_dir also exists

    # Seed setting (important for reproducibility, especially in DDP)
    torch.manual_seed(1337 + ddp_local_rank) # Offset seed by rank
    torch.cuda.manual_seed(1337 + ddp_local_rank)

    # Determine device type and AMP dtype
    device_type = "cuda" if "cuda" in str(args.device) else "cpu"
    pt_dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=pt_dtype)
    Logger(f"Using device: {args.device}, dtype: {args.dtype}")

    # WandB setup
    wandb = None
    if args.use_wandb and (not ddp or dist.get_rank() == 0):
        try:
            import wandb
            # Construct wandb run name
            run_name_parts = [
                f"Dim{args.dim}",
                f"L{args.n_layers}",
                f"Seq{args.max_seq_len}",
                f"BS{args.batch_size*args.accumulation_steps*(dist.get_world_size() if ddp else 1)}", # Effective batch size
                f"LR{args.learning_rate:.1e}",
                f"Dtype{args.dtype}",
            ]
            if args.use_moe: run_name_parts.append("MoE")
            if args.fp8: run_name_parts.append("FP8")
            args.wandb_run_name = "-".join(run_name_parts)

            wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=args)
            Logger(f"WandB initialized for project '{args.wandb_project}', run '{args.wandb_run_name}'")
        except ImportError:
            Logger("WandB not installed, skipping WandB logging.")

    # --- Model and Data ---
    lm_config = LMConfig(
        dim=args.dim,
        n_layers=args.n_layers,
        max_seq_len=args.max_seq_len,
        use_moe=args.use_moe,
        fp8=args.fp8, # Pass fp8 flag to config
        batch_size=args.batch_size # Add batch_size
    )
    # Initialize model, passing fp8_enabled based on args.fp8 AND has_te
    model, tokenizer = init_model(lm_config, fp8_enabled=(args.fp8 and has_te))

    # Define FP8 recipe (only if FP8 is enabled and TE is available)
    fp8_recipe_instance = None
    if args.fp8 and has_te:
        fp8_recipe_instance = DelayedScaling(
            margin=0,
            fp8_format=Format.HYBRID, # E4M3 for fwd, E5M2 for bwd
            amax_history_len=16,      # Example value
            amax_compute_algo="max"   # Example value
        )
        Logger(f"FP8 Recipe configured: {fp8_recipe_instance}")


    # --- Dataloader ---
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    train_sampler = DistributedSampler(train_ds, shuffle=True) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False, # Keep last batch even if smaller
        shuffle=(train_sampler is None), # Shuffle only if not using DDP sampler
        num_workers=args.num_workers,
        sampler=train_sampler
    )
    iter_per_epoch = len(train_loader) // args.accumulation_steps # Number of optimizer steps per epoch
    tokens_per_iter = args.batch_size * lm_config.max_seq_len * args.accumulation_steps * (dist.get_world_size() if ddp else 1)
    Logger(f"Effective batch size: {args.batch_size * args.accumulation_steps * (dist.get_world_size() if ddp else 1)}")
    Logger(f"Tokens per optimizer step: {tokens_per_iter / 1e6:.3f} Million")
    Logger(f"Optimizer steps per epoch: {iter_per_epoch}")

    # --- Optimizer and Scaler ---
    # Filter out parameters that don't require gradients (e.g., buffers)
    optim_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(optim_params, lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.1) # Common AdamW settings
    # GradScaler for mixed precision (AMP) - enabled based on dtype
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    Logger(f"Optimizer: AdamW, GradScaler enabled: {scaler.is_enabled()}")

    # Wrap model with DDP if enabled
    if ddp:
        # find_unused_parameters can be needed for MoE/conditional computation
        # Set buffer ignore list if needed (e.g., for pos_cis)
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"} # Example if pos_cis causes issues
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank], find_unused_parameters=lm_config.use_moe)
        Logger("Model wrapped with DistributedDataParallel.")

    # --- Training Loop ---
    Logger("Starting training...")
    for epoch in range(args.epochs):
        if ddp:
            train_sampler.set_epoch(epoch) # Ensure proper shuffling with DDP sampler
        train_epoch(epoch, wandb, fp8_recipe_instance)

    # --- Cleanup ---
    if ddp:
        dist.destroy_process_group()
    if wandb is not None and (not ddp or dist.get_rank() == 0):
        wandb.finish()
    Logger("Training finished.")