import os
import json
import random
from pathlib import Path

import torch
import torch.nn             as nn
import torch.distributed    as dist





################################
#####  Distributed Setup   #####
################################


def setup_distributed(backend="nccl"):
    """
    Initialize distributed training environment for DGX Spark multi-node setup.
    
    Uses NCCL backend for GPU-to-GPU communication over ConnectX-7 200Gb/s network.
    Environment variables should be set by torchrun/torch.distributed.launch.
    """
    if not dist.is_initialized():
        # Check if running in distributed mode
        if "RANK" not in os.environ:
            return None
        
        rank       = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        # Set device before init
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        
        # Initialize process group
        dist.init_process_group(
            backend    = backend,
            rank       = rank,
            world_size = world_size,
        )
        
        return {
            "rank":       rank,
            "world_size": world_size,
            "local_rank": local_rank,
            "is_main":    rank == 0,
        }
    
    return {
        "rank":       dist.get_rank(),
        "world_size": dist.get_world_size(),
        "local_rank": int(os.environ.get("LOCAL_RANK", 0)),
        "is_main":    dist.get_rank() == 0,
    }


def cleanup_distributed():

    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():

    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_world_size():

    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def barrier():
    """Synchronization barrier for distributed training"""
    if dist.is_initialized():
        dist.barrier()


def all_reduce_mean(tensor):
    """Average tensor across all processes"""
    if not dist.is_initialized():
        return tensor
    
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= get_world_size()
    return tensor


def broadcast_object(obj, src=0):
    """Broadcast Python object from src to all processes"""
    if not dist.is_initialized():
        return obj
    
    obj_list = [obj]
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]


############################
#####  Model Wrapping  #####
############################


def wrap_model_distributed(model, device, find_unused_params=False,
                           gradient_as_bucket_view=True, static_graph=False):
    """Wrap model with DistributedDataParallel for multi-GPU training"""

    if not dist.is_initialized():
        return model.to(device)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    model = model.to(device)
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids              = [local_rank],
        output_device           = local_rank,
        find_unused_parameters  = find_unused_params,
        gradient_as_bucket_view = gradient_as_bucket_view,
        static_graph            = static_graph,
    )

    return model


def unwrap_model(model):
    """Unwrap model from DistributedDataParallel."""
    if hasattr(model, 'module'):
        return model.module
    return model


################################
#####  Training Functions  #####
################################


def train_epoch(model, dataloader, optimizer, scheduler, device, grad_accum=1, max_grad_norm=1.0):
    """Train for one epoch with BF16 mixed precision and gradient accumulation"""
    model.train()
    total_loss = 0
    num_steps  = 0
    
    # BF16 for training (model stays FP32, compute in BF16)
    dtype  = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == torch.float16))
    
    optimizer.zero_grad()
    
    for step, batch in enumerate(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        with torch.amp.autocast('cuda', dtype=dtype):
            outputs = model(**batch) if not hasattr(model, 'forward_finetune') else model(**batch)
            
            if isinstance(outputs, dict):
                loss = outputs["loss"]
            else:
                loss = outputs.loss
            
            loss = loss / grad_accum
        
        if dtype == torch.float16:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        if (step + 1) % grad_accum == 0:
            if dtype == torch.float16:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            num_steps += 1
        
        total_loss += loss.item() * grad_accum
    
    avg_loss = total_loss / len(dataloader)
    
    # Average loss across processes in distributed training
    if dist.is_initialized():
        loss_tensor = torch.tensor([avg_loss], device=device)
        all_reduce_mean(loss_tensor)
        avg_loss = loss_tensor.item()
    
    return avg_loss


def train_epoch_seq2seq(model, dataloader, optimizer, scheduler, device, grad_accum=1, max_grad_norm=1.0):
    """Train for one epoch with BF16 mixed precision and distributed support"""
    model.train()
    total_loss = 0

    dtype  = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == torch.float16))

    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.amp.autocast('cuda', dtype=dtype):
            outputs = model(
                input_ids = batch["input_ids"],
                labels    = batch["labels"],
            )

            loss = outputs["loss"] / grad_accum

        if dtype == torch.float16:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % grad_accum == 0:
            if dtype == torch.float16:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum

    avg_loss = total_loss / len(dataloader)

    if dist.is_initialized():
        loss_tensor = torch.tensor([avg_loss], device=device)
        all_reduce_mean(loss_tensor)
        avg_loss = loss_tensor.item()

    return avg_loss


def train_epoch_seq2seq_distributed(
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    grad_accum    = 1,
    max_grad_norm = 1.0,
    scaler        = None,
    use_amp       = True,
):
    """Train for one epoch with full distributed support for DGX Spark"""

    model.train()
    total_loss     = 0.0
    total_moe_loss = 0.0
    num_batches    = 0

    dtype = torch.bfloat16 if (torch.cuda.is_bf16_supported() and use_amp) else torch.float32
    if scaler is None and dtype == torch.float16:
        scaler = torch.amp.GradScaler('cuda')

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(dataloader):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        with torch.amp.autocast('cuda', dtype=dtype, enabled=use_amp):
            outputs = model(
                input_ids = batch["input_ids"],
                labels    = batch["labels"],
            )

            if isinstance(outputs, dict):
                loss     = outputs["loss"]
                moe_loss = outputs.get("moe_loss", 0.0)
            else:
                loss     = outputs.loss
                moe_loss = getattr(outputs, 'moe_loss', 0.0)

            scaled_loss = loss / grad_accum

        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        if (step + 1) % grad_accum == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss     += loss.detach().item()
        total_moe_loss += moe_loss if isinstance(moe_loss, (int, float)) else moe_loss.item()
        num_batches    += 1

    avg_loss     = total_loss / max(num_batches, 1)
    avg_moe_loss = total_moe_loss / max(num_batches, 1)

    if dist.is_initialized():
        metrics      = torch.tensor([avg_loss, avg_moe_loss, num_batches], device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        world_size   = get_world_size()
        avg_loss     = metrics[0].item() / world_size
        avg_moe_loss = metrics[1].item() / world_size

    return avg_loss, avg_moe_loss


def evaluate(model, dataloader, device):
    """Evaluate model on validation set with BF16."""
    model.eval()
    total_loss = 0
    correct    = 0
    total      = 0
    
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            with torch.amp.autocast('cuda', dtype=dtype):
                outputs = model(**batch)
                
                if isinstance(outputs, dict):
                    loss = outputs["loss"]
                else:
                    loss = outputs.loss
            
            total_loss += loss.item()
            
            if isinstance(outputs, dict) and "logits" in outputs:
                logits   = outputs["logits"]
                preds    = torch.argmax(logits, dim=-1)
                correct += (preds == batch["labels"]).sum().item()
                total   += batch["labels"].size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total if total > 0 else None
    
    # Synchronize metrics in distributed training
    if dist.is_initialized():
        metrics = torch.tensor([avg_loss, correct, total], device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        world_size = get_world_size()
        avg_loss   = metrics[0].item() / world_size
        if metrics[2].item() > 0:
            accuracy = metrics[1].item() / metrics[2].item()
    
    return {"loss": avg_loss, "accuracy": accuracy}


def evaluate_seq2seq(model, dataloader, device):
    """Evaluate model with BF16 and distributed support"""
    model.eval()
    total_loss  = 0
    num_batches = 0

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast('cuda', dtype=dtype):
                outputs = model(
                    input_ids = batch["input_ids"],
                    labels    = batch["labels"],
                )

            if isinstance(outputs, dict):
                total_loss += outputs["loss"].item()
            else:
                total_loss += outputs.loss.item()
            num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)

    if dist.is_initialized():
        loss_tensor = torch.tensor([avg_loss, num_batches], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        world_size = get_world_size()
        avg_loss   = loss_tensor[0].item() / world_size

    model.train()
    return avg_loss


def evaluate_seq2seq_distributed(model, dataloader, device, use_amp=True):
    """Evaluate model with full distributed support for DGX Spark"""

    model.eval()
    total_loss     = 0.0
    total_moe_loss = 0.0
    num_batches    = 0

    dtype = torch.bfloat16 if (torch.cuda.is_bf16_supported() and use_amp) else torch.float32

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            with torch.amp.autocast('cuda', dtype=dtype, enabled=use_amp):
                outputs = model(
                    input_ids = batch["input_ids"],
                    labels    = batch["labels"],
                )

            if isinstance(outputs, dict):
                total_loss     += outputs["loss"].item()
                total_moe_loss += outputs.get("moe_loss", 0.0)
                if isinstance(total_moe_loss, torch.Tensor):
                    total_moe_loss = total_moe_loss.item()
            else:
                total_loss += outputs.loss.item()

            num_batches += 1

    avg_loss     = total_loss / max(num_batches, 1)
    avg_moe_loss = total_moe_loss / max(num_batches, 1)

    if dist.is_initialized():
        metrics    = torch.tensor([avg_loss, avg_moe_loss, num_batches], device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        world_size = get_world_size()
        avg_loss     = metrics[0].item() / world_size
        avg_moe_loss = metrics[1].item() / world_size

    model.train()
    return avg_loss, avg_moe_loss


def validate_prefixlm(model, val_loader, device, dtype, is_dist=False):
    """Run validation loop for prefix-LM models with distributed sync"""

    model.eval()
    total_val_loss = 0
    num_batches    = 0

    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

            with torch.amp.autocast('cuda', dtype=dtype):
                outputs = model(
                    input_ids  = batch['input_ids'],
                    labels     = batch['labels'],
                    prefix_len = batch.get('prefix_len', 0),
                )

            total_val_loss += outputs['loss'].item()
            num_batches    += 1

            del outputs, batch

    val_loss = total_val_loss / max(num_batches, 1)

    if is_dist:
        val_tensor = torch.tensor([val_loss], device=device)
        all_reduce_mean(val_tensor)
        val_loss = val_tensor.item()

    model.train()
    return val_loss


##################################
#####  Checkpoint Functions  #####
##################################


def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device="cpu"):
    """Load model, optimizer, and scheduler states from checkpoint"""
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle DDP wrapped models
    target_model = unwrap_model(model)
    target_model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    epoch         = checkpoint.get("epoch", 0)
    best_val_loss = checkpoint.get("config", {}).get("best_val_loss", float('inf'))
    
    if is_main_process():
        print(f"Loaded checkpoint from {checkpoint_path} (epoch {epoch})")
    
    return {
        "epoch":         epoch,
        "best_val_loss": best_val_loss,
        "config":        checkpoint.get("config", {})
    }


def save_checkpoint(model, optimizer, scheduler, epoch, save_path, config=None, global_step=None):
    """Save model, optimizer, and scheduler states (only on main process)"""

    if not is_main_process():
        return

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Unwrap DDP model for saving
    target_model = unwrap_model(model)

    checkpoint = {
        "epoch":                epoch,
        "model_state_dict":     target_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if global_step is not None:
        checkpoint["global_step"] = global_step

    if config:
        checkpoint["config"] = config

    torch.save(checkpoint, save_path)


def save_checkpoint_distributed(model, optimizer, scheduler, epoch, save_path, config=None):
    """
    Save checkpoint with distributed training support.
    Only main process saves, others wait at barrier.
    """
    barrier()  # Ensure all processes are synced
    
    if is_main_process():
        save_checkpoint(model, optimizer, scheduler, epoch, save_path, config)
    
    barrier()  # Wait for save to complete


def save_final_model(model, tokenizer, model_path, output_dir):
    """Save final model weights, tokenizer, and config"""

    final_model = unwrap_model(model)
    final_model.save(output_dir / 'pytorch_model.bin')
    tokenizer.save_pretrained(output_dir)

    model_path = Path(model_path)
    with open(model_path / 'config.json') as f:
        model_config = json.load(f)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(model_config, f, indent=2)


###################################
#####  Model Setup Functions  #####
###################################


def setup_gene_prediction_model(model_path, tokenizer, device):
    """Setup model for gene prediction task."""
    
    base_model = EncoderDecoderModel.from_pretrained(model_path)
    base_model.resize_token_embeddings(len(tokenizer))
    
    class ModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, input_ids, attention_mask, labels=None):
            return self.model(
                input_ids      = input_ids,
                attention_mask = attention_mask,
                labels         = labels,
            )
    
    model = ModelWrapper(base_model)
    model.to(device)
    return model


def prepare_tokenizer(model_path):
    """Load tokenizer (tokens already hardcoded during init)"""

    from lib.tokenizer.hf import GeneTokenizer
    return GeneTokenizer(model_path)


def apply_mxfp8(model):
    """Apply MXFP8 quantization to eligible Linear layers"""

    from torchao.prototype.mx_formats import MXLinearConfig
    from torchao.quantization import quantize_

    mx_config = MXLinearConfig.from_recipe_name("mxfp8_cublas")

    def mxfp8_filter(mod, fqn):
        if not isinstance(mod, torch.nn.Linear):
            return False
        if mod.in_features % 32 != 0 or mod.out_features % 32 != 0:
            return False
        return True

    quantize_(model, mx_config, filter_fn=mxfp8_filter)
    n_mx = sum(1 for _, m in model.named_modules() if type(m).__name__ == 'MXLinear')

    if is_main_process():
        print(f"  MXFP8: converted {n_mx} Linear layers to MXLinear")


def _newton_schulz(G, steps=5):
    """Batched Newton-Schulz orthogonalization (supports ndim >= 2)"""

    a, b, c = (3.4445, -4.7750, 2.0315)
    X       = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class MuonE2E:
    """End-to-end Muon optimizer for ALL parameter types

    ndim >= 2: Newton-Schulz ortho + aspect ratio scaling (includes embeddings)
    ndim == 1: L2-normalized momentum
    """

    def __init__(self, model, lr, weight_decay, momentum=0.95):

        self._params_2d = []
        self._params_1d = []

        for p in model.parameters():
            if not p.requires_grad:
                continue
            if p.ndim >= 2:
                self._params_2d.append(p)
            else:
                self._params_1d.append(p)

        self._bufs_2d = [torch.zeros_like(p) for p in self._params_2d]
        self._bufs_1d = [torch.zeros_like(p) for p in self._params_1d]
        self._momentum = momentum
        self._wd       = weight_decay

        self._lr_proxy  = torch.optim.SGD([torch.zeros(1)], lr=lr)
        self._schedulers = []

    def step(self):

        lr = self._lr_proxy.param_groups[0]['lr']

        for p, buf in zip(self._params_2d, self._bufs_2d):
            if p.grad is None:
                continue
            g = p.grad

            buf.lerp_(g, 1 - self._momentum)
            update = g.lerp_(buf, self._momentum)

            update = _newton_schulz(update)
            update *= max(1, p.size(-2) / p.size(-1)) ** 0.5

            p.data.mul_(1 - lr * self._wd)
            p.data.add_(update, alpha=-lr)

        for p, buf in zip(self._params_1d, self._bufs_1d):
            if p.grad is None:
                continue
            g = p.grad

            buf.lerp_(g, 1 - self._momentum)
            update = g.lerp_(buf, self._momentum)

            nrm = update.norm() + 1e-7
            update = update / nrm

            p.data.mul_(1 - lr * self._wd)
            p.data.add_(update, alpha=-lr)

    def zero_grad(self, set_to_none=False):

        for p in self._params_2d:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()

        for p in self._params_1d:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()

    def state_dict(self):

        return {
            'muon_e2e': {
                'bufs_2d':  [buf.clone() for buf in self._bufs_2d],
                'bufs_1d':  [buf.clone() for buf in self._bufs_1d],
                'lr_proxy': self._lr_proxy.state_dict(),
            }
        }

    def load_state_dict(self, state_dict):
        """Load state; silently skip incompatible checkpoints (AdamW, MuonAdamW)"""

        if 'muon_e2e' not in state_dict:
            return
        s = state_dict['muon_e2e']
        if 'bufs_2d' in s:
            for buf, saved in zip(self._bufs_2d, s['bufs_2d']):
                buf.copy_(saved)
        if 'bufs_1d' in s:
            for buf, saved in zip(self._bufs_1d, s['bufs_1d']):
                buf.copy_(saved)
        if 'lr_proxy' in s:
            self._lr_proxy.load_state_dict(s['lr_proxy'])

    def create_schedulers(self, schedule_fn, warmup_steps, total_steps):

        self._schedulers = [schedule_fn(self._lr_proxy, warmup_steps, total_steps)]

    def step_schedulers(self):

        for s in self._schedulers:
            s.step()

    def get_last_lr(self):

        return self._schedulers[0].get_last_lr() if self._schedulers else [0.0]

    def parameters(self):

        yield from self._params_2d
        yield from self._params_1d

    @property
    def param_groups(self):

        return self._lr_proxy.param_groups


def create_optimizer(model, lr, weight_decay):
    """Create Muon E2E optimizer"""

    opt = MuonE2E(model, lr, weight_decay)
    if is_main_process():
        n_2d = sum(p.numel() for p in opt._params_2d)
        n_1d = sum(p.numel() for p in opt._params_1d)
        print(f"  MuonE2E: {n_2d + n_1d:,} params (lr={lr})")
        print(f"    2D/3D: {len(opt._params_2d)} tensors, {n_2d:,} params (Newton-Schulz)")
        print(f"    1D:    {len(opt._params_1d)} tensors, {n_1d:,} params (L2-norm)")
    return opt


###############################
#####  Utility Functions  #####
###############################


def get_device():

    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return torch.device(f"cuda:{local_rank}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def print_rank0(*args, **kwargs):
    """Print only from rank 0 process"""
    if is_main_process():
        print(*args, **kwargs)


def log_metrics(metrics, epoch, step=None, prefix=""):
    """Log metrics only from main process"""
    if not is_main_process():
        return
    
    if step is not None:
        header = f"[Epoch {epoch}, Step {step}]"
    else:
        header = f"[Epoch {epoch}]"
    
    metric_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                             for k, v in metrics.items()])
    print(f"{prefix}{header} {metric_str}")


def set_seeds(seed):
    """Set random seeds for reproducibility"""

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.set_float32_matmul_precision('high')