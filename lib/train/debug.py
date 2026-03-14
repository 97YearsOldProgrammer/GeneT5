import json
import math
import time

import torch
import torch.distributed as dist


def check_nan(loss_val, step, epoch, rank, flight_file=None, batch=None, dump_dir=None):
    """Check loss for NaN/Inf — abort with diagnostics if found"""

    is_bad = math.isnan(loss_val) or math.isinf(loss_val)
    if not is_bad:
        return False

    print(f"\n{'!'*60}", flush=True)
    print(f"  [rank {rank}] NaN/Inf DETECTED at epoch {epoch+1} step {step}", flush=True)
    print(f"  loss = {loss_val}", flush=True)

    if batch is not None and dump_dir is not None:
        dump_path = dump_dir / f'_crash_batch_rank{rank}_e{epoch}_s{step}.pt'
        dump = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                dump[k] = v.cpu()
            else:
                dump[k] = v
        torch.save(dump, dump_path)
        print(f"  Batch dumped: {dump_path}", flush=True)

    if flight_file:
        flight_file.write(f'{{"CRASH":"NaN","s":{step},"e":{epoch},"r":{rank},'
                          f'"t":{int(time.time())}}}\n')
        flight_file.flush()

    print(f"{'!'*60}\n", flush=True)
    return True


def check_rank_health(loss_val, device, step, rank):
    """Cross-rank NaN check via allreduce — detects rank divergence"""

    if not dist.is_initialized():
        return False

    is_bad = 1.0 if (math.isnan(loss_val) or math.isinf(loss_val)) else 0.0
    flag   = torch.tensor([is_bad], device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)

    if flag.item() > 0 and is_bad == 0:
        print(f"\n[rank {rank}] WARNING: Another rank has NaN at step {step}!", flush=True)
        return True
    return flag.item() > 0


def grad_stats(model):
    """Compute gradient norm and max — no GPU sync, just reads .grad"""

    total_norm = 0.0
    max_val    = 0.0
    nan_params = []

    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.data
        pn = g.norm(2).item()
        pm = g.abs().max().item()

        if math.isnan(pn) or math.isinf(pn):
            nan_params.append(name)

        total_norm += pn ** 2
        max_val     = max(max_val, pm)

    total_norm = total_norm ** 0.5
    return {
        'grad_norm': total_norm,
        'grad_max':  max_val,
        'nan_params': nan_params,
    }


def activation_stats(model, raw_model):
    """Collect encoder output stats — call after forward, costs one .item()"""

    stats = {}
    hook_handles = []

    def make_hook(name):
        def hook(module, inp, out):
            if torch.is_tensor(out):
                t = out.detach().float()
                stats[name] = {
                    'mean': t.mean().item(),
                    'std':  t.std().item(),
                    'max':  t.abs().max().item(),
                    'has_nan': bool(torch.isnan(t).any().item()),
                }
        return hook

    # Hook on encoder output and a few internal layers
    enc = raw_model.encoder
    hook_handles.append(enc.register_forward_hook(make_hook('encoder_out')))

    n_layers = len(enc.layers)
    for idx in [0, n_layers // 2, n_layers - 1]:
        hook_handles.append(
            enc.layers[idx].register_forward_hook(make_hook(f'layer_{idx}'))
        )

    return stats, hook_handles


def format_debug_line(step, loss, grad_norm, grad_max, nan_params=None):
    """Single-line debug summary"""

    parts = [f"s={step}", f"loss={loss:.6f}", f"gnorm={grad_norm:.4f}", f"gmax={grad_max:.4f}"]
    if nan_params:
        parts.append(f"NaN_GRADS={nan_params[:3]}")
    return " | ".join(parts)
