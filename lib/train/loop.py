import os
import random

import torch

from lib.train.distributed import is_main_process, all_reduce_mean


################################
#####  Training Functions  #####
################################


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


###################################
#####  Model Setup Functions  #####
###################################


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


def set_seeds(seed):
    """Set random seeds for reproducibility"""

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.set_float32_matmul_precision('high')
