import os

import torch
import torch.nn          as nn
import torch.distributed as dist


################################
#####  Distributed Setup   #####
################################


def setup_distributed(backend="nccl"):
    """Initialize distributed training environment for DGX Spark multi-node setup"""

    if not dist.is_initialized():
        if "RANK" not in os.environ:
            return None

        rank       = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

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
    """Unwrap model from DistributedDataParallel"""
    if hasattr(model, 'module'):
        return model.module
    return model
