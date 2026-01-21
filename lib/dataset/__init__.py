# optional torch-dependent imports
try:
    from .dataset import (
        LazyDataset,
        NoisedDataset,
        SmartBatchSampler,
        DistributedSmartBatchSampler,
        DistributedSamplerWrapper,
        DynamicPaddingCollator,
        create_distributed_dataloader,
        set_dataloader_epoch,
    )
except ImportError:
    LazyDataset                  = None
    NoisedDataset                = None
    SmartBatchSampler            = None
    DistributedSmartBatchSampler = None
    DistributedSamplerWrapper    = None
    DynamicPaddingCollator       = None
    create_distributed_dataloader = None
    set_dataloader_epoch         = None


__all__ = [
    
    # dataset (torch-dependent)
    "LazyDataset",
    "NoisedDataset",
    "SmartBatchSampler",
    "DistributedSmartBatchSampler",
    "DistributedSamplerWrapper",
    "DynamicPaddingCollator",
    "create_distributed_dataloader",
    "set_dataloader_epoch",
    
    # noising
    "NoisingConfig",
    "GFFNoiser",
    
    # binary
    "BinaryDataset",
]