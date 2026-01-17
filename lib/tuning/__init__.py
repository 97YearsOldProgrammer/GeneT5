from ._parser import (
    parse_fasta,
    parse_gff,
    group_features_by_seqid,
    group_features_by_parent,
    create_gene_prediction_dataset,
    create_rna_classification_dataset,
    save_dataset,
    load_dataset,
    format_annotation_target,
    RNA_CLASSES,
    GENE_FEATURE_TYPES,
    RNA_FEATURE_TYPES,
    anti,
)

from ._chunking import (
    load_tokenizer_config,
    get_existing_tokens,
    find_missing_rna_tokens,
    append_tokens_to_config,
    update_tokenizer_with_rna_classes,
    estimate_tokens,
    estimate_gff_tokens,
    find_gene_boundaries,
    chunk_sequence_with_overlap,
    chunk_gff_with_overlap,
    should_chunk_annotation,
    validate_chunks,
    create_gene_prediction_dataset_with_chunking,
    DEFAULT_WINDOW_SIZE,
    DEFAULT_STRIDE,
    DEFAULT_OVERLAP_TOKENS,
    DEFAULT_MAX_GFF_LINES,
)

from .dataset import (
    MixedTaskDataset,
    SmartBatchSampler,
    DynamicPaddingCollator,
)


__all__ = [
    # Parser - parsing functions
    "parse_fasta",
    "parse_gff",
    "group_features_by_seqid",
    "group_features_by_parent",
    
    # Parser - dataset creation
    "create_gene_prediction_dataset",
    "create_rna_classification_dataset",
    "format_annotation_target",
    
    # Parser - I/O
    "save_dataset",
    "load_dataset",
    
    # Parser - constants
    "RNA_CLASSES",
    "GENE_FEATURE_TYPES",
    "RNA_FEATURE_TYPES",
    
    # Parser - utilities
    "anti",
    
    # Chunking - tokenizer functions
    "load_tokenizer_config",
    "get_existing_tokens",
    "find_missing_rna_tokens",
    "append_tokens_to_config",
    "update_tokenizer_with_rna_classes",
    
    # Chunking - chunking functions
    "estimate_tokens",
    "estimate_gff_tokens",
    "find_gene_boundaries",
    "chunk_sequence_with_overlap",
    "chunk_gff_with_overlap",
    "should_chunk_annotation",
    
    # Chunking - advanced dataset creation
    "create_gene_prediction_dataset_with_chunking",
    
    # Chunking - validation
    "validate_chunks",
    
    # Chunking - constants
    "DEFAULT_WINDOW_SIZE",
    "DEFAULT_STRIDE",
    "DEFAULT_OVERLAP_TOKENS",
    "DEFAULT_MAX_GFF_LINES",
    
    # Dataset - PyTorch classes
    "MixedTaskDataset",
    "SmartBatchSampler",
    "DynamicPaddingCollator",
]


__version__ = "1.0.0"