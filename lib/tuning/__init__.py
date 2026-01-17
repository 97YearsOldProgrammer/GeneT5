from ._parser import (
    anti,
    parse_fasta,
    parse_gff,
    group_features_by_seqid,
    group_features_by_parent,
    format_annotation_target,
    create_gene_prediction_dataset,
    create_rna_classification_dataset,
    save_dataset,
    load_dataset,
    GENE_FEATURE_TYPES,
    RNA_CLASSES,
    RNA_FEATURE_TYPES,
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
    build_gene_hierarchy,
    group_features_by_gene_simple,
    create_gene_prediction_dataset_with_chunking,
)

# optional torch-dependent imports
try:
    from .dataset import (
        MixedTaskDataset,
        SmartBatchSampler,
        DynamicPaddingCollator,
    )
except ImportError:
    MixedTaskDataset       = None
    SmartBatchSampler      = None
    DynamicPaddingCollator = None

__all__ = [
    # parser
    "anti",
    "parse_fasta",
    "parse_gff",
    "group_features_by_seqid",
    "group_features_by_parent",
    "format_annotation_target",
    "create_gene_prediction_dataset",
    "create_rna_classification_dataset",
    "save_dataset",
    "load_dataset",
    "GENE_FEATURE_TYPES",
    "RNA_CLASSES",
    "RNA_FEATURE_TYPES",
    # chunking
    "load_tokenizer_config",
    "get_existing_tokens",
    "find_missing_rna_tokens",
    "append_tokens_to_config",
    "update_tokenizer_with_rna_classes",
    "estimate_tokens",
    "estimate_gff_tokens",
    "find_gene_boundaries",
    "chunk_sequence_with_overlap",
    "chunk_gff_with_overlap",
    "should_chunk_annotation",
    "validate_chunks",
    "build_gene_hierarchy",
    "group_features_by_gene_simple",
    "create_gene_prediction_dataset_with_chunking",
    # dataset
    "MixedTaskDataset",
    "SmartBatchSampler",
    "DynamicPaddingCollator",
]