from ._parser import (
    anti,
    parse_fasta,
    parse_gff,
    group_features_by_seqid,
    group_features_by_parent,
    build_transcript_map,
    build_feature_hierarchy,
    format_annotation_target,
    create_gene_prediction_dataset,
    extract_feature_types,
    extract_biotypes,
    save_dataset,
    load_dataset,
    GENE_FEATURE_TYPES,
)

from ._chunking import (
    estimate_tokens,
    estimate_gff_tokens,
    find_gene_boundaries,
    chunk_sequence_with_overlap,
    chunk_gff_with_overlap,
    should_chunk_annotation,
    validate_chunks,
    build_transcript_to_gene_map,
    build_transcript_info,
    group_features_by_gene_with_biotype,
    create_gene_prediction_dataset_with_chunking,
    format_annotation_target_chunked,
)

# optional torch-dependent imports
try:
    from .dataset import (
        LazyDataset,
        MixedTaskDataset,
        SmartBatchSampler,
        DynamicPaddingCollator,
    )
except ImportError:
    LazyDataset            = None
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
    "build_transcript_map",
    "build_feature_hierarchy",
    "format_annotation_target",
    "create_gene_prediction_dataset",
    "extract_feature_types",
    "extract_biotypes",
    "save_dataset",
    "load_dataset",
    "GENE_FEATURE_TYPES",
    
    # chunking
    "estimate_tokens",
    "estimate_gff_tokens",
    "find_gene_boundaries",
    "chunk_sequence_with_overlap",
    "chunk_gff_with_overlap",
    "should_chunk_annotation",
    "validate_chunks",
    "build_transcript_to_gene_map",
    "build_transcript_info",
    "group_features_by_gene_with_biotype",
    "create_gene_prediction_dataset_with_chunking",
    "format_annotation_target_chunked",
    
    # dataset
    "LazyDataset",
    "MixedTaskDataset",
    "SmartBatchSampler",
    "DynamicPaddingCollator",
]