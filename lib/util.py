import sys
from pathlib import Path


#######################
#####  Bake Data  #####
#######################


def print_run_stats(stats, chunk_stats, validation, output_path):
    """Print comprehensive run statistics"""
    print(f"\n{'='*60}")
    print("Run Statistics")
    print(f"{'='*60}")
    
    print(f"\nChunking:")
    print(f"  Total chunks:     {chunk_stats['total_chunks']}")
    print(f"  Backtrack events: {chunk_stats['backtrack_count']}")
    
    if chunk_stats["genes_per_chunk"]:
        avg_genes = sum(chunk_stats["genes_per_chunk"]) / len(chunk_stats["genes_per_chunk"])
        print(f"  Avg genes/chunk:  {avg_genes:.2f}")
    
    if chunk_stats["chunk_sizes"]:
        avg_size = sum(chunk_stats["chunk_sizes"]) / len(chunk_stats["chunk_sizes"])
        print(f"  Avg chunk size:   {avg_size/1000:.1f} kb")
    
    print(f"\nData Augmentation:")
    print(f"  Raw chunks:       {stats['raw_count']}")
    print(f"  Augmented chunks: {stats['aug_count']}")
    print(f"  Total samples:    {stats['total_samples']}")
    
    print(f"\nOutput:")
    print(f"  Binary file:      {output_path}")
    print(f"  File size:        {stats.get('file_size', 0) / 1024:.1f} KB")
    
    if validation:
        ds.print_validation_stats(validation)