# Gene Finder Survey

Reference for GeneT5 gene finder design decisions

## What Gene Finders Predict

Gene finders predict the **full internal structure** of genes from raw DNA

| Level | What | Example |
|-------|------|---------|
| Gene boundary | start/end of the gene locus | chr1:10000-15000 |
| Exon boundaries | each exon's start/end coordinates | exon1: 10200-10800, exon2: 11500-12300 |
| CDS boundaries | where coding sequence starts/ends within exons | CDS starts at 10400 (after 5'UTR) |
| Strand | which DNA strand | + or - |
| Gene type | protein_coding, lncrna, etc | protein_coding |

Introns are implied (gaps between exons), UTRs are implied (exon regions outside CDS)

## Evaluation Metrics

- **Nucleotide F1** -- per-base classification (coding vs non-coding)
- **Exon F1** -- each exon boundary must be exactly correct
- **Gene F1** -- ALL exons of a gene must be exactly correct (strictest)

## Modern DL Gene Finders

### Tiberius (2024, SOTA)

- Paper: [Bioinformatics 2024](https://doi.org/10.1093/bioinformatics/btae685)
- Code: [github.com/Gaius-Augustus/Tiberius](https://github.com/Gaius-Augustus/Tiberius)
- From the AUGUSTUS team (successor)
- Architecture: CNN + LSTM + differentiable HMM (end-to-end)
- Parameters: ~8M
- **Train window: 10kb** (9,999 bp)
- **Inference window: 500kb** (500,004 bp)
- Processes one strand at a time, reverse complement for other strand
- HMM layer with 15 states enforces valid gene structure
- F1: **89.7% exon, 55.1% gene**

### Helixer (2021-2025)

- Paper: [Nature Methods 2025](https://doi.org/10.1038/s41592-025-02939-1)
- Code: [github.com/usadellab/Helixer](https://github.com/usadellab/Helixer)
- Architecture: 4-layer BiLSTM + HMM post-processing
- Parameters: ~5.4M
- **Train window: 20kb** (20,000 bp)
- 10 bases per timestep
- Overlapping sliding window at inference, averaged softmax at boundaries
- Cross-species: single model for fungi/plants/vertebrates/invertebrates
- F1: **72.9% exon, 19.3% gene**

### AUGUSTUS (2003-present, classic)

- HMM-based (no deep learning)
- Window: 200kb default (`--maxDNAPieceSize`)
- Species-specific parameter training
- Still widely used in annotation pipelines (BRAKER3)

## Key Design Patterns

1. **Train short, infer long** -- all modern DL gene finders train on 10-20kb windows and extrapolate to 200-500kb at inference
2. **Overlapping windows** -- stitch predictions at inference by overlapping and averaging
3. **HMM post-processing** -- enforce biological constraints (valid exon/intron/CDS structure)
4. **Per-base classification** -- most predict a class per nucleotide, then decode into gene structure
5. **CNN+LSTM dominant** -- no transformer-based gene finders exist yet (GeneT5 would be the first)

## GeneT5 Approach

| Aspect | Tiberius | Helixer | GeneT5 |
|--------|----------|---------|--------|
| Architecture | CNN+LSTM+HMM | BiLSTM+HMM | Bidirectional transformer + MDLM diffusion |
| Backbone | CNN+LSTM | BiLSTM | DNABERT-2 (pretrained, MoE upcycled) |
| Compression | None | 10bp/timestep | None (full sequence) |
| Decoding | HMM (15 states) | HMM post-proc | Iterative parallel unmasking |
| Train window | 10kb | 20kb | 20kb |
| Inference window | 500kb | variable | extrapolate (perceiver scales) |
| Output format | per-base class | per-base class | compressed GFF tokens |
| Parameters | ~8M | ~5.4M | ~168M |

### Key difference

Tiberius and Helixer predict a **class per nucleotide** (intergenic/intron/exon/CDS), then decode structure with HMM

GeneT5 directly generates **compressed GFF annotation** via masked diffusion, no per-base classification needed
