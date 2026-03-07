# Entropy Model v2 Analysis

## Architecture

DNAEntropyModel trained at the base (nucleotide) level.

| Parameter | Value |
|-----------|-------|
| Parameters | 21M |
| Dimension | 512 |
| Layers | 8 |
| Heads | 8 |
| FF Dim | 1024 |
| Window | 512 |

## Training

- 5 epochs, distributed on 2x DGX Spark (GB10, 96GB unified memory each)
- Raw FASTA from 50 species across bacteria, fungi, and metazoa
- Best validation loss: 1.3284 (epoch 4)

---

## Benchmark Results

### Entropy Distribution

| Statistic | Value |
|-----------|-------|
| Mean | 1.2486 |
| Std | 0.2352 |
| Min | 0.0177 |
| Max | 1.5261 |

### Percentiles

| Percentile | Entropy |
|-----------|---------|
| p1 | 0.105 |
| p5 | 0.780 |
| p10 | 1.114 |
| p25 | 1.255 |
| p50 | 1.324 |
| p75 | 1.357 |
| p90 | 1.373 |
| p95 | 1.379 |
| p99 | 1.384 |

### Distribution Shape

The entropy histogram is bimodal. The vast majority of positions fall in the [1.30, 1.38] range, producing a sharp spike. Above 1.39 there is a cliff: almost no positions exceed this value. This compressed dynamic range is a direct consequence of the 4-base alphabet (see v3 rationale below).

---

## Threshold Sweep

Compression is measured relative to a fixed patch size of 8 bp (BLT paper default).

| Threshold | Avg bp/patch | Patches/seq | vs Fixed-8 |
|-----------|-------------|-------------|------------|
| 0.80 | 1.1 | high | -86.3% |
| 0.90 | 1.4 | high | -82.5% |
| 1.00 | 2.1 | high | -73.8% |
| 1.10 | 3.2 | high | -60.0% |
| 1.20 | 4.4 | medium | -45.0% |
| 1.30 | 5.8 | medium | -27.5% |
| 1.34 | 7.0 | medium | -14.4% |
| 1.35 | 8.1 | medium | +1.3% |
| 1.36 | 10.2 | low | +21.3% |
| 1.37 | 14.9 | low | +46.4% |
| 1.38 | 37.5 | very low | +78.6% |
| 1.39 | cliff | 1 patch | degenerate |
| 1.50 | cliff | 1 patch | degenerate |

Key observation: the usable threshold window is extremely narrow — only a 0.03-unit range (1.35 to 1.38) separates near-fixed behavior from degenerate single-patch behavior.

---

## BLT Paper Comparison

The BLT paper uses threshold 1.34 for English text over a 256-byte vocabulary.

| System | Threshold | Max Entropy | Threshold / Max | Regime |
|--------|-----------|-------------|-----------------|--------|
| BLT (English, 256-byte vocab) | 1.34 | 8.0 bits | 17% | aggressive compression |
| DNA v2 (base-level, 4-base vocab) | 1.35 | 2.0 bits | 68% | near-maximum entropy |

At 68% of maximum entropy the model is triggering patches only for positions that are near-maximally uncertain. This leaves almost no headroom. The BLT paper's 17% regime is fundamentally different: it carves out structure from a much richer entropy space.

---

## Biological Signal Analysis

### Strand Symmetry

Forward vs. reverse complement entropy correlation: 0.985. The model learned strand symmetry without explicit supervision.

### Codon Frame Awareness

Per-frame mean entropy across all three reading frames (offsets 0, 1, 2) is identical at approximately 1.248. The base-level model has no codon awareness.

### Splice Site Motifs

GT/AG canonical splice site dinucleotides produce only +0.026 entropy above genomic background. This is not a meaningful signal and does not produce reliable patch boundaries at exon-intron junctions.

### High-Entropy Peak Composition

Positions with entropy > 1.35 show uniform 4-mer distributions with no enriched sequence motifs. The high-entropy tail reflects general sequence complexity rather than specific biological features.

---

## Per-Species Entropy

| Species | Mean Entropy | Notes |
|---------|-------------|-------|
| A. nidulans | 1.3474 | Fungus, highest across dataset |
| B. subtilis | 1.3294 | Bacteria |
| S. cerevisiae | 1.3221 | Yeast |
| P. aeruginosa | 1.1717 | Bacteria, GC-biased genome |
| N. vectensis | 1.1332 | Sea anemone, repetitive genome, lowest |

Species with GC bias (P. aeruginosa) or repetitive content (N. vectensis) show lower mean entropy, which is expected. Fungal and yeast genomes cluster near the dataset mean.

---

## Codon-Level Entropy Model (v3) Rationale

The base-level model has a fundamental architectural limitation. With a 4-base alphabet the theoretical maximum entropy is log2(4) = 2.0 bits. In practice genomic sequences are near-maximally complex, so almost all positions land near the ceiling. This produces the narrow dynamic range observed in the threshold sweep.

### Solution: 3-mer Codon Tokenization

Train on non-overlapping 3-mer codons (64-token vocabulary).

| Property | Base-level (v2) | Codon-level (v3) |
|----------|----------------|-----------------|
| Vocabulary | 4 | 64 |
| Max entropy | 2.0 bits | 6.0 bits |
| Threshold 1.34 / Max | 67% | 22% |
| Regime match to BLT paper | No | Yes |

At 22% of maximum entropy the threshold operates in the same compression regime as the BLT paper. This restores dynamic range and makes threshold selection robust.

### Training Augmentation

- Train on all 3 reading frames (offset 0, 1, 2) per window to avoid frame-of-reference bias
- Reverse complement augmentation is not required; v2 already demonstrated the model learns strand symmetry

### Biological Motivation

Codon boundaries are biologically meaningful. Each 3-mer corresponds to an amino acid in coding sequence. A codon-level model may naturally learn CDS vs. non-CDS entropy differences that base-level models cannot resolve due to ceiling effects.
