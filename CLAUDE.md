# GeneT5 Project Instructions

## Coding Standards

### Visual Formatting
- Vertical alignment: align all `=` operators across consecutive lines
- No top-level file descriptions or metadata comments
- Add blank line between docstring and function body

### Code Structure & Imports
- All `def` functions go in `lib/` modules; executables only import and run
- Use aliased imports: `import tokenizer as tk` (not `from x import y`)
- Standard library: `import math` directly
- No type hints in function signatures

### Documentation & Comments
- Single concise line only
- No trailing periods
- Do not describe return values
- Minimal complexity explanations

### Testing & Execution
- Verify with minimal self-contained test case
- No PyTorch in tests (avoid env errors)
- Use standard libraries or lightweight alternatives

## Testing Workflow

- Create test scripts in `tests/` during development
- Run with fake/synthetic data
- **Delete test files after feature confirmed working**
- Prefix temporary tests: `test_tmp_*` or `test_dev_*`

## Project Structure

```
train/         - SFT/GRPO training pipeline
  finet          - SFT fine-tuning
  grpo           - GRPO reinforcement learning
  prep_grpo      - Multi-species GRPO data prep
  bake_data      - Data baking for SFT
  sft.sh         - Distributed SFT launcher
  bake.sh        - Distributed bake launcher
init/          - Model initialization and GPU setup
  init_model.py  - Model init
  gpu-exec       - Run commands inside gt5 container from host
  gpu/           - Container launch and migration scripts
    migrate_data.sh     - Host data layout migration
    start-gt5.sh        - GeneT5 master container
    start-gt5-worker.sh - GeneT5 worker container
    start-fa.sh         - FlashAlignment container
    start-pqc.sh        - ProteinQC container
lib/           - Core library code
  model/         - Model architectures and builders
    genet5.py      - GeneT5 model architecture
    build.py       - Model builder (GeneT5 from DNABERT-2)
  tokenizer/     - Tokenization
    hf.py          - GeneTokenizer (DNABERT-2 extended)
  data/          - Data loading, packing, dynamic padding
  blocks/        - Model components (transformer layers, MoE, attention)
  train/         - Training loop, logging, memory monitoring, eval hooks
    loop.py        - Distributed training, optimizers, checkpoints
    logger.py      - CSV + stdout training logger
    memwatch.py    - Background memory monitor
    eval_hook.py   - Checkpoint evaluation with F1 metrics
  inference/     - Inference engine and output parsing
    engine.py      - GeneT5Inference wrapper, device detection
    output.py      - Model output parser, GFF3 conversion
  grpo/          - GRPO reinforcement learning
    algo.py        - GRPO dataset, loss, advantage computation
    reward.py      - Exon/gene F1 reward functions
  bake/          - Data baking utilities
    databaker.py   - Species processing, tokenizer expansion
    time.py        - Time formatting helpers
  nosing/        - Post-processing (exon/intron/protein extraction)
  util/          - Re-export aggregator (import lib.util as util still works)
eval/          - Evaluation (BUSCO, F1, GFF parsing, eval scripts)
  eval_f1        - F1 evaluation (nucleotide/exon/gene)
  eval_busco     - BUSCO evaluation
tests/         - Development benchmarks (temporary)
docs/          - Design documents
```

## Data Layout

Inside the `gt5` container, `/workspace/code` is the project source and `/workspace/data/`
subdirectories are individually mounted from the host. Host data is root-owned (created by Docker).

```
/workspace/
  code/                   - Project source (mounted from host Code/GeneT5)
  data/                   - Per-project data (mounted from host Data/GeneT5)
    raw/                    - Raw GFF + FASTA by species (symlink to shared genome/raw)
      {Genus.species}/        - Binomial naming (e.g. H.sapiens, D.melanogaster)
        fna.gz                - Compressed genome FASTA
        gff.gz                - Compressed gene annotations
        fna                   - Decompressed FASTA
        fna.fai               - FASTA index (created by pyfaidx)
    baked/                  - Packed training datasets
      {run_name}/             - Convention: {mon}{DD}_s{N}_w{W}_p{P}
        {species}/              - Per-species directory
          shard_000.tar         - WebDataset training shards
          stats.json            - Species baking stats
          gene_index.json       - Gene index sidecar (avoids re-parsing GFF)
        validation.bin          - Merged held-out species binary
        eval.json               - Eval samples (diverse coding genes)
        bake_config.json        - Full baking config (species, params, totals)
    model/                  - Model checkpoints
      init/                   - Initialized (untrained) models
        init_dense_24L/         - Dense 24L init from DNABERT-2
      {mon}{DD}_*_t{N}/       - Experiment runs (date + trial number)
        finetune_config.json    - Full training config
        training_log.csv        - Step-level training metrics
        eval_log.csv            - Per-epoch evaluation metrics
        memory_*.csv            - Memory monitoring logs
        best_model.pt           - Best checkpoint (by val loss)
        checkpoint_latest.pt    - Latest epoch checkpoint
        checkpoint_step_*.pt    - Step-level checkpoints
    logs/                   - Baking and misc logs
      baker/                  - Per-species baking logs
      sft/                    - SFT training logs
```

## Output Directory Management

All model outputs MUST go under `/workspace/data/model/` with descriptive names:
- SFT experiments: `{mon}{DD}_*_t{N}/` (month + date + trial)
- NEVER save to random ad-hoc directories outside `data/model/`

## Training Pipeline

Full pipeline from raw genomes to trained models:

```
Step 1: INIT MODEL
  init/init_model.py → /workspace/data/model/init/init_dense_24L/
  Builds GeneT5 from DNABERT-2 encoder weights (dense 24L)

Step 2: BAKE SFT DATA
  train/bake_data → /workspace/data/baked/{run}/
  Raw GFF+FASTA → tokenized WebDataset shards (.tar)
  Distributed: train/bake.sh --worker <IP>

Step 3: SFT TRAINING
  train/diffusion_finet → /workspace/data/model/{exp}/
  Trains GeneT5 on baked tar shards
  Distributed: train/sft.sh --worker <IP>

Step 4: EVALUATION
  eval/eval_f1 → nucleotide/exon/gene F1 scores
  eval/eval_busco → BUSCO completeness assessment

Step 5: GRPO (reinforcement learning)
  train/prep_grpo → prepare multi-species GRPO data
  train/grpo → GRPO training with exon/gene F1 rewards
```

## Sparse Attention Constraints

- Block size: power of 2 and >= 16
- Valid: 16, 32, 64, 128
- Window size: divisible by block size

## torch.compile

- Use `dynamic=None` (not `True`) to avoid compilation hangs on GB10
- NGC 25.12 requires `cluster_dims` monkeypatch in `triton_heuristics.py`
- Inductor cache at `/tmp/torchinductor_root/` — reusable across runs
- First batch compiles (mark_dynamic on seq dim), then all shapes use cached kernels

## Hardware

- 2x DGX Spark (GB10, 128GB unified memory each)
- Master: spark-1089 (192.168.100.10), Worker: spark-0b7c (192.168.100.11)
- NCCL over ConnectX-7 InfiniBand (RDMA)
- NGC pytorch:26.02-py3-igpu container on both nodes
- Per-project containers: `gt5` (GeneT5), `fa` (FlashAlignment), `pqc` (ProteinQC)
- Container launch: `init/gpu/start-gt5.sh` (master), `init/gpu/start-gt5-worker.sh` (worker)
- Distributed training: `train/sft.sh` handles one-click launch, auto-patches, cleanup

## Training Logs & Experiments

- Experiment doc: `docs/sft_experiments.md` (all trials, configs, results, comparisons)
- Current init: `/workspace/data/model/init/init_dense_24L/` (230M dense, 24 layers)
- Current baked data: `/workspace/data/baked/mar12_s51_w20k_p18k/` (51 species, packed_tar)
- Key findings from Feb trials: aggressive LR (3e-4) + small effective batch (256) = fast convergence but severe overfitting; conservative LR (1e-4) + large batch (1024) + label smoothing = better generalization

## Lessons Learned

- BLT entropy patching (2026-02-22 to 2026-02-23): DNA's 4-base alphabet compresses entropy into a 0.04-nat useful band (threshold 1.34-1.37), giving only 12.6% compression vs BPE with extreme species variance. Reverted to BPE tokenization.

## Commands

Inside the `gt5` container (paths relative to `/workspace/code`):

- Init model: `python init/init_model.py --help`
- Bake data: `python train/bake_data /workspace/data/raw --tokenizer <path> [--train] [--eval] [--val_species X]`
- Fine-tune: `python train/diffusion_finet <data_dir> <output_dir> <model_path>`
- GRPO: `python train/grpo --help`
- Prep GRPO: `python train/prep_grpo --help`
- Eval F1: `python eval/eval_f1 --help`
- Eval BUSCO: `python eval/eval_busco --help`
- Distributed SFT: `train/sft.sh <data_dir> <output_dir> <model_path> --worker <IP> [flags]`
- Distributed bake: `train/bake.sh --worker <IP> --tokenizer <path> /workspace/data/raw [args...]`
- Run tests: `PYTHONPATH=. python tests/<test_file>.py`

Container management (from DGX host):

- Launch GeneT5 master: `bash ~/Code/GeneT5/init/gpu/start-gt5.sh`
- Launch GeneT5 worker: `bash ~/Code/GeneT5/init/gpu/start-gt5-worker.sh --daemon`
- Migrate data layout: `bash ~/Code/GeneT5/init/gpu/migrate_data.sh`
