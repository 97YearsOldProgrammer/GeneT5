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
init/          - Model initialization
  init_model.py  - Model init
  setup-nfs.sh   - NFS setup
lib/           - Core library code
  model/         - Model architectures and builders
    seq2seq.py     - GeneT5 decoder-only architecture
    build.py       - Model builder (GeneT5 from DNABERT-2)
  tokenizer/     - Tokenization
    hf.py          - GeneTokenizer (DNABERT-2 extended)
  data/          - Data loading, packing, dynamic padding
  blocks/        - Model components (encoder, decoder, MoE, sparse attention)
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

```
/workspace/
  raw/                    - Raw GFF + FASTA by species
    {Genus.species}/        - Binomial naming (e.g. H.sapiens, D.melanogaster)
      fna.gz                - Compressed genome FASTA
      gff.gz                - Compressed gene annotations
      fna                   - Decompressed FASTA
      fna.fai               - FASTA index (created by pyfaidx)

  baked/                  - Packed training datasets
    GeneT5/                 - GeneT5 SFT pipeline baked data
      {run_name}/           - Convention: feb{DD}_s{N}_v{V}_e{V}
        {species}/            - Per-species directory
          shard_000.tar       - WebDataset training shards
          stats.json          - Species baking stats
          gene_index.json     - Gene index sidecar (avoids re-parsing GFF)
        validation.bin        - Merged held-out species binary
        eval.json             - Eval samples (diverse coding genes)
        bake_config.json      - Full baking config (species, params, totals)

  model/                  - Model checkpoints
    GeneT5/
      init/                 - Initialized (untrained) models
        init_moe4/            - MoE-4 expert init from DNABERT-2
      feb{DD}_*_t{N}/       - Experiment runs (date + trial number)
        finetune_config.json  - Full training config
        training_log.csv      - Step-level training metrics
        eval_log.csv          - Per-epoch evaluation metrics
        memory_*.csv          - Memory monitoring logs
        best_model.pt         - Best checkpoint (by val loss)
        checkpoint_latest.pt  - Latest epoch checkpoint
        checkpoint_step_*.pt  - Step-level checkpoints

  logs/                   - Baking and misc logs
    baker/                  - Per-species baking logs
```

## Output Directory Management

All model outputs MUST go under `/workspace/model/GeneT5/` with descriptive names:
- SFT experiments: `feb{DD}_*_t{N}/` (date + trial)
- NEVER save to random ad-hoc directories outside `model/GeneT5/`

## Training Pipeline

Full pipeline from raw genomes to trained models:

```
Step 1: INIT MODEL
  init/init_model.py → /workspace/model/GeneT5/init/init_moe4/
  Builds GeneT5 from DNABERT-2 encoder weights

Step 2: BAKE SFT DATA
  train/bake_data → /workspace/baked/GeneT5/{run}/
  Raw GFF+FASTA → tokenized WebDataset shards (.tar)
  Distributed: train/bake.sh --worker <IP>

Step 3: SFT TRAINING
  train/finet → /workspace/model/GeneT5/{exp}/
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

- 2x DGX Spark (GB10, 96GB unified memory each)
- Master: 192.168.100.10, Worker: 192.168.100.11
- NCCL over ConnectX-7 InfiniBand (RDMA)
- NGC pytorch:25.12-py3 container on both nodes
- Container name: `gt5`, launch scripts: `start-gt5.sh` (master), `start-worker.sh` (worker)
- Distributed training: `train/sft.sh` handles one-click launch, auto-patches, cleanup

## Training Logs & Experiments

- Experiment doc: `docs/sft_experiments.md` (all trials, configs, results, comparisons)
- Previous trial notes: `/workspace/logs/GeneT5/notes/trial1_feb10_run1.md`
- Current run outputs: `/workspace/model/GeneT5/feb12_trail1/` (training_log.csv, eval_log.csv, memory_*.csv)
- Key finding: aggressive LR (3e-4) + small effective batch (256) = fast train convergence but severe overfitting (val 20x train); conservative LR (1e-4) + large batch (1024) + label smoothing = better generalization

## Lessons Learned

- BLT entropy patching (2026-02-22 to 2026-02-23): DNA's 4-base alphabet compresses entropy into a 0.04-nat useful band (threshold 1.34-1.37), giving only 12.6% compression vs BPE with extreme species variance. Reverted to BPE tokenization.

## Commands

- Init model: `PYTHONPATH=. python init/init_model.py --help`
- Bake data: `PYTHONPATH=. python train/bake_data <raw_dir> --tokenizer <path> [--train] [--eval] [--val_species X]`
- Fine-tune: `PYTHONPATH=. python train/finet <data_dir> <output_dir> <model_path>`
- GRPO: `PYTHONPATH=. python train/grpo --help`
- Prep GRPO: `PYTHONPATH=. python train/prep_grpo --help`
- Eval F1: `PYTHONPATH=. python eval/eval_f1 --help`
- Eval BUSCO: `PYTHONPATH=. python eval/eval_busco --help`
- Distributed SFT: `train/sft.sh <data_dir> <output_dir> <model_path> --worker <IP> [flags]`
- Distributed bake: `train/bake.sh --worker <IP> --tokenizer <path> <raw_dir> [args...]`
- Run tests: `PYTHONPATH=. python tests/<test_file>.py`
