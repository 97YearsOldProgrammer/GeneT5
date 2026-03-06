import subprocess
import pathlib
import json

import lib.data as ds

from lib.bake._discovery import decompress_to_temp, find_genome_files, BakeJob


#################################
#####  Species Processing  ######
#################################


def run_parse_data(species_name, fasta_path, gff_path, output_dir, limit, log_dir, tokenizer_path=None, n_workers=1, compress=None):
    """Run parse_data.py for a single species"""

    cmd = [
        "python3", "train/parse_data.py",
        str(fasta_path),
        str(gff_path),
        str(output_dir),
        "--limit", str(limit),
        "--n_workers", str(n_workers),
        "--canonical_only",
        "--fast_tokenizer",
    ]

    if tokenizer_path:
        cmd.extend(["--tokenizer", str(tokenizer_path)])

    if compress:
        cmd.extend(["--compress", compress])

    log_file = log_dir / f"{species_name}.log"

    try:
        with open(log_file, 'w') as log:
            log.write(f"{'='*20} PARSING: {species_name} {'='*20}\n")
            log.write(f"FASTA: {fasta_path}\n")
            log.write(f"GFF:   {gff_path}\n")
            log.write(f"Limit: {limit} bp\n")
            log.write(f"{'='*60}\n\n")
            log.flush()

            result = subprocess.run(
                cmd,
                stdout = log,
                stderr = subprocess.STDOUT,
                text   = True,
            )

        # Extract error from log if failed
        error_msg = None
        if result.returncode != 0:
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()

                # Filter out header lines
                content_lines = [l.strip() for l in lines if l.strip() and not l.startswith('=')]

                # Check if only header present (crashed immediately)
                if len(content_lines) <= 4:
                    error_msg = "Crashed immediately (likely OOM or file error)"
                else:
                    # Look for traceback or error
                    for i, line in enumerate(lines):
                        if 'Error' in line or 'Exception' in line or 'Killed' in line:
                            error_msg = line.strip()[:100]
                            break
                        if 'MemoryError' in line:
                            error_msg = "MemoryError (OOM)"
                            break

                    # Fallback: get last non-empty line
                    if error_msg is None:
                        non_empty = [l.strip() for l in lines if l.strip()]
                        error_msg = non_empty[-1][:100] if non_empty else f"Exit code {result.returncode}"
            except Exception:
                error_msg = f"Exit code {result.returncode}"

        return {
            "species":  species_name,
            "success":  result.returncode == 0,
            "error":    error_msg,
            "log_file": str(log_file),
            "output":   str(output_dir),
        }

    except Exception as e:
        return {
            "species":  species_name,
            "success":  False,
            "error":    str(e),
            "log_file": str(log_file),
        }


def convert_binary_to_tar(binary_path, species_name, maxcount=50000):
    """Convert training.bin to WebDataset tar shards, return sample count"""

    import webdataset as wds

    binary_path = pathlib.Path(binary_path)
    shard_dir   = binary_path.parent
    pattern     = str(shard_dir / f"{species_name}-%06d.tar")

    total_written = 0
    with wds.ShardWriter(pattern, maxcount=maxcount) as sink:
        for idx, chunk in ds.iter_binary(binary_path):
            safe_seqid = chunk.seqid.replace(".", "_")
            key        = f"{safe_seqid}_{chunk.start}_{chunk.end}_{idx:06d}"

            meta = {
                "seqid":    chunk.seqid,
                "start":    chunk.start,
                "end":      chunk.end,
                "gene_ids": chunk.gene_ids,
            }

            sink.write({
                "__key__":    key,
                "input.txt":  chunk.get_input_text(),
                "target.txt": chunk.get_target_text(),
                "meta.json":  json.dumps(meta),
            })
            total_written += 1

    # Remove the binary after successful conversion
    binary_path.unlink()

    return total_written


def convert_binary_to_packed_tar(binary_path, species_name, tokenizer,
                                  max_seq_len=8192, maxcount=50000):
    """SPFHP-pack pre-tokenized binary chunks into fixed-length tar shards

    Shortest-Pack-First Histogram Packing (Krell et al. 2021): operates on
    the length histogram for near-optimal packing (<1% waste). Loads all
    samples into RAM for global optimization — no sort_buffer needed.
    Uses cached token IDs from binary (no re-tokenization).
    """

    import numpy as np
    import webdataset as wds
    from threading import Thread
    from queue import Queue

    binary_path = pathlib.Path(binary_path)
    shard_dir   = binary_path.parent
    pattern     = str(shard_dir / f"{species_name}-%06d.tar")
    pad_id      = tokenizer.pad_token_id

    total_written    = 0
    fallback_count   = 0

    # ── Phase 1: Load all samples ──────────────────────────────────
    # Prefetch in background thread
    prefetch_q = Queue(maxsize=4096)

    def _prefetch_worker():
        for idx, in_ids, tgt_ids in ds.iter_binary_token_ids(binary_path):
            prefetch_q.put((idx, in_ids, tgt_ids))
        prefetch_q.put(None)

    reader = Thread(target=_prefetch_worker, daemon=True)
    reader.start()

    all_samples = []

    while True:
        item = prefetch_q.get()
        if item is None:
            break

        idx, input_ids, target_ids = item

        if input_ids is not None:
            full_ids = input_ids + target_ids
            pfx_len  = len(input_ids)
        else:
            fallback_count += 1
            chunk      = ds.read_chunk_at_index(binary_path, idx)
            input_ids  = tokenizer.encode(chunk.get_input_text(),  add_special_tokens=False)
            target_ids = tokenizer.encode(chunk.get_target_text(), add_special_tokens=False)
            full_ids   = input_ids + target_ids
            pfx_len    = len(input_ids)

        if len(full_ids) > max_seq_len:
            full_ids = full_ids[:max_seq_len]
            pfx_len  = min(pfx_len, max_seq_len)

        all_samples.append((np.array(full_ids, dtype=np.int32), pfx_len))

    reader.join()
    total_raw = len(all_samples)

    if total_raw == 0:
        binary_path.unlink()
        return 0, 0

    # ── Phase 2: SPFHP — shortest-pack-first with O(log n) lookup ──
    from sortedcontainers import SortedList

    # Group samples by length, process longest first
    by_length = {}
    for i, (ids, plen) in enumerate(all_samples):
        slen = len(ids)
        by_length.setdefault(slen, []).append(i)

    sorted_lengths = sorted(by_length.keys(), reverse=True)

    # SortedList of (used_space, bin_idx) — O(log n) bisect for best-fit
    bins       = []
    open_bins  = SortedList()  # (used, bin_idx) sorted ascending by used

    for slen in sorted_lengths:
        indices = by_length[slen]

        for sample_idx in indices:
            ids, plen = all_samples[sample_idx]

            # Best-fit: find the fullest bin that still has room
            # We need used + slen <= max_seq_len, i.e. used <= max_seq_len - slen
            max_used = max_seq_len - slen
            placed   = False

            if open_bins:
                # Bisect right to find the rightmost bin with used <= max_used
                pos = open_bins.bisect_right((max_used, float('inf'))) - 1
                if pos >= 0:
                    used, bidx = open_bins[pos]
                    if used + slen <= max_seq_len:
                        open_bins.pop(pos)
                        start = bins[bidx][1]
                        bins[bidx][0][start:start + slen] = ids
                        bins[bidx][1] += slen
                        bins[bidx][2].append(slen)
                        bins[bidx][3].append(plen)
                        new_remaining = max_seq_len - bins[bidx][1]
                        # Only re-add if bin can still fit something (min sample ~ 100 tokens)
                        if new_remaining >= 100:
                            open_bins.add((bins[bidx][1], bidx))
                        placed = True

            if not placed:
                arr = np.empty(max_seq_len, dtype=np.int32)
                arr[:slen] = ids
                bidx = len(bins)
                bins.append([arr, slen, [slen], [plen]])
                if max_seq_len - slen >= 100:
                    open_bins.add((slen, bidx))

    # ── Phase 3: Write all bins to tar ─────────────────────────────
    sink = wds.ShardWriter(pattern, maxcount=maxcount)

    total_real_tokens = 0
    total_pad_tokens  = 0

    for b in bins:
        arr, used, seq_lens, pfx_lens = b
        arr[used:] = pad_id
        sink.write({
            "__key__":    f"pack_{total_written:06d}",
            "packed.npy": arr,
            "meta.json":  json.dumps({"seq_lens": seq_lens,
                                      "prefix_lens": pfx_lens}),
        })
        total_written    += 1
        total_real_tokens += used
        total_pad_tokens  += max_seq_len - used

    sink.close()

    # Report packing efficiency
    total_tokens = total_real_tokens + total_pad_tokens
    waste_pct    = 100 * total_pad_tokens / total_tokens if total_tokens else 0
    print(f"  SPFHP: {total_raw} samples -> {total_written} packs "
          f"(waste {waste_pct:.1f}%, {total_pad_tokens:,} pad tokens)")

    if fallback_count > 0:
        print(f"  WARNING: {fallback_count}/{total_raw} chunks lacked cached IDs (re-tokenized)")

    binary_path.unlink()
    return total_written, total_raw


def process_species(job):
    """Worker function for parallel species processing

    Accepts a BakeJob dataclass instance
    """

    species_raw_dir = pathlib.Path(job.raw_dir) / job.species

    if not species_raw_dir.exists():
        return {
            "species": job.species,
            "success": False,
            "error":   f"Directory not found: {species_raw_dir}",
        }

    fasta_file, gff_file = find_genome_files(species_raw_dir)

    if fasta_file is None or gff_file is None:
        return {
            "species": job.species,
            "success": False,
            "error":   f"Missing FASTA or GFF in: {species_raw_dir}",
        }

    output_dir = pathlib.Path(job.output_dir) / job.species
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pre-decompress gzipped files for faster parsing
    decompressed_files = []
    try:
        if str(fasta_file).endswith('.gz'):
            fasta_file = decompress_to_temp(fasta_file, output_dir)
            decompressed_files.append(fasta_file)
        if str(gff_file).endswith('.gz'):
            gff_file = decompress_to_temp(gff_file, output_dir)
            decompressed_files.append(gff_file)
    except Exception:
        pass

    result = run_parse_data(
        job.species, fasta_file, gff_file, output_dir, job.window_size,
        pathlib.Path(job.log_dir), job.tokenizer, job.n_workers,
        job.compress,
    )

    # Clean up decompressed files
    for f in decompressed_files:
        try:
            if f.exists():
                f.unlink()
        except Exception:
            pass

    # Convert binary to tar shards if requested and parse succeeded
    if result["success"] and job.output_format == "tar":
        binary_path = output_dir / "training.bin"
        if binary_path.exists():
            try:
                total = convert_binary_to_tar(binary_path, job.species)
                result["total_samples"] = total
            except Exception as e:
                result["success"] = False
                result["error"]   = f"Tar conversion failed: {e}"

    # Packed tar format: tokenize + pack at bake time
    if result["success"] and job.output_format == "packed_tar":
        binary_path = output_dir / "training.bin"
        if binary_path.exists():
            try:
                import lib.tokenizer.hf as tk_mod
                tokenizer    = tk_mod.GeneTokenizer(pathlib.Path(job.tokenizer))
                packed, raw  = convert_binary_to_packed_tar(
                    binary_path, job.species, tokenizer,
                    max_seq_len=job.pack_seq_len,
                )
                result["total_packed_samples"] = packed
                result["total_raw_samples"]    = raw
                result["total_samples"]        = packed
            except Exception as e:
                result["success"] = False
                result["error"]   = f"Packed tar conversion failed: {e}"

    return result


####################################
#####  Tokenizer Expansion  ########
####################################


def run_tokenizer_expansion(token_file, tokenizer_path, output_path=None, dry_run=False):
    """Run append_tk.py to expand tokenizer with new tokens"""

    token_path = pathlib.Path(token_file)

    if not token_path.exists():
        return {
            "success": False,
            "error":   f"Token file not found: {token_file}",
        }

    with open(token_path, 'r') as f:
        tokens = [line.strip() for line in f if line.strip()]

    if not tokens:
        return {
            "success": True,
            "added":   0,
            "message": "No tokens to add (file empty)",
        }

    cmd = [
        "python3", "bin/append_tk.py",
        token_file,
        tokenizer_path,
    ]

    if output_path:
        cmd.extend(["--output", output_path])

    if dry_run:
        cmd.append("--dry_run")

    try:
        result = subprocess.run(
            cmd,
            capture_output = True,
            text           = True,
        )

        return {
            "success":        result.returncode == 0,
            "stdout":         result.stdout,
            "stderr":         result.stderr,
            "tokens_in_file": len(tokens),
        }

    except Exception as e:
        return {
            "success": False,
            "error":   str(e),
        }
