import csv
import json
import time
import pathlib
from collections import Counter

import torch

import lib.util._output as output_lib


class CheckpointEvaluator:
    """Runs inference on held-out eval data and computes exon/gene F1 metrics"""

    def __init__(self, eval_data_path, tokenizer, num_samples=20, max_length=512,
                 temperature=1.0, top_k=50, top_p=0.9):

        with open(eval_data_path, 'r') as f:
            all_samples = json.load(f)

        self.samples     = all_samples[:num_samples]
        self.tokenizer   = tokenizer
        self.max_length  = max_length
        self.temperature = temperature
        self.top_k       = top_k
        self.top_p       = top_p
        self.parser      = output_lib.ModelOutputParser(strict=False)

    @torch.no_grad()
    def evaluate(self, model, device, dtype):
        """Run inference on eval samples and compute F1 metrics"""

        was_training = model.training
        model.eval()

        t0              = time.time()
        all_pred_exons  = []
        all_ref_exons   = []
        all_pred_genes  = []
        all_ref_genes   = []
        total_predicted = 0
        total_ref       = 0

        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id

        for sample in self.samples:
            sequence     = sample["sequence"]
            ref_features = sample["ref_features"]

            # Encode input DNA sequence
            encoded   = self.tokenizer.encode(sequence)
            input_ids = torch.tensor([encoded], dtype=torch.long, device=device)

            # Generate
            with torch.amp.autocast(device.type, dtype=dtype):
                generated = model.generate(
                    encoder_input_ids = input_ids,
                    max_length        = self.max_length,
                    temperature       = self.temperature,
                    top_k             = self.top_k,
                    top_p             = self.top_p,
                    bos_token_id      = bos_id,
                    eos_token_id      = eos_id,
                    pad_token_id      = pad_id,
                )

            # Decode output tokens to text
            output_ids = generated[0].tolist()
            raw_output = self.tokenizer.decode(output_ids)

            # Clean and parse
            cleaned      = self._clean_output(raw_output)
            parsed_genes = self.parser.parse_sequence(cleaned)

            # Extract predicted exon DNA strings
            pred_exon_dna = set()
            pred_gene_map = {}
            for gi, gene in enumerate(parsed_genes):
                for exon_seq in gene.exons:
                    pred_exon_dna.add(exon_seq)
                if gene.exons:
                    pred_gene_map[gi] = set(gene.exons)

            # Extract reference exon DNA from sequence + ref_features
            ref_exon_dna = set()
            ref_gene_map = {}
            for rf in ref_features:
                if rf.get("type", "exon").lower() != "exon":
                    continue
                dna  = sequence[rf["start"]:rf["end"]]
                ref_exon_dna.add(dna)
                gkey = rf.get("gene_idx", rf.get("gene_id", 0))
                if gkey not in ref_gene_map:
                    ref_gene_map[gkey] = set()
                ref_gene_map[gkey].add(dna)

            total_predicted += len(pred_exon_dna)
            total_ref       += len(ref_exon_dna)

            all_pred_exons.append(pred_exon_dna)
            all_ref_exons.append(ref_exon_dna)

            # Gene signatures: frozenset of exon DNA strings per gene
            pred_gene_sigs = Counter(frozenset(exons) for exons in pred_gene_map.values() if exons)
            ref_gene_sigs  = Counter(frozenset(exons) for exons in ref_gene_map.values() if exons)

            all_pred_genes.append(pred_gene_sigs)
            all_ref_genes.append(ref_gene_sigs)

        # Aggregate exon F1
        exon_metrics = self._compute_aggregate_f1(all_pred_exons, all_ref_exons)

        # Aggregate gene F1
        gene_metrics = self._compute_aggregate_f1(all_pred_genes, all_ref_genes)

        elapsed = time.time() - t0

        if was_training:
            model.train()

        return {
            "exon_f1":        exon_metrics["f1"],
            "exon_precision": exon_metrics["precision"],
            "exon_recall":    exon_metrics["recall"],
            "exon_tp":        exon_metrics["tp"],
            "exon_fp":        exon_metrics["fp"],
            "exon_fn":        exon_metrics["fn"],
            "gene_f1":        gene_metrics["f1"],
            "gene_precision": gene_metrics["precision"],
            "gene_recall":    gene_metrics["recall"],
            "gene_tp":        gene_metrics["tp"],
            "gene_fp":        gene_metrics["fp"],
            "gene_fn":        gene_metrics["fn"],
            "num_samples":    len(self.samples),
            "num_predicted":  total_predicted,
            "num_ref":        total_ref,
            "eval_sec":       round(elapsed, 1),
        }

    def _compute_aggregate_f1(self, all_pred, all_ref):
        """Compute aggregate F1 across all samples (handles set or Counter)"""

        total_tp = 0
        total_fp = 0
        total_fn = 0

        for pred_set, ref_set in zip(all_pred, all_ref):
            if isinstance(pred_set, Counter):
                # Counter intersection: min of counts for matching keys
                common   = pred_set & ref_set
                tp       = sum(common.values())
                fp       = sum((pred_set - ref_set).values())
                fn       = sum((ref_set - pred_set).values())
            else:
                tp = len(pred_set & ref_set)
                fp = len(pred_set - ref_set)
                fn = len(ref_set - pred_set)
            total_tp += tp
            total_fp += fp
            total_fn += fn

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "f1":        round(f1, 4),
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "tp":        total_tp,
            "fp":        total_fp,
            "fn":        total_fn,
        }

    def _clean_output(self, raw_output):
        """Clean raw model output for parsing"""

        output = raw_output.strip()
        output = output.replace("<BOS>", "<bos>")
        output = output.replace("<EOS>", "<eos>")
        output = output.replace("[BOS]", "<bos>")
        output = output.replace("[EOS]", "<eos>")
        output = output.replace("[+]", "<+>")
        output = output.replace("[-]", "<->")
        output = output.replace("[exon]", "<exon>")
        return output


class EvalLogger:
    """CSV logger for eval metrics"""

    FIELDS = [
        "timestamp", "epoch", "global_step",
        "exon_f1", "exon_precision", "exon_recall",
        "gene_f1", "gene_precision", "gene_recall",
        "num_samples", "num_predicted", "num_ref", "eval_sec",
    ]

    def __init__(self, output_dir):

        self.log_path = pathlib.Path(output_dir) / "eval_log.csv"
        self._file    = None
        self._writer  = None

        if not self.log_path.exists():
            self._file   = open(self.log_path, 'w', newline='')
            self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
            self._writer.writeheader()
            self._file.flush()
        else:
            self._file   = open(self.log_path, 'a', newline='')
            self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)

    def log(self, epoch, global_step, metrics):
        """Log eval metrics to CSV"""

        row = {
            "timestamp":      time.strftime("%Y-%m-%d %H:%M:%S"),
            "epoch":          epoch,
            "global_step":    global_step,
            "exon_f1":        metrics.get("exon_f1", 0),
            "exon_precision": metrics.get("exon_precision", 0),
            "exon_recall":    metrics.get("exon_recall", 0),
            "gene_f1":        metrics.get("gene_f1", 0),
            "gene_precision": metrics.get("gene_precision", 0),
            "gene_recall":    metrics.get("gene_recall", 0),
            "num_samples":    metrics.get("num_samples", 0),
            "num_predicted":  metrics.get("num_predicted", 0),
            "num_ref":        metrics.get("num_ref", 0),
            "eval_sec":       metrics.get("eval_sec", 0),
        }
        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        """Close the log file"""

        if self._file:
            self._file.close()
            self._file = None

    def __del__(self):
        self.close()
