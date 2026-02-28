import csv
import json
import time
import pathlib
from collections import Counter

import torch

import lib.inference.output           as output_lib
import lib.inference.diffusion_engine as diff_engine


class CheckpointEvaluator:
    """Runs diffusion inference on held-out data and computes exon F1 metrics"""

    def __init__(self, eval_data_path, tokenizer, num_samples=20,
                 max_target_len=2048, num_steps=32, temperature=1.0):

        with open(eval_data_path, 'r') as f:
            all_samples = json.load(f)

        self.samples        = all_samples[:num_samples]
        self.tokenizer      = tokenizer
        self.max_target_len = max_target_len
        self.num_steps      = num_steps
        self.temperature    = temperature
        self.parser         = output_lib.ModelOutputParser(strict=False)

    @torch.no_grad()
    def evaluate(self, model, device, dtype):
        """Run diffusion inference on eval samples and compute F1"""

        was_training = model.training
        model.train(False)

        t0              = time.time()
        all_pred_exons  = []
        all_ref_exons   = []
        total_predicted = 0
        total_ref       = 0

        mask_id = self.tokenizer.mask_token_id
        pad_id  = self.tokenizer.pad_token_id

        for sample in self.samples:
            sequence     = sample["sequence"]
            ref_features = sample["ref_features"]

            # Encode input DNA
            encoded    = self.tokenizer.encode(sequence, add_special_tokens=False)
            bos_id     = self.tokenizer.bos_token_id
            prefix_ids = encoded + [bos_id]
            prefix_ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)

            # Diffusion inference
            with torch.amp.autocast(device.type, dtype=dtype):
                generated = diff_engine.diffusion_generate(
                    model          = model,
                    prefix_ids     = prefix_ids,
                    target_len     = self.max_target_len,
                    mask_token_id  = mask_id,
                    pad_token_id   = pad_id,
                    num_steps      = self.num_steps,
                    temperature    = self.temperature,
                )

            # Decode target portion
            output_ids = generated[0, prefix_ids.size(1):].tolist()
            # Remove padding
            output_ids = [t for t in output_ids if t != pad_id]
            raw_output = self.tokenizer.decode(output_ids)

            # Parse
            parsed = self.parser.parse_sequence(raw_output)

            # Extract predicted exon DNA
            pred_exon_dna = set()
            for feat in parsed:
                if feat.kind == "exon":
                    pred_exon_dna.add(feat.dna)

            # Extract reference exon DNA
            ref_exon_dna = set()
            for rf in ref_features:
                if rf.get("type", "exon").lower() != "exon":
                    continue
                dna = sequence[rf["start"]:rf["end"]]
                ref_exon_dna.add(dna)

            total_predicted += len(pred_exon_dna)
            total_ref       += len(ref_exon_dna)

            all_pred_exons.append(pred_exon_dna)
            all_ref_exons.append(ref_exon_dna)

        # Aggregate exon F1
        exon_metrics = self._compute_aggregate_f1(all_pred_exons, all_ref_exons)

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
            "gene_f1":        0.0,
            "gene_precision": 0.0,
            "gene_recall":    0.0,
            "gene_tp":        0,
            "gene_fp":        0,
            "gene_fn":        0,
            "num_samples":    len(self.samples),
            "num_predicted":  total_predicted,
            "num_ref":        total_ref,
            "eval_sec":       round(elapsed, 1),
        }

    def _compute_aggregate_f1(self, all_pred, all_ref):
        """Compute aggregate F1 across all samples"""

        total_tp = 0
        total_fp = 0
        total_fn = 0

        for pred_set, ref_set in zip(all_pred, all_ref):
            if isinstance(pred_set, Counter):
                common = pred_set & ref_set
                tp     = sum(common.values())
                fp     = sum((pred_set - ref_set).values())
                fn     = sum((ref_set - pred_set).values())
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


class EvalLogger:
    """CSV logger for evaluation metrics"""

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
        """Log evaluation metrics to CSV"""

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

        if self._file:
            self._file.close()
            self._file = None

    def __del__(self):
        self.close()
