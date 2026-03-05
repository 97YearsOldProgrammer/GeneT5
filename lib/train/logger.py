import time
import csv
import psutil
import torch
from collections import deque
from pathlib import Path


SPARK = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"

STEP_FIELDS = [
    'timestamp',   'elapsed_sec', 'epoch',       'global_step',
    'batch',       'num_batches', 'train_loss',  'val_loss',
    'lr',          'batch_per_sec', 'eta_sec',
    'gpu_alloc_gb', 'gpu_max_gb', 'ram_used_gb',
]


def _sparkline(values, width=20):
    """Unicode sparkline from recent loss values"""

    if len(values) < 2:
        return ""
    recent = list(values)[-width:]
    lo, hi = min(recent), max(recent)
    rng    = hi - lo or 1.0
    return "".join(SPARK[min(int((v - lo) / rng * 7), 7)] for v in recent)


def _bar(pct, width=20):
    """Unicode progress bar"""

    filled = int(pct / 100 * width)
    if filled >= width:
        return "\u2501" * width
    if filled == 0:
        return "\u2500" * width
    return "\u2501" * filled + "\u2578" + "\u2500" * (width - filled - 1)


def _fmt_eta(seconds):
    """Format ETA as human-readable string"""

    if seconds <= 0:
        return "done"
    if seconds >= 3600:
        return f"{seconds / 3600:.1f}h"
    if seconds >= 60:
        return f"{seconds / 60:.0f}m"
    return f"{seconds:.0f}s"


def _fmt_duration(seconds):
    """Format duration as Xh Ym Zs"""

    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


class TrainLogger:
    """
    Structured CSV + formatted stdout logger for training runs

    Writes one CSV row per log event. Prints single-line progress with
    sparkline loss trend, progress bar, smoothed ETA. Rank-0 only.
    Stores full history for end-of-training summary.
    """

    def __init__(self, output_dir, filename="training_log.csv",
                 resume=False, total_epochs=1):

        self.log_path      = Path(output_dir) / filename
        self.start_time    = time.time()
        self.total_epochs  = total_epochs
        self._loss_hist    = deque(maxlen=50)
        self._speed_ema    = None

        # Full history for end-of-training summary
        self._all_steps    = []
        self._epoch_rows   = []

        if resume and self.log_path.exists():
            self._file   = open(self.log_path, 'a', newline='')
            self._writer = csv.DictWriter(self._file, fieldnames=STEP_FIELDS)
        else:
            self._file   = open(self.log_path, 'w', newline='')
            self._writer = csv.DictWriter(self._file, fieldnames=STEP_FIELDS)
            self._writer.writeheader()

        self._file.flush()

    def _memory_stats(self):
        """Collect current memory snapshot"""

        mem   = psutil.virtual_memory()
        stats = {'ram_used_gb': f'{mem.used / 1e9:.2f}'}

        if torch.cuda.is_available():
            stats['gpu_alloc_gb'] = f'{torch.cuda.memory_allocated() / 1e9:.2f}'
            stats['gpu_max_gb']   = f'{torch.cuda.max_memory_allocated() / 1e9:.2f}'

        return stats

    def _elapsed(self):

        return time.time() - self.start_time

    def _smooth_speed(self, speed):
        """EMA-smoothed batch throughput"""

        if self._speed_ema is None:
            self._speed_ema = speed
        else:
            self._speed_ema = 0.9 * self._speed_ema + 0.1 * speed
        return self._speed_ema

    def log_step(self, epoch, global_step, batch, num_batches,
                 loss, lr, batch_per_sec):
        """Log training step: CSV row + formatted progress line"""

        elapsed   = self._elapsed()
        speed     = self._smooth_speed(batch_per_sec)
        remaining = (num_batches - batch) / speed if speed > 0 else 0
        mem       = self._memory_stats()
        pct       = batch * 100 / num_batches if num_batches > 0 else 0

        self._loss_hist.append(loss)
        self._all_steps.append({
            'step': global_step, 'loss': loss, 'lr': lr,
            'speed': batch_per_sec, 'elapsed': elapsed,
        })

        row = {
            'timestamp':    f'{time.time():.0f}',
            'elapsed_sec':  f'{elapsed:.1f}',
            'epoch':        epoch,
            'global_step':  global_step,
            'batch':        batch,
            'num_batches':  num_batches,
            'train_loss':   f'{loss:.6f}',
            'val_loss':     '',
            'lr':           f'{lr:.2e}',
            'batch_per_sec': f'{batch_per_sec:.2f}',
            'eta_sec':      f'{remaining:.0f}',
            **mem,
        }
        self._writer.writerow(row)
        self._file.flush()

        bar   = _bar(pct)
        spark = _sparkline(self._loss_hist)
        gpu   = f"GPU {mem['gpu_alloc_gb']}/{mem['gpu_max_gb']}GB" if 'gpu_alloc_gb' in mem else ""

        print(f"  Ep {epoch}/{self.total_epochs} {bar} {pct:5.1f}% "
              f"\u2502 loss {loss:.4f} {spark} "
              f"\u2502 {speed:.1f} b/s "
              f"\u2502 ETA {_fmt_eta(remaining)} "
              f"\u2502 {gpu}")

    def log_epoch(self, epoch, train_loss, val_loss, lr):
        """Log epoch summary row"""

        mem = self._memory_stats()

        self._epoch_rows.append({
            'epoch': epoch, 'train': train_loss, 'val': val_loss, 'lr': lr,
        })

        row = {
            'timestamp':    f'{time.time():.0f}',
            'elapsed_sec':  f'{self._elapsed():.1f}',
            'epoch':        epoch,
            'global_step':  '',
            'batch':        '',
            'num_batches':  '',
            'train_loss':   f'{train_loss:.6f}',
            'val_loss':     f'{val_loss:.6f}',
            'lr':           f'{lr:.2e}',
            'batch_per_sec': '',
            'eta_sec':      '',
            **mem,
        }
        self._writer.writerow(row)
        self._file.flush()

        spark = _sparkline(self._loss_hist)
        print(f"\n  \u2500\u2500 Epoch {epoch} \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
              f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
              f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")
        print(f"  train {train_loss:.4f} \u2502 val {val_loss:.4f} \u2502 lr {lr:.1e} \u2502 {spark}")
        print()

    def log_summary(self, best_val_loss=None):
        """Print end-of-training summary with full loss curve and stats"""

        duration = self._elapsed()
        steps    = self._all_steps
        epochs   = self._epoch_rows

        if not steps:
            return

        losses = [s['loss'] for s in steps]
        speeds = [s['speed'] for s in steps]

        print(f"\n  \u2550\u2550 Training Summary \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
              f"\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
              f"\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550")

        # Stats line
        best_step = min(steps, key=lambda s: s['loss'])
        parts = [
            f"Duration: {_fmt_duration(duration)}",
            f"Steps: {steps[-1]['step']}",
            f"Avg speed: {sum(speeds) / len(speeds):.1f} b/s",
        ]
        if best_val_loss is not None:
            parts.append(f"Best val: {best_val_loss:.4f}")
        print(f"  {' \u2502 '.join(parts)}")

        # Full-width loss sparkline (all logged points)
        full_spark = _sparkline(losses, width=60)
        print(f"\n  Loss curve ({len(losses)} points, step {steps[0]['step']}"
              f"\u2192{steps[-1]['step']}):")
        print(f"  {full_spark}")
        print(f"  {losses[0]:.4f} {'.' * 50} {losses[-1]:.4f}")

        # Per-epoch table
        if epochs:
            print(f"\n  {'Ep':>4}  {'Train':>8}  {'Val':>8}  {'LR':>10}  {'Delta':>8}")
            print(f"  {'─' * 4}  {'─' * 8}  {'─' * 8}  {'─' * 10}  {'─' * 8}")
            prev_val = None
            for e in epochs:
                delta = ""
                if prev_val is not None:
                    d = e['val'] - prev_val
                    delta = f"{d:+.4f}"
                best_marker = " *" if e['val'] == best_val_loss else ""
                print(f"  {e['epoch']:>4}  {e['train']:>8.4f}  {e['val']:>8.4f}"
                      f"  {e['lr']:>10.1e}  {delta:>8}{best_marker}")
                prev_val = e['val']

        # Loss at key milestones (first, 25%, 50%, 75%, last)
        n = len(steps)
        if n >= 5:
            indices   = [0, n // 4, n // 2, 3 * n // 4, n - 1]
            print(f"\n  {'Step':>8}  {'Loss':>8}  {'LR':>10}  {'Speed':>8}  {'Time':>10}")
            print(f"  {'─' * 8}  {'─' * 8}  {'─' * 10}  {'─' * 8}  {'─' * 10}")
            for i in indices:
                s = steps[i]
                print(f"  {s['step']:>8}  {s['loss']:>8.4f}  {s['lr']:>10.1e}"
                      f"  {s['speed']:>6.1f}/s  {_fmt_duration(s['elapsed']):>10}")

        print(f"  {'═' * 49}")

    def close(self):

        if self._file and not self._file.closed:
            self._file.close()

    def __del__(self):

        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def create_train_logger(output_dir, filename="training_log.csv",
                        resume=False, total_epochs=1):
    """Create a training logger, appends if resuming from checkpoint"""

    return TrainLogger(output_dir, filename, resume=resume, total_epochs=total_epochs)
