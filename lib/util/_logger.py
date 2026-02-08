import time
import csv
import psutil
import torch
from pathlib import Path


STEP_FIELDS = [
    'timestamp',   'elapsed_sec', 'epoch',       'global_step',
    'batch',       'num_batches', 'train_loss',  'val_loss',
    'lr',          'batch_per_sec', 'eta_sec',
    'gpu_alloc_gb', 'gpu_max_gb', 'ram_used_gb',
]


class TrainLogger:
    """
    Structured CSV + stdout logger for training runs

    Writes one CSV row per log event (step progress or epoch summary).
    Prints human-readable lines to stdout. Rank-0 only.
    """

    def __init__(self, output_dir, filename="training_log.csv"):

        self.log_path   = Path(output_dir) / filename
        self.start_time = time.time()
        self._file      = open(self.log_path, 'w', newline='')
        self._writer    = csv.DictWriter(self._file, fieldnames=STEP_FIELDS)
        self._writer.writeheader()
        self._file.flush()

    def _memory_stats(self):
        """Collect current memory snapshot"""

        mem = psutil.virtual_memory()
        stats = {'ram_used_gb': f'{mem.used / 1e9:.2f}'}

        if torch.cuda.is_available():
            stats['gpu_alloc_gb'] = f'{torch.cuda.memory_allocated() / 1e9:.2f}'
            stats['gpu_max_gb']   = f'{torch.cuda.max_memory_allocated() / 1e9:.2f}'

        return stats

    def _elapsed(self):

        return time.time() - self.start_time

    def log_step(self, epoch, global_step, batch, num_batches,
                 loss, lr, batch_per_sec):
        """Log a training progress point and print formatted line"""

        elapsed     = self._elapsed()
        remaining   = (num_batches - batch) / batch_per_sec if batch_per_sec > 0 else 0
        mem         = self._memory_stats()
        pct         = batch * 100 // num_batches

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

        # Human-readable stdout
        mem_parts = [f"RAM: {mem['ram_used_gb']}GB"]
        if 'gpu_alloc_gb' in mem:
            mem_parts.append(f"GPU: {mem['gpu_alloc_gb']}/{mem['gpu_max_gb']}GB")
        mem_str = " | ".join(mem_parts)

        print(f"  [{pct:3d}%] batch {batch}/{num_batches} | "
              f"loss: {loss:.4f} | lr: {lr:.2e} | "
              f"speed: {batch_per_sec:.1f} batch/s | "
              f"ETA: {remaining/3600:.1f}h | {mem_str}")

    def log_epoch(self, epoch, train_loss, val_loss, lr):
        """Log epoch summary row"""

        mem = self._memory_stats()

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

        print(f"  Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {lr:.2e}")

    def close(self):

        if self._file and not self._file.closed:
            self._file.close()

    def __del__(self):

        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def create_train_logger(output_dir, filename="training_log.csv"):
    """Create a training logger that writes to output_dir/filename"""

    return TrainLogger(output_dir, filename)
