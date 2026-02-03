import time
import threading
import csv
import psutil
import torch
from pathlib import Path


class MemoryWatcher:
    """
    Background memory monitor for training runs

    Records RAM and GPU memory to CSV at adaptive intervals:
    - First 5 minutes: every 5 seconds (catch early issues)
    - After 5 minutes: every 30 seconds (reduce overhead)
    """

    def __init__(self, log_path, warmup_duration=300, warmup_interval=5, normal_interval=30):
        self.log_path        = Path(log_path)
        self.warmup_duration = warmup_duration
        self.warmup_interval = warmup_interval
        self.normal_interval = normal_interval
        self.start_time      = None
        self.running         = False
        self.thread          = None

    def _get_memory_stats(self):
        mem = psutil.virtual_memory()
        stats = {
            'timestamp':     time.time(),
            'elapsed_sec':   time.time() - self.start_time,
            'ram_used_gb':   mem.used / 1e9,
            'ram_total_gb':  mem.total / 1e9,
            'ram_percent':   mem.percent,
            'ram_avail_gb':  mem.available / 1e9,
        }

        if torch.cuda.is_available():
            stats['gpu_allocated_gb'] = torch.cuda.memory_allocated() / 1e9
            stats['gpu_reserved_gb']  = torch.cuda.memory_reserved() / 1e9
            stats['gpu_max_alloc_gb'] = torch.cuda.max_memory_allocated() / 1e9

        return stats

    def _monitor_loop(self):
        fieldnames = [
            'timestamp', 'elapsed_sec', 'ram_used_gb', 'ram_total_gb',
            'ram_percent', 'ram_avail_gb', 'gpu_allocated_gb',
            'gpu_reserved_gb', 'gpu_max_alloc_gb'
        ]

        with open(self.log_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            while self.running:
                stats = self._get_memory_stats()
                writer.writerow(stats)
                f.flush()

                elapsed = time.time() - self.start_time
                if elapsed < self.warmup_duration:
                    interval = self.warmup_interval
                else:
                    interval = self.normal_interval

                time.sleep(interval)

    def start(self):
        if self.running:
            return

        self.start_time = time.time()
        self.running    = True
        self.thread     = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        print(f"  Memory watcher started: {self.log_path}")
        print(f"    First {self.warmup_duration}s: every {self.warmup_interval}s")
        print(f"    After: every {self.normal_interval}s")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        print(f"  Memory watcher stopped. Log: {self.log_path}")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


def create_memory_watcher(output_dir, prefix="memory"):
    """Create a memory watcher that logs to output_dir/prefix_TIMESTAMP.csv"""

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path  = Path(output_dir) / f"{prefix}_{timestamp}.csv"
    return MemoryWatcher(log_path)
