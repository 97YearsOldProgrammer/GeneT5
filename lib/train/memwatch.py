import time
import threading
import csv
import psutil
import torch
from pathlib import Path


FIELDS = [
    'timestamp',      'elapsed_sec',
    'ram_used_gb',    'ram_total_gb',   'ram_pct',    'ram_avail_gb',
    'swap_used_gb',   'swap_total_gb',  'swap_pct',
    'gpu_alloc_gb',   'gpu_peak_gb',
]


class MemoryWatcher:
    """
    Background memory monitor for training runs

    Records RAM, swap, and GPU memory to CSV with pressure-adaptive sampling:
    - Normal: every 30 seconds
    - High pressure (>80% RAM or swap active): every 5 seconds
    - Critical (>90% RAM): every 2 seconds
    """

    def __init__(self, log_path, normal_interval=30, high_interval=5, critical_interval=2):

        self.log_path          = Path(log_path)
        self.normal_interval   = normal_interval
        self.high_interval     = high_interval
        self.critical_interval = critical_interval
        self.start_time        = None
        self.running           = False
        self.thread            = None
        self._file             = None
        self._writer           = None
        self._prev_alloc       = 0.0

    def _get_memory_stats(self):

        now  = time.time()
        mem  = psutil.virtual_memory()
        swap = psutil.swap_memory()

        stats = {
            'timestamp':    f'{now:.0f}',
            'elapsed_sec':  f'{now - self.start_time:.1f}',
            'ram_used_gb':  f'{mem.used / 1e9:.2f}',
            'ram_total_gb': f'{mem.total / 1e9:.2f}',
            'ram_pct':      f'{mem.percent:.1f}',
            'ram_avail_gb': f'{mem.available / 1e9:.2f}',
            'swap_used_gb': f'{swap.used / 1e9:.2f}',
            'swap_total_gb': f'{swap.total / 1e9:.2f}',
            'swap_pct':     f'{swap.percent:.1f}',
        }

        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1e9
            peak  = torch.cuda.max_memory_allocated() / 1e9

            # Reset peak tracking when allocation drops (new phase)
            if alloc < self._prev_alloc * 0.5 and self._prev_alloc > 1.0:
                torch.cuda.reset_peak_memory_stats()
                peak = alloc

            self._prev_alloc       = alloc
            stats['gpu_alloc_gb']  = f'{alloc:.2f}'
            stats['gpu_peak_gb']   = f'{peak:.2f}'
        else:
            stats['gpu_alloc_gb']  = ''
            stats['gpu_peak_gb']   = ''

        return stats, mem.percent, swap.percent

    def _choose_interval(self, ram_pct, swap_pct):
        """Sample faster when memory pressure is high"""

        if ram_pct > 90:
            return self.critical_interval
        if ram_pct > 80 or swap_pct > 10:
            return self.high_interval
        return self.normal_interval

    def _monitor_loop(self):

        self._file   = open(self.log_path, 'w', newline='')
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDS)
        self._writer.writeheader()
        self._file.flush()

        while self.running:
            stats, ram_pct, swap_pct = self._get_memory_stats()
            self._writer.writerow(stats)
            self._file.flush()

            interval = self._choose_interval(ram_pct, swap_pct)
            time.sleep(interval)

        self._file.close()
        self._file = None

    def start(self):

        if self.running:
            return

        self.start_time = time.time()
        self.running    = True
        self.thread     = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        print(f"  Memory watcher started: {self.log_path}")
        print(f"    Normal: {self.normal_interval}s | High (>80%): {self.high_interval}s | Critical (>90%): {self.critical_interval}s")

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

    def __del__(self):

        if self.running:
            self.running = False
        if self._file and not self._file.closed:
            self._file.close()


def create_memory_watcher(output_dir, prefix="memory"):
    """Create a memory watcher that logs to output_dir/prefix_TIMESTAMP.csv"""

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path  = Path(output_dir) / f"{prefix}_{timestamp}.csv"
    return MemoryWatcher(log_path)
