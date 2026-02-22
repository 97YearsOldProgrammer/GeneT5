import time


def format_time(seconds):
    """Format seconds into human readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def format_rate(count, seconds):
    """Format processing rate"""
    if seconds <= 0:
        return "∞/s"
    rate = count / seconds
    if rate >= 1000:
        return f"{rate/1000:.1f}k/s"
    return f"{rate:.0f}/s"