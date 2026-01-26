import time

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


#############################
#####  RAM Monitoring   #####
#############################


def get_memory_usage_pct():
    """Get current RAM usage percentage"""
    
    if not HAS_PSUTIL:
        return 0.0  # Can't monitor without psutil
    
    return psutil.virtual_memory().percent


def get_memory_info():
    """Get detailed memory info"""
    
    if not HAS_PSUTIL:
        return {"available_gb": "unknown", "used_pct": 0.0, "total_gb": "unknown"}
    
    mem = psutil.virtual_memory()
    return {
        "total_gb":     mem.total / (1024**3),
        "available_gb": mem.available / (1024**3),
        "used_pct":     mem.percent,
    }


def wait_for_memory(limit_pct, check_interval=2.0, max_wait=600.0):
    """Wait until RAM usage drops below limit
    
    Returns:
        True if memory freed up, False if timeout
    """
    
    if not HAS_PSUTIL:
        return True  # No monitoring, proceed
    
    waited = 0.0
    while get_memory_usage_pct() > limit_pct:
        if waited >= max_wait:
            return False  # Timeout
        
        time.sleep(check_interval)
        waited += check_interval
    
    return True


def can_submit_new_work(limit_pct):
    """Check if RAM usage allows submitting new work"""
    
    if not HAS_PSUTIL:
        return True
    
    return get_memory_usage_pct() < limit_pct
