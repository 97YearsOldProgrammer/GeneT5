import random


def split_validation(chunks, val_ratio=0.05, seed=42):
    """Randomly split chunks into train and validation sets"""

    random.seed(seed)
    n       = len(chunks)
    num_val = max(1, int(n * val_ratio))
    indices = set(random.sample(range(n), min(num_val, n)))

    train = [c for i, c in enumerate(chunks) if i not in indices]
    val   = [c for i, c in enumerate(chunks) if i in indices]

    return train, val
