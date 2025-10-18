import numpy as np


def get_random_block(N=16, batch=256):
    """Generates a random batch of blocks in binary {-1, 1}."""
    return 2 * np.random.randint(2, size=(batch, N, 1)) - 1