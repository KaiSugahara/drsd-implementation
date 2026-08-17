import numpy as np


def hard_routing(probs: np.ndarray) -> np.ndarray:
    labels = np.argmax(probs, axis=1)
    hard = np.zeros_like(probs, dtype=int)
    hard[np.arange(probs.shape[0]), labels] = 1
    return hard
