import numpy as np
from numpy import typing as npt
from tqdm import trange, tqdm


# Matrix shapes:
# - example: (n_features, 1)
# - label: (n_classes, 1)
# - weights: (n_features, n_classes)
def train_once(example: npt.NDArray[np.float32], label: npt.NDArray[np.int8], weights: npt.NDArray[np.float32],
               learning_rate: float = 0.01) -> None:
    scores = weights.T @ example
    scores -= scores.max()  # For numerical stability, to prevent overflows
    probs = np.exp(scores)
    probs /= probs.sum()

    gradient = example @ (probs - label).T

    weights -= learning_rate * gradient


# Matrix shapes:
# - examples: (n_examples, n_features)
# - labels: (n_examples, n_classes)
# - weights: (n_features, n_classes)
def train(examples: npt.NDArray[np.float32], labels: npt.NDArray[np.int8], weights: npt.NDArray[np.float32],
          start_learning_rate: float = 0.1, learning_rate_decay=0.9999, epochs: int = 1) -> None:
    learning_rate = start_learning_rate
    for _ in trange(epochs, desc='Epoch'):
        for i, (example, label) in enumerate(zip(tqdm(examples, desc='Examples'), labels)):
            train_once(example[:, None], label[:, None], weights, learning_rate)
            learning_rate = learning_rate * learning_rate_decay
