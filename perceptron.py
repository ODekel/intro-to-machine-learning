import numpy as np
import numpy.typing as npt
from tqdm import trange, tqdm


# Matrix shapes:
# - example: (n_features, 1)
# - label: (n_classes, 1)
# - weights: (n_features, n_classes)
def train(example: npt.NDArray[np.float32], label: npt.NDArray[np.int8], weights: npt.NDArray[np.float32]) -> None:
    misclassified = (np.sign(weights.T @ example).astype(np.int8) != label).squeeze()
    weights[:, misclassified] += example @ label[misclassified].T


# Matrix shapes:
# - examples: (n_examples, n_features)
# - labels: (n_examples, n_classes)
# - weights: (n_features, n_classes)
def pocket_train(examples: npt.NDArray[np.float32], labels: npt.NDArray[np.int8], weights: npt.NDArray[np.float32],
                 epochs: int = 1, test_interval: int = 100) -> None:
    pocket = weights.copy()
    best_score = test(examples, labels, weights)
    for _ in trange(epochs, desc='Epoch'):
        pbar = tqdm(examples, desc='Examples')
        pbar.set_postfix({'Best Score': best_score})
        for i, (example, label) in enumerate(zip(pbar, labels)):
            train(example[:, None], label[:, None], weights)
            if i % test_interval == 0:
                curr_score = test(examples, labels, weights)
                if curr_score > best_score:
                    np.copyto(pocket, weights)
                    best_score = curr_score
                    pbar.set_postfix({'Best Score': best_score})
    np.copyto(weights, pocket)


# Matrix shapes:
# - examples: (n_examples, n_features)
# - labels: (n_examples, n_classes)
# - weights: (n_features, n_classes)
def test(examples: npt.NDArray[np.float32], labels: npt.NDArray[np.int8], weights: npt.NDArray[np.float32]) -> int:
    return ((examples @ weights).argmax(axis=1) == labels.argmax(axis=1)).sum()
