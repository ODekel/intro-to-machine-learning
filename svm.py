import numpy as np
from numpy import typing as npt
from tqdm import trange


# Matrix shapes:
# - examples: (n_examples, n_features)
# - labels: (n_examples, n_classes)  (expected to be 1 for positive, -1 for negative)
# - weights: (n_features, n_classes) (including bias feature)
def train(examples: npt.NDArray[np.float32], labels: npt.NDArray[np.int8], weights: npt.NDArray[np.float32],
          epochs: int = 20, learning_rate: float = 0.01, lambda_param: float = 0.01) -> None:
    n_examples, n_features_to_train = examples.shape
    n_classes = labels.shape[1]

    # 1. Linear SVM Training (One-vs-Rest) using Gradient Descent
    w_svm = np.zeros((n_features_to_train, n_classes), dtype=np.float32)

    batch_size = 200
    for _ in trange(epochs, desc='SVM Epoch'):
        indices = np.random.permutation(n_examples)
        x_shuffled = examples[indices]
        labels_shuffled = labels[indices]

        for i in range(0, n_examples, batch_size):
            x_batch = x_shuffled[i:i + batch_size]
            y_batch = labels_shuffled[i:i + batch_size]

            for c in range(n_classes):
                y = y_batch[:, c]

                # Forward pass: compute margin (bias is organically included in X)
                margin = y * (x_batch @ w_svm[:, c])

                # Misclassified or inside margin
                misclassified = margin < 1

                # Gradients
                # L2 Penalty only applies to features, not the assumed bias (index 0)
                # We scale the L2 penalty correctly to not implicitly depend on batch size.
                d_w = np.zeros_like(w_svm[:, c])
                d_w[1:] = 2 * lambda_param * w_svm[1:, c] / x_batch.shape[0]

                if np.any(misclassified):
                    # Gradients from hinge loss
                    d_w -= np.sum(y[misclassified, None] * x_batch[misclassified], axis=0) / x_batch.shape[0]

                # Update weights
                w_svm[:, c] -= learning_rate * d_w

    # 2. Assemble final weights into the designated argument
    weights[:, :] = w_svm
