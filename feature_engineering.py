import numpy as np
from numpy import typing as npt


def fit_pca(x: npt.NDArray[np.float32], n_components: int):
    mean = np.mean(x, axis=0)
    x_centered = x - mean

    cov = np.cov(x_centered, rowvar=False, dtype=np.float32)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    idx = np.argsort(eigenvalues)[::-1]
    p = eigenvectors[:, idx][:, :n_components]

    return p, mean


def apply_pca(x: npt.NDArray[np.float32], p: npt.NDArray[np.float32], mean: npt.NDArray[np.float32]
              ) -> npt.NDArray[np.float32]:
    return (x - mean) @ p


def expand_poly(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    n_samples, n_features = x.shape
    poly_list = [x]
    for i in range(n_features):
        poly_list.append(x[:, i:] * x[:, i:i + 1])
    poly_matrix = np.concat(poly_list, axis=1)
    return np.concat([np.ones((n_samples, 1), dtype=np.float32), poly_matrix], axis=1)
