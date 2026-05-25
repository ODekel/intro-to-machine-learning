import numpy as np
from tensorflow.keras.datasets import fashion_mnist

from perceptron import pocket_train as perceptron_pocket_train
from softmax_regression import train as smr_train
from svm import train as svm_train
from vis import display_stats
from feature_engineering import fit_pca, apply_pca, expand_poly

# Load dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print("Training set:", x_train.shape, x_train.dtype, y_train.shape, y_train.dtype)
print("Test set:", x_test.shape, x_test.dtype, y_test.shape, y_test.dtype)

x_train.shape = (x_train.shape[0], -1)
x_test.shape = (x_test.shape[0], -1)

x_train = (x_train / 255)
x_test = (x_test / 255)

x_train = np.concat([np.ones((x_train.shape[0], 1)), x_train], dtype=np.float32, axis=1)
x_test = np.concat([np.ones((x_test.shape[0], 1)), x_test], dtype=np.float32, axis=1)

y_train_neg_one_one = -np.ones((y_train.shape[0], 10), dtype=np.int8)
y_test_neg_one_one = -np.ones((y_test.shape[0], 10), dtype=np.int8)
y_train_zero_one = np.zeros((y_train.shape[0], 10), dtype=np.int8)
y_test_zero_one = np.zeros((y_test.shape[0], 10), dtype=np.int8)

y_train_neg_one_one[np.arange(y_train.shape[0]), y_train] = 1
y_test_neg_one_one[np.arange(y_test.shape[0]), y_test] = 1
y_train_zero_one[np.arange(y_train.shape[0]), y_train] = 1
y_test_zero_one[np.arange(y_test.shape[0]), y_test] = 1

print("Training set:", x_train.shape, x_train.dtype, y_train_neg_one_one.shape, y_train_neg_one_one.dtype)
print("Test set:", x_test.shape, x_test.dtype, y_test_neg_one_one.shape, y_test_neg_one_one.dtype)


perceptron_weights = np.zeros((x_train.shape[1], y_train_neg_one_one.shape[1]), dtype=np.float32)
perceptron_pocket_train(x_train, y_train_neg_one_one, perceptron_weights)
display_stats(x_test, y_test_neg_one_one, perceptron_weights)

smr_weights = np.zeros((x_train.shape[1], y_train_zero_one.shape[1]), dtype=np.float32)
smr_train(x_train, y_train_zero_one, smr_weights, epochs=3)
display_stats(x_test, y_test_zero_one, smr_weights)


print("Computing Polynomial Features (Degree 2) on PCA data...")
# 1. First run PCA to reduce dimensions (784 is too large for O(n^2) poly features)
x_tr = x_train[:, 1:]
x_te = x_test[:, 1:]

p, mean = fit_pca(x_tr, 60)

# Normalise the PCA outputs so polynomials don't explode exponentially
# (helps condition the gradient descent)
x_tr_pca_raw = apply_pca(x_tr, p, mean)
x_te_pca_raw = apply_pca(x_te, p, mean)

pca_std = np.std(x_tr_pca_raw, axis=0) + 1e-8
x_tr_pca = x_tr_pca_raw / pca_std
x_te_pca = x_te_pca_raw / pca_std
x_train_poly = expand_poly(x_tr_pca)
x_test_poly = expand_poly(x_te_pca)

svm_poly_weights = np.zeros((x_train_poly.shape[1], y_train_neg_one_one.shape[1]), dtype=np.float32)
# Using a larger learning rate and more epochs since features are normalised
svm_train(x_train_poly, y_train_neg_one_one, svm_poly_weights, epochs=35, batch_size=100, learning_rate=0.01,
          lambda_param=0.0001)
display_stats(x_test_poly, y_test_neg_one_one, svm_poly_weights)
