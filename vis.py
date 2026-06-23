import numpy as np
from matplotlib import pyplot as plt
from numpy import typing as npt


def display_stats(x_test: npt.NDArray[np.float32], y_test: npt.NDArray[np.int8], weights: npt.NDArray[np.float32]
                  ) -> None:
    y_pred = (x_test @ weights).argmax(axis=1)
    y_true = y_test.argmax(axis=1)

    confusion_matrix = np.zeros((10, 10), dtype=int)
    for t, p in zip(y_true, y_pred):
        confusion_matrix[t, p] += 1

    accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
    print(f"Accuracy: {accuracy}")

    tpr = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)

    # Visualization
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                   'Ankle boot']

    plt.figure(figsize=(14, 6))

    # Confusion Matrix Subplot
    plt.subplot(1, 2, 1)
    plt.imshow(confusion_matrix, interpolation='nearest', cmap='Blues')
    plt.title(f'Confusion Matrix (Accuracy: {accuracy:.2%})')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    thresh = confusion_matrix.max() / 2.
    for i, j in np.ndindex(confusion_matrix.shape):
        plt.text(j, i, format(confusion_matrix[i, j], 'd'),
                 horizontalalignment="center", verticalalignment="center",
                 color="white" if confusion_matrix[i, j] > thresh else "black")

    # TPR Subplot
    plt.subplot(1, 2, 2)
    bars = plt.bar(tick_marks, tpr, color='skyblue')
    plt.title('True Positive Rate per Class')
    plt.xlabel('Class')
    plt.ylabel('TPR')
    plt.xticks(tick_marks, class_names)
    plt.ylim(0, 1.1)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{height:.2f}',
                 ha='center', va='bottom')

    plt.tight_layout()
    plt.show()
