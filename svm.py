import numpy as np


def train_svm(X: np.ndarray, y: np.ndarray, learning_rate: float, epochs: int, lambda_reg: float, weight_0: float, weight_1: float, logging: bool = True) -> tuple[np.ndarray, float]:
    y_svm = np.where(y == 0, -1, 1)
    sample_weights = np.where(y == 0, weight_0, weight_1)

    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0

    for epoch in range(epochs):
        scores = X.dot(w) + b
        distances = y_svm * scores

        mask = (distances < 1).astype(float)
        weighted_y = mask * y_svm * sample_weights

        dw = (2 * lambda_reg * w) - (X.T.dot(weighted_y)) / n_samples
        db = -np.sum(weighted_y) / n_samples

        w -= learning_rate * dw
        b -= learning_rate * db

        if logging and epoch % (epochs // 10) == 0:

            print(f"Epoch {epoch}, Loss: {lambda_reg * np.sum(w**2) + np.mean(sample_weights * np.maximum(0, 1 - distances))}")

    return w, b

def predict_svm(X: np.ndarray, w:np.ndarray, b: float) -> np.ndarray:
    scores = X.dot(w) + b
    return np.where(scores >= 0, 1, 0)