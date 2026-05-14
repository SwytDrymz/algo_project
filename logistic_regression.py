import numpy as np
from data_prep import tfidf_sparse, tokenize
from typing import Union
from scipy.sparse import csr_matrix

def sigmoid(z):
    return np.where(z >= 0, 
                    1 / (1 + np.exp(-z)), 
                    np.exp(z) / (1 + np.exp(z)))

def compute_loss(y, y_pred) -> float:
    m = len(y)
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)) / m


def get_gradients(X: np.ndarray, y: np.ndarray, y_pred: np.ndarray, w: np.ndarray, lambda_reg: float, weight_0: float, weight_1: float) -> tuple[np.ndarray, float]:
    m = len(y)

    sample_weights = np.where(y == 1, weight_1, weight_0)
    weighted_errors = sample_weights * (y_pred - y)

    dw = (1 / m) * X.T.dot(weighted_errors) + (lambda_reg / m) * w
    db = (1 / m) * np.sum(weighted_errors)
    return np.asarray(dw).flatten(), db

def train_lr(X: np.ndarray, y: np.ndarray, learning_rate: float, epochs: int, lambda_reg: float, weight_0: float, weight_1: float, logging: bool = True) -> tuple[np.ndarray, float]:
    n_features = X.shape[1]
    w = np.zeros(n_features)
    b = 0

    for epoch in range(epochs):
        z = X.dot(w) + b
        y_pred = sigmoid(z)
        dw, db = get_gradients(X, y, y_pred, w, lambda_reg, weight_0, weight_1)
        w -= learning_rate * dw
        b -= learning_rate * db

        if logging and epoch % (epochs // 10) == 0:
            loss = compute_loss(y, y_pred)
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    return w, b

def predict_lr(X: Union[np.ndarray, csr_matrix], w: np.ndarray, b: float) -> np.ndarray:
    z = X.dot(w) + b
    return sigmoid(z)

def predict_new_sentence(sentence: str, vocab: list[str], w: np.ndarray, b: float, idf: np.ndarray) -> float:
    tokens = tokenize(sentence)
    X_new, _ = tfidf_sparse([tokens], vocab, idf)
    prob = predict_lr(X_new, w, b)
    return prob[0]