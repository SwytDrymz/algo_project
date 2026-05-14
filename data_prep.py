import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
from wordcloud import WordCloud
import math
from collections import Counter
from scipy.sparse import csr_matrix, diags
from typing import Optional, Tuple

def tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text.split()

def vocabulary(corpus: list[list[str]]) -> list[str]:
    vocab = set()
    for doc in corpus:
        vocab.update(doc)
    return sorted(list(vocab))


def tfidf_sparse(corpus: list[list[str]], vocab: list[str], idf: Optional[np.ndarray] = None) -> Tuple[csr_matrix, np.ndarray]:
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    n_vocab = len(vocab)
    n_docs = len(corpus)

    rows, cols, data = [], [], []
    df_counts = np.zeros(len(vocab), dtype=int)

    for doc_idx, doc in enumerate(corpus):
        counts = Counter(doc)
        doc_len = len(doc)
        for word, count in counts.items():
            if word in word_to_idx:
                idx = word_to_idx[word]
                rows.append(doc_idx)
                cols.append(idx)
                data.append(count / doc_len)
                df_counts[idx] += 1

    tf_matrix = csr_matrix((data, (rows, cols)), shape=(n_docs, n_vocab))
    
    if idf is None:
        idf = np.log(n_docs / (df_counts + 1))
    return tf_matrix.multiply(idf.reshape(1, -1)), idf
        
def plot_sentiment_distribution(df: pd.DataFrame) -> None:
    counts = df["Sentiment"].value_counts()
    plt.figure(figsize=(8, 6))
    plt.pie(counts, labels = counts.index.tolist())
    plt.title("Distribuce sentimentu")
    plt.show()


def plot_cloud(corpus: list[list[str]]) -> None:
    text = " ".join(token for tokens in corpus for token in tokens)
    wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=80).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  
    plt.title("Financial Sentiment Word Cloud")
    plt.show()


def bar_top_words(n: int, corpus: list[list[str]]) -> None:
    word_counts = {}
    for doc in corpus:
         for word in doc:
              word_counts[word] = word_counts.get(word, 0) + 1

    top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:n]
    words, counts = zip(*top_words)

    plt.bar(words, counts)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")

    cm = np.array([[tn, fp], [fn, tp]])
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)
    tick_marks = [0, 1]
    plt.xticks(tick_marks, ["Negative", "Positive"])
    plt.yticks(tick_marks, ["Negative", "Positive"])

    plt.text(0, 0, str(tn), ha="center", va="center", color="black")
    plt.text(1, 0, str(fp), ha="center", va="center", color="black")
    plt.text(0, 1, str(fn), ha="center", va="center", color="black")
    plt.text(1, 1, str(tp), ha="center", va="center", color="black")

    plt.show()