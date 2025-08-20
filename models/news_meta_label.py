import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np


def train_news_meta_classifier(primary_probs, sentiment_scores, outcomes) -> RandomForestClassifier:
    """Train a meta-classifier to judge predictions under sentiment conflict."""

    probs = pd.Series(primary_probs, name="prob").astype(float)
    sent = pd.Series(sentiment_scores, name="sentiment").astype(float)
    outcomes = pd.Series(outcomes).astype(int)

    primary_pred = (probs > 0.5).astype(int)
    sentiment_pred = (sent > 0).astype(int)
    conflict = (primary_pred != sentiment_pred).astype(int)

    X = pd.concat([probs, sent, conflict.rename("conflict")], axis=1)
    y = (primary_pred == outcomes).astype(int)

    clf = RandomForestClassifier(n_estimators=50, random_state=0)
    clf.fit(X, y)
    return clf


def predict_trust(model: RandomForestClassifier, prob: float, sentiment: float) -> bool:
    """Predict whether to trust a primary prediction given sentiment."""

    primary_pred = int(prob > 0.5)
    sentiment_pred = int(sentiment > 0)
    conflict = int(primary_pred != sentiment_pred)
    X = pd.DataFrame({"prob": [prob], "sentiment": [sentiment], "conflict": [conflict]})
    return bool(model.predict(X)[0])


__all__ = ["train_news_meta_classifier", "predict_trust"]

