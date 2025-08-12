import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def train_meta_classifier(features: pd.DataFrame, primary_labels) -> RandomForestClassifier:
    """Train a meta-classifier to judge primary model reliability.

    Parameters
    ----------
    features : pd.DataFrame
        Feature matrix used by the meta classifier.  Must include a ``prob``
        column representing the primary model's predicted probability and may
        include additional columns (e.g. ``confidence``).
    primary_labels : array-like
        True binary outcomes for each prediction from the primary model.

    Returns
    -------
    RandomForestClassifier
        A classifier that predicts ``1`` when the primary model's prediction
        is likely to be correct and ``0`` otherwise.
    """

    # ensure dataframe
    X = pd.DataFrame(features).copy()
    if "prob" not in X.columns:
        raise ValueError("features must include a 'prob' column")

    # derive meta labels: 1 when primary prediction matches actual outcome
    primary_pred = (X["prob"] > 0.5).astype(int)
    y = (primary_pred == pd.Series(primary_labels).astype(int)).astype(int)

    clf = RandomForestClassifier(n_estimators=50, random_state=0)
    clf.fit(X, y)
    return clf
