import numpy as np
import pandas as pd
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

pytest.importorskip("sklearn")
from models.meta_label import train_meta_classifier


def test_meta_label_integration_reduces_false_positives():
    probs = np.linspace(0, 1, 200)
    y_true = (probs > 0.7).astype(int)
    features = pd.DataFrame({"prob": probs, "confidence": np.ones_like(probs)})
    clf = train_meta_classifier(features, y_true)

    base_preds = probs > 0.5
    fp_raw = ((base_preds == 1) & (y_true == 0)).sum()
    meta_preds = clf.predict(features[base_preds])
    fp_filtered = ((meta_preds == 1) & (y_true[base_preds] == 0)).sum()
    assert fp_filtered < fp_raw
