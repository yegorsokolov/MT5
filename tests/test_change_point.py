import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from analysis.change_point import ChangePointDetector


def test_change_point_detection(tmp_path):
    rng = np.random.default_rng(0)
    data = np.concatenate([rng.normal(0, 1, 50), rng.normal(3, 1, 50)])
    df = pd.DataFrame({"feat": data})

    detector = ChangePointDetector(penalty=5.0, threshold=0.5)
    cps = detector.detect(df)

    assert "feat" in cps
    # should detect a breakpoint near the middle
    assert any(abs(bp - 50) <= 5 for bp in cps["feat"])

    detector.record(df, out_dir=tmp_path)
    latest = tmp_path / "latest.json"
    assert latest.exists()
    recorded = json.loads(latest.read_text())
    assert "feat" in recorded and recorded["feat"]
