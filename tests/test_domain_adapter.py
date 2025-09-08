import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.domain_adapter import DomainAdapter


def test_domain_adapter_alignment():
    rng = np.random.default_rng(0)
    source = pd.DataFrame(rng.normal(size=(500, 3)), columns=list("abc"))
    # target distribution shifted and scaled
    target = pd.DataFrame(
        rng.normal(loc=5, scale=2, size=(500, 3)), columns=list("abc")
    )

    adapter = DomainAdapter()
    adapter.fit_source(source)
    adapter.update_target(target)

    before_mean = np.linalg.norm(source.mean().to_numpy() - target.mean().to_numpy())
    before_cov = np.linalg.norm(
        np.cov(source.to_numpy().T, bias=True) - np.cov(target.to_numpy().T, bias=True)
    )

    aligned = adapter.transform(target)

    after_mean = np.linalg.norm(source.mean().to_numpy() - aligned.mean().to_numpy())
    after_cov = np.linalg.norm(
        np.cov(source.to_numpy().T, bias=True) - np.cov(aligned.to_numpy().T, bias=True)
    )

    assert after_mean < before_mean
    assert after_cov < before_cov
