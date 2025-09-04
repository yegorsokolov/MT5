from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass
class BayesianForecaster:
    """Simple Bayesian forecaster for returns/volatility.

    This implements a Normal-Inverse-Gamma model for returns with
    unknown mean and variance. The posterior predictive distribution
    is used to compute credible intervals for the next return.
    """

    mu0: float = 0.0
    kappa0: float = 1.0
    alpha0: float = 1.0
    beta0: float = 1.0

    def fit(self, returns: np.ndarray) -> None:
        """Update posterior hyper-parameters given observed returns."""
        returns = np.asarray(returns)
        n = len(returns)
        if n == 0:
            raise ValueError("returns array must be non-empty")
        mean = returns.mean()
        var = returns.var(ddof=1)

        self.kappa_n = self.kappa0 + n
        self.mu_n = (self.kappa0 * self.mu0 + n * mean) / self.kappa_n
        self.alpha_n = self.alpha0 + n / 2
        self.beta_n = (
            self.beta0
            + 0.5 * (n - 1) * var
            + 0.5 * (self.kappa0 * n * (mean - self.mu0) ** 2) / self.kappa_n
        )

    def forecast(
        self,
        cred_level: float = 0.9,
        n_samples: int = 2000,
        log: bool = True,
    ) -> dict:
        """Return posterior predictive mean and credible interval.

        Parameters
        ----------
        cred_level : float
            Desired credible interval level (default 0.9).
        n_samples : int
            Number of Monte Carlo samples from the posterior predictive.
        log : bool
            If ``True``, append diagnostics to ``reports/bayesian_forecasts``.
        """
        if not hasattr(self, "mu_n"):
            raise RuntimeError("fit must be called before forecast")

        # sample sigma^2 from Inv-Gamma(alpha_n, beta_n)
        sigma2 = 1 / np.random.gamma(self.alpha_n, 1 / self.beta_n, size=n_samples)
        mu = np.random.normal(self.mu_n, np.sqrt(sigma2 / self.kappa_n))
        predictive = np.random.normal(mu, np.sqrt(sigma2))
        mean_pred = predictive.mean()
        lower = np.quantile(predictive, (1 - cred_level) / 2)
        upper = np.quantile(predictive, 1 - (1 - cred_level) / 2)

        if log:
            self._log_diagnostics(mean_pred, lower, upper)

        return {"mean": float(mean_pred), "lower": float(lower), "upper": float(upper)}

    def _log_diagnostics(self, mean: float, lower: float, upper: float) -> None:
        """Append diagnostics to reports/bayesian_forecasts."""
        report_dir = Path("reports/bayesian_forecasts")
        report_dir.mkdir(parents=True, exist_ok=True)
        file = report_dir / "posterior_diagnostics.csv"
        df = pd.DataFrame(
            [
                {
                    "timestamp": pd.Timestamp.utcnow(),
                    "mean": mean,
                    "lower": lower,
                    "upper": upper,
                }
            ]
        )
        header = not file.exists()
        df.to_csv(file, index=False, mode="a", header=header)
