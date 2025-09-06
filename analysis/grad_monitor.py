import logging
import time
from pathlib import Path
from typing import Iterable, Tuple

try:  # optional dependency
    import torch
    from torch import nn
except Exception:  # pragma: no cover - torch may be stubbed
    torch = None  # type: ignore
    nn = None  # type: ignore

try:  # optional dependency
    from stable_baselines3.common.callbacks import BaseCallback
except Exception:  # pragma: no cover - optional dependency
    class BaseCallback:  # type: ignore
        def __init__(self, *args, **kwargs) -> None:
            self.model = None
            self.n_calls = 0

        def __call__(self, *args, **kwargs):  # pragma: no cover - fallback
            return True


class GradientMonitor:
    """Track gradient norms and detect exploding/vanishing trends."""

    def __init__(
        self,
        explode: float = 1e3,
        vanish: float = 1e-6,
        window: int = 10,
        out_dir: Path | str = Path("reports") / "gradients",
    ) -> None:
        self.explode = float(explode)
        self.vanish = float(vanish)
        self.window = int(window)
        self.history: list[float] = []
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def _compute_norm(self, params: Iterable["nn.Parameter"]) -> float:
        if torch is None:  # pragma: no cover - torch unavailable
            return 0.0
        total = 0.0
        for p in params:
            if p is not None and getattr(p, "grad", None) is not None:
                param_norm = p.grad.data.norm(2).item()
                total += param_norm ** 2
        return total ** 0.5

    def track(self, params: Iterable["nn.Parameter"]) -> Tuple[str | None, float]:
        """Record gradient norm and return detected trend."""

        norm = self._compute_norm(params)
        self.history.append(norm)
        trend = None
        if len(self.history) >= self.window:
            recent = self.history[-self.window:]
            if all(n > self.explode for n in recent):
                trend = "explode"
            elif all(n < self.vanish for n in recent):
                trend = "vanish"
        if trend:
            self.logger.warning("Gradient %s detected (norm=%.4e)", trend, norm)
        return trend, norm

    def plot(self, tag: str) -> Path | None:
        """Persist a plot of recorded gradient norms and return the path."""

        if not self.history:
            return None
        try:  # optional dependency
            import matplotlib.pyplot as plt
        except Exception:  # pragma: no cover - matplotlib optional
            return None
        self.out_dir.mkdir(parents=True, exist_ok=True)
        plt.figure()
        plt.plot(self.history)
        plt.xlabel("step")
        plt.ylabel("gradient norm")
        plt.title(f"Gradient norms ({tag})")
        path = self.out_dir / f"{tag}_{int(time.time())}.png"
        plt.savefig(path)
        plt.close()
        return path


class GradMonitorCallback(BaseCallback):
    """Stable-Baselines callback integrating :class:`GradientMonitor`."""

    def __init__(
        self,
        monitor: GradientMonitor,
        check_freq: int = 100,
        min_lr: float = 1e-6,
        decay: float = 0.5,
        growth: float = 2.0,
    ) -> None:
        super().__init__()
        self.monitor = monitor
        self.check_freq = int(check_freq)
        self.min_lr = float(min_lr)
        self.decay = float(decay)
        self.growth = float(growth)

    def _on_step(self) -> bool:  # type: ignore[override]
        self.n_calls += 1
        if self.n_calls % self.check_freq != 0:
            return True
        policy = getattr(self.model, "policy", None)
        if policy is None or torch is None:
            return True
        trend, _ = self.monitor.track(policy.parameters())
        opt = getattr(policy, "optimizer", None)
        if trend == "explode" and opt is not None:
            for group in opt.param_groups:
                group["lr"] *= self.decay
                if group["lr"] < self.min_lr:
                    self.logger.error("Gradient explosion; aborting training")
                    return False
        elif trend == "vanish" and opt is not None:
            for group in opt.param_groups:
                group["lr"] *= self.growth
        return True
