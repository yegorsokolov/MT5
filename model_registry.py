"""Dynamic model selection based on system capabilities."""

import asyncio
import importlib.util
import logging
import sys
import weakref
import gc
import time
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence
import types

import joblib
from prediction_cache import PredictionCache

from analysis.inference_latency import InferenceLatency
from analytics.metrics_store import model_cache_hit, model_unload
from models import residual_learner

# ``models.mixture_of_experts`` depends on no heavy libraries but importing the
# ``models`` package would trigger optional imports like ``torch``. Import the
# module directly from its file path to keep tests lightweight.
sys.modules.setdefault(
    "analytics.metrics_store",
    types.SimpleNamespace(
        record_metric=lambda *a, **k: None,
        model_cache_hit=lambda: None,
        model_unload=lambda: None,
    ),
)

_moe_spec = importlib.util.spec_from_file_location(
    "mixture_of_experts", Path(__file__).with_name("models") / "mixture_of_experts.py"
)
_moe = importlib.util.module_from_spec(_moe_spec)
assert _moe_spec and _moe_spec.loader
sys.modules["mixture_of_experts"] = _moe
_moe_spec.loader.exec_module(_moe)  # type: ignore

ExpertSpec = _moe.ExpertSpec
GatingNetwork = _moe.GatingNetwork
MacroExpert = _moe.MacroExpert
MeanReversionExpert = _moe.MeanReversionExpert
TrendExpert = _moe.TrendExpert

# ``utils.resource_monitor`` lives inside a package whose ``__init__`` pulls in
# heavy optional dependencies. Import the module directly from its file path to
# avoid importing ``utils`` itself and keep tests lightweight.
_spec = importlib.util.spec_from_file_location(
    "resource_monitor", Path(__file__).with_name("utils") / "resource_monitor.py"
)
_rm = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_rm)  # type: ignore

ResourceCapabilities = _rm.ResourceCapabilities
ResourceMonitor = _rm.ResourceMonitor
monitor = _rm.monitor

from telemetry import get_tracer, get_meter

tracer = get_tracer(__name__)
meter = get_meter(__name__)
_refresh_counter = meter.create_counter(
    "model_refreshes", description="Number of model refresh operations"
)

TIERS = {"lite": 0, "standard": 1, "gpu": 2, "hpc": 3}

# Maximum acceptable moving-average latency (seconds) per capability tier
LATENCY_THRESHOLDS: Dict[str, float] = {
    "lite": 0.1,
    "standard": 0.1,
    "gpu": 0.1,
    "hpc": 0.1,
}


@dataclass
class ModelVariant:
    """A specific model implementation and its resource needs."""

    name: str
    requirements: ResourceCapabilities
    quantized: Optional[str] = None
    remote_only: bool = False
    weights: Optional[Path] = None
    quantized_weights: Optional[Path] = None
    quantized_requirements: Optional[ResourceCapabilities] = None

    def __post_init__(self) -> None:
        base = Path(__file__).with_name("models")
        if self.weights is None:
            self.weights = base / f"{self.name}.pkl"
        if self.quantized and self.quantized_weights is None:
            self.quantized_weights = base / f"{self.quantized}.pkl"

    def is_supported(self, capabilities: ResourceCapabilities) -> bool:
        """Return True if system capabilities meet this variant's needs."""
        return (
            capabilities.cpus >= self.requirements.cpus
            and capabilities.memory_gb >= self.requirements.memory_gb
            and (not self.requirements.has_gpu or capabilities.has_gpu)
        )

    def quantized_is_supported(self, capabilities: ResourceCapabilities) -> bool:
        """Return True if quantized variant fits within available resources."""
        req = self.quantized_requirements or self.requirements
        return (
            capabilities.cpus >= req.cpus
            and capabilities.memory_gb >= req.memory_gb
            and (not req.has_gpu or capabilities.has_gpu)
        )


# Registry of task -> model variants ordered from heaviest to lightest
MODEL_REGISTRY: Dict[str, List[ModelVariant]] = {
    "sentiment": [
        ModelVariant(
            name="sentiment_large",
            requirements=ResourceCapabilities(8, 32, True, gpu_count=1),
            quantized="sentiment_large_quantized",
            remote_only=True,
            quantized_requirements=ResourceCapabilities(4, 16, False, gpu_count=0),
        ),
        ModelVariant(
            name="sentiment_medium",
            requirements=ResourceCapabilities(4, 16, True, gpu_count=1),
            quantized="sentiment_medium_quantized",
            quantized_requirements=ResourceCapabilities(2, 8, False, gpu_count=0),
        ),
        ModelVariant(
            name="sentiment_small",
            requirements=ResourceCapabilities(2, 4, False, gpu_count=0),
            quantized="sentiment_small_quantized",
            quantized_requirements=ResourceCapabilities(1, 2, False, gpu_count=0),
        ),
    ],
    "rl_policy": [
        ModelVariant(
            name="rl_large",
            requirements=ResourceCapabilities(16, 64, True, gpu_count=1),
            quantized="rl_large_quantized",
            remote_only=True,
            quantized_requirements=ResourceCapabilities(8, 32, False, gpu_count=0),
        ),
        ModelVariant(
            name="rl_medium",
            requirements=ResourceCapabilities(8, 32, True, gpu_count=1),
            quantized="rl_medium_quantized",
            quantized_requirements=ResourceCapabilities(4, 16, False, gpu_count=0),
        ),
        ModelVariant(
            name="rl_large_distilled",
            requirements=ResourceCapabilities(4, 16, False, gpu_count=0),
            quantized="rl_large_distilled_quantized",
            quantized_requirements=ResourceCapabilities(2, 8, False, gpu_count=0),
        ),
        ModelVariant(
            name="rl_small",
            requirements=ResourceCapabilities(2, 4, False, gpu_count=0),
            quantized="rl_small_quantized",
            quantized_requirements=ResourceCapabilities(1, 2, False, gpu_count=0),
        ),
        ModelVariant(name="baseline", requirements=ResourceCapabilities(1, 1, False, gpu_count=0)),
    ],
    "trade_exit": [
        ModelVariant(
            name="exit_transformer",
            requirements=ResourceCapabilities(8, 32, True, gpu_count=1),
            quantized="exit_transformer_quantized",
            remote_only=True,
            quantized_requirements=ResourceCapabilities(4, 16, False, gpu_count=0),
        ),
        ModelVariant(
            name="exit_transformer_distilled",
            requirements=ResourceCapabilities(4, 16, False, gpu_count=0),
            quantized="exit_transformer_distilled_quantized",
            quantized_requirements=ResourceCapabilities(2, 8, False, gpu_count=0),
        ),
        ModelVariant(
            name="exit_gbm",
            requirements=ResourceCapabilities(2, 4, False, gpu_count=0),
            quantized="exit_gbm_quantized",
            quantized_requirements=ResourceCapabilities(1, 2, False, gpu_count=0),
        ),
        ModelVariant(name="exit_baseline", requirements=ResourceCapabilities(1, 1, False, gpu_count=0)),
    ],
}


class ModelRegistry:
    """Selects appropriate model variants based on available resources."""

    def __init__(
        self,
        monitor: ResourceMonitor = monitor,
        auto_refresh: bool = True,
        latency: Optional[InferenceLatency] = None,
        cfg: Optional[Dict[str, Any]] = None,
        cache: Optional[PredictionCache] = None,
        regime_getter: Callable[[], int] | None = None,
    ) -> None:
        self.monitor = monitor
        self.selected: Dict[str, ModelVariant] = {}
        # Keep track of previous variants so we can rollback if a canary fails
        self._previous: Dict[str, ModelVariant] = {}
        self._task: Optional[asyncio.Task] = None
        self.logger = logging.getLogger(__name__)
        self._models: Dict[str, Any] = {}
        self._finalizers: Dict[str, weakref.finalize] = {}
        self._last_used: Dict[str, float] = {}
        self._weights: Dict[str, Path] = {}
        # Metadata associated with stored policies or models.  This allows
        # training routines to annotate artefacts with useful information such
        # as feature sets or training regimes.
        self._policy_meta: Dict[str, Dict[str, Any]] = {}
        self._variant_by_name: Dict[str, ModelVariant] = {}
        self.latency = latency or InferenceLatency()
        self.cfg = cfg or {}
        self._cache_ttl = float(self.cfg.get("model_cache_ttl", 0))
        if cache is not None:
            self.cache = cache
        else:
            self.cache = PredictionCache(
                int(self.cfg.get("pred_cache_size", 0)),
                self.cfg.get("pred_cache_policy", "lru"),
            )
        self.breach_checks = 3
        self.recovery_checks = 5
        self.regime_getter = regime_getter or (lambda: 0)
        self._latency_breach: Dict[str, int] = {}
        self._latency_recover: Dict[str, int] = {}
        self._latency_history: Dict[str, List[ModelVariant]] = {}
        for variants in MODEL_REGISTRY.values():
            for v in variants:
                self._variant_by_name[v.name] = v
                if v.weights:
                    self._weights[v.name] = Path(v.weights)
                if v.quantized:
                    self._variant_by_name[v.quantized] = v
                    if v.quantized_weights:
                        self._weights[v.quantized] = Path(v.quantized_weights)
        self.moe = GatingNetwork(
            [
                ExpertSpec(TrendExpert(), ResourceCapabilities(2, 4, False, gpu_count=0)),
                ExpertSpec(
                    MeanReversionExpert(), ResourceCapabilities(1, 2, False, gpu_count=0)
                ),
                ExpertSpec(MacroExpert(), ResourceCapabilities(1, 1, False, gpu_count=0)),
            ]
        )
        self._pick_models()
        if auto_refresh:
            self.monitor.start()
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.get_event_loop()
            queue = self.monitor.subscribe()
            self._task = loop.create_task(self._watch(queue))

    # Registration -----------------------------------------------------
    def register_variants(self, task: str, variants: List[ModelVariant]) -> None:
        """Register *variants* for ``task`` and immediately evaluate them.

        This is primarily used by strategy evolution experiments where new
        contenders are generated at runtime.  Variants should be ordered from
        richest to lightest implementation.
        """

        MODEL_REGISTRY[task] = variants
        for v in variants:
            self._variant_by_name[v.name] = v
            if v.weights:
                self._weights[v.name] = Path(v.weights)
            if v.quantized:
                self._variant_by_name[v.quantized] = v
                if v.quantized_weights:
                    self._weights[v.quantized] = Path(v.quantized_weights)
        # Re-run selection to include the new task
        self._pick_models()

    # ------------------------------------------------------------------
    def register_policy(
        self, name: str, path: str | Path, metadata: Dict[str, Any]
    ) -> None:
        """Register a trained RL policy under ``name`` with accompanying metadata.

        The policy path is stored so that executors can later load the model
        dynamically.  ``metadata`` is persisted next to the weights file for
        external introspection and also kept in‑memory for quick access.
        """

        p = Path(path)
        self._weights[name] = p
        self._policy_meta[name] = metadata
        try:
            with open(p.with_suffix(".json"), "w") as fh:
                json.dump(metadata, fh)
        except Exception:  # pragma: no cover - best effort
            self.logger.exception("Failed to write metadata for %s", name)

    # ------------------------------------------------------------------
    def get_policy_path(self, name: str | None = None) -> Optional[Path]:
        """Return the weights path for the active RL policy.

        When ``name`` is ``None`` the currently selected variant for the
        ``rl_policy`` task is used.
        """

        if name is None:
            try:
                name = self.get("rl_policy")
            except Exception:
                return None
        return self._weights.get(name)

    # ------------------------------------------------------------------
    def save_model(
        self,
        name: str,
        model: Any,
        metadata: Dict[str, Any] | None = None,
        path: str | Path | None = None,
    ) -> Path:
        """Serialize ``model`` to ``path`` and register it under ``name``.

        Parameters
        ----------
        name:
            Identifier used to later retrieve the model.
        model:
            Object implementing the policy to be persisted.
        metadata:
            Optional dictionary describing the model.
        path:
            Target file location.  When omitted the model is stored under
            ``models/<name>.pkl`` relative to the repository root.
        """

        p = Path(path) if path is not None else Path("models") / f"{name}.pkl"
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, p)
        self.register_policy(name, p, metadata or {})
        return p

    def _pick_models(self) -> None:
        with tracer.start_as_current_span("pick_models"):
            caps = self.monitor.capabilities
            for task, variants in MODEL_REGISTRY.items():
                baseline = variants[-1]
                chosen = baseline
                for variant in variants:
                    if variant.is_supported(caps):
                        chosen = variant
                        break
                prev = self.selected.get(task)
                if prev != chosen:
                    if chosen is baseline:
                        self.logger.warning("Falling back to baseline for %s", task)
                    elif prev and prev.name == baseline.name:
                        self.logger.info("Restored %s for %s", chosen.name, task)
                self.selected[task] = chosen
            _refresh_counter.add(1)

    async def _watch(self, queue: asyncio.Queue[str]) -> None:
        """Re-evaluate models whenever resource information is refreshed."""

        prev = self.monitor.capability_tier
        while True:
            tier = await queue.get()
            if tier != prev:
                self.logger.info("Capability tier changed to %s; re-evaluating models", tier)
                if TIERS.get(tier, 0) < TIERS.get(prev, 0):
                    self._evict_oversized()
            else:
                # Hardware may have improved within the same tier; still re-check
                self.logger.info("Resource capabilities refreshed; re-evaluating models")
            self._pick_models()
            self._purge_unused()
            try:
                self.reload_tuned_params()
            except Exception:  # pragma: no cover - safety
                pass
            prev = tier

    def get(self, task: str) -> str:
        """Return the chosen model variant for the given task."""
        variant = self.selected[task]
        tier = self.monitor.capability_tier
        caps = self.monitor.capabilities
        if (
            variant.quantized
            and variant.quantized_is_supported(caps)
            and TIERS.get(tier, 0) <= TIERS["lite"]
        ):
            return variant.quantized
        return variant.name

    def refresh(self) -> None:
        """Manually re-run model selection using current capabilities."""
        with tracer.start_as_current_span("refresh_models"):
            self._evict_oversized()
            self._pick_models()
            self._purge_unused()
            try:
                self.reload_tuned_params()
            except Exception:  # pragma: no cover - safety
                pass

    def reload_tuned_params(self) -> None:
        """Reload latest tuned hyper-parameters for the current regime."""

        regime = int(self.regime_getter())
        try:  # pragma: no cover - optional persistence
            from models import model_store
            from models.hot_reload import hot_reload

            store = model_store._ensure_store()  # type: ignore[attr-defined]
            latest = None
            for path in sorted(store.glob("tuned_*.json"), reverse=True):
                try:
                    with open(path) as fh:
                        meta = json.load(fh)
                except Exception:
                    continue
                if int(meta.get("regime", -1)) == regime:
                    latest = meta
                    break
            if latest and "params" in latest:
                hot_reload(latest["params"])
                self.logger.info("Reloaded tuned params for regime %s", regime)
        except Exception:
            self.logger.debug("No tuned params for regime %s", regime)

    def report_failure(self, task: str) -> None:
        """Report that the active model for ``task`` has crashed.

        Switch to the baseline variant and log the event.
        """

        variants = MODEL_REGISTRY.get(task)
        if not variants:
            return
        baseline = variants[-1]
        prev = self.selected.get(task)
        if prev != baseline:
            self.logger.error(
                "Model %s for %s crashed; using baseline", prev.name if prev else "unknown", task
            )
            self.selected[task] = baseline
            self._purge_unused()

    def requires_remote(self, task: str) -> bool:
        """Return True if ``task`` should be executed remotely."""
        return self._remote_variant(task) is not None

    # Internal ------------------------------------------------------------
    def _remote_variant(self, task: str) -> Optional[ModelVariant]:
        variants = MODEL_REGISTRY.get(task)
        if not variants:
            return None
        top = variants[0]
        if top.remote_only and not top.is_supported(self.monitor.capabilities):
            return top
        return None

    def _load_model(self, name: str) -> Any:
        model = self._models.get(name)
        if model is None:
            path = self._weights.get(name)
            if path is None:
                raise KeyError(f"Unknown model {name}")
            model = joblib.load(path)
            self._models[name] = model
            self._finalizers[name] = weakref.finalize(model, self._models.pop, name, None)
            try:
                residual_learner.train_from_cache(name)
            except Exception:  # pragma: no cover - training is best effort
                self.logger.exception("Residual training failed for %s", name)
        else:
            model_cache_hit()
        self._last_used[name] = time.time()
        return model

    def _purge_unused(self) -> None:
        active = {self.get(task) for task in self.selected}
        now = time.time()
        for name in list(self._models):
            ttl_expired = self._cache_ttl and now - self._last_used.get(name, now) > self._cache_ttl
            if name not in active or ttl_expired:
                self._unload_model(name)
        gc.collect()

    def _unload_model(self, name: str) -> None:
        fin = self._finalizers.pop(name, None)
        if fin is not None:
            fin()
        self._models.pop(name, None)
        self._last_used.pop(name, None)
        model_unload()

    def _evict_oversized(self) -> None:
        caps = self.monitor.capabilities
        for name in list(self._models):
            variant = self._variant_by_name.get(name)
            if variant and not variant.is_supported(caps):
                self._unload_model(name)

    def _check_latency(self, task: str, model_name: str) -> None:
        tier = self.monitor.capability_tier
        threshold = LATENCY_THRESHOLDS.get(tier)
        if threshold is None:
            return
        avg = self.latency.moving_average(model_name)
        if avg > threshold:
            cnt = self._latency_breach.get(task, 0) + 1
            self._latency_breach[task] = cnt
            self._latency_recover[task] = 0
            if cnt >= self.breach_checks:
                self._downgrade(task, avg)
                self._latency_breach[task] = 0
        else:
            cnt = self._latency_recover.get(task, 0) + 1
            self._latency_recover[task] = cnt
            self._latency_breach[task] = 0
            if cnt >= self.recovery_checks:
                self._upgrade(task, avg)
                self._latency_recover[task] = 0

    def _downgrade(self, task: str, avg: float) -> None:
        current = self.selected.get(task)
        if not current:
            return
        remote = self._remote_variant(task)
        if remote is not None and current.name != remote.name:
            self.logger.warning(
                "Latency high for %s (%.3fs); offloading to remote %s", task, avg, remote.name
            )
            self._latency_history.setdefault(task, []).append(current)
            self.selected[task] = remote
            self._purge_unused()
            return
        variants = MODEL_REGISTRY.get(task, [])
        try:
            idx = variants.index(current)
        except ValueError:
            return
        if idx + 1 < len(variants):
            new_variant = variants[idx + 1]
            self.logger.warning(
                "Latency high for %s (%.3fs); switching from %s to %s",
                task,
                avg,
                current.name,
                new_variant.name,
            )
            self._latency_history.setdefault(task, []).append(current)
            self.selected[task] = new_variant
            self._purge_unused()

    def _upgrade(self, task: str, avg: float) -> None:
        history = self._latency_history.get(task)
        if not history:
            return
        prev = history.pop()
        if not prev.is_supported(self.monitor.capabilities):
            history.append(prev)
            return
        self.logger.info(
            "Latency normalised for %s (%.3fs); restoring %s", task, avg, prev.name
        )
        self.selected[task] = prev
        self._purge_unused()
        if not history:
            del self._latency_history[task]

    def predict_mixture(self, history: Any, regime: float) -> float:
        """Predict using the mixture-of-experts gating network."""
        with tracer.start_as_current_span("predict_mixture"):
            return self.moe.predict(history, regime, self.monitor.capabilities)

    # ------------------------------------------------------------------
    def _feature_hash(self, features: Any) -> int:
        """Return a stable hash for ``features`` used by the prediction cache."""
        if hasattr(features, "to_dict"):
            records = features.to_dict(orient="records")
        elif isinstance(features, Sequence):
            records = list(features)
        else:
            records = [features]
        return hash(json.dumps(records, sort_keys=True))

    def predict(
        self, task: str, features: Any, loader=None, batch_size: int | None = None
    ) -> Any:
        """Return predictions for ``task`` using local or remote models.

        ``batch_size`` provides a unified batching interface for both local and
        remote models.  When provided, inputs are chunked into batches of this
        size and processed sequentially.
        """
        model_name = self.get(task)
        with tracer.start_as_current_span("predict"):
            variant = self._variant_by_name.get(model_name)
            remote_variant = self._remote_variant(task)
            use_remote = False
            remote_name = model_name

            if variant and not variant.is_supported(self.monitor.capabilities):
                use_remote = True
                remote_name = variant.name
            elif remote_variant is not None:
                use_remote = True
                remote_name = remote_variant.name

            active_name = remote_name if use_remote else model_name

            key = None
            if self.cache.maxsize > 0:
                try:
                    key = self._feature_hash(features)
                    cached = self.cache.get(key)
                    if cached is not None:
                        return cached
                except Exception:
                    key = None

            start = time.perf_counter()
            try:
                if use_remote:
                    import importlib.util

                    spec = importlib.util.spec_from_file_location(
                        "remote_client", Path(__file__).with_name("models") / "remote_client.py"
                    )
                    rc = importlib.util.module_from_spec(spec)
                    assert spec and spec.loader
                    spec.loader.exec_module(rc)  # type: ignore
                    result = rc.predict(remote_name, features, batch_size=batch_size)
                else:
                    model = loader(model_name) if loader else self._load_model(model_name)
                    if batch_size is not None and batch_size > 0:
                        # Process features in batches for local models
                        if hasattr(features, "iloc"):
                            feats = features
                            preds = []
                            for i in range(0, len(feats), batch_size):
                                chunk = feats.iloc[i : i + batch_size]
                                if hasattr(model, "predict_proba"):
                                    preds.extend(model.predict_proba(chunk))
                                else:
                                    preds.extend(model.predict(chunk))
                            result = preds
                        else:
                            records = list(features) if isinstance(features, Sequence) else [features]
                            preds = []
                            for i in range(0, len(records), batch_size):
                                chunk = records[i : i + batch_size]
                                if hasattr(model, "predict_proba"):
                                    preds.extend(model.predict_proba(chunk))  # type: ignore[arg-type]
                                else:
                                    preds.extend(model.predict(chunk))  # type: ignore[arg-type]
                            result = preds
                    else:
                        if hasattr(model, "predict_proba"):
                            result = model.predict_proba(features)
                        else:
                            result = model.predict(features)
                try:
                    residual_model = residual_learner.load(active_name)
                    if residual_model is not None:
                        residual = residual_learner.predict(features, result, residual_model)
                        result = result + residual
                except Exception:  # pragma: no cover - best effort
                    self.logger.exception("Residual inference failed for %s", active_name)
            finally:
                elapsed = time.perf_counter() - start
                self.latency.record(active_name, elapsed)
                self._check_latency(task, active_name)

            if key is not None:
                self.cache.set(key, result)
            return result

    # ------------------------------------------------------------------
    def promote(self, task: str, model_name: str) -> None:
        """Promote a candidate model to production for ``task``."""

        prev = self.selected.get(task)
        if prev:
            self._previous[task] = prev
            requirements = prev.requirements
        else:
            requirements = ResourceCapabilities(0, 0, False, gpu_count=0)
        self.selected[task] = ModelVariant(model_name, requirements)

    # ------------------------------------------------------------------
    def rollback(self, task: str) -> None:
        """Rollback to the previously selected model for ``task``."""

        prev = self._previous.get(task)
        if prev:
            self.logger.info("Reverting to previous model %s for %s", prev.name, task)
            self.selected[task] = prev
            del self._previous[task]


# ---------------------------------------------------------------------------
# Module-level helper for convenient selection refresh
# ---------------------------------------------------------------------------
_GLOBAL_REGISTRY = ModelRegistry(monitor, auto_refresh=False)


def select_models() -> list[str]:
    """Refresh and return the active model variant names."""

    _GLOBAL_REGISTRY.refresh()
    return [_GLOBAL_REGISTRY.get(task) for task in _GLOBAL_REGISTRY.selected]


def register_policy(name: str, path: str | Path, metadata: Dict[str, Any]) -> None:
    """Convenience wrapper around :meth:`ModelRegistry.register_policy`."""

    _GLOBAL_REGISTRY.register_policy(name, path, metadata)


def get_policy_path(name: str | None = None) -> Optional[Path]:
    """Return the stored policy path from the global registry."""

    return _GLOBAL_REGISTRY.get_policy_path(name)


def save_model(
    name: str,
    model: Any,
    metadata: Dict[str, Any] | None = None,
    path: str | Path | None = None,
) -> Path:
    """Serialize ``model`` and register it in the global registry."""

    return _GLOBAL_REGISTRY.save_model(name, model, metadata or {}, path)
