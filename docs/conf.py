import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "MT5"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
    "myst_parser",
]

autodoc_mock_imports = [
    "torch",
    "tensorflow",
    "sklearn",
    "xgboost",
    "lightgbm",
    "catboost",
    "matplotlib",
    "pytorch_lightning",
    "cvxpy",
    "cvxopt",
    "statsmodels",
    "prophet",
    "gluonts",
    "pandas",
    "numpy",
    "mlflow",
    "joblib",
    "prometheus_client",
    "cryptography",
    "pydantic",
    "filelock",
    "utils",
    "data",
    "features",
]

exclude_patterns = [
    "deployment/*",
    "strategies/*",
    "config.md",
    "key_management.md",
    "message_bus.md",
    "monitoring.md",
    "online_updates.md",
    "strategy_approval.md",
    "config.html",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

html_theme = "sphinx_rtd_theme"

suppress_warnings = ["autodoc.mocked_object"]
