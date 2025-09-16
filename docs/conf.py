import os
import sys

os.environ.setdefault("MT5_DOCS_BUILD", "1")
sys.path.insert(0, os.path.abspath(".."))

project = "MT5"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
    "myst_parser",
]
autosummary_generate = True

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
    "strategy_training.md",
    "config.html",
    "EXTENDING.md",
    "api/features.rst",
    "testing/*",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

html_theme = "sphinx_rtd_theme"

suppress_warnings = ["autodoc.mocked_object"]


def autodoc_skip_member(app, what, name, obj, skip, options):
    if name == "check_symbols":
        return True
    return None


def setup(app):
    app.connect("autodoc-skip-member", autodoc_skip_member)
