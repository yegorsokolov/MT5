import yaml
from pathlib import Path

def test_cloud_init_yaml_valid() -> None:
    path = Path(__file__).resolve().parents[1] / "deploy" / "cloud-init.yaml"
    with path.open() as fh:
        yaml.safe_load(fh)
