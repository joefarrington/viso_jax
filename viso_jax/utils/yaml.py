import yaml
from typing import Dict, Any


def to_yaml(data: Dict, filepath: str) -> None:
    """Write a dict to YAML file"""
    with open(filepath, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def from_yaml(filepath: str) -> Dict[str, Any]:
    """Load from a YAML file"""
    with open(filepath) as f:
        data = yaml.safe_load(f)
    return data
