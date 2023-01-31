import yaml


def to_yaml(data: dict, filepath: str) -> None:
    """Write a dict to YAML file"""
    with open(filepath, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def from_yaml(filepath: str) -> dict:
    """Load from a YAML file"""
    with open(filepath) as f:
        data = yaml.safe_load(f)
    return data
