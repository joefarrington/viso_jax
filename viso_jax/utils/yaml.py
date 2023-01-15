import yaml


def to_yaml(data, filepath):
    """A function to write to YAML file"""
    with open(filepath, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def from_yaml(filepath):
    """A function to load from a YAML file"""
    with open(filepath) as f:
        data = yaml.safe_load(f)
    return data
