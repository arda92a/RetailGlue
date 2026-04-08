import os
import yaml
from types import SimpleNamespace

# Root directory of the RetailGlue repository (one level above retailglue/)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_path(relative_path: str) -> str:
    """Resolve a relative path against the repository root."""
    if os.path.isabs(relative_path):
        return relative_path
    return os.path.join(ROOT_DIR, relative_path)


def _dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_dict_to_namespace(i) for i in d]
    return d


def load_config(config_path: str = None) -> SimpleNamespace:
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    return _dict_to_namespace(data)


_config = None


def get_config(config_path: str = None) -> SimpleNamespace:
    global _config
    if _config is None:
        _config = load_config(config_path)
    return _config
