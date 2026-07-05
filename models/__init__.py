from .dlinear import DLinear

# Future: from .patchtst import PatchTST

_MODEL_REGISTRY = {
    "DLinear": DLinear,
}


def build_model(config):
    """Instantiate the backbone model from config."""
    name = config.model_name
    assert name in _MODEL_REGISTRY, (
        f"Model '{name}' is not registered. Available: {list(_MODEL_REGISTRY.keys())}"
    )
    return _MODEL_REGISTRY[name](config)
