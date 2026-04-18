"""Custom model modules for wind turbine blade damage detection.

Registers CBAM and related attention modules with the ultralytics framework
so they can be referenced by name in YAML model configurations.
"""

from .cbam import CBAM, ChannelAttention, SpatialAttention

__all__ = ["CBAM", "ChannelAttention", "SpatialAttention"]


def register_custom_modules() -> None:
    """Register custom modules into the ultralytics nn.modules namespace.

    Call this function before loading any custom YAML model so that
    ``ultralytics.nn.tasks.parse_model`` can resolve the module names.
    """
    import ultralytics.nn.modules as _unn

    _unn.CBAM = CBAM
    _unn.ChannelAttention = ChannelAttention
    _unn.SpatialAttention = SpatialAttention
