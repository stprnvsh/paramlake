"""
ParamLake Zarr - Track, store, and analyze deep learning model parameters.
"""

__version__ = "0.1.0"

# Import main decorator
from paramlake.decorators.model_decorator import paramlake

# Import analyzers
from paramlake.storage.zarr_analyzer import ZarrModelAnalyzer

# Import Icechunk analyzer if available
try:
    from paramlake.storage.icechunk_analyzer import IcechunkModelAnalyzer
    HAS_ICECHUNK = True
except ImportError:
    HAS_ICECHUNK = False
    
__all__ = ["paramlake", "ZarrModelAnalyzer"]
if HAS_ICECHUNK:
    __all__.append("IcechunkModelAnalyzer") 