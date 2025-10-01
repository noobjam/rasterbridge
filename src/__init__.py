"""
rasterbridge - Bridge geospatial TIFF rasters and Parquet dataframes
"""

from .io_backends import TifHandler, ParquetHandler
from .plotting import plot_band, plot_composite

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.1.0"

__all__ = [
    "TifHandler",
    "ParquetHandler",
    "plot_band",
    "plot_composite",
    "__version__",
]