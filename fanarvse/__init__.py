"""FanarVSE: memory-light GPU vector search with error bars from coding theory."""

from .config import PRESETS, IndexConfig, SearchParams
from .index import Index, ground_truth, recall

__version__ = "0.1.0"
__all__ = ["Index", "IndexConfig", "SearchParams", "PRESETS", "ground_truth", "recall", "__version__"]
