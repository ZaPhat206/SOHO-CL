"""Method-specific feature maps for generic analytic backends."""

from .fly import FLYAnalyticLearner
from .ranpac import RanPACAnalyticLearner

__all__ = ["FLYAnalyticLearner", "RanPACAnalyticLearner"]
