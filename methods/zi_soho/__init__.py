"""Zero-inflated sufficient-statistic SOHO."""

from .learner import ZISOHOLearner, create_learner

METHODS = {"wta_ncm", "support_only", "active_gaussian", "hurdle"}

__all__ = ["METHODS", "ZISOHOLearner", "create_learner"]
