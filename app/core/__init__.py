"""Core workflow and state management."""
from .state import BidState

# Note: workflow is imported lazily to avoid circular imports with agents
# Use: from app.core.workflow import build_estimator_graph

__all__ = ['BidState']
