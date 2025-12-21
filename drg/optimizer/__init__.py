"""DSPy optimizer module for iterative learning and improvement."""

from .optimizer import (
    OptimizerConfig,
    DRGOptimizer,
    create_optimizer,
    evaluate_extraction,
)
from .metrics import (
    ExtractionMetrics,
    calculate_metrics,
    compare_metrics,
)

__all__ = [
    "OptimizerConfig",
    "DRGOptimizer",
    "create_optimizer",
    "evaluate_extraction",
    "ExtractionMetrics",
    "calculate_metrics",
    "compare_metrics",
]

