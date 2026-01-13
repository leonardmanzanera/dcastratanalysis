"""
DCA Strategy Analysis - Moteur d'Optimisation de Stratégies DCA Hybrides
"""

from .data_loader import load_data, compute_indicators, get_market_regime
from .strategy_core import simulate_dca_strategy, DCAStrategyParams
from .metrics import calmar_ratio, sortino_ratio, max_drawdown

__version__ = "1.0.0"
__all__ = [
    "load_data",
    "compute_indicators", 
    "get_market_regime",
    "simulate_dca_strategy",
    "DCAStrategyParams",
    "calmar_ratio",
    "sortino_ratio",
    "max_drawdown",
]
