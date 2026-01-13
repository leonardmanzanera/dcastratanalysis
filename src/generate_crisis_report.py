"""
Crisis Report Generator - Sniper Agressif vs Pure DCA (2007-2010).
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import prepare_simulation_data
from strategy_core import simulate_dca_strategy
from metrics import compute_all_metrics, drawdown_series
from strategies_catalog import get_strategy_by_name

def generate_crisis_report():
    ticker = "SPY"
    start_date = "2007-01-01"
    end_date = "2010-01-01"
    monthly_amount = 500.0
    
    # Load data
    prices, indicators, is_bull, dates = prepare_simulation_data(ticker, start_date, end_date)
    
    # Define strategies
    sniper_strat = get_strategy_by_name("AggR_8x_RSI75_P50_CD4")
    
    # Simulation: Sniper
    res_s = simulate_dca_strategy(
        prices, indicators, is_bull,
        monthly_amount, 0.0, True, 
        sniper_strat.rebalance_rsi_trigger,
        sniper_strat.rebalance_profit_trigger,
        sniper_strat.rebalance_pct,
        True, 0.3,
        sniper_strat.dca_multiplier,
        sniper_strat.signal_indicator,
        sniper_strat.signal_threshold,
        sniper_strat.cooldown_months
    )
    
    # Simulation: DCA
    res_d = simulate_dca_strategy(
        prices, indicators, is_bull,
        monthly_amount, 0.0, False,
        80, 0.5, 0.2, False, 1.0, 1, 0, 0, 1
    )
    
    # Unpack correctly
    equity_s, cash_s, shares_s, wc_s, invested_s, sold_s, n_buys_s, n_sells_s = res_s
    equity_d, cash_d, shares_d, wc_d, invested_d, sold_d, n_buys_d, n_sells_d = res_d
    
    # Metrics
    metrics_s = compute_all_metrics(equity_s, invested_s)
    metrics_d = compute_all_metrics(equity_d, invested_d)
    
    # Save Results
    results_dir = Path(__file__).parent.parent / "results"
    
    # Plotting
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Equity Curve
    ax1.plot(dates, equity_s, label="Sniper Agressif", color="#00e676", linewidth=2)
    ax1.plot(dates, equity_d, label="DCA Classique", color="#90a4ae", linewidth=1.5, linestyle="--")
    ax1.set_title("Survie pendant la Crise de 2008 (SPY)")
    ax1.set_ylabel("Valeur du Portefeuille (€)")
    ax1.legend()
    ax1.grid(alpha=0.2)
    
    # Drawdown
    dd_s = drawdown_series(equity_s) * 100
    dd_d = drawdown_series(equity_d) * 100
    ax2.fill_between(dates, -dd_s, 0, color="#00e676", alpha=0.3, label="DD Sniper")
    ax2.fill_between(dates, -dd_d, 0, color="#ef5350", alpha=0.3, label="DD DCA")
    ax2.set_title("Drawdown Comparatif (%)")
    ax2.set_ylabel("Perte depuis ATH (%)")
    ax2.legend()
    ax2.grid(alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(results_dir / "crisis_chart_2008.png")
    
    # Output results
    print("\n--- CRISIS METRICS (2007-2010) ---")
    print(f"invested={invested_s}")
    print(f"final_s={equity_s[-1]}")
    print(f"max_dd_s={metrics_s['max_drawdown']}")
    print(f"calmar_s={metrics_s['calmar_ratio']}")
    print(f"final_d={equity_d[-1]}")
    print(f"max_dd_d={metrics_d['max_drawdown']}")
    print(f"calmar_d={metrics_d['calmar_ratio']}")

if __name__ == "__main__":
    generate_crisis_report()
