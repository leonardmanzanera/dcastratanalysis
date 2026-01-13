"""
Benchmark Runner - Exécute toutes les stratégies sur plusieurs périodes.

Ce script:
1. Charge les 250 stratégies du catalogue
2. Les exécute sur SPY pour 5, 7 et 10 ans
3. Calcule les métriques de performance
4. Sauvegarde les résultats en Parquet
"""

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import prepare_simulation_data
from strategy_core import simulate_dca_strategy
from metrics import compute_all_metrics, calmar_ratio, sortino_ratio, sharpe_ratio, max_drawdown
from strategies_catalog import get_all_strategies, StrategyConfig


# Configuration
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_single_strategy(
    strategy: StrategyConfig,
    prices: np.ndarray,
    indicators: np.ndarray,
    is_bull: np.ndarray,
    period_name: str
) -> Dict:
    """
    Exécute une stratégie unique et retourne les métriques.
    """
    try:
        results = simulate_dca_strategy(
            prices,
            indicators,
            is_bull,
            strategy.monthly_amount,
            strategy.initial_war_chest,
            strategy.enable_rebalancing,
            strategy.rebalance_rsi_trigger,
            strategy.rebalance_profit_trigger,
            strategy.rebalance_pct,
            strategy.use_regime_filter,
            strategy.bear_multiplier_reduction,
            strategy.dca_multiplier,
            strategy.signal_indicator,
            strategy.signal_threshold,
            strategy.cooldown_months,
        )
        
        equity_curve = results[0]
        total_invested = results[4]
        years = len(equity_curve) / 252.0
        
        # Compute metrics
        metrics = compute_all_metrics(equity_curve, total_invested)
        
        return {
            "strategy_name": strategy.name,
            "category": strategy.category,
            "period": period_name,
            "years": years,
            "total_invested": total_invested,
            "final_value": metrics["final_value"],
            "profit": metrics["profit"],
            "profit_pct": metrics["profit_pct"],
            "cagr": metrics["cagr"],
            "max_drawdown": metrics["max_drawdown"],
            "calmar_ratio": metrics["calmar_ratio"],
            "sortino_ratio": metrics["sortino_ratio"],
            "sharpe_ratio": metrics["sharpe_ratio"],
            "n_buys": results[6],
            "n_sells": results[7],
            # Strategy params for reference
            "dca_multiplier": strategy.dca_multiplier,
            "signal_indicator": strategy.signal_indicator,
            "signal_threshold": strategy.signal_threshold,
            "enable_rebalancing": strategy.enable_rebalancing,
            "use_regime_filter": strategy.use_regime_filter,
            "cooldown_months": strategy.cooldown_months,
            "status": "success"
        }
        
    except Exception as e:
        return {
            "strategy_name": strategy.name,
            "category": strategy.category,
            "period": period_name,
            "status": "error",
            "error": str(e)
        }


def run_benchmark(
    ticker: str = "SPY",
    periods: List[Tuple[str, str, str]] = None,
    monthly_amount: float = 500.0,
    max_strategies: int = None
) -> pd.DataFrame:
    """
    Exécute le benchmark complet sur toutes les stratégies.
    
    Args:
        ticker: Symbole du ticker
        periods: Liste de (nom, start, end) pour les périodes
        monthly_amount: Montant mensuel DCA
        max_strategies: Limite le nombre de stratégies (pour tests)
    
    Returns:
        DataFrame avec tous les résultats
    """
    if periods is None:
        periods = [
            ("5Y", "2019-01-01", "2024-01-01"),
            ("7Y", "2017-01-01", "2024-01-01"),
            ("10Y", "2014-01-01", "2024-01-01"),
        ]
    
    print("=" * 70)
    print("🚀 BENCHMARK DES 250 STRATÉGIES DCA")
    print("=" * 70)
    print(f"\nTicker: {ticker}")
    print(f"Périodes: {', '.join([p[0] for p in periods])}")
    print()
    
    # Get strategies
    strategies = get_all_strategies()
    if max_strategies:
        strategies = strategies[:max_strategies]
    
    print(f"📊 {len(strategies)} stratégies à tester")
    print(f"📅 {len(periods)} périodes")
    print(f"⏱️ Total: {len(strategies) * len(periods)} simulations")
    print()
    
    all_results = []
    
    for period_name, start_date, end_date in periods:
        print(f"\n{'='*50}")
        print(f"📅 Période: {period_name} ({start_date} → {end_date})")
        print(f"{'='*50}")
        
        # Load data for this period
        print("Chargement des données...")
        try:
            prices, indicators, is_bull, dates = prepare_simulation_data(
                ticker, start_date, end_date
            )
            print(f"  ✅ {len(prices)} jours de trading")
        except Exception as e:
            print(f"  ❌ Erreur: {e}")
            continue
        
        # Update monthly amount in strategies
        for s in strategies:
            s.monthly_amount = monthly_amount
        
        # Run all strategies for this period
        print(f"\nExécution des {len(strategies)} stratégies...")
        
        for strategy in tqdm(strategies, desc=f"  {period_name}", ncols=80):
            result = run_single_strategy(
                strategy, prices, indicators, is_bull, period_name
            )
            all_results.append(result)
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Filter successful results
    df_success = df[df["status"] == "success"].copy()
    
    print(f"\n{'='*70}")
    print("📊 RÉSUMÉ")
    print(f"{'='*70}")
    print(f"Simulations réussies: {len(df_success)} / {len(df)}")
    
    if len(df_success) > 0:
        # Top 10 by Calmar Ratio (10Y if available, else longest period)
        best_period = periods[-1][0]  # Dernière = plus longue
        df_best = df_success[df_success["period"] == best_period].copy()
        
        if len(df_best) > 0:
            df_best = df_best.sort_values("calmar_ratio", ascending=False)
            
            print(f"\n🏆 TOP 10 STRATÉGIES ({best_period}):")
            print("-" * 70)
            for i, row in df_best.head(10).iterrows():
                print(f"  {row['strategy_name'][:40]:<40} "
                      f"Calmar: {row['calmar_ratio']:>6.2f}  "
                      f"CAGR: {row['cagr']:>5.1f}%")
    
    return df


def save_results(df: pd.DataFrame, filename: str = "benchmark_results.parquet"):
    """Sauvegarde les résultats."""
    filepath = RESULTS_DIR / filename
    df.to_parquet(filepath, index=False)
    print(f"\n💾 Résultats sauvegardés: {filepath}")
    
    # Also save CSV for easy viewing
    csv_path = RESULTS_DIR / filename.replace(".parquet", ".csv")
    df.to_csv(csv_path, index=False)
    print(f"💾 CSV sauvegardé: {csv_path}")
    
    return filepath


def load_results(filename: str = "benchmark_results.parquet") -> pd.DataFrame:
    """Charge les résultats existants."""
    filepath = RESULTS_DIR / filename
    if filepath.exists():
        return pd.read_parquet(filepath)
    else:
        raise FileNotFoundError(f"Fichier non trouvé: {filepath}")


def print_category_summary(df: pd.DataFrame):
    """Affiche un résumé par catégorie."""
    df_success = df[df["status"] == "success"]
    
    print("\n📊 PERFORMANCE PAR CATÉGORIE:")
    print("-" * 70)
    
    for period in df_success["period"].unique():
        df_period = df_success[df_success["period"] == period]
        
        print(f"\n{period}:")
        summary = df_period.groupby("category").agg({
            "calmar_ratio": ["mean", "max"],
            "cagr": ["mean", "max"],
            "max_drawdown": ["mean", "min"],
            "final_value": "mean"
        }).round(2)
        
        print(summary.to_string())


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="Benchmark des 250 stratégies DCA"
    )
    parser.add_argument(
        "--ticker", type=str, default="SPY",
        help="Ticker à analyser (default: SPY)"
    )
    parser.add_argument(
        "--monthly", type=float, default=500.0,
        help="Montant mensuel DCA (default: 500)"
    )
    parser.add_argument(
        "--max-strategies", type=int, default=None,
        help="Limite le nombre de stratégies (pour tests)"
    )
    parser.add_argument(
        "--periods", type=str, default="5,7,10",
        help="Périodes à tester en années (default: 5,7,10)"
    )
    
    args = parser.parse_args()
    
    # Parse periods
    period_years = [int(p) for p in args.periods.split(",")]
    periods = []
    for years in period_years:
        start_year = 2024 - years
        periods.append((
            f"{years}Y",
            f"{start_year}-01-01",
            "2024-01-01"
        ))
    
    start_time = time.time()
    
    # Run benchmark
    results = run_benchmark(
        ticker=args.ticker,
        periods=periods,
        monthly_amount=args.monthly,
        max_strategies=args.max_strategies
    )
    
    # Save results
    save_results(results)
    
    # Print summary
    print_category_summary(results)
    
    elapsed = time.time() - start_time
    print(f"\n⏱️ Temps total: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
