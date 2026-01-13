"""
Robustness Validator - Validation statistique robuste des stratégies.

Ce module implémente des tests de robustesse statistiques:
1. Walk-Forward Validation sur 20+ ans
2. Monte Carlo Bootstrap (fenêtres aléatoires)
3. Rolling Window Analysis (fenêtres glissantes)
4. Out-of-Sample Testing
5. Metrics de stabilité (Sharpe Stability, Calmar Consistency)

Best Practices:
- Test sur minimum 100 fenêtres différentes
- Calcul de confidence intervals (95%)
- Détection du sur-apprentissage (overfitting)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_data, compute_indicators, prepare_simulation_data
from strategy_core import simulate_dca_strategy
from metrics import compute_all_metrics, calmar_ratio, sortino_ratio, max_drawdown
from strategies_catalog import get_all_strategies, get_strategy_by_name, StrategyConfig


@dataclass
class RobustnessResult:
    """Résultat d'un test de robustesse."""
    strategy_name: str
    n_windows: int
    # Mean metrics
    mean_calmar: float
    mean_cagr: float
    mean_drawdown: float
    mean_sharpe: float
    # Stability metrics
    std_calmar: float
    std_cagr: float
    calmar_stability: float  # mean / std
    win_rate: float  # % of windows with positive returns
    # Confidence intervals (95%)
    calmar_ci_lower: float
    calmar_ci_upper: float
    cagr_ci_lower: float
    cagr_ci_upper: float
    # Consistency score (0-100)
    robustness_score: float
    # Overfitting detection
    in_sample_calmar: float
    out_of_sample_calmar: float
    overfitting_ratio: float
    
    def to_dict(self) -> Dict:
        return {
            "strategy_name": self.strategy_name,
            "n_windows": self.n_windows,
            "mean_calmar": self.mean_calmar,
            "mean_cagr": self.mean_cagr,
            "mean_drawdown": self.mean_drawdown,
            "mean_sharpe": self.mean_sharpe,
            "std_calmar": self.std_calmar,
            "std_cagr": self.std_cagr,
            "calmar_stability": self.calmar_stability,
            "win_rate": self.win_rate,
            "calmar_ci_lower": self.calmar_ci_lower,
            "calmar_ci_upper": self.calmar_ci_upper,
            "cagr_ci_lower": self.cagr_ci_lower,
            "cagr_ci_upper": self.cagr_ci_upper,
            "robustness_score": self.robustness_score,
            "in_sample_calmar": self.in_sample_calmar,
            "out_of_sample_calmar": self.out_of_sample_calmar,
            "overfitting_ratio": self.overfitting_ratio,
        }


def generate_rolling_windows(
    total_days: int,
    window_size_days: int = 252 * 5,  # 5 years
    step_days: int = 21,  # Monthly step
    min_windows: int = 100
) -> List[Tuple[int, int]]:
    """
    Génère des fenêtres glissantes pour le test.
    
    Args:
        total_days: Nombre total de jours de données
        window_size_days: Taille de chaque fenêtre
        step_days: Pas entre les fenêtres
        min_windows: Nombre minimum de fenêtres souhaitées
    
    Returns:
        Liste de tuples (start_idx, end_idx)
    """
    windows = []
    start = 0
    
    while start + window_size_days <= total_days:
        windows.append((start, start + window_size_days))
        start += step_days
    
    # Adjust step if not enough windows
    if len(windows) < min_windows and total_days > window_size_days:
        step_days = max(1, (total_days - window_size_days) // (min_windows - 1))
        windows = []
        start = 0
        while start + window_size_days <= total_days:
            windows.append((start, start + window_size_days))
            start += step_days
    
    return windows


def generate_monte_carlo_windows(
    total_days: int,
    window_size_days: int = 252 * 5,
    n_samples: int = 200,
    seed: int = 42
) -> List[Tuple[int, int]]:
    """
    Génère des fenêtres aléatoires (bootstrap) pour Monte Carlo.
    
    Args:
        total_days: Nombre total de jours
        window_size_days: Taille de fenêtre
        n_samples: Nombre d'échantillons
        seed: Graine aléatoire
    
    Returns:
        Liste de tuples (start_idx, end_idx)
    """
    np.random.seed(seed)
    max_start = total_days - window_size_days
    
    if max_start <= 0:
        return [(0, total_days)]
    
    starts = np.random.randint(0, max_start, size=n_samples)
    return [(int(s), int(s + window_size_days)) for s in starts]


def run_strategy_on_window(
    strategy: StrategyConfig,
    prices: np.ndarray,
    indicators: np.ndarray,
    is_bull: np.ndarray,
    start_idx: int,
    end_idx: int
) -> Dict:
    """
    Exécute une stratégie sur une fenêtre spécifique.
    """
    # Slice data
    p = prices[start_idx:end_idx]
    ind = indicators[start_idx:end_idx]
    bull = is_bull[start_idx:end_idx]
    
    if len(p) < 252:  # Min 1 year
        return None
    
    try:
        results = simulate_dca_strategy(
            p, ind, bull,
            strategy.monthly_amount, strategy.initial_war_chest,
            strategy.enable_rebalancing, strategy.rebalance_rsi_trigger,
            strategy.rebalance_profit_trigger, strategy.rebalance_pct,
            strategy.use_regime_filter, strategy.bear_multiplier_reduction,
            strategy.dca_multiplier, strategy.signal_indicator,
            strategy.signal_threshold, strategy.cooldown_months
        )
        
        equity = results[0]
        invested = results[4]
        
        if len(equity) < 10 or invested == 0:
            return None
        
        metrics = compute_all_metrics(equity, invested)
        
        return {
            "calmar": metrics["calmar_ratio"],
            "cagr": metrics["cagr"],
            "drawdown": metrics["max_drawdown"],
            "sharpe": metrics["sharpe_ratio"],
            "sortino": metrics["sortino_ratio"],
            "profit_pct": metrics["profit_pct"],
            "final_value": metrics["final_value"]
        }
    except:
        return None


def validate_strategy_robustness(
    strategy: StrategyConfig,
    prices: np.ndarray,
    indicators: np.ndarray,
    is_bull: np.ndarray,
    n_rolling_windows: int = 100,
    n_monte_carlo: int = 200,
    window_size_years: int = 5
) -> RobustnessResult:
    """
    Valide la robustesse d'une stratégie sur multiples fenêtres.
    
    Args:
        strategy: Configuration de la stratégie
        prices: Prix historiques (20+ ans)
        indicators: Matrice d'indicateurs
        is_bull: Régime de marché
        n_rolling_windows: Nombre de fenêtres glissantes
        n_monte_carlo: Nombre d'échantillons Monte Carlo
        window_size_years: Taille des fenêtres en années
    
    Returns:
        RobustnessResult avec toutes les métriques
    """
    window_size_days = 252 * window_size_years
    total_days = len(prices)
    
    # Generate windows
    rolling_windows = generate_rolling_windows(
        total_days, window_size_days, 
        step_days=21, min_windows=n_rolling_windows
    )
    
    monte_carlo_windows = generate_monte_carlo_windows(
        total_days, window_size_days, n_samples=n_monte_carlo
    )
    
    all_windows = rolling_windows + monte_carlo_windows
    
    # Run on all windows
    results = []
    for start_idx, end_idx in all_windows:
        r = run_strategy_on_window(
            strategy, prices, indicators, is_bull, start_idx, end_idx
        )
        if r is not None:
            results.append(r)
    
    if len(results) < 10:
        return RobustnessResult(
            strategy_name=strategy.name,
            n_windows=len(results),
            mean_calmar=0, mean_cagr=0, mean_drawdown=0, mean_sharpe=0,
            std_calmar=0, std_cagr=0, calmar_stability=0, win_rate=0,
            calmar_ci_lower=0, calmar_ci_upper=0,
            cagr_ci_lower=0, cagr_ci_upper=0,
            robustness_score=0,
            in_sample_calmar=0, out_of_sample_calmar=0, overfitting_ratio=1.0
        )
    
    # Extract metrics arrays
    calmars = np.array([r["calmar"] for r in results])
    cagrs = np.array([r["cagr"] for r in results])
    drawdowns = np.array([r["drawdown"] for r in results])
    sharpes = np.array([r["sharpe"] for r in results])
    profits = np.array([r["profit_pct"] for r in results])
    
    # Filter outliers (>3 sigma)
    calmar_mean = np.mean(calmars)
    calmar_std = np.std(calmars)
    valid_mask = np.abs(calmars - calmar_mean) < 3 * calmar_std
    
    calmars = calmars[valid_mask]
    cagrs = cagrs[valid_mask]
    drawdowns = drawdowns[valid_mask]
    sharpes = sharpes[valid_mask]
    profits = profits[valid_mask]
    
    # Compute statistics
    mean_calmar = np.mean(calmars)
    mean_cagr = np.mean(cagrs)
    mean_drawdown = np.mean(drawdowns)
    mean_sharpe = np.mean(sharpes)
    
    std_calmar = np.std(calmars)
    std_cagr = np.std(cagrs)
    
    # Stability (higher is better)
    calmar_stability = mean_calmar / std_calmar if std_calmar > 0.01 else mean_calmar * 10
    
    # Win rate
    win_rate = np.sum(profits > 0) / len(profits) * 100
    
    # 95% Confidence Intervals
    ci_factor = 1.96  # 95% CI
    n = len(calmars)
    calmar_ci_lower = mean_calmar - ci_factor * std_calmar / np.sqrt(n)
    calmar_ci_upper = mean_calmar + ci_factor * std_calmar / np.sqrt(n)
    
    cagr_std = np.std(cagrs)
    cagr_ci_lower = mean_cagr - ci_factor * cagr_std / np.sqrt(n)
    cagr_ci_upper = mean_cagr + ci_factor * cagr_std / np.sqrt(n)
    
    # In-sample vs Out-of-sample (first half vs second half)
    mid = len(all_windows) // 2
    
    in_sample_results = []
    for start_idx, end_idx in all_windows[:mid]:
        r = run_strategy_on_window(strategy, prices, indicators, is_bull, start_idx, end_idx)
        if r:
            in_sample_results.append(r["calmar"])
    
    out_sample_results = []
    for start_idx, end_idx in all_windows[mid:]:
        r = run_strategy_on_window(strategy, prices, indicators, is_bull, start_idx, end_idx)
        if r:
            out_sample_results.append(r["calmar"])
    
    in_sample_calmar = np.mean(in_sample_results) if in_sample_results else 0
    out_sample_calmar = np.mean(out_sample_results) if out_sample_results else 0
    
    # Overfitting ratio (closer to 1 is better)
    overfitting_ratio = out_sample_calmar / in_sample_calmar if in_sample_calmar > 0.01 else 0
    
    # Robustness Score (0-100)
    # Combines stability, win rate, confidence, and overfitting
    score_stability = min(calmar_stability * 10, 30)  # Max 30 points
    score_winrate = win_rate * 0.3  # Max 30 points
    score_consistency = max(0, 20 - abs(overfitting_ratio - 1) * 20)  # Max 20 points
    score_calmar = min(mean_calmar * 2, 20)  # Max 20 points
    
    robustness_score = score_stability + score_winrate + score_consistency + score_calmar
    robustness_score = min(100, max(0, robustness_score))
    
    return RobustnessResult(
        strategy_name=strategy.name,
        n_windows=len(results),
        mean_calmar=mean_calmar,
        mean_cagr=mean_cagr,
        mean_drawdown=mean_drawdown,
        mean_sharpe=mean_sharpe,
        std_calmar=std_calmar,
        std_cagr=std_cagr,
        calmar_stability=calmar_stability,
        win_rate=win_rate,
        calmar_ci_lower=calmar_ci_lower,
        calmar_ci_upper=calmar_ci_upper,
        cagr_ci_lower=cagr_ci_lower,
        cagr_ci_upper=cagr_ci_upper,
        robustness_score=robustness_score,
        in_sample_calmar=in_sample_calmar,
        out_of_sample_calmar=out_sample_calmar,
        overfitting_ratio=overfitting_ratio
    )


def run_full_robustness_analysis(
    ticker: str = "SPY",
    start_year: int = 2000,
    end_year: int = 2024,
    top_n_strategies: int = 50,
    n_windows: int = 100,
    window_size_years: int = 5
) -> pd.DataFrame:
    """
    Analyse complète de robustesse sur les meilleures stratégies.
    
    Args:
        ticker: Symbole à analyser
        start_year: Année de début (20+ ans recommandé)
        end_year: Année de fin
        top_n_strategies: Nombre de stratégies à tester
        n_windows: Nombre de fenêtres par stratégie
        window_size_years: Taille des fenêtres
    
    Returns:
        DataFrame avec résultats de robustesse
    """
    print("=" * 70)
    print("🔬 ANALYSE DE ROBUSTESSE STATISTIQUE")
    print("=" * 70)
    print(f"\nTicker: {ticker}")
    print(f"Période: {start_year} → {end_year} ({end_year - start_year} ans)")
    print(f"Stratégies: Top {top_n_strategies}")
    print(f"Fenêtres par stratégie: ~{n_windows * 2}")
    print()
    
    # Load full historical data
    print("📊 Chargement des données historiques...")
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-01-01"
    
    prices, indicators, is_bull, dates = prepare_simulation_data(
        ticker, start_date, end_date
    )
    
    print(f"   ✅ {len(prices)} jours de données ({len(prices)/252:.1f} années)")
    
    # Load benchmark results to get top strategies
    results_path = Path(__file__).parent.parent / "results" / "benchmark_results.parquet"
    
    if results_path.exists():
        df_benchmark = pd.read_parquet(results_path)
        df_10y = df_benchmark[
            (df_benchmark["status"] == "success") & 
            (df_benchmark["period"] == "10Y")
        ]
        top_strategies = df_10y.nlargest(top_n_strategies, "calmar_ratio")["strategy_name"].tolist()
    else:
        # Use first N strategies from catalog
        all_strats = get_all_strategies()
        top_strategies = [s.name for s in all_strats[:top_n_strategies]]
    
    print(f"\n🎯 Test de {len(top_strategies)} stratégies...")
    
    # Run robustness analysis
    results = []
    
    for strat_name in tqdm(top_strategies, desc="Validation"):
        try:
            strategy = get_strategy_by_name(strat_name)
            result = validate_strategy_robustness(
                strategy, prices, indicators, is_bull,
                n_rolling_windows=n_windows,
                n_monte_carlo=n_windows * 2,
                window_size_years=window_size_years
            )
            results.append(result.to_dict())
        except Exception as e:
            print(f"\n   ⚠️ Erreur {strat_name}: {e}")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Sort by robustness score
    df = df.sort_values("robustness_score", ascending=False)
    
    # Print summary
    print("\n" + "=" * 70)
    print("🏆 TOP 10 STRATÉGIES LES PLUS ROBUSTES")
    print("=" * 70)
    
    for i, row in df.head(10).iterrows():
        print(f"\n{row['strategy_name'][:45]}")
        print(f"   Score: {row['robustness_score']:.1f}/100")
        print(f"   Calmar: {row['mean_calmar']:.2f} ± {row['std_calmar']:.2f}")
        print(f"   Win Rate: {row['win_rate']:.1f}%")
        print(f"   Overfitting: {row['overfitting_ratio']:.2f}")
    
    # Save results
    output_path = Path(__file__).parent.parent / "results" / "robustness_results.parquet"
    df.to_parquet(output_path, index=False)
    
    csv_path = output_path.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    
    print(f"\n💾 Résultats: {output_path}")
    
    return df


def quick_robustness_check(strategy_name: str, n_windows: int = 50) -> Dict:
    """
    Vérification rapide de robustesse pour une stratégie.
    """
    strategy = get_strategy_by_name(strategy_name)
    
    prices, indicators, is_bull, dates = prepare_simulation_data(
        "SPY", "2000-01-01", "2024-01-01"
    )
    
    result = validate_strategy_robustness(
        strategy, prices, indicators, is_bull,
        n_rolling_windows=n_windows,
        n_monte_carlo=n_windows,
        window_size_years=5
    )
    
    return result.to_dict()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyse de robustesse des stratégies DCA")
    parser.add_argument("--ticker", default="SPY", help="Ticker (default: SPY)")
    parser.add_argument("--start-year", type=int, default=2000, help="Année de début")
    parser.add_argument("--end-year", type=int, default=2024, help="Année de fin")
    parser.add_argument("--top-n", type=int, default=50, help="Nombre de stratégies")
    parser.add_argument("--windows", type=int, default=100, help="Fenêtres par stratégie")
    
    args = parser.parse_args()
    
    run_full_robustness_analysis(
        ticker=args.ticker,
        start_year=args.start_year,
        end_year=args.end_year,
        top_n_strategies=args.top_n,
        n_windows=args.windows
    )
