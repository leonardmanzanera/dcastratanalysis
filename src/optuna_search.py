"""
Optuna Search Module - Optimisation Bayésienne des hyperparamètres DCA.

Ce module permet de :
- Explorer l'espace des hyperparamètres avec Optuna
- Valider les résultats avec Walk-Forward
- Comparer automatiquement Rebalancing vs Accumulation
- Sauvegarder les meilleurs résultats
"""

import os
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from data_loader import load_data, compute_indicators, get_market_regime, prepare_simulation_data
from strategy_core import simulate_dca_strategy, DCAStrategyParams, run_backtest
from metrics import calmar_ratio, sortino_ratio, max_drawdown, compute_all_metrics


# Configuration
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def create_objective(
    prices: np.ndarray,
    indicators: np.ndarray,
    is_bull_market: np.ndarray,
    monthly_amount: float = 500.0,
    objective_metric: str = "calmar"
):
    """
    Crée une fonction objective pour Optuna.
    
    Args:
        prices: Array des prix
        indicators: Matrice des indicateurs
        is_bull_market: Array du régime
        monthly_amount: Montant mensuel
        objective_metric: 'calmar' ou 'sortino'
    
    Returns:
        Fonction objective pour Optuna
    """
    
    def objective(trial: optuna.Trial) -> float:
        """Fonction objective pour un trial Optuna."""
        
        # ===== HYPERPARAMÈTRES À OPTIMISER =====
        
        # Rebalancing (on/off)
        enable_rebalancing = trial.suggest_categorical(
            "enable_rebalancing", [True, False]
        )
        
        # Paramètres conditionnels de rebalancing
        if enable_rebalancing:
            rebalance_rsi_trigger = trial.suggest_int(
                "rebalance_rsi_trigger", 70, 90
            )
            rebalance_pct = trial.suggest_float(
                "rebalance_pct", 0.1, 0.5, step=0.05
            )
            rebalance_profit_trigger = trial.suggest_float(
                "rebalance_profit_trigger", 0.3, 1.0, step=0.1
            )
        else:
            rebalance_rsi_trigger = 80
            rebalance_pct = 0.2
            rebalance_profit_trigger = 0.5
        
        # Filtre de régime
        use_regime_filter = trial.suggest_categorical(
            "use_regime_filter", [True, False]
        )
        
        if use_regime_filter:
            bear_multiplier_reduction = trial.suggest_float(
                "bear_multiplier_reduction", 0.3, 0.8, step=0.1
            )
        else:
            bear_multiplier_reduction = 0.5
        
        # Paramètres de levier
        dca_multiplier = trial.suggest_int(
            "dca_multiplier", 2, 12
        )
        
        signal_indicator = trial.suggest_categorical(
            "signal_indicator", [0, 1, 2]  # RSI, MACD, Bollinger
        )
        
        signal_threshold = trial.suggest_int(
            "signal_threshold", 20, 40
        )
        
        # Cooldown
        cooldown_months = trial.suggest_int(
            "cooldown_months", 1, 6
        )
        
        # ===== EXÉCUTION DE LA SIMULATION =====
        
        results = simulate_dca_strategy(
            prices,
            indicators,
            is_bull_market,
            monthly_amount,
            0.0,  # initial_war_chest
            enable_rebalancing,
            rebalance_rsi_trigger,
            rebalance_profit_trigger,
            rebalance_pct,
            use_regime_filter,
            bear_multiplier_reduction,
            dca_multiplier,
            signal_indicator,
            signal_threshold,
            cooldown_months,
        )
        
        equity_curve = results[0]
        total_invested = results[4]
        
        # ===== CALCUL DE LA MÉTRIQUE OBJECTIVE =====
        
        years = len(equity_curve) / 252.0
        
        if objective_metric == "calmar":
            score = calmar_ratio(equity_curve, years)
        elif objective_metric == "sortino":
            score = sortino_ratio(equity_curve)
        else:
            raise ValueError(f"Métrique inconnue: {objective_metric}")
        
        # Pénaliser les scores infinis ou NaN
        if np.isinf(score) or np.isnan(score):
            score = 0.0
        
        # Limiter les scores extrêmes
        score = min(score, 100.0)
        
        # Sauvegarder des métriques additionnelles
        trial.set_user_attr("final_value", float(equity_curve[-1]))
        trial.set_user_attr("total_invested", float(total_invested))
        trial.set_user_attr("max_drawdown", float(max_drawdown(equity_curve)))
        trial.set_user_attr("n_buys", int(results[6]))
        trial.set_user_attr("n_sells", int(results[7]))
        
        return score
    
    return objective


def run_optimization(
    ticker: str,
    start: str,
    end: str,
    n_trials: int = 100,
    monthly_amount: float = 500.0,
    objective_metric: str = "calmar",
    study_name: Optional[str] = None,
    n_jobs: int = 1
) -> optuna.Study:
    """
    Lance l'optimisation Optuna.
    
    Args:
        ticker: Symbole du ticker
        start: Date de début
        end: Date de fin
        n_trials: Nombre de trials
        monthly_amount: Montant mensuel
        objective_metric: Métrique à optimiser
        study_name: Nom de l'étude
        n_jobs: Parallélisme
    
    Returns:
        Étude Optuna complète
    """
    print(f"\n{'='*60}")
    print(f"OPTIMISATION DCA - {ticker}")
    print(f"{'='*60}")
    print(f"Période: {start} → {end}")
    print(f"Trials: {n_trials}")
    print(f"Métrique: {objective_metric}")
    print(f"{'='*60}\n")
    
    # Charger les données
    print("Chargement des données...")
    prices, indicators, is_bull, dates = prepare_simulation_data(
        ticker, start, end
    )
    print(f"  - {len(prices)} jours de trading")
    print(f"  - {(is_bull.sum() / len(is_bull) * 100):.1f}% en Bull Market")
    
    # Créer l'étude
    if study_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        study_name = f"dca_{ticker}_{timestamp}"
    
    storage = f"sqlite:///{RESULTS_DIR / 'optuna_study.db'}"
    
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True
    )
    
    # Créer l'objective
    objective = create_objective(
        prices, indicators, is_bull,
        monthly_amount, objective_metric
    )
    
    # Lancer l'optimisation
    print(f"\nDébut de l'optimisation ({n_trials} trials)...\n")
    
    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=True
    )
    
    return study


def print_results(study: optuna.Study):
    """Affiche les résultats de l'optimisation."""
    
    print(f"\n{'='*60}")
    print("MEILLEURS RÉSULTATS")
    print(f"{'='*60}")
    
    best = study.best_trial
    
    print(f"\n📊 Score Optimal: {best.value:.4f}")
    print(f"\n🎯 Hyperparamètres:")
    for key, value in best.params.items():
        print(f"   {key}: {value}")
    
    print(f"\n📈 Métriques:")
    print(f"   Valeur Finale: €{best.user_attrs['final_value']:,.2f}")
    print(f"   Total Investi: €{best.user_attrs['total_invested']:,.2f}")
    print(f"   Max Drawdown: {best.user_attrs['max_drawdown']*100:.2f}%")
    print(f"   Nb Achats: {best.user_attrs['n_buys']}")
    print(f"   Nb Ventes: {best.user_attrs['n_sells']}")
    
    # Comparer Rebalancing vs Accumulation
    print(f"\n{'='*60}")
    print("COMPARAISON REBALANCING vs ACCUMULATION")
    print(f"{'='*60}")
    
    rebal_trials = [t for t in study.trials if t.params.get('enable_rebalancing', False)]
    accum_trials = [t for t in study.trials if not t.params.get('enable_rebalancing', False)]
    
    if rebal_trials and accum_trials:
        best_rebal = max(rebal_trials, key=lambda t: t.value if t.value else 0)
        best_accum = max(accum_trials, key=lambda t: t.value if t.value else 0)
        
        print(f"\n  Meilleur Rebalancing:")
        print(f"    Score: {best_rebal.value:.4f}")
        print(f"    Valeur: €{best_rebal.user_attrs['final_value']:,.2f}")
        
        print(f"\n  Meilleure Accumulation:")
        print(f"    Score: {best_accum.value:.4f}")
        print(f"    Valeur: €{best_accum.user_attrs['final_value']:,.2f}")
        
        if best_rebal.value > best_accum.value:
            winner = "REBALANCING"
            diff = best_rebal.value - best_accum.value
        else:
            winner = "ACCUMULATION"
            diff = best_accum.value - best_rebal.value
        
        print(f"\n  🏆 Gagnant: {winner} (+{diff:.4f})")


def export_results(study: optuna.Study, filepath: Path):
    """Exporte les résultats en CSV."""
    
    # Créer un DataFrame avec tous les trials
    rows = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            row = {
                "trial_number": trial.number,
                "score": trial.value,
                **trial.params,
                **trial.user_attrs
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.sort_values("score", ascending=False)
    df.to_csv(filepath, index=False)
    print(f"\nRésultats exportés: {filepath}")


def walk_forward_validation(
    ticker: str,
    start: str,
    end: str,
    train_years: int = 2,
    test_years: int = 1,
    n_trials: int = 50,
    monthly_amount: float = 500.0
) -> pd.DataFrame:
    """
    Validation Walk-Forward avec fenêtres glissantes.
    
    Args:
        ticker: Symbole du ticker
        start: Date de début globale
        end: Date de fin globale
        train_years: Années d'entraînement
        test_years: Années de test
        n_trials: Trials par fenêtre
        monthly_amount: Montant mensuel
    
    Returns:
        DataFrame avec résultats de chaque fenêtre
    """
    from datetime import datetime, timedelta
    from dateutil.relativedelta import relativedelta
    
    start_date = datetime.strptime(start, "%Y-%m-%d")
    end_date = datetime.strptime(end, "%Y-%m-%d")
    
    window_results = []
    window_id = 0
    
    current_start = start_date
    
    while True:
        train_end = current_start + relativedelta(years=train_years)
        test_start = train_end
        test_end = test_start + relativedelta(years=test_years)
        
        if test_end > end_date:
            break
        
        print(f"\n{'='*60}")
        print(f"WINDOW {window_id}: Train {current_start.date()} → {train_end.date()}")
        print(f"              Test  {test_start.date()} → {test_end.date()}")
        print(f"{'='*60}")
        
        # Optimiser sur la période d'entraînement
        study = run_optimization(
            ticker,
            current_start.strftime("%Y-%m-%d"),
            train_end.strftime("%Y-%m-%d"),
            n_trials=n_trials,
            monthly_amount=monthly_amount,
            study_name=f"wf_{ticker}_{window_id}_train"
        )
        
        best_params = study.best_params
        
        # Tester sur la période de test
        prices, indicators, is_bull, dates = prepare_simulation_data(
            ticker,
            test_start.strftime("%Y-%m-%d"),
            test_end.strftime("%Y-%m-%d")
        )
        
        # Appliquer les meilleurs paramètres
        results = simulate_dca_strategy(
            prices, indicators, is_bull,
            monthly_amount, 0.0,
            best_params.get('enable_rebalancing', False),
            best_params.get('rebalance_rsi_trigger', 80),
            best_params.get('rebalance_profit_trigger', 0.5),
            best_params.get('rebalance_pct', 0.2),
            best_params.get('use_regime_filter', True),
            best_params.get('bear_multiplier_reduction', 0.5),
            best_params.get('dca_multiplier', 5),
            best_params.get('signal_indicator', 0),
            best_params.get('signal_threshold', 30),
            best_params.get('cooldown_months', 3),
        )
        
        equity_curve = results[0]
        years = len(equity_curve) / 252.0
        
        window_results.append({
            "window_id": window_id,
            "train_start": current_start.strftime("%Y-%m-%d"),
            "train_end": train_end.strftime("%Y-%m-%d"),
            "test_start": test_start.strftime("%Y-%m-%d"),
            "test_end": test_end.strftime("%Y-%m-%d"),
            "train_score": study.best_value,
            "test_calmar": calmar_ratio(equity_curve, years),
            "test_final_value": equity_curve[-1],
            **best_params
        })
        
        # Avancer de 6 mois
        current_start += relativedelta(months=6)
        window_id += 1
    
    return pd.DataFrame(window_results)


def main():
    """Point d'entrée principal."""
    
    parser = argparse.ArgumentParser(
        description="Optimisation Bayésienne des stratégies DCA"
    )
    parser.add_argument(
        "--ticker", type=str, default="SPY",
        help="Ticker à analyser (default: SPY)"
    )
    parser.add_argument(
        "--start", type=str, default="2015-01-01",
        help="Date de début (default: 2015-01-01)"
    )
    parser.add_argument(
        "--end", type=str, default="2024-01-01",
        help="Date de fin (default: 2024-01-01)"
    )
    parser.add_argument(
        "--n-trials", type=int, default=100,
        help="Nombre de trials Optuna (default: 100)"
    )
    parser.add_argument(
        "--monthly-amount", type=float, default=500.0,
        help="Montant mensuel DCA (default: 500)"
    )
    parser.add_argument(
        "--metric", type=str, default="calmar",
        choices=["calmar", "sortino"],
        help="Métrique à optimiser (default: calmar)"
    )
    parser.add_argument(
        "--walk-forward", action="store_true",
        help="Activer validation Walk-Forward"
    )
    parser.add_argument(
        "--n-jobs", type=int, default=1,
        help="Parallélisme (default: 1)"
    )
    
    args = parser.parse_args()
    
    if args.walk_forward:
        # Mode Walk-Forward
        results = walk_forward_validation(
            args.ticker,
            args.start,
            args.end,
            train_years=2,
            test_years=1,
            n_trials=args.n_trials,
            monthly_amount=args.monthly_amount
        )
        print("\n" + "="*60)
        print("RÉSULTATS WALK-FORWARD")
        print("="*60)
        print(results.to_string())
        
        # Exporter
        filepath = RESULTS_DIR / f"walk_forward_{args.ticker}.csv"
        results.to_csv(filepath, index=False)
        print(f"\nRésultats exportés: {filepath}")
    
    else:
        # Mode optimisation standard
        study = run_optimization(
            args.ticker,
            args.start,
            args.end,
            n_trials=args.n_trials,
            monthly_amount=args.monthly_amount,
            objective_metric=args.metric,
            n_jobs=args.n_jobs
        )
        
        print_results(study)
        
        # Exporter les résultats
        filepath = RESULTS_DIR / f"optimization_{args.ticker}.csv"
        export_results(study, filepath)


if __name__ == "__main__":
    main()
