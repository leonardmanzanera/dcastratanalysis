"""
Metrics Module - Calcul des métriques de performance pour le backtesting.

Ce module fournit des fonctions optimisées pour calculer :
- Calmar Ratio (CAGR / Max Drawdown)
- Sortino Ratio (Excess Return / Downside Deviation)
- Maximum Drawdown
- Sharpe Ratio
- Statistiques diverses
"""

import numpy as np
from numba import njit
from typing import Tuple, Dict


@njit
def max_drawdown(equity_curve: np.ndarray) -> float:
    """
    Calcule le Maximum Drawdown d'une courbe d'équité.
    
    Args:
        equity_curve: Array des valeurs du portefeuille
    
    Returns:
        Maximum Drawdown en valeur absolue (0.0 à 1.0)
    """
    if len(equity_curve) == 0:
        return 0.0
    
    peak = equity_curve[0]
    max_dd = 0.0
    
    for value in equity_curve:
        if value > peak:
            peak = value
        dd = (peak - value) / peak
        if dd > max_dd:
            max_dd = dd
    
    return max_dd


@njit
def drawdown_series(equity_curve: np.ndarray) -> np.ndarray:
    """
    Calcule la série complète de drawdown.
    
    Args:
        equity_curve: Array des valeurs du portefeuille
    
    Returns:
        Array des drawdowns (0.0 à 1.0) pour chaque point
    """
    n = len(equity_curve)
    if n == 0:
        return np.zeros(0)
    
    drawdowns = np.zeros(n)
    peak = equity_curve[0]
    
    for i in range(n):
        if equity_curve[i] > peak:
            peak = equity_curve[i]
        drawdowns[i] = (peak - equity_curve[i]) / peak
    
    return drawdowns


@njit
def cagr(
    initial_value: float,
    final_value: float,
    years: float
) -> float:
    """
    Calcule le Compound Annual Growth Rate (CAGR).
    
    Args:
        initial_value: Valeur initiale du portefeuille
        final_value: Valeur finale du portefeuille
        years: Nombre d'années
    
    Returns:
        CAGR en pourcentage (ex: 0.10 = 10%)
    """
    if initial_value <= 0 or final_value <= 0 or years <= 0:
        return 0.0
    
    return (final_value / initial_value) ** (1 / years) - 1


@njit
def calmar_ratio(
    equity_curve: np.ndarray,
    years: float
) -> float:
    """
    Calcule le Calmar Ratio (CAGR / Max Drawdown).
    
    Le Calmar Ratio mesure le rendement ajusté au risque.
    Plus il est élevé, meilleur est le ratio rendement/risque.
    
    Args:
        equity_curve: Array des valeurs du portefeuille
        years: Nombre d'années de la période
    
    Returns:
        Calmar Ratio (peut être infini si pas de drawdown)
    """
    if len(equity_curve) < 2 or years <= 0:
        return 0.0
    
    # CAGR
    initial = equity_curve[0]
    final = equity_curve[-1]
    annual_return = cagr(initial, final, years)
    
    # Max Drawdown
    mdd = max_drawdown(equity_curve)
    
    if mdd == 0:
        return float('inf') if annual_return > 0 else 0.0
    
    return annual_return / mdd


@njit
def daily_returns(equity_curve: np.ndarray) -> np.ndarray:
    """
    Calcule les rendements journaliers.
    
    Args:
        equity_curve: Array des valeurs du portefeuille
    
    Returns:
        Array des rendements (longueur = n - 1)
    """
    n = len(equity_curve)
    if n < 2:
        return np.zeros(0)
    
    returns = np.zeros(n - 1)
    for i in range(1, n):
        if equity_curve[i - 1] != 0:
            returns[i - 1] = (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
    
    return returns


@njit
def downside_deviation(
    returns: np.ndarray,
    target: float = 0.0
) -> float:
    """
    Calcule la déviation downside (volatilité des rendements négatifs).
    
    Args:
        returns: Array des rendements
        target: Rendement cible (par défaut 0)
    
    Returns:
        Downside deviation annualisée
    """
    if len(returns) == 0:
        return 0.0
    
    # Calculer les rendements sous le target
    downside_returns = np.zeros(len(returns))
    count = 0
    
    for i in range(len(returns)):
        if returns[i] < target:
            downside_returns[count] = (returns[i] - target) ** 2
            count += 1
    
    if count == 0:
        return 0.0
    
    # Moyenne des carrés des rendements négatifs
    mean_sq = 0.0
    for i in range(count):
        mean_sq += downside_returns[i]
    mean_sq /= count
    
    # Annualiser (252 jours de trading)
    return np.sqrt(mean_sq * 252)


@njit
def sortino_ratio(
    equity_curve: np.ndarray,
    risk_free_rate: float = 0.02
) -> float:
    """
    Calcule le Sortino Ratio.
    
    Sortino = (Return - Risk Free Rate) / Downside Deviation
    
    Le Sortino Ratio est similaire au Sharpe mais ne pénalise
    que la volatilité négative (downside risk).
    
    Args:
        equity_curve: Array des valeurs du portefeuille
        risk_free_rate: Taux sans risque annuel (par défaut 2%)
    
    Returns:
        Sortino Ratio
    """
    if len(equity_curve) < 2:
        return 0.0
    
    # Calcul des rendements journaliers
    returns = daily_returns(equity_curve)
    
    if len(returns) == 0:
        return 0.0
    
    # Rendement annualisé
    total_return = equity_curve[-1] / equity_curve[0] - 1
    years = len(returns) / 252.0
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    
    # Excess return
    excess_return = annual_return - risk_free_rate
    
    # Downside deviation
    dd = downside_deviation(returns, target=risk_free_rate / 252)
    
    if dd == 0:
        return float('inf') if excess_return > 0 else 0.0
    
    return excess_return / dd


@njit
def sharpe_ratio(
    equity_curve: np.ndarray,
    risk_free_rate: float = 0.02
) -> float:
    """
    Calcule le Sharpe Ratio.
    
    Sharpe = (Return - Risk Free Rate) / Volatility
    
    Args:
        equity_curve: Array des valeurs du portefeuille
        risk_free_rate: Taux sans risque annuel (par défaut 2%)
    
    Returns:
        Sharpe Ratio
    """
    if len(equity_curve) < 2:
        return 0.0
    
    # Calcul des rendements journaliers
    returns = daily_returns(equity_curve)
    
    if len(returns) == 0:
        return 0.0
    
    # Rendement annualisé
    total_return = equity_curve[-1] / equity_curve[0] - 1
    years = len(returns) / 252.0
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    
    # Excess return
    excess_return = annual_return - risk_free_rate
    
    # Volatilité annualisée
    mean_return = 0.0
    for r in returns:
        mean_return += r
    mean_return /= len(returns)
    
    variance = 0.0
    for r in returns:
        variance += (r - mean_return) ** 2
    variance /= len(returns)
    
    volatility = np.sqrt(variance * 252)
    
    if volatility == 0:
        return float('inf') if excess_return > 0 else 0.0
    
    return excess_return / volatility


def compute_all_metrics(
    equity_curve: np.ndarray,
    total_invested: float,
    risk_free_rate: float = 0.02
) -> Dict[str, float]:
    """
    Calcule toutes les métriques de performance.
    
    Args:
        equity_curve: Array des valeurs du portefeuille
        total_invested: Montant total investi
        risk_free_rate: Taux sans risque
    
    Returns:
        Dict avec toutes les métriques
    """
    if len(equity_curve) < 2:
        return {
            "total_return": 0.0,
            "total_return_pct": 0.0,
            "cagr": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0,
            "sortino_ratio": 0.0,
            "sharpe_ratio": 0.0,
            "total_invested": total_invested,
            "final_value": 0.0,
            "profit": 0.0,
        }
    
    years = len(equity_curve) / 252.0
    initial = equity_curve[0]
    final = equity_curve[-1]
    
    # Calcul des métriques
    total_return = final - initial
    total_return_pct = (final / initial - 1) * 100 if initial > 0 else 0
    annual_return = cagr(initial, final, years)
    mdd = max_drawdown(equity_curve)
    cal_ratio = calmar_ratio(equity_curve, years)
    sort_ratio = sortino_ratio(equity_curve, risk_free_rate)
    shp_ratio = sharpe_ratio(equity_curve, risk_free_rate)
    
    # Profit réel (vs investi)
    profit = final - total_invested
    profit_pct = (final / total_invested - 1) * 100 if total_invested > 0 else 0
    
    return {
        "total_return": total_return,
        "total_return_pct": total_return_pct,
        "cagr": annual_return * 100,  # En pourcentage
        "max_drawdown": mdd * 100,     # En pourcentage
        "calmar_ratio": cal_ratio,
        "sortino_ratio": sort_ratio,
        "sharpe_ratio": shp_ratio,
        "total_invested": total_invested,
        "final_value": final,
        "profit": profit,
        "profit_pct": profit_pct,
        "years": years,
    }


def format_metrics_table(metrics: Dict[str, float]) -> str:
    """
    Formate les métriques en tableau lisible.
    
    Args:
        metrics: Dict des métriques
    
    Returns:
        String formaté
    """
    lines = [
        "=" * 50,
        " PERFORMANCE METRICS",
        "=" * 50,
        f" Total Invested:     €{metrics['total_invested']:,.2f}",
        f" Final Value:        €{metrics['final_value']:,.2f}",
        f" Profit:             €{metrics['profit']:,.2f} ({metrics['profit_pct']:.2f}%)",
        "-" * 50,
        f" CAGR:               {metrics['cagr']:.2f}%",
        f" Max Drawdown:       {metrics['max_drawdown']:.2f}%",
        f" Calmar Ratio:       {metrics['calmar_ratio']:.2f}",
        f" Sortino Ratio:      {metrics['sortino_ratio']:.2f}",
        f" Sharpe Ratio:       {metrics['sharpe_ratio']:.2f}",
        "-" * 50,
        f" Period:             {metrics['years']:.2f} years",
        "=" * 50,
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    # Test du module
    print("=== Test Metrics Module ===\n")
    
    # Simuler une courbe d'équité
    np.random.seed(42)
    n_days = 252 * 5  # 5 ans
    
    # Simulation d'une courbe avec tendance haussière et volatilité
    returns = np.random.normal(0.0003, 0.015, n_days)  # ~7.5% annuel
    equity = np.zeros(n_days)
    equity[0] = 10000  # Capital initial
    
    for i in range(1, n_days):
        equity[i] = equity[i-1] * (1 + returns[i])
    
    print(f"Capital initial: €{equity[0]:,.2f}")
    print(f"Capital final: €{equity[-1]:,.2f}")
    print()
    
    # Calcul des métriques
    metrics = compute_all_metrics(equity, total_invested=10000)
    print(format_metrics_table(metrics))
    
    # Test individuel
    print("\n=== Tests individuels ===")
    print(f"Max Drawdown: {max_drawdown(equity) * 100:.2f}%")
    print(f"Calmar Ratio: {calmar_ratio(equity, 5.0):.2f}")
    print(f"Sortino Ratio: {sortino_ratio(equity):.2f}")
    print(f"Sharpe Ratio: {sharpe_ratio(equity):.2f}")
