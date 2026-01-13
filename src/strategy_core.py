"""
Strategy Core Module - Moteur de simulation DCA optimisé avec Numba.

Ce module implémente la logique "stateful" du DCA amélioré :
- Tiered Cash System (Flux Mensuel + War Chest)
- Market Regime Filter (Bull/Bear)
- Conditional Rebalancing
- Cool-down Mechanism

Toutes les fonctions critiques sont compilées avec Numba @njit pour
des performances optimales lors des 100+ simulations.
"""

import numpy as np
from numba import njit
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
from enum import IntEnum


class SignalIndicator(IntEnum):
    """Indicateurs de signal pour achat levier."""
    RSI = 0
    MACD = 1
    BOLLINGER = 2
    STOCHASTIC = 3


@dataclass
class DCAStrategyParams:
    """
    Paramètres de la stratégie DCA.
    
    Ces paramètres correspondent à l'espace de recherche Optuna.
    """
    # Cash management
    monthly_amount: float = 500.0        # Montant mensuel DCA
    initial_war_chest: float = 0.0       # Réserve initiale
    
    # Rebalancing
    enable_rebalancing: bool = False     # Activer prise de profit
    rebalance_rsi_trigger: int = 80      # Seuil RSI pour vendre
    rebalance_profit_trigger: float = 0.5  # Gain annuel déclencheur (50%)
    rebalance_pct: float = 0.2           # % du portfolio à vendre
    
    # Market Regime
    use_regime_filter: bool = True       # Adapter selon Bull/Bear
    bear_multiplier_reduction: float = 0.5  # Réduction du levier en Bear
    
    # Leverage buy
    dca_multiplier: int = 5              # Multiplicateur achat levier
    signal_indicator: int = 0            # 0=RSI, 1=MACD, 2=BB
    signal_threshold: int = 30           # Niveau déclenchement
    
    # Cool-down
    cooldown_months: int = 3             # Pause après achat levier
    
    def to_numba_arrays(self) -> Tuple:
        """Convertit les paramètres en arrays pour Numba."""
        return (
            self.monthly_amount,
            self.initial_war_chest,
            self.enable_rebalancing,
            self.rebalance_rsi_trigger,
            self.rebalance_profit_trigger,
            self.rebalance_pct,
            self.use_regime_filter,
            self.bear_multiplier_reduction,
            self.dca_multiplier,
            self.signal_indicator,
            self.signal_threshold,
            self.cooldown_months,
        )


@njit
def is_new_month(dates_ordinal: np.ndarray, idx: int) -> bool:
    """
    Vérifie si l'indice correspond au premier jour d'un nouveau mois.
    
    Args:
        dates_ordinal: Array des dates en format ordinal
        idx: Indice courant
    
    Returns:
        True si nouveau mois
    """
    if idx == 0:
        return True
    
    # Approximation: nouveau mois si différence > 20 jours depuis dernier achat mensuel
    # Plus précis serait de parser les dates, mais Numba a des limitations
    return False  # Sera géré par le compteur de jours


@njit
def check_buy_signal(
    indicators: np.ndarray,
    idx: int,
    signal_indicator: int,
    signal_threshold: int,
    is_bull: bool,
    use_regime_filter: bool
) -> bool:
    """
    Vérifie si un signal d'achat levier est déclenché.
    
    Args:
        indicators: Matrice [n_days, n_indicators] avec RSI, MACD, BB_Pct, Stoch
        idx: Indice courant
        signal_indicator: Type d'indicateur (0=RSI, 1=MACD, 2=BB)
        signal_threshold: Seuil de déclenchement
        is_bull: Régime de marché actuel
        use_regime_filter: Utiliser le filtre de régime
    
    Returns:
        True si signal d'achat levier
    """
    # Si filtre de régime actif et marché Bear, être plus strict
    threshold = signal_threshold
    if use_regime_filter and not is_bull:
        threshold = threshold * 0.8  # Plus strict en Bear
    
    if signal_indicator == 0:  # RSI
        rsi = indicators[idx, 0]
        return rsi < threshold
    
    elif signal_indicator == 1:  # MACD
        macd_hist = indicators[idx, 1]
        # Signal si histogramme très négatif puis remonte
        if idx > 0:
            prev_hist = indicators[idx - 1, 1]
            return macd_hist > prev_hist and macd_hist < 0
        return False
    
    elif signal_indicator == 2:  # Bollinger
        bb_pct = indicators[idx, 2]
        # Signal si prix sous bande inférieure
        return bb_pct < (threshold / 100.0)
    
    elif signal_indicator == 3:  # Stochastic
        stoch_k = indicators[idx, 3]
        return stoch_k < threshold
    
    return False


@njit
def check_sell_signal(
    indicators: np.ndarray,
    idx: int,
    rebalance_rsi_trigger: int,
    current_profit_pct: float,
    rebalance_profit_trigger: float
) -> bool:
    """
    Vérifie si un signal de vente (rebalancing) est déclenché.
    
    Args:
        indicators: Matrice des indicateurs
        idx: Indice courant
        rebalance_rsi_trigger: Seuil RSI pour vente
        current_profit_pct: Profit courant en %
        rebalance_profit_trigger: Seuil de profit pour vente
    
    Returns:
        True si signal de vente
    """
    rsi = indicators[idx, 0]
    
    # Vendre si RSI > seuil OU profit > seuil
    return rsi > rebalance_rsi_trigger or current_profit_pct > rebalance_profit_trigger


@njit
def simulate_dca_strategy(
    prices: np.ndarray,
    indicators: np.ndarray,
    is_bull_market: np.ndarray,
    # Cash params
    monthly_amount: float,
    initial_war_chest: float,
    # Rebalancing params
    enable_rebalancing: bool,
    rebalance_rsi_trigger: int,
    rebalance_profit_trigger: float,
    rebalance_pct: float,
    # Regime params
    use_regime_filter: bool,
    bear_multiplier_reduction: float,
    # Leverage params
    dca_multiplier: int,
    signal_indicator: int,
    signal_threshold: int,
    # Cooldown
    cooldown_months: int,
    # Trading days per month (approximation)
    trading_days_per_month: int = 21
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, int, int]:
    """
    Simule une stratégie DCA complète sur les données.
    
    Returns:
        Tuple contenant:
        - equity_curve: Valeur totale du portefeuille
        - cash_curve: Cash disponible (Monthly + War Chest)
        - shares_curve: Nombre d'actions détenues
        - war_chest_curve: Réserve stratégique
        - total_invested: Montant total investi
        - total_sold: Montant total vendu (rebalancing)
        - n_buys: Nombre d'achats
        - n_sells: Nombre de ventes
    """
    n = len(prices)
    
    # Initialisation des arrays de sortie
    equity_curve = np.zeros(n)
    cash_curve = np.zeros(n)
    shares_curve = np.zeros(n)
    war_chest_curve = np.zeros(n)
    
    # État du portefeuille
    shares = 0.0                    # Actions détenues
    monthly_cash = monthly_amount   # Cash mensuel disponible
    war_chest = initial_war_chest   # Réserve stratégique
    total_invested = 0.0            # Total cash injecté
    total_sold = 0.0                # Total vendu (rebalancing)
    
    # Compteurs
    days_since_monthly = 0          # Jours depuis dernier achat mensuel
    cooldown_remaining = 0          # Jours de cooldown restants
    n_buys = 0
    n_sells = 0
    
    # Première injection de cash
    total_invested += monthly_amount
    
    for i in range(n):
        price = prices[i]
        is_bull = is_bull_market[i]
        
        # ===== 1. MISE À JOUR MENSUELLE DU CASH =====
        days_since_monthly += 1
        if days_since_monthly >= trading_days_per_month:
            # Nouveau mois: ajouter le flux mensuel
            monthly_cash += monthly_amount
            total_invested += monthly_amount
            days_since_monthly = 0
        
        # ===== 2. VÉRIFICATION REBALANCING (si activé) =====
        if enable_rebalancing and shares > 0:
            # Calculer le profit courant
            portfolio_value = shares * price
            cost_basis = total_invested - total_sold  # Approximation
            if cost_basis > 0:
                current_profit_pct = (portfolio_value / cost_basis) - 1
            else:
                current_profit_pct = 0.0
            
            # Vérifier le signal de vente
            if check_sell_signal(
                indicators, i,
                rebalance_rsi_trigger,
                current_profit_pct,
                rebalance_profit_trigger
            ):
                # Vendre X% du portfolio
                shares_to_sell = shares * rebalance_pct
                sell_amount = shares_to_sell * price
                
                shares -= shares_to_sell
                war_chest += sell_amount  # Profit va dans War Chest
                total_sold += sell_amount
                n_sells += 1
        
        # ===== 3. VÉRIFICATION COOLDOWN =====
        if cooldown_remaining > 0:
            # En pause, décrémenter le compteur
            cooldown_remaining -= 1
        else:
            # ===== 4. VÉRIFICATION SIGNAL ACHAT LEVIER =====
            if check_buy_signal(
                indicators, i,
                signal_indicator, signal_threshold,
                is_bull, use_regime_filter
            ):
                # Calculer le multiplicateur effectif
                effective_multiplier = dca_multiplier
                if use_regime_filter and not is_bull:
                    effective_multiplier = int(dca_multiplier * bear_multiplier_reduction)
                    effective_multiplier = max(1, effective_multiplier)
                
                # Montant de l'achat levier
                leverage_amount = monthly_amount * effective_multiplier
                
                # Utiliser War Chest si disponible, sinon Monthly Cash
                available_cash = war_chest + monthly_cash
                buy_amount = min(leverage_amount, available_cash)
                
                if buy_amount > 0:
                    # Acheter avec priorité War Chest
                    if buy_amount <= war_chest:
                        war_chest -= buy_amount
                    else:
                        remainder = buy_amount - war_chest
                        war_chest = 0
                        monthly_cash = max(0, monthly_cash - remainder)
                    
                    # Exécuter l'achat
                    shares_bought = buy_amount / price
                    shares += shares_bought
                    n_buys += 1
                    
                    # Activer le cooldown
                    cooldown_remaining = cooldown_months * trading_days_per_month
            
            # ===== 5. ACHAT DCA STANDARD =====
            elif monthly_cash > 0:
                # Achat standard avec le cash mensuel
                buy_amount = monthly_cash
                shares_bought = buy_amount / price
                shares += shares_bought
                monthly_cash = 0
                n_buys += 1
        
        # ===== 6. MISE À JOUR DES COURBES =====
        portfolio_value = shares * price
        total_cash = monthly_cash + war_chest
        
        equity_curve[i] = portfolio_value + total_cash
        cash_curve[i] = total_cash
        shares_curve[i] = shares
        war_chest_curve[i] = war_chest
    
    return (
        equity_curve,
        cash_curve,
        shares_curve,
        war_chest_curve,
        total_invested,
        total_sold,
        n_buys,
        n_sells
    )


def run_backtest(
    prices: np.ndarray,
    indicators: np.ndarray,
    is_bull_market: np.ndarray,
    params: DCAStrategyParams
) -> Dict:
    """
    Lance un backtest avec les paramètres donnés.
    
    Args:
        prices: Array des prix
        indicators: Matrice des indicateurs
        is_bull_market: Array booléen du régime
        params: Paramètres de la stratégie
    
    Returns:
        Dict avec résultats du backtest
    """
    results = simulate_dca_strategy(
        prices,
        indicators,
        is_bull_market,
        params.monthly_amount,
        params.initial_war_chest,
        params.enable_rebalancing,
        params.rebalance_rsi_trigger,
        params.rebalance_profit_trigger,
        params.rebalance_pct,
        params.use_regime_filter,
        params.bear_multiplier_reduction,
        params.dca_multiplier,
        params.signal_indicator,
        params.signal_threshold,
        params.cooldown_months,
    )
    
    (equity_curve, cash_curve, shares_curve, war_chest_curve,
     total_invested, total_sold, n_buys, n_sells) = results
    
    return {
        "equity_curve": equity_curve,
        "cash_curve": cash_curve,
        "shares_curve": shares_curve,
        "war_chest_curve": war_chest_curve,
        "total_invested": total_invested,
        "total_sold": total_sold,
        "n_buys": n_buys,
        "n_sells": n_sells,
        "final_value": equity_curve[-1],
        "final_shares": shares_curve[-1],
    }


def compare_strategies(
    prices: np.ndarray,
    indicators: np.ndarray,
    is_bull_market: np.ndarray,
    monthly_amount: float = 500.0
) -> Tuple[Dict, Dict]:
    """
    Compare la stratégie avec et sans rebalancing.
    
    Args:
        prices: Array des prix
        indicators: Matrice des indicateurs
        is_bull_market: Array du régime
        monthly_amount: Montant mensuel
    
    Returns:
        Tuple (results_accumulation, results_rebalancing)
    """
    # Stratégie Accumulation Pure
    params_accum = DCAStrategyParams(
        monthly_amount=monthly_amount,
        enable_rebalancing=False,
        use_regime_filter=True,
        dca_multiplier=5,
        signal_indicator=0,  # RSI
        signal_threshold=30,
    )
    results_accum = run_backtest(prices, indicators, is_bull_market, params_accum)
    
    # Stratégie Rebalancing
    params_rebal = DCAStrategyParams(
        monthly_amount=monthly_amount,
        enable_rebalancing=True,
        rebalance_rsi_trigger=80,
        rebalance_pct=0.2,
        use_regime_filter=True,
        dca_multiplier=5,
        signal_indicator=0,  # RSI
        signal_threshold=30,
    )
    results_rebal = run_backtest(prices, indicators, is_bull_market, params_rebal)
    
    return results_accum, results_rebal


if __name__ == "__main__":
    # Test du module
    print("=== Test Strategy Core ===\n")
    
    # Générer des données synthétiques pour test
    np.random.seed(42)
    n_days = 252 * 5  # 5 ans
    
    # Prix simulé (tendance haussière avec volatilité)
    returns = np.random.normal(0.0004, 0.015, n_days)
    prices = np.zeros(n_days)
    prices[0] = 100.0
    for i in range(1, n_days):
        prices[i] = prices[i-1] * (1 + returns[i])
    
    # Indicateurs simulés
    indicators = np.column_stack([
        np.random.uniform(20, 80, n_days),  # RSI
        np.random.normal(0, 0.5, n_days),   # MACD Hist
        np.random.uniform(0, 1, n_days),    # BB Pct
        np.random.uniform(20, 80, n_days),  # Stoch
    ])
    
    # Régime de marché (80% Bull)
    is_bull = np.random.random(n_days) > 0.2
    
    # Test comparaison
    print("Comparaison des stratégies...")
    results_accum, results_rebal = compare_strategies(
        prices, indicators, is_bull, monthly_amount=500.0
    )
    
    print(f"\n{'='*50}")
    print("ACCUMULATION PURE")
    print(f"{'='*50}")
    print(f"  Total Investi:  €{results_accum['total_invested']:,.2f}")
    print(f"  Valeur Finale:  €{results_accum['final_value']:,.2f}")
    print(f"  Profit:         €{results_accum['final_value'] - results_accum['total_invested']:,.2f}")
    print(f"  Nb Achats:      {results_accum['n_buys']}")
    print(f"  Nb Ventes:      {results_accum['n_sells']}")
    
    print(f"\n{'='*50}")
    print("AVEC REBALANCING")
    print(f"{'='*50}")
    print(f"  Total Investi:  €{results_rebal['total_invested']:,.2f}")
    print(f"  Valeur Finale:  €{results_rebal['final_value']:,.2f}")
    print(f"  Profit:         €{results_rebal['final_value'] - results_rebal['total_invested']:,.2f}")
    print(f"  Total Vendu:    €{results_rebal['total_sold']:,.2f}")
    print(f"  Nb Achats:      {results_rebal['n_buys']}")
    print(f"  Nb Ventes:      {results_rebal['n_sells']}")
    
    # Déterminer le gagnant
    print(f"\n{'='*50}")
    if results_accum['final_value'] > results_rebal['final_value']:
        diff = results_accum['final_value'] - results_rebal['final_value']
        print(f"🏆 GAGNANT: Accumulation Pure (+€{diff:,.2f})")
    else:
        diff = results_rebal['final_value'] - results_accum['final_value']
        print(f"🏆 GAGNANT: Rebalancing (+€{diff:,.2f})")
