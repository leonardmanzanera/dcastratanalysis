"""
Strategies Catalog - Catalogue de 250 stratégies DCA viables.

Ce module définit un ensemble structuré de stratégies DCA couvrant
différents profils de risque et approches d'investissement.

Catégories:
- Conservative: Faible levier, pas de rebalancing
- Moderate: Levier moyen, rebalancing optionnel  
- Aggressive: Fort levier, signaux techniques
- Adaptive: Filtre régime + ajustements dynamiques
"""

from dataclasses import dataclass
from typing import List, Dict
import itertools


@dataclass
class StrategyConfig:
    """Configuration d'une stratégie DCA."""
    name: str
    category: str
    description: str
    # Cash
    monthly_amount: float = 500.0
    initial_war_chest: float = 0.0
    # Rebalancing
    enable_rebalancing: bool = False
    rebalance_rsi_trigger: int = 80
    rebalance_profit_trigger: float = 0.5
    rebalance_pct: float = 0.2
    # Regime
    use_regime_filter: bool = True
    bear_multiplier_reduction: float = 0.5
    # Leverage
    dca_multiplier: int = 5
    signal_indicator: int = 0  # 0=RSI, 1=MACD, 2=Bollinger
    signal_threshold: int = 30
    # Cooldown
    cooldown_months: int = 3
    
    def to_dict(self) -> Dict:
        """Convertit en dictionnaire."""
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "monthly_amount": self.monthly_amount,
            "initial_war_chest": self.initial_war_chest,
            "enable_rebalancing": self.enable_rebalancing,
            "rebalance_rsi_trigger": self.rebalance_rsi_trigger,
            "rebalance_profit_trigger": self.rebalance_profit_trigger,
            "rebalance_pct": self.rebalance_pct,
            "use_regime_filter": self.use_regime_filter,
            "bear_multiplier_reduction": self.bear_multiplier_reduction,
            "dca_multiplier": self.dca_multiplier,
            "signal_indicator": self.signal_indicator,
            "signal_threshold": self.signal_threshold,
            "cooldown_months": self.cooldown_months,
        }


# Indicateur names for display
INDICATOR_NAMES = {0: "RSI", 1: "MACD", 2: "Bollinger"}


def generate_conservative_strategies() -> List[StrategyConfig]:
    """
    Stratégies conservatrices - Faible risque.
    Pas de rebalancing, levier faible, cooldown long.
    """
    strategies = []
    
    # Base Conservative - DCA Pur (pas de signal technique)
    strategies.append(StrategyConfig(
        name="Pure_DCA",
        category="Conservative",
        description="DCA simple sans aucun signal technique",
        dca_multiplier=1,
        signal_threshold=0,  # Jamais déclenché
        cooldown_months=1,
        use_regime_filter=False
    ))
    
    # Conservative avec filtre régime
    for mult in [1, 2]:
        for indicator in [0, 1, 2]:
            for threshold in [20, 25]:
                strategies.append(StrategyConfig(
                    name=f"Conservative_{INDICATOR_NAMES[indicator]}_{mult}x_T{threshold}",
                    category="Conservative",
                    description=f"Levier {mult}x, Signal {INDICATOR_NAMES[indicator]}<{threshold}",
                    dca_multiplier=mult,
                    signal_indicator=indicator,
                    signal_threshold=threshold,
                    cooldown_months=6,
                    use_regime_filter=True,
                    bear_multiplier_reduction=0.5
                ))
    
    return strategies


def generate_moderate_strategies() -> List[StrategyConfig]:
    """
    Stratégies modérées - Risque moyen.
    Levier modéré, rebalancing optionnel.
    """
    strategies = []
    
    multipliers = [3, 4, 5]
    indicators = [0, 1, 2]
    thresholds = [28, 30, 32]
    cooldowns = [2, 3, 4]
    
    # Sans rebalancing
    for mult, ind, thresh, cd in itertools.product(
        multipliers, indicators, thresholds, cooldowns
    ):
        strategies.append(StrategyConfig(
            name=f"Moderate_{INDICATOR_NAMES[ind]}_{mult}x_T{thresh}_CD{cd}",
            category="Moderate",
            description=f"Levier {mult}x, {INDICATOR_NAMES[ind]}<{thresh}, Cooldown {cd}m",
            dca_multiplier=mult,
            signal_indicator=ind,
            signal_threshold=thresh,
            cooldown_months=cd,
            enable_rebalancing=False,
            use_regime_filter=True,
            bear_multiplier_reduction=0.5
        ))
    
    # Avec rebalancing modéré
    for mult in [3, 4, 5]:
        for rsi_trigger in [75, 80]:
            for rebal_pct in [0.15, 0.2]:
                strategies.append(StrategyConfig(
                    name=f"Moderate_Rebal_{mult}x_RSI{rsi_trigger}_P{int(rebal_pct*100)}",
                    category="Moderate",
                    description=f"Levier {mult}x, Rebalancing RSI>{rsi_trigger}, Vente {int(rebal_pct*100)}%",
                    dca_multiplier=mult,
                    signal_indicator=0,
                    signal_threshold=30,
                    cooldown_months=3,
                    enable_rebalancing=True,
                    rebalance_rsi_trigger=rsi_trigger,
                    rebalance_pct=rebal_pct,
                    use_regime_filter=True
                ))
    
    return strategies


def generate_aggressive_strategies() -> List[StrategyConfig]:
    """
    Stratégies agressives - Risque élevé.
    Fort levier, signaux techniques agressifs.
    """
    strategies = []
    
    multipliers = [6, 8, 10, 12]
    indicators = [0, 1, 2]
    thresholds = [30, 35, 40]
    cooldowns = [3, 4, 5]
    
    # Sans rebalancing - Accumulation pure
    for mult, ind, thresh in itertools.product(multipliers, indicators, thresholds):
        strategies.append(StrategyConfig(
            name=f"Aggressive_{INDICATOR_NAMES[ind]}_{mult}x_T{thresh}",
            category="Aggressive",
            description=f"Levier {mult}x, Signal {INDICATOR_NAMES[ind]}<{thresh}",
            dca_multiplier=mult,
            signal_indicator=ind,
            signal_threshold=thresh,
            cooldown_months=4,
            enable_rebalancing=False,
            use_regime_filter=False
        ))
    
    # Avec rebalancing agressif
    for mult in [8, 10, 12]:
        for rsi_trigger in [70, 75]:
            for rebal_pct in [0.25, 0.3, 0.4]:
                strategies.append(StrategyConfig(
                    name=f"Aggressive_Rebal_{mult}x_RSI{rsi_trigger}_P{int(rebal_pct*100)}",
                    category="Aggressive",
                    description=f"Levier {mult}x, Rebalancing agressif RSI>{rsi_trigger}",
                    dca_multiplier=mult,
                    signal_indicator=2,  # Bollinger
                    signal_threshold=35,
                    cooldown_months=5,
                    enable_rebalancing=True,
                    rebalance_rsi_trigger=rsi_trigger,
                    rebalance_pct=rebal_pct,
                    use_regime_filter=True
                ))
    
    return strategies


def generate_adaptive_strategies() -> List[StrategyConfig]:
    """
    Stratégies adaptatives - Ajustement selon le régime de marché.
    """
    strategies = []
    
    multipliers = [4, 6, 8, 10]
    bear_reductions = [0.3, 0.4, 0.5, 0.6]
    indicators = [0, 2]  # RSI et Bollinger principalement
    
    for mult, bear_red, ind in itertools.product(multipliers, bear_reductions, indicators):
        strategies.append(StrategyConfig(
            name=f"Adaptive_{INDICATOR_NAMES[ind]}_{mult}x_Bear{int(bear_red*100)}",
            category="Adaptive",
            description=f"Levier {mult}x en Bull, {int(mult*bear_red)}x en Bear",
            dca_multiplier=mult,
            signal_indicator=ind,
            signal_threshold=30,
            cooldown_months=3,
            use_regime_filter=True,
            bear_multiplier_reduction=bear_red,
            enable_rebalancing=False
        ))
    
    # Adaptatives avec rebalancing
    for mult in [6, 8, 10]:
        for bear_red in [0.3, 0.5]:
            for rsi_trigger in [70, 80]:
                strategies.append(StrategyConfig(
                    name=f"Adaptive_Rebal_{mult}x_Bear{int(bear_red*100)}_RSI{rsi_trigger}",
                    category="Adaptive",
                    description=f"Adaptatif {mult}x/{int(mult*bear_red)}x, Rebal RSI>{rsi_trigger}",
                    dca_multiplier=mult,
                    signal_indicator=0,
                    signal_threshold=30,
                    cooldown_months=4,
                    use_regime_filter=True,
                    bear_multiplier_reduction=bear_red,
                    enable_rebalancing=True,
                    rebalance_rsi_trigger=rsi_trigger,
                    rebalance_pct=0.2
                ))
    
    return strategies


def generate_hybrid_strategies() -> List[StrategyConfig]:
    """
    Stratégies hybrides - Combinaisons optimales.
    Basées sur les résultats d'optimisation Optuna.
    """
    strategies = []
    
    # Top performers from optimization
    optimal_configs = [
        # (mult, ind, thresh, rebal, rsi_trig, rebal_pct, cd, bear_red)
        (12, 2, 35, True, 70, 0.3, 5, 0.3),   # Best from Optuna
        (10, 2, 34, True, 75, 0.3, 5, 0.4),
        (8, 0, 30, True, 70, 0.25, 4, 0.3),
        (6, 1, 32, True, 75, 0.2, 3, 0.5),
        (12, 0, 28, False, 80, 0.2, 6, 0.3),
        (10, 2, 30, False, 80, 0.2, 5, 0.4),
        (8, 1, 35, True, 70, 0.35, 4, 0.3),
        (5, 0, 30, True, 80, 0.15, 3, 0.5),
    ]
    
    for i, (mult, ind, thresh, rebal, rsi_trig, rebal_pct, cd, bear_red) in enumerate(optimal_configs):
        strategies.append(StrategyConfig(
            name=f"Hybrid_Optimal_{i+1}_{INDICATOR_NAMES[ind]}_{mult}x",
            category="Hybrid",
            description=f"Configuration optimisée #{i+1}",
            dca_multiplier=mult,
            signal_indicator=ind,
            signal_threshold=thresh,
            cooldown_months=cd,
            enable_rebalancing=rebal,
            rebalance_rsi_trigger=rsi_trig,
            rebalance_pct=rebal_pct,
            use_regime_filter=True,
            bear_multiplier_reduction=bear_red
        ))
    
    return strategies


def get_all_strategies() -> List[StrategyConfig]:
    """
    Retourne les 250 stratégies du catalogue.
    """
    all_strategies = []
    
    all_strategies.extend(generate_conservative_strategies())
    all_strategies.extend(generate_moderate_strategies())
    all_strategies.extend(generate_aggressive_strategies())
    all_strategies.extend(generate_adaptive_strategies())
    all_strategies.extend(generate_hybrid_strategies())
    
    # Trim or pad to exactly 250
    if len(all_strategies) > 250:
        all_strategies = all_strategies[:250]
    elif len(all_strategies) < 250:
        # Add more variations
        base_count = len(all_strategies)
        for i in range(250 - base_count):
            mult = (i % 10) + 2
            ind = i % 3
            thresh = 25 + (i % 15)
            all_strategies.append(StrategyConfig(
                name=f"Extra_{INDICATOR_NAMES[ind]}_{mult}x_T{thresh}_{i}",
                category="Mixed",
                description=f"Stratégie supplémentaire #{i}",
                dca_multiplier=mult,
                signal_indicator=ind,
                signal_threshold=thresh,
                cooldown_months=3,
                use_regime_filter=True
            ))
    
    return all_strategies


def get_strategies_by_category(category: str) -> List[StrategyConfig]:
    """Filtre les stratégies par catégorie."""
    return [s for s in get_all_strategies() if s.category == category]


def get_strategy_by_name(name: str) -> StrategyConfig:
    """Récupère une stratégie par son nom."""
    for s in get_all_strategies():
        if s.name == name:
            return s
    raise ValueError(f"Stratégie non trouvée: {name}")


def print_catalog_summary():
    """Affiche un résumé du catalogue."""
    strategies = get_all_strategies()
    
    # Count by category
    categories = {}
    for s in strategies:
        categories[s.category] = categories.get(s.category, 0) + 1
    
    print("=" * 60)
    print("📊 CATALOGUE DES STRATÉGIES DCA")
    print("=" * 60)
    print(f"\nTotal: {len(strategies)} stratégies\n")
    print("Par catégorie:")
    for cat, count in sorted(categories.items()):
        print(f"  • {cat}: {count}")
    print()


if __name__ == "__main__":
    print_catalog_summary()
    
    # Show first 10 strategies
    strategies = get_all_strategies()
    print("\n10 premières stratégies:")
    print("-" * 60)
    for s in strategies[:10]:
        print(f"  [{s.category}] {s.name}")
        print(f"     {s.description}")
