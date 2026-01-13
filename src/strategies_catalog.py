"""
Strategies Catalog 1000 - Catalogue de 1000 stratégies DCA avancées.

Objectif: Battre le DCA classique tout en limitant le risque (Drawdown).

Catégories:
- Conservative: Faible levier, stabilité maximale
- Moderate: Équilibre rendement/risque
- Aggressive: Fort levier, signaux techniques
- Adaptive: Ajustement dynamique Bull/Bear
- Hybrid: Combinaisons optimisées Optuna
- RiskManaged: Focus sur Calmar Ratio élevé
- MomentumBased: Signaux MACD et tendance
- VolatilityBased: Exploitation des Bollinger Bands
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


INDICATOR_NAMES = {0: "RSI", 1: "MACD", 2: "BB"}


def generate_conservative_strategies() -> List[StrategyConfig]:
    """Stratégies conservatrices - Stabilité maximale."""
    strategies = []
    
    # DCA Pure baseline
    strategies.append(StrategyConfig(
        name="Pure_DCA",
        category="Conservative",
        description="DCA classique sans signal",
        dca_multiplier=1,
        signal_threshold=0,
        cooldown_months=1,
        use_regime_filter=False,
        enable_rebalancing=False
    ))
    
    # Low leverage with regime filter
    for mult in [1, 2, 3]:
        for indicator in [0, 1, 2]:
            for threshold in [20, 25, 30]:
                for bear_red in [0.3, 0.5, 0.7]:
                    strategies.append(StrategyConfig(
                        name=f"Cons_{INDICATOR_NAMES[indicator]}_{mult}x_T{threshold}_B{int(bear_red*100)}",
                        category="Conservative",
                        description=f"Conservatif {mult}x, Bear reduction {int(bear_red*100)}%",
                        dca_multiplier=mult,
                        signal_indicator=indicator,
                        signal_threshold=threshold,
                        cooldown_months=6,
                        use_regime_filter=True,
                        bear_multiplier_reduction=bear_red,
                        enable_rebalancing=False
                    ))
    
    return strategies


def generate_moderate_strategies() -> List[StrategyConfig]:
    """Stratégies modérées - Équilibre rendement/risque."""
    strategies = []
    
    multipliers = [3, 4, 5, 6]
    indicators = [0, 1, 2]
    thresholds = [25, 28, 30, 32, 35]
    cooldowns = [2, 3, 4, 5]
    
    # Without rebalancing
    for mult, ind, thresh, cd in itertools.product(multipliers, indicators, thresholds[:3], cooldowns[:2]):
        strategies.append(StrategyConfig(
            name=f"Mod_{INDICATOR_NAMES[ind]}_{mult}x_T{thresh}_CD{cd}",
            category="Moderate",
            description=f"Modéré {mult}x, {INDICATOR_NAMES[ind]}<{thresh}",
            dca_multiplier=mult,
            signal_indicator=ind,
            signal_threshold=thresh,
            cooldown_months=cd,
            enable_rebalancing=False,
            use_regime_filter=True,
            bear_multiplier_reduction=0.5
        ))
    
    # With rebalancing
    for mult in multipliers:
        for rsi in [70, 75, 80, 85]:
            for pct in [0.1, 0.15, 0.2, 0.25]:
                strategies.append(StrategyConfig(
                    name=f"ModR_{mult}x_RSI{rsi}_P{int(pct*100)}",
                    category="Moderate",
                    description=f"Modéré Rebal {mult}x, RSI>{rsi}",
                    dca_multiplier=mult,
                    signal_indicator=0,
                    signal_threshold=30,
                    cooldown_months=3,
                    enable_rebalancing=True,
                    rebalance_rsi_trigger=rsi,
                    rebalance_pct=pct,
                    use_regime_filter=True
                ))
    
    return strategies


def generate_aggressive_strategies() -> List[StrategyConfig]:
    """Stratégies agressives - Fort levier."""
    strategies = []
    
    multipliers = [6, 8, 10, 12, 15]
    indicators = [0, 1, 2]
    thresholds = [28, 30, 32, 35, 38, 40]
    
    # Without rebalancing
    for mult, ind, thresh in itertools.product(multipliers, indicators, thresholds):
        strategies.append(StrategyConfig(
            name=f"Agg_{INDICATOR_NAMES[ind]}_{mult}x_T{thresh}",
            category="Aggressive",
            description=f"Agressif {mult}x, {INDICATOR_NAMES[ind]}<{thresh}",
            dca_multiplier=mult,
            signal_indicator=ind,
            signal_threshold=thresh,
            cooldown_months=4,
            enable_rebalancing=False,
            use_regime_filter=False
        ))
    
    # With rebalancing - TARGET: Beat DCA with lower risk
    for mult in [8, 10, 12]:
        for rsi in [65, 70, 75, 80]:
            for pct in [0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
                for cd in [3, 4, 5, 6]:
                    strategies.append(StrategyConfig(
                        name=f"AggR_{mult}x_RSI{rsi}_P{int(pct*100)}_CD{cd}",
                        category="Aggressive",
                        description=f"Agressif Rebal {mult}x, RSI>{rsi}, Vente {int(pct*100)}%",
                        dca_multiplier=mult,
                        signal_indicator=2,  # Bollinger
                        signal_threshold=35,
                        cooldown_months=cd,
                        enable_rebalancing=True,
                        rebalance_rsi_trigger=rsi,
                        rebalance_pct=pct,
                        use_regime_filter=True,
                        bear_multiplier_reduction=0.3
                    ))
    
    return strategies


def generate_adaptive_strategies() -> List[StrategyConfig]:
    """Stratégies adaptatives - Ajustement Bull/Bear."""
    strategies = []
    
    multipliers = [4, 6, 8, 10, 12]
    bear_reductions = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    indicators = [0, 2]
    
    for mult, bear_red, ind in itertools.product(multipliers, bear_reductions, indicators):
        strategies.append(StrategyConfig(
            name=f"Adapt_{INDICATOR_NAMES[ind]}_{mult}x_B{int(bear_red*100)}",
            category="Adaptive",
            description=f"Adaptatif {mult}x Bull, {int(mult*bear_red)}x Bear",
            dca_multiplier=mult,
            signal_indicator=ind,
            signal_threshold=30,
            cooldown_months=3,
            use_regime_filter=True,
            bear_multiplier_reduction=bear_red,
            enable_rebalancing=False
        ))
    
    # Adaptive with rebalancing
    for mult in [6, 8, 10, 12]:
        for bear_red in [0.2, 0.3, 0.5]:
            for rsi in [65, 70, 75, 80]:
                for pct in [0.15, 0.2, 0.25, 0.3]:
                    strategies.append(StrategyConfig(
                        name=f"AdaptR_{mult}x_B{int(bear_red*100)}_RSI{rsi}_P{int(pct*100)}",
                        category="Adaptive",
                        description=f"Adaptatif Rebal {mult}x, Bear {int(bear_red*100)}%",
                        dca_multiplier=mult,
                        signal_indicator=0,
                        signal_threshold=30,
                        cooldown_months=4,
                        use_regime_filter=True,
                        bear_multiplier_reduction=bear_red,
                        enable_rebalancing=True,
                        rebalance_rsi_trigger=rsi,
                        rebalance_pct=pct
                    ))
    
    return strategies


def generate_risk_managed_strategies() -> List[StrategyConfig]:
    """Stratégies gestion du risque - Focus Calmar Ratio."""
    strategies = []
    
    # High rebalancing frequency to lock profits
    for mult in [5, 6, 8, 10]:
        for rsi in [65, 68, 70, 72, 75]:
            for pct in [0.15, 0.2, 0.25, 0.3]:
                for bear in [0.2, 0.3, 0.4]:
                    strategies.append(StrategyConfig(
                        name=f"Risk_{mult}x_RSI{rsi}_P{int(pct*100)}_B{int(bear*100)}",
                        category="RiskManaged",
                        description=f"Risk-Managed {mult}x, Profit Lock RSI>{rsi}",
                        dca_multiplier=mult,
                        signal_indicator=2,  # Bollinger for entries
                        signal_threshold=32,
                        cooldown_months=4,
                        use_regime_filter=True,
                        bear_multiplier_reduction=bear,
                        enable_rebalancing=True,
                        rebalance_rsi_trigger=rsi,
                        rebalance_pct=pct
                    ))
    
    return strategies


def generate_momentum_strategies() -> List[StrategyConfig]:
    """Stratégies momentum - MACD based."""
    strategies = []
    
    for mult in [4, 6, 8, 10]:
        for thresh in [25, 28, 30, 32, 35]:
            for cd in [2, 3, 4, 5]:
                strategies.append(StrategyConfig(
                    name=f"Mom_MACD_{mult}x_T{thresh}_CD{cd}",
                    category="Momentum",
                    description=f"Momentum MACD {mult}x",
                    dca_multiplier=mult,
                    signal_indicator=1,  # MACD
                    signal_threshold=thresh,
                    cooldown_months=cd,
                    use_regime_filter=True,
                    bear_multiplier_reduction=0.4,
                    enable_rebalancing=False
                ))
    
    # With rebalancing
    for mult in [6, 8, 10]:
        for rsi in [70, 75, 80]:
            for pct in [0.2, 0.25, 0.3]:
                strategies.append(StrategyConfig(
                    name=f"MomR_MACD_{mult}x_RSI{rsi}_P{int(pct*100)}",
                    category="Momentum",
                    description=f"Momentum Rebal MACD {mult}x",
                    dca_multiplier=mult,
                    signal_indicator=1,
                    signal_threshold=30,
                    cooldown_months=4,
                    use_regime_filter=True,
                    bear_multiplier_reduction=0.3,
                    enable_rebalancing=True,
                    rebalance_rsi_trigger=rsi,
                    rebalance_pct=pct
                ))
    
    return strategies


def generate_volatility_strategies() -> List[StrategyConfig]:
    """Stratégies volatilité - Bollinger Bands."""
    strategies = []
    
    for mult in [4, 6, 8, 10, 12]:
        for thresh in [25, 30, 35, 40]:
            for cd in [3, 4, 5, 6]:
                strategies.append(StrategyConfig(
                    name=f"Vol_BB_{mult}x_T{thresh}_CD{cd}",
                    category="Volatility",
                    description=f"Volatility BB {mult}x, Threshold {thresh}",
                    dca_multiplier=mult,
                    signal_indicator=2,  # Bollinger
                    signal_threshold=thresh,
                    cooldown_months=cd,
                    use_regime_filter=True,
                    bear_multiplier_reduction=0.4,
                    enable_rebalancing=False
                ))
    
    # With rebalancing
    for mult in [8, 10, 12]:
        for rsi in [68, 72, 75, 78]:
            for pct in [0.2, 0.25, 0.3, 0.35]:
                strategies.append(StrategyConfig(
                    name=f"VolR_BB_{mult}x_RSI{rsi}_P{int(pct*100)}",
                    category="Volatility",
                    description=f"Volatility Rebal BB {mult}x",
                    dca_multiplier=mult,
                    signal_indicator=2,
                    signal_threshold=35,
                    cooldown_months=5,
                    use_regime_filter=True,
                    bear_multiplier_reduction=0.3,
                    enable_rebalancing=True,
                    rebalance_rsi_trigger=rsi,
                    rebalance_pct=pct
                ))
    
    return strategies


def generate_hybrid_optimal_strategies() -> List[StrategyConfig]:
    """Stratégies hybrides optimales - Best from Optuna."""
    strategies = []
    
    # Grid search around optimal parameters
    for mult in [8, 10, 12]:
        for ind in [0, 2]:
            for thresh in [30, 32, 35]:
                for rsi in [68, 70, 72, 75]:
                    for pct in [0.25, 0.3, 0.35]:
                        for bear in [0.25, 0.3, 0.35]:
                            for cd in [4, 5, 6]:
                                strategies.append(StrategyConfig(
                                    name=f"Hyb_{INDICATOR_NAMES[ind]}_{mult}x_RSI{rsi}_P{int(pct*100)}_B{int(bear*100)}_CD{cd}",
                                    category="Hybrid",
                                    description=f"Hybrid {mult}x optimized",
                                    dca_multiplier=mult,
                                    signal_indicator=ind,
                                    signal_threshold=thresh,
                                    cooldown_months=cd,
                                    use_regime_filter=True,
                                    bear_multiplier_reduction=bear,
                                    enable_rebalancing=True,
                                    rebalance_rsi_trigger=rsi,
                                    rebalance_pct=pct
                                ))
    
    return strategies


def get_all_strategies(max_count: int = 1000) -> List[StrategyConfig]:
    """Retourne les 1000 stratégies du catalogue."""
    all_strategies = []
    
    all_strategies.extend(generate_conservative_strategies())
    all_strategies.extend(generate_moderate_strategies())
    all_strategies.extend(generate_aggressive_strategies())
    all_strategies.extend(generate_adaptive_strategies())
    all_strategies.extend(generate_risk_managed_strategies())
    all_strategies.extend(generate_momentum_strategies())
    all_strategies.extend(generate_volatility_strategies())
    all_strategies.extend(generate_hybrid_optimal_strategies())
    
    # Trim to max_count
    if len(all_strategies) > max_count:
        all_strategies = all_strategies[:max_count]
    
    return all_strategies


def get_strategies_by_category(category: str) -> List[StrategyConfig]:
    return [s for s in get_all_strategies() if s.category == category]


def get_strategy_by_name(name: str) -> StrategyConfig:
    for s in get_all_strategies():
        if s.name == name:
            return s
    raise ValueError(f"Stratégie non trouvée: {name}")


def print_catalog_summary():
    strategies = get_all_strategies()
    
    categories = {}
    for s in strategies:
        categories[s.category] = categories.get(s.category, 0) + 1
    
    print("=" * 60)
    print("📊 CATALOGUE DES 1000 STRATÉGIES DCA AVANCÉES")
    print("=" * 60)
    print(f"\nTotal: {len(strategies)} stratégies\n")
    print("Par catégorie:")
    for cat, count in sorted(categories.items()):
        print(f"  • {cat}: {count}")
    print()


if __name__ == "__main__":
    print_catalog_summary()
