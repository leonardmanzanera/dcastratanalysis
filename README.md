# Moteur d'Optimisation de Stratégies DCA Hybrides

## Description

Moteur de backtesting vectorisé haute fréquence pour stratégies d'investissement DCA (Dollar Cost Averaging). Ce projet permet de générer, simuler et comparer 100+ variations de stratégies pour identifier la configuration optimale sur différents indices mondiaux.

## Fonctionnalités

- **Backtesting Vectorisé** : Simulation haute performance avec VectorBT et Numba
- **Optimisation Bayésienne** : Exploration intelligente de l'espace des hyperparamètres avec Optuna
- **Tiered Cash System** : Gestion avancée du cash (Flux Mensuel + War Chest)
- **Market Regime Filter** : Adaptation automatique selon les conditions de marché (Bull/Bear)
- **Conditional Rebalancing** : Prise de profit optionnelle basée sur RSI
- **Cool-down Mechanism** : Pause DCA après achats leviers

## Installation

```bash
# Cloner le repository
git clone git@github.com:leonardmanzanera/dcastratanalysis.git
cd dcastratanalysis

# Installer les dépendances
pip install -e .
```

## Structure du Projet

```
dcastratanalysis/
├── src/
│   ├── data_loader.py      # Ingestion données + indicateurs
│   ├── strategy_core.py    # Moteur Numba
│   ├── optuna_search.py    # Optimisation
│   └── metrics.py          # Calmar, Sortino, Drawdown
├── notebooks/
│   └── analysis.ipynb      # Visualisation
├── data/cache/             # Cache données OHLCV
├── results/                # Résultats Optuna
└── tests/                  # Tests unitaires
```

## Utilisation

### 1. Lancer l'optimisation

```bash
python src/optuna_search.py --n-trials 100 --ticker SPY
```

### 2. Analyser les résultats

Ouvrir le notebook `notebooks/analysis.ipynb` pour visualiser :
- Courbes d'équité des meilleures stratégies
- Comparaison Rebalancing vs Accumulation
- Heatmap des hyperparamètres

## Hyperparamètres Optimisés

| Paramètre | Type | Plage |
|-----------|------|-------|
| `enable_rebalancing` | Bool | [True, False] |
| `rebalance_rsi_trigger` | Int | [70, 90] |
| `rebalance_pct` | Float | [0.1, 0.5] |
| `use_regime_filter` | Bool | [True, False] |
| `dca_multiplier` | Int | [2, 12] |
| `signal_indicator` | Categorical | ['RSI', 'MACD', 'Bollinger'] |
| `signal_threshold` | Int | [20, 40] |

## Licence

MIT
