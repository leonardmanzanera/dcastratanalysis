"""
Streamlit Dashboard - Interface interactive pour l'optimisation DCA.

Fonctionnalités:
- Sélection du ticker et période
- Configuration des hyperparamètres
- Visualisation des courbes d'équité
- Comparaison Rebalancing vs Accumulation
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_data, compute_indicators, prepare_simulation_data, TICKER_MAPPING
from strategy_core import simulate_dca_strategy, compare_strategies, DCAStrategyParams
from metrics import compute_all_metrics, max_drawdown, drawdown_series


# Page config
st.set_page_config(
    page_title="DCA Strategy Optimizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #3d3d5c;
    }
    .winner-badge {
        background: linear-gradient(90deg, #00c853 0%, #69f0ae 100%);
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # Header
    st.markdown('<h1 class="main-header">📈 DCA Strategy Optimizer</h1>', unsafe_allow_html=True)
    st.markdown("*Moteur d'optimisation de stratégies DCA hybrides avec backtesting vectorisé*")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Ticker selection
        ticker_name = st.selectbox(
            "📊 Indice",
            options=list(TICKER_MAPPING.keys()),
            index=0
        )
        ticker = TICKER_MAPPING[ticker_name]
        
        # Date range
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "📅 Début",
                value=datetime(2018, 1, 1),
                min_value=datetime(2000, 1, 1)
            )
        with col2:
            end_date = st.date_input(
                "📅 Fin",
                value=datetime(2024, 1, 1),
                max_value=datetime.now()
            )
        
        st.divider()
        
        # DCA Parameters
        st.subheader("💰 Paramètres DCA")
        monthly_amount = st.slider(
            "Montant mensuel (€)",
            min_value=100, max_value=2000, value=500, step=50
        )
        
        st.divider()
        
        # Strategy Parameters
        st.subheader("🎯 Stratégie")
        
        enable_rebalancing = st.toggle("Activer Rebalancing", value=False)
        
        if enable_rebalancing:
            rebalance_rsi = st.slider("RSI Trigger", 70, 95, 80)
            rebalance_pct = st.slider("% à Vendre", 0.1, 0.5, 0.2, 0.05)
        else:
            rebalance_rsi = 80
            rebalance_pct = 0.2
        
        use_regime = st.toggle("Filtre de Régime (SMA200)", value=True)
        
        st.divider()
        
        # Leverage Parameters
        st.subheader("⚡ Levier")
        
        signal_type = st.selectbox(
            "Signal d'achat",
            options=["RSI", "MACD", "Bollinger"],
            index=0
        )
        signal_indicator = {"RSI": 0, "MACD": 1, "Bollinger": 2}[signal_type]
        
        signal_threshold = st.slider("Seuil de signal", 20, 40, 30)
        dca_multiplier = st.slider("Multiplicateur", 2, 12, 5)
        cooldown_months = st.slider("Cooldown (mois)", 1, 6, 3)
        
        st.divider()
        
        run_button = st.button("🚀 Lancer le Backtest", type="primary", use_container_width=True)
    
    # Main content
    if run_button:
        with st.spinner("Chargement des données..."):
            try:
                prices, indicators, is_bull, dates = prepare_simulation_data(
                    ticker,
                    start_date.strftime("%Y-%m-%d"),
                    end_date.strftime("%Y-%m-%d")
                )
                
                if len(prices) == 0:
                    st.error("Pas assez de données après calcul des indicateurs. Essayez une période plus longue.")
                    return
                    
            except Exception as e:
                st.error(f"Erreur lors du chargement: {e}")
                return
        
        with st.spinner("Simulation en cours..."):
            # Run custom strategy
            results_custom = simulate_dca_strategy(
                prices, indicators, is_bull,
                float(monthly_amount), 0.0,
                enable_rebalancing, rebalance_rsi, 0.5, rebalance_pct,
                use_regime, 0.5,
                dca_multiplier, signal_indicator, signal_threshold,
                cooldown_months
            )
            
            # Run baseline (simple DCA)
            results_baseline = simulate_dca_strategy(
                prices, indicators, is_bull,
                float(monthly_amount), 0.0,
                False, 80, 0.5, 0.2,
                False, 0.5,
                1, 0, 30, 0  # No leverage, no cooldown
            )
        
        # Extract results
        equity_custom = results_custom[0]
        equity_baseline = results_baseline[0]
        total_invested = results_custom[4]
        
        # Calculate metrics
        metrics_custom = compute_all_metrics(equity_custom, total_invested)
        metrics_baseline = compute_all_metrics(equity_baseline, total_invested)
        
        # Display results
        st.header("📊 Résultats du Backtest")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "💰 Valeur Finale",
                f"€{metrics_custom['final_value']:,.0f}",
                f"{metrics_custom['profit_pct']:.1f}%"
            )
        
        with col2:
            st.metric(
                "📈 CAGR",
                f"{metrics_custom['cagr']:.1f}%"
            )
        
        with col3:
            st.metric(
                "📉 Max Drawdown",
                f"{metrics_custom['max_drawdown']:.1f}%"
            )
        
        with col4:
            st.metric(
                "⚖️ Calmar Ratio",
                f"{metrics_custom['calmar_ratio']:.2f}"
            )
        
        st.divider()
        
        # Charts
        tab1, tab2, tab3 = st.tabs(["📈 Courbes d'Équité", "📉 Drawdown", "📊 Comparaison"])
        
        with tab1:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=dates, y=equity_custom,
                name="Stratégie Optimisée",
                line=dict(color="#667eea", width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=dates, y=equity_baseline,
                name="DCA Simple",
                line=dict(color="#94a3b8", width=1, dash="dash")
            ))
            
            # Invested amount line
            invested_line = np.cumsum(
                [monthly_amount if i % 21 == 0 else 0 for i in range(len(prices))]
            ) + monthly_amount
            
            fig.add_trace(go.Scatter(
                x=dates, y=invested_line,
                name="Total Investi",
                line=dict(color="#64748b", width=1, dash="dot")
            ))
            
            fig.update_layout(
                title="Évolution du Portefeuille",
                xaxis_title="Date",
                yaxis_title="Valeur (€)",
                template="plotly_dark",
                height=500,
                legend=dict(x=0.01, y=0.99)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            dd_custom = drawdown_series(equity_custom) * 100
            dd_baseline = drawdown_series(equity_baseline) * 100
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=dates, y=-dd_custom,
                name="Stratégie Optimisée",
                fill="tozeroy",
                line=dict(color="#667eea", width=1)
            ))
            
            fig.add_trace(go.Scatter(
                x=dates, y=-dd_baseline,
                name="DCA Simple",
                line=dict(color="#94a3b8", width=1, dash="dash")
            ))
            
            fig.update_layout(
                title="Drawdown",
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                template="plotly_dark",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Comparison table
            comparison_data = {
                "Métrique": ["Valeur Finale", "Profit", "CAGR", "Max DD", "Calmar", "Sortino"],
                "Stratégie Optimisée": [
                    f"€{metrics_custom['final_value']:,.0f}",
                    f"€{metrics_custom['profit']:,.0f}",
                    f"{metrics_custom['cagr']:.1f}%",
                    f"{metrics_custom['max_drawdown']:.1f}%",
                    f"{metrics_custom['calmar_ratio']:.2f}",
                    f"{metrics_custom['sortino_ratio']:.2f}"
                ],
                "DCA Simple": [
                    f"€{metrics_baseline['final_value']:,.0f}",
                    f"€{metrics_baseline['profit']:,.0f}",
                    f"{metrics_baseline['cagr']:.1f}%",
                    f"{metrics_baseline['max_drawdown']:.1f}%",
                    f"{metrics_baseline['calmar_ratio']:.2f}",
                    f"{metrics_baseline['sortino_ratio']:.2f}"
                ]
            }
            
            df_comparison = pd.DataFrame(comparison_data)
            
            # Determine winner
            if metrics_custom['calmar_ratio'] > metrics_baseline['calmar_ratio']:
                winner = "🏆 Stratégie Optimisée"
                diff = metrics_custom['final_value'] - metrics_baseline['final_value']
            else:
                winner = "🏆 DCA Simple"
                diff = metrics_baseline['final_value'] - metrics_custom['final_value']
            
            st.success(f"**Gagnant:** {winner} (+€{diff:,.0f})")
            st.dataframe(df_comparison, use_container_width=True, hide_index=True)
        
        # Transaction stats
        st.divider()
        st.subheader("📋 Statistiques des Transactions")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Nombre d'Achats", results_custom[6])
        with col2:
            st.metric("Nombre de Ventes", results_custom[7])
        with col3:
            st.metric("Total Investi", f"€{total_invested:,.0f}")
    
    else:
        # Welcome screen
        st.info("👈 Configurez les paramètres dans la barre latérale et cliquez sur **Lancer le Backtest**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🎯 Fonctionnalités
            
            - **Tiered Cash System**: Gestion séparée du flux mensuel et de la réserve stratégique
            - **Market Regime Filter**: Adaptation automatique Bull/Bear
            - **Conditional Rebalancing**: Prise de profit optionnelle
            - **Cool-down Mechanism**: Pause après achats leviers
            """)
        
        with col2:
            st.markdown("""
            ### 📊 Indicateurs Disponibles
            
            - **RSI**: Relative Strength Index
            - **MACD**: Moving Average Convergence Divergence
            - **Bollinger Bands**: Bandes de volatilité
            - **SMA 200**: Filtre de tendance long terme
            """)


if __name__ == "__main__":
    main()
