"""
Dashboard Results - Visualisation interactive des résultats de benchmark.

Ce dashboard Streamlit permet de:
1. Visualiser le Top 10 des stratégies par métrique
2. Comparer plusieurs stratégies côte à côte
3. Analyser les performances par catégorie
4. Filtrer et explorer les 250 stratégies
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from strategies_catalog import get_all_strategies, INDICATOR_NAMES
from data_loader import prepare_simulation_data
from strategy_core import simulate_dca_strategy
from metrics import compute_all_metrics, drawdown_series


# Configuration
RESULTS_DIR = Path(__file__).parent.parent / "results"
BENCHMARK_FILE = RESULTS_DIR / "benchmark_results.parquet"


# Page config
st.set_page_config(
    page_title="DCA Strategies Benchmark",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00c6ff 0%, #0072ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .metric-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #0f3460;
        text-align: center;
    }
    .category-badge {
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .stDataFrame {
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_benchmark_results():
    """Charge les résultats du benchmark."""
    if BENCHMARK_FILE.exists():
        return pd.read_parquet(BENCHMARK_FILE)
    return None


def run_benchmark_if_needed():
    """Exécute le benchmark si les résultats n'existent pas."""
    if not BENCHMARK_FILE.exists():
        st.warning("⚠️ Résultats de benchmark non trouvés. Exécutez d'abord:")
        st.code("python3 src/benchmark_runner.py", language="bash")
        
        if st.button("🚀 Lancer le Benchmark (peut prendre quelques minutes)"):
            with st.spinner("Exécution du benchmark..."):
                import benchmark_runner
                results = benchmark_runner.run_benchmark(max_strategies=50)  # Quick run
                benchmark_runner.save_results(results)
                st.success("✅ Benchmark terminé!")
                st.rerun()
        return None
    return load_benchmark_results()


def render_overview(df: pd.DataFrame):
    """Affiche la vue d'ensemble."""
    st.markdown("## 📊 Vue d'Ensemble")
    
    # Filter to successful results
    df_success = df[df["status"] == "success"].copy()
    
    # Period selector
    periods = sorted(df_success["period"].unique())
    selected_period = st.selectbox("📅 Période", periods, index=len(periods)-1)
    
    df_period = df_success[df_success["period"] == selected_period]
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📈 Stratégies Testées",
            len(df_period),
            f"{len(df_period['category'].unique())} catégories"
        )
    
    with col2:
        best_calmar = df_period["calmar_ratio"].max()
        st.metric("🏆 Meilleur Calmar", f"{best_calmar:.2f}")
    
    with col3:
        avg_cagr = df_period["cagr"].mean()
        st.metric("📊 CAGR Moyen", f"{avg_cagr:.1f}%")
    
    with col4:
        avg_dd = df_period["max_drawdown"].mean()
        st.metric("📉 DD Moyen", f"{avg_dd:.1f}%")
    
    st.divider()
    
    # Top 10 strategies
    st.markdown("### 🏆 Top 10 Stratégies (par Calmar Ratio)")
    
    top_10 = df_period.nlargest(10, "calmar_ratio")[[
        "strategy_name", "category", "calmar_ratio", "cagr", 
        "max_drawdown", "final_value", "sharpe_ratio"
    ]].copy()
    
    top_10.columns = ["Stratégie", "Catégorie", "Calmar", "CAGR %", "Max DD %", "Valeur Finale", "Sharpe"]
    top_10["Valeur Finale"] = top_10["Valeur Finale"].apply(lambda x: f"€{x:,.0f}")
    top_10["Calmar"] = top_10["Calmar"].round(2)
    top_10["CAGR %"] = top_10["CAGR %"].round(1)
    top_10["Max DD %"] = top_10["Max DD %"].round(1)
    top_10["Sharpe"] = top_10["Sharpe"].round(2)
    
    st.dataframe(top_10, use_container_width=True, hide_index=True)
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution of Calmar Ratios
        fig = px.histogram(
            df_period, x="calmar_ratio", nbins=30,
            color="category",
            title="Distribution des Calmar Ratios",
            labels={"calmar_ratio": "Calmar Ratio", "count": "Nombre"}
        )
        fig.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Average metrics by category
        cat_summary = df_period.groupby("category").agg({
            "calmar_ratio": "mean",
            "cagr": "mean",
            "max_drawdown": "mean"
        }).round(2).reset_index()
        
        fig = px.bar(
            cat_summary, x="category", y="calmar_ratio",
            color="category",
            title="Calmar Ratio Moyen par Catégorie",
            labels={"calmar_ratio": "Calmar Ratio", "category": "Catégorie"}
        )
        fig.update_layout(template="plotly_dark", height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


def render_comparison(df: pd.DataFrame):
    """Permet de comparer plusieurs stratégies."""
    st.markdown("## 🔄 Comparaison de Stratégies")
    
    df_success = df[df["status"] == "success"].copy()
    
    # Period selector
    periods = sorted(df_success["period"].unique())
    selected_period = st.selectbox("📅 Période", periods, index=len(periods)-1, key="compare_period")
    
    df_period = df_success[df_success["period"] == selected_period]
    
    # Strategy selector
    all_strategies = sorted(df_period["strategy_name"].unique())
    
    # Pre-select top 5
    top_5 = df_period.nlargest(5, "calmar_ratio")["strategy_name"].tolist()
    
    selected_strategies = st.multiselect(
        "🎯 Sélectionnez les stratégies à comparer",
        options=all_strategies,
        default=top_5[:3]
    )
    
    if not selected_strategies:
        st.info("👆 Sélectionnez au moins une stratégie")
        return
    
    # Comparison table
    df_compare = df_period[df_period["strategy_name"].isin(selected_strategies)][[
        "strategy_name", "category", "final_value", "profit_pct", "cagr",
        "max_drawdown", "calmar_ratio", "sortino_ratio", "sharpe_ratio",
        "n_buys", "n_sells", "dca_multiplier", "signal_indicator"
    ]].copy()
    
    # Format
    df_compare["signal_indicator"] = df_compare["signal_indicator"].map(INDICATOR_NAMES)
    df_compare["final_value"] = df_compare["final_value"].apply(lambda x: f"€{x:,.0f}")
    
    df_compare.columns = [
        "Stratégie", "Catégorie", "Valeur Finale", "Profit %", "CAGR %",
        "Max DD %", "Calmar", "Sortino", "Sharpe", "Achats", "Ventes",
        "Multiplicateur", "Signal"
    ]
    
    st.dataframe(df_compare, use_container_width=True, hide_index=True)
    
    # Radar chart comparison
    if len(selected_strategies) >= 2:
        st.markdown("### 📊 Comparaison Radar")
        
        metrics_for_radar = ["CAGR %", "Calmar", "Sortino", "Sharpe"]
        
        fig = go.Figure()
        
        for strategy in selected_strategies:
            row = df_compare[df_compare["Stratégie"] == strategy].iloc[0]
            values = [row["CAGR %"], row["Calmar"], row["Sortino"], row["Sharpe"]]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics_for_radar,
                fill='toself',
                name=strategy[:30]
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            template="plotly_dark",
            height=500,
            title="Comparaison Multi-Métrique"
        )
        
        st.plotly_chart(fig, use_container_width=True)


def render_category_analysis(df: pd.DataFrame):
    """Analyse par catégorie."""
    st.markdown("## 📁 Analyse par Catégorie")
    
    df_success = df[df["status"] == "success"].copy()
    
    # Period selector
    periods = sorted(df_success["period"].unique())
    selected_period = st.selectbox("📅 Période", periods, index=len(periods)-1, key="cat_period")
    
    df_period = df_success[df_success["period"] == selected_period]
    
    # Category summary
    cat_summary = df_period.groupby("category").agg({
        "strategy_name": "count",
        "calmar_ratio": ["mean", "max", "std"],
        "cagr": ["mean", "max"],
        "max_drawdown": ["mean", "min"],
        "final_value": "mean"
    }).round(2)
    
    cat_summary.columns = [
        "Nb Stratégies", "Calmar Moy", "Calmar Max", "Calmar Std",
        "CAGR Moy", "CAGR Max", "DD Moy", "DD Min", "Valeur Moy"
    ]
    
    st.dataframe(cat_summary, use_container_width=True)
    
    # Box plots
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.box(
            df_period, x="category", y="calmar_ratio",
            color="category",
            title="Distribution Calmar par Catégorie"
        )
        fig.update_layout(template="plotly_dark", height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(
            df_period, x="category", y="cagr",
            color="category",
            title="Distribution CAGR par Catégorie"
        )
        fig.update_layout(template="plotly_dark", height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Scatter: Risk vs Return
    st.markdown("### ⚖️ Risque vs Rendement")
    
    fig = px.scatter(
        df_period, x="max_drawdown", y="cagr",
        color="category", size="calmar_ratio",
        hover_name="strategy_name",
        title="CAGR vs Max Drawdown (taille = Calmar Ratio)",
        labels={"max_drawdown": "Max Drawdown %", "cagr": "CAGR %"}
    )
    fig.update_layout(template="plotly_dark", height=500)
    st.plotly_chart(fig, use_container_width=True)


def render_strategy_explorer(df: pd.DataFrame):
    """Explorateur de stratégies avec filtres."""
    st.markdown("## 🔍 Explorateur de Stratégies")
    
    df_success = df[df["status"] == "success"].copy()
    
    # Filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        periods = sorted(df_success["period"].unique())
        selected_period = st.selectbox("📅 Période", periods, index=len(periods)-1, key="exp_period")
    
    with col2:
        categories = ["Toutes"] + sorted(df_success["category"].unique())
        selected_category = st.selectbox("📁 Catégorie", categories)
    
    with col3:
        min_calmar = st.number_input("Min Calmar", value=0.0, step=0.5)
    
    with col4:
        max_dd = st.number_input("Max Drawdown %", value=100.0, step=5.0)
    
    # Apply filters
    df_filtered = df_success[df_success["period"] == selected_period]
    
    if selected_category != "Toutes":
        df_filtered = df_filtered[df_filtered["category"] == selected_category]
    
    df_filtered = df_filtered[df_filtered["calmar_ratio"] >= min_calmar]
    df_filtered = df_filtered[df_filtered["max_drawdown"] <= max_dd]
    
    # Sort options
    sort_by = st.selectbox(
        "Trier par",
        ["calmar_ratio", "cagr", "sharpe_ratio", "sortino_ratio", "final_value"],
        format_func=lambda x: {
            "calmar_ratio": "Calmar Ratio",
            "cagr": "CAGR",
            "sharpe_ratio": "Sharpe Ratio",
            "sortino_ratio": "Sortino Ratio",
            "final_value": "Valeur Finale"
        }.get(x, x)
    )
    
    df_sorted = df_filtered.sort_values(sort_by, ascending=False)
    
    st.markdown(f"**{len(df_sorted)} stratégies correspondent aux critères**")
    
    # Display
    display_cols = [
        "strategy_name", "category", "calmar_ratio", "cagr", "max_drawdown",
        "sharpe_ratio", "final_value", "dca_multiplier", "enable_rebalancing"
    ]
    
    df_display = df_sorted[display_cols].copy()
    df_display.columns = [
        "Stratégie", "Catégorie", "Calmar", "CAGR %", "Max DD %",
        "Sharpe", "Valeur Finale", "Multiplicateur", "Rebalancing"
    ]
    
    df_display["Valeur Finale"] = df_display["Valeur Finale"].apply(lambda x: f"€{x:,.0f}")
    df_display["Calmar"] = df_display["Calmar"].round(2)
    df_display["CAGR %"] = df_display["CAGR %"].round(1)
    df_display["Max DD %"] = df_display["Max DD %"].round(1)
    df_display["Sharpe"] = df_display["Sharpe"].round(2)
    
    st.dataframe(df_display, use_container_width=True, hide_index=True, height=500)


def render_period_comparison(df: pd.DataFrame):
    """Compare les performances sur différentes périodes."""
    st.markdown("## 📅 Comparaison par Période")
    
    df_success = df[df["status"] == "success"].copy()
    
    # Strategy selector
    all_strategies = sorted(df_success["strategy_name"].unique())
    
    # Pre-select best overall
    best_avg = df_success.groupby("strategy_name")["calmar_ratio"].mean().nlargest(3).index.tolist()
    
    selected_strategies = st.multiselect(
        "🎯 Sélectionnez les stratégies",
        options=all_strategies,
        default=best_avg
    )
    
    if not selected_strategies:
        st.info("👆 Sélectionnez au moins une stratégie")
        return
    
    df_selected = df_success[df_success["strategy_name"].isin(selected_strategies)]
    
    # Pivot table
    pivot = df_selected.pivot_table(
        index="strategy_name",
        columns="period",
        values=["calmar_ratio", "cagr", "max_drawdown"],
        aggfunc="first"
    ).round(2)
    
    st.markdown("### Performance par Période")
    st.dataframe(pivot, use_container_width=True)
    
    # Line chart
    fig = px.line(
        df_selected, x="period", y="calmar_ratio",
        color="strategy_name",
        markers=True,
        title="Évolution du Calmar Ratio par Période"
    )
    fig.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig, use_container_width=True)


def render_equity_curves(df: pd.DataFrame):
    """Visualise les courbes d'équité comparées au DCA passif."""
    st.markdown("## 📈 Courbes d'Évolution du Portefeuille")
    st.markdown("*Comparez l'évolution de vos stratégies sélectionnées vs le DCA passif*")
    
    df_success = df[df["status"] == "success"].copy()
    
    # Period selector
    col1, col2 = st.columns([1, 3])
    
    with col1:
        periods = sorted(df_success["period"].unique())
        selected_period = st.selectbox("📅 Période", periods, index=len(periods)-1, key="equity_period")
    
    df_period = df_success[df_success["period"] == selected_period]
    
    # Get Pure_DCA as baseline
    baseline_name = "Pure_DCA"
    baseline_row = df_period[df_period["strategy_name"] == baseline_name]
    
    if len(baseline_row) == 0:
        # Fallback: use first conservative strategy
        baseline_row = df_period[df_period["category"] == "Conservative"].head(1)
        baseline_name = baseline_row["strategy_name"].values[0] if len(baseline_row) > 0 else None
    
    # Strategy selector
    all_strategies = sorted(df_period["strategy_name"].unique())
    top_5 = df_period.nlargest(5, "calmar_ratio")["strategy_name"].tolist()
    
    # Ensure baseline is in defaults
    defaults = [s for s in top_5[:3] if s != baseline_name]
    if baseline_name:
        defaults = [baseline_name] + defaults[:2]
    
    selected_strategies = st.multiselect(
        "🎯 Sélectionnez les stratégies à visualiser (le DCA passif sera toujours affiché en référence)",
        options=all_strategies,
        default=defaults
    )
    
    if not selected_strategies:
        st.info("👆 Sélectionnez au moins une stratégie")
        return
    
    # Ensure Pure_DCA is always included
    if baseline_name and baseline_name not in selected_strategies:
        selected_strategies = [baseline_name] + selected_strategies
    
    # Load data for simulation
    period_map = {"5Y": ("2019-01-01", "2024-01-01"), 
                  "7Y": ("2017-01-01", "2024-01-01"),
                  "10Y": ("2014-01-01", "2024-01-01")}
    
    start_date, end_date = period_map.get(selected_period, ("2019-01-01", "2024-01-01"))
    
    with st.spinner("Chargement des données et simulation..."):
        try:
            prices, indicators, is_bull, dates = prepare_simulation_data("SPY", start_date, end_date)
        except Exception as e:
            st.error(f"Erreur: {e}")
            return
    
    # Get strategy configs and run simulations
    strategies_data = {}
    
    for strat_name in selected_strategies:
        try:
            strat = get_strategy_by_name(strat_name)
            
            results = simulate_dca_strategy(
                prices, indicators, is_bull,
                strat.monthly_amount, strat.initial_war_chest,
                strat.enable_rebalancing, strat.rebalance_rsi_trigger,
                strat.rebalance_profit_trigger, strat.rebalance_pct,
                strat.use_regime_filter, strat.bear_multiplier_reduction,
                strat.dca_multiplier, strat.signal_indicator,
                strat.signal_threshold, strat.cooldown_months
            )
            
            strategies_data[strat_name] = {
                "equity": results[0],
                "invested": results[4],
                "category": strat.category
            }
        except Exception as e:
            st.warning(f"Erreur pour {strat_name}: {e}")
    
    if not strategies_data:
        st.error("Aucune stratégie n'a pu être simulée")
        return
    
    # Create equity curves chart
    st.markdown("### 📊 Évolution de la Valorisation")
    
    fig = go.Figure()
    
    colors = {
        "Conservative": "#94a3b8",
        "Moderate": "#3b82f6",
        "Aggressive": "#ef4444",
        "Adaptive": "#10b981",
        "RiskManaged": "#f59e0b",
        "Momentum": "#8b5cf6",
        "Volatility": "#ec4899",
        "Hybrid": "#06b6d4",
        "Mixed": "#6b7280"
    }
    
    for strat_name, data in strategies_data.items():
        is_baseline = strat_name == baseline_name
        color = colors.get(data["category"], "#ffffff")
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=data["equity"],
            name=strat_name[:40],
            line=dict(
                color="#94a3b8" if is_baseline else color,
                width=3 if is_baseline else 2,
                dash="dot" if is_baseline else "solid"
            ),
            opacity=0.7 if is_baseline else 1.0
        ))
    
    # Add invested amount line
    invested_cumsum = np.zeros(len(prices))
    monthly_amount = 500.0
    for i in range(len(prices)):
        if i % 21 == 0:  # Approximation mensuelle
            invested_cumsum[i:] += monthly_amount
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=invested_cumsum,
        name="Capital Investi",
        line=dict(color="#64748b", width=1, dash="dash"),
        opacity=0.5
    ))
    
    fig.update_layout(
        template="plotly_dark",
        height=600,
        title=f"Évolution du Portefeuille ({selected_period})",
        xaxis_title="Date",
        yaxis_title="Valeur (€)",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.5)"),
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Drawdown comparison
    st.markdown("### 📉 Comparaison des Drawdowns")
    
    fig_dd = go.Figure()
    
    for strat_name, data in strategies_data.items():
        dd = drawdown_series(data["equity"]) * 100
        is_baseline = strat_name == baseline_name
        color = colors.get(data["category"], "#ffffff")
        
        fig_dd.add_trace(go.Scatter(
            x=dates,
            y=-dd,
            name=strat_name[:40],
            fill="tozeroy" if is_baseline else None,
            line=dict(
                color="#94a3b8" if is_baseline else color,
                width=2 if is_baseline else 1
            ),
            opacity=0.3 if is_baseline else 0.8
        ))
    
    fig_dd.update_layout(
        template="plotly_dark",
        height=400,
        title="Drawdown (perte depuis le plus haut)",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode="x unified"
    )
    
    st.plotly_chart(fig_dd, use_container_width=True)
    
    # Performance table
    st.markdown("### 📋 Tableau Comparatif")
    
    comparison_data = []
    baseline_final = strategies_data.get(baseline_name, {}).get("equity", [0])[-1] if baseline_name else 0
    
    for strat_name, data in strategies_data.items():
        equity = data["equity"]
        invested = data["invested"]
        final = equity[-1] if len(equity) > 0 else 0
        
        # Get row from benchmark results
        row = df_period[df_period["strategy_name"] == strat_name].iloc[0] if len(df_period[df_period["strategy_name"] == strat_name]) > 0 else None
        
        comparison_data.append({
            "Stratégie": strat_name,
            "Catégorie": data["category"],
            "Valeur Finale": f"€{final:,.0f}",
            "vs DCA Passif": f"{((final - baseline_final) / baseline_final * 100):+.1f}%" if baseline_final > 0 and strat_name != baseline_name else "Baseline",
            "Max Drawdown": f"{row['max_drawdown']:.1f}%" if row is not None else "N/A",
            "Calmar Ratio": f"{row['calmar_ratio']:.2f}" if row is not None else "N/A",
            "CAGR": f"{row['cagr']:.1f}%" if row is not None else "N/A"
        })
    
    df_table = pd.DataFrame(comparison_data)
    st.dataframe(df_table, use_container_width=True, hide_index=True)
    
    # Winner announcement
    if len(strategies_data) > 1 and baseline_name:
        best_strat = max(
            [(k, v["equity"][-1]) for k, v in strategies_data.items() if k != baseline_name],
            key=lambda x: x[1]
        )
        
        if best_strat[1] > baseline_final:
            gain = best_strat[1] - baseline_final
            st.success(f"🏆 **Gagnant**: {best_strat[0]} (+€{gain:,.0f} vs DCA passif)")
        else:
            st.info(f"📊 Le DCA passif reste le plus performant en valeur absolue sur cette période")


def main():
    """Point d'entrée principal."""
    st.markdown('<h1 class="main-header">📊 DCA Strategies Benchmark</h1>', unsafe_allow_html=True)
    st.markdown("*Analyse comparative de 1000 stratégies DCA avancées sur SPY*")
    
    # Load results
    df = run_benchmark_if_needed()
    
    if df is None:
        return
    
    # Sidebar navigation
    with st.sidebar:
        st.header("📑 Navigation")
        page = st.radio(
            "Page",
            ["Vue d'Ensemble", "📈 Courbes d'Équité", "Comparaison", "Par Catégorie", "Explorateur", "Par Période"],
            label_visibility="collapsed"
        )
        
        st.divider()
        
        # Quick stats
        df_success = df[df["status"] == "success"]
        st.metric("Total Résultats", len(df_success))
        st.metric("Catégories", len(df_success["category"].unique()))
        st.metric("Périodes", len(df_success["period"].unique()))
    
    # Render selected page
    if page == "Vue d'Ensemble":
        render_overview(df)
    elif page == "📈 Courbes d'Équité":
        render_equity_curves(df)
    elif page == "Comparaison":
        render_comparison(df)
    elif page == "Par Catégorie":
        render_category_analysis(df)
    elif page == "Explorateur":
        render_strategy_explorer(df)
    elif page == "Par Période":
        render_period_comparison(df)


if __name__ == "__main__":
    main()
