"""
Data Loader Module - Ingestion de données OHLCV et calcul des indicateurs techniques.

Ce module gère :
- Téléchargement des données via yfinance
- Cache local pour éviter les appels API répétés
- Calcul de la matrice d'indicateurs (SMA, EMA, RSI, MACD, Bollinger, etc.)
- Détermination du régime de marché (Bull/Bear)
"""

import os
import hashlib
from pathlib import Path
from typing import Optional, Tuple, List
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
import talib


# Configuration du cache
CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _get_cache_path(ticker: str, start: str, end: str) -> Path:
    """Génère un chemin de cache unique basé sur les paramètres."""
    cache_key = hashlib.md5(f"{ticker}_{start}_{end}".encode()).hexdigest()[:12]
    return CACHE_DIR / f"{ticker}_{cache_key}.parquet"


def load_data(
    ticker: str,
    start: str,
    end: str,
    use_cache: bool = True,
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Charge les données OHLCV pour un ticker donné.
    
    Args:
        ticker: Symbole du ticker (ex: 'SPY', 'QQQ', '^FCHI' pour CAC40)
        start: Date de début (format: 'YYYY-MM-DD')
        end: Date de fin (format: 'YYYY-MM-DD')
        use_cache: Utiliser le cache local si disponible
        interval: Intervalle temporel ('1d', '1wk', '1mo')
    
    Returns:
        DataFrame avec colonnes: Open, High, Low, Close, Volume, Adj Close
    """
    cache_path = _get_cache_path(ticker, start, end)
    
    # Vérifier le cache
    if use_cache and cache_path.exists():
        df = pd.read_parquet(cache_path)
        print(f"[Cache] Données chargées depuis {cache_path}")
        return df
    
    # Télécharger depuis yfinance
    print(f"[Download] Téléchargement de {ticker} ({start} → {end})...")
    df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
    
    if df.empty:
        raise ValueError(f"Aucune donnée trouvée pour {ticker}")
    
    # Nettoyer les colonnes multi-index si présentes
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Sauvegarder dans le cache
    if use_cache:
        df.to_parquet(cache_path)
        print(f"[Cache] Données sauvegardées dans {cache_path}")
    
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule la matrice complète d'indicateurs techniques.
    
    Args:
        df: DataFrame avec colonnes OHLCV
    
    Returns:
        DataFrame enrichi avec tous les indicateurs
    """
    df = df.copy()
    close = df["Close"].values.astype(float)
    high = df["High"].values.astype(float)
    low = df["Low"].values.astype(float)
    
    # ==================== TREND INDICATORS ====================
    
    # Simple Moving Averages
    df["SMA_50"] = talib.SMA(close, timeperiod=50)
    df["SMA_200"] = talib.SMA(close, timeperiod=200)
    
    # Exponential Moving Averages
    df["EMA_12"] = talib.EMA(close, timeperiod=12)
    df["EMA_26"] = talib.EMA(close, timeperiod=26)
    df["EMA_50"] = talib.EMA(close, timeperiod=50)
    
    # Ichimoku Cloud (manual calculation - TA-Lib doesn't have it)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9).max()
    period9_low = pd.Series(low).rolling(window=9).min()
    df["ICHI_Tenkan"] = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26).max()
    period26_low = pd.Series(low).rolling(window=26).min()
    df["ICHI_Kijun"] = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, shifted 26 periods forward
    df["ICHI_Senkou_A"] = ((df["ICHI_Tenkan"] + df["ICHI_Kijun"]) / 2).shift(26)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, shifted 26 periods
    period52_high = pd.Series(high).rolling(window=52).max()
    period52_low = pd.Series(low).rolling(window=52).min()
    df["ICHI_Senkou_B"] = ((period52_high + period52_low) / 2).shift(26)
    
    # Chikou Span (Lagging Span): Close price shifted 26 periods back
    df["ICHI_Chikou"] = df["Close"].shift(-26)
    
    # Cloud Signal: Price above the Cloud (bullish)
    senkou_max = df[["ICHI_Senkou_A", "ICHI_Senkou_B"]].max(axis=1)
    senkou_min = df[["ICHI_Senkou_A", "ICHI_Senkou_B"]].min(axis=1)
    df["ICHI_Above_Cloud"] = (df["Close"] > senkou_max).astype(int)
    df["ICHI_Below_Cloud"] = (df["Close"] < senkou_min).astype(int)
    df["ICHI_In_Cloud"] = ((df["Close"] >= senkou_min) & (df["Close"] <= senkou_max)).astype(int)
    
    # ==================== MOMENTUM INDICATORS ====================
    
    # RSI (Relative Strength Index)
    df["RSI_14"] = talib.RSI(close, timeperiod=14)
    
    # MACD (Moving Average Convergence Divergence)
    macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df["MACD"] = macd
    df["MACD_Signal"] = macd_signal
    df["MACD_Hist"] = macd_hist
    
    # Stochastic Oscillator
    slowk, slowd = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
    df["STOCH_K"] = slowk
    df["STOCH_D"] = slowd
    
    # ==================== VOLATILITY INDICATORS ====================
    
    # ATR (Average True Range)
    df["ATR_14"] = talib.ATR(high, low, close, timeperiod=14)
    
    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df["BB_Upper"] = bb_upper
    df["BB_Middle"] = bb_middle
    df["BB_Lower"] = bb_lower
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]
    df["BB_Pct"] = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"])
    
    # ==================== MARKET REGIME ====================
    
    # Position relative à SMA 200
    df["Above_SMA200"] = (df["Close"] > df["SMA_200"]).astype(int)
    
    # Distance à SMA 200 (en %)
    df["Dist_SMA200_Pct"] = ((df["Close"] - df["SMA_200"]) / df["SMA_200"]) * 100
    
    # ==================== DERIVED SIGNALS ====================
    
    # RSI Oversold/Overbought signals
    df["RSI_Oversold"] = (df["RSI_14"] < 30).astype(int)
    df["RSI_Overbought"] = (df["RSI_14"] > 70).astype(int)
    
    # MACD Crossover signals
    df["MACD_Cross_Up"] = ((df["MACD"] > df["MACD_Signal"]) & 
                           (df["MACD"].shift(1) <= df["MACD_Signal"].shift(1))).astype(int)
    df["MACD_Cross_Down"] = ((df["MACD"] < df["MACD_Signal"]) & 
                             (df["MACD"].shift(1) >= df["MACD_Signal"].shift(1))).astype(int)
    
    # Bollinger Band signals
    df["BB_Below_Lower"] = (df["Close"] < df["BB_Lower"]).astype(int)
    df["BB_Above_Upper"] = (df["Close"] > df["BB_Upper"]).astype(int)
    
    return df


def get_market_regime(
    df: pd.DataFrame,
    method: str = "sma200"
) -> np.ndarray:
    """
    Détermine le régime de marché (Bull/Bear) pour chaque période.
    
    Args:
        df: DataFrame avec indicateurs calculés
        method: Méthode de détermination ('sma200', 'sma50_200_cross')
    
    Returns:
        Array booléen: True = Bull Market, False = Bear Market
    """
    if method == "sma200":
        # Bull si prix > SMA 200
        return (df["Close"] > df["SMA_200"]).values
    
    elif method == "sma50_200_cross":
        # Bull si SMA 50 > SMA 200 (Golden Cross)
        return (df["SMA_50"] > df["SMA_200"]).values
    
    else:
        raise ValueError(f"Méthode inconnue: {method}")


def get_weekly_regime(df: pd.DataFrame) -> np.ndarray:
    """
    Calcule le régime de marché basé sur SMA 200 Weekly.
    
    Args:
        df: DataFrame avec données journalières
    
    Returns:
        Array booléen aligné sur les dates du DataFrame original
    """
    # Resample to weekly
    weekly = df.resample("W").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    })
    
    # Calculate weekly SMA 200
    weekly["SMA_200_W"] = talib.SMA(weekly["Close"].values.astype(float), timeperiod=200)
    weekly["Bull_Weekly"] = weekly["Close"] > weekly["SMA_200_W"]
    
    # Forward fill to daily
    regime_weekly = weekly["Bull_Weekly"].reindex(df.index, method="ffill")
    
    return regime_weekly.values


def prepare_simulation_data(
    ticker: str,
    start: str,
    end: str,
    use_cache: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Prépare toutes les données nécessaires pour la simulation Numba.
    
    Args:
        ticker: Symbole du ticker
        start: Date de début
        end: Date de fin
        use_cache: Utiliser le cache
    
    Returns:
        Tuple contenant:
        - prices: Array des prix de clôture
        - indicators: Matrice des indicateurs normalisés
        - is_bull: Array booléen du régime de marché
        - dates: Index des dates
    """
    # Charger les données
    df = load_data(ticker, start, end, use_cache)
    
    # Calculer les indicateurs
    df = compute_indicators(df)
    
    # Supprimer les NaN (début de série avec SMA 200)
    df = df.dropna()
    
    # Extraire les arrays pour Numba
    prices = df["Close"].values.astype(np.float64)
    
    # Matrice d'indicateurs pour les signaux
    # Index: 0=RSI, 1=MACD_Hist, 2=BB_Pct, 3=Stoch_K
    indicators = np.column_stack([
        df["RSI_14"].values,
        df["MACD_Hist"].values,
        df["BB_Pct"].values,
        df["STOCH_K"].values if "STOCH_K" in df.columns else np.zeros(len(df)),
    ]).astype(np.float64)
    
    # Régime de marché
    is_bull = get_market_regime(df, method="sma200")
    
    return prices, indicators, is_bull, df.index


def load_multiple_tickers(
    tickers: List[str],
    start: str,
    end: str,
    use_cache: bool = True
) -> dict:
    """
    Charge les données pour plusieurs tickers.
    
    Args:
        tickers: Liste des symboles
        start: Date de début
        end: Date de fin
        use_cache: Utiliser le cache
    
    Returns:
        Dict mapping ticker → DataFrame
    """
    data = {}
    for ticker in tickers:
        try:
            df = load_data(ticker, start, end, use_cache)
            df = compute_indicators(df)
            data[ticker] = df
            print(f"[OK] {ticker}: {len(df)} lignes")
        except Exception as e:
            print(f"[ERROR] {ticker}: {e}")
    
    return data


# Ticker mappings for common indices
TICKER_MAPPING = {
    "SPY": "SPY",           # S&P 500 ETF
    "QQQ": "QQQ",           # Nasdaq 100 ETF
    "CAC40": "^FCHI",       # CAC 40 (Paris)
    "DAX": "^GDAXI",        # DAX (Frankfurt)
    "FTSE100": "^FTSE",     # FTSE 100 (London)
    "EURUSD": "EURUSD=X",   # EUR/USD
    "BTCUSD": "BTC-USD",    # Bitcoin
    "ETHUSD": "ETH-USD",    # Ethereum
}

# Currency mappings for non-USD indices
CURRENCY_MAPPING = {
    "^FCHI": "EUR",         # CAC40 is in EUR
    "^GDAXI": "EUR",        # DAX is in EUR
    "^FTSE": "GBP",         # FTSE100 is in GBP
    "^N225": "JPY",         # Nikkei is in JPY
}


def get_ticker_symbol(name: str) -> str:
    """Convertit un nom d'indice en symbole yfinance."""
    return TICKER_MAPPING.get(name.upper(), name)


def load_with_currency_conversion(
    ticker: str,
    start: str,
    end: str,
    target_currency: str = "USD",
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Charge les données OHLCV et applique la conversion de devise si nécessaire.
    
    Pour les indices non-USD (CAC40, DAX, FTSE), les prix sont automatiquement
    convertis en USD en utilisant le taux de change approprié.
    
    Args:
        ticker: Symbole du ticker
        start: Date de début
        end: Date de fin
        target_currency: Devise cible (par défaut USD)
        use_cache: Utiliser le cache
    
    Returns:
        DataFrame avec prix convertis en devise cible
    """
    # Charger les données de l'indice
    df = load_data(ticker, start, end, use_cache)
    
    # Vérifier si une conversion est nécessaire
    base_currency = CURRENCY_MAPPING.get(ticker)
    
    if base_currency is None or base_currency == target_currency:
        # Pas de conversion nécessaire
        return df
    
    # Déterminer la paire de devises
    fx_pair = f"{base_currency}{target_currency}=X"
    
    print(f"[FX] Conversion {base_currency} → {target_currency}...")
    
    try:
        # Charger le taux de change
        fx_data = load_data(fx_pair, start, end, use_cache)
        
        # Aligner les dates (forward fill pour les jours sans cotation forex)
        fx_rate = fx_data["Close"].reindex(df.index, method="ffill")
        
        # Convertir les colonnes de prix
        for col in ["Open", "High", "Low", "Close"]:
            if col in df.columns:
                df[col] = df[col] * fx_rate
        
        # Marquer comme converti
        df.attrs["converted_from"] = base_currency
        df.attrs["converted_to"] = target_currency
        
        print(f"[FX] Conversion appliquée. Taux moyen: {fx_rate.mean():.4f}")
        
    except Exception as e:
        print(f"[FX] Erreur de conversion: {e}")
        print(f"[FX] Utilisation des prix en devise d'origine ({base_currency})")
    
    return df


if __name__ == "__main__":
    # Test du module
    print("=== Test Data Loader ===\n")
    
    # Charger SPY
    df = load_data("SPY", "2020-01-01", "2024-01-01")
    print(f"\nDonnées brutes: {df.shape}")
    print(df.head())
    
    # Calculer les indicateurs
    df = compute_indicators(df)
    print(f"\nAvec indicateurs: {df.shape}")
    print(f"Colonnes: {list(df.columns)}")
    
    # Régime de marché
    regime = get_market_regime(df)
    bull_pct = regime.sum() / len(regime) * 100
    print(f"\n% Bull Market: {bull_pct:.1f}%")
    
    # Préparer pour simulation
    prices, indicators, is_bull, dates = prepare_simulation_data(
        "SPY", "2020-01-01", "2024-01-01"
    )
    print(f"\nDonnées simulation:")
    print(f"  - Prices: {prices.shape}")
    print(f"  - Indicators: {indicators.shape}")
    print(f"  - Is Bull: {is_bull.shape}")
