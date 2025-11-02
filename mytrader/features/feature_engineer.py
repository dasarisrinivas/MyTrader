"""Feature engineering for technical indicators and sentiment fusion."""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = avg_loss.replace(0, np.nan)
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.replace([np.inf, -np.inf], np.nan).astype(float).fillna(50.0)
    return rsi


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average Directional Index."""
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    
    plus_dm = high - prev_high
    minus_dm = prev_low - low
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    plus_dm[(plus_dm < minus_dm)] = 0
    minus_dm[(minus_dm < plus_dm)] = 0
    
    atr = _atr(high, low, close, period)
    
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
    
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    
    return adx.fillna(0)


def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14, smooth_k: int = 3) -> tuple[pd.Series, pd.Series]:
    """Calculate Stochastic Oscillator."""
    lowest_low = low.rolling(window=period).min()
    highest_high = high.rolling(window=period).max()
    
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    k = k.rolling(window=smooth_k).mean()  # %K
    d = k.rolling(window=3).mean()  # %D
    
    return k.fillna(50), d.fillna(50)


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add comprehensive technical indicators to price data."""
    enriched = df.copy()
    close = enriched["close"]
    high = enriched["high"]
    low = enriched["low"]
    volume = enriched["volume"]

    # RSI
    enriched["RSI_14"] = _rsi(close, 14)

    # MACD
    ema_fast = _ema(close, 12)
    ema_slow = _ema(close, 26)
    macd = ema_fast - ema_slow
    signal = _ema(macd, 9)
    enriched["MACD_12_26_9"] = macd
    enriched["MACDsignal_12_26_9"] = signal
    enriched["MACDhist_12_26_9"] = macd - signal

    # EMAs
    enriched["EMA_21"] = _ema(close, 21)
    enriched["EMA_50"] = _ema(close, 50)
    enriched["EMA_200"] = _ema(close, 200)

    # Bollinger Bands
    rolling_mean = close.rolling(window=20).mean()
    rolling_std = close.rolling(window=20).std(ddof=0)
    enriched["BB_mid_20_2"] = rolling_mean
    enriched["BB_upper_20_2"] = rolling_mean + 2 * rolling_std
    enriched["BB_lower_20_2"] = rolling_mean - 2 * rolling_std
    
    # Bollinger %B
    bb_range = enriched["BB_upper_20_2"] - enriched["BB_lower_20_2"]
    enriched["BB_percent"] = ((close - enriched["BB_lower_20_2"]) / bb_range).fillna(0.5)

    # VWAP (cumulative)
    typical_price = (high + low + close) / 3
    cumulative_vwap = (typical_price * volume).cumsum() / volume.cumsum().replace(0, pd.NA)
    enriched["VWAP"] = cumulative_vwap.ffill()
    
    # Daily VWAP reset (better for intraday)
    try:
        if hasattr(enriched.index, 'date'):
            enriched['date'] = enriched.index.date
            vwap_daily = enriched.groupby('date', group_keys=False).apply(
                lambda x: ((x['high'] + x['low'] + x['close']) / 3 * x['volume']).cumsum() / x['volume'].cumsum(),
                include_groups=False
            )
            enriched['VWAP_daily'] = vwap_daily
            enriched.drop('date', axis=1, inplace=True)
        else:
            enriched['VWAP_daily'] = enriched['VWAP']
    except Exception:
        # Fallback if daily VWAP calculation fails
        enriched['VWAP_daily'] = enriched['VWAP']

    # ATR (Average True Range) - volatility
    enriched["ATR_14"] = _atr(high, low, close, 14)

    # ADX (Average Directional Index) - trend strength
    enriched["ADX_14"] = _adx(high, low, close, 14)

    # Stochastic Oscillator
    stoch_k, stoch_d = _stochastic(high, low, close, 14, 3)
    enriched["STOCH_K"] = stoch_k
    enriched["STOCH_D"] = stoch_d

    # Volume indicators
    enriched["volume_sma_20"] = volume.rolling(20).mean()
    enriched["volume_ratio"] = (volume / enriched["volume_sma_20"]).fillna(1.0)
    enriched["volume_delta"] = volume.diff().fillna(0)

    # Price momentum and volatility
    enriched["volatility"] = close.pct_change().rolling(20).std().fillna(0)
    enriched["rolling_return"] = close.pct_change(20)
    enriched["momentum_10"] = close.pct_change(10)
    
    # Price relative to moving averages
    enriched["price_to_sma_20"] = (close / rolling_mean).fillna(1.0)
    enriched["price_to_ema_50"] = (close / enriched["EMA_50"]).fillna(1.0)
    
    # Use pandas_ta if available for additional indicators
    if HAS_PANDAS_TA:
        try:
            # OBV (On Balance Volume)
            obv = ta.obv(close, volume)
            if obv is not None:
                enriched["OBV"] = obv
            
            # CMF (Chaikin Money Flow)
            cmf = ta.cmf(high, low, close, volume, length=20)
            if cmf is not None:
                enriched["CMF"] = cmf
                
            # Supertrend
            supertrend = ta.supertrend(high, low, close, length=10, multiplier=3)
            if supertrend is not None and not supertrend.empty:
                enriched["SUPERTREND"] = supertrend.iloc[:, 0]
                
        except Exception:
            pass  # Silently skip if pandas_ta indicators fail
    
    return enriched


def merge_sentiment(df: pd.DataFrame, sentiment: pd.DataFrame) -> pd.DataFrame:
    if sentiment.empty:
        return df
    resampled = sentiment.resample("1min").mean().ffill()
    if "sentiment" in resampled.columns:
        score = resampled["sentiment"]
    else:
        score = resampled.mean(axis=1)
    combined = df.join(score.rename("sentiment_score"), how="left")
    combined["sentiment_score"] = combined["sentiment_score"].ffill().fillna(0.0)
    return combined


def engineer_features(price_df: pd.DataFrame, sentiment_df: pd.DataFrame | None = None) -> pd.DataFrame:
    base = add_technical_indicators(price_df)
    if sentiment_df is not None:
        base = merge_sentiment(base, sentiment_df)
    base.dropna(inplace=True)
    return base
