# core/screener/normalizer.py
import pandas as pd
import numpy as np

def calculate_z_scores(historical_prices: list, current_price: float, historical_volumes: list, current_volume: float) -> dict:
    """
    Normalizes asset volatility and volume using Z-Scores.
    A Z-Score > 1.5 indicates a statistically significant breakout.
    """
    try:
        # 1. Price Normalization
        prices_series = pd.Series(historical_prices)
        price_mean = prices_series.mean()
        price_std = prices_series.std()
        
        # Prevent division by zero if an asset has completely flatlined
        if price_std == 0:
            price_z_score = 0.0
        else:
            price_z_score = (current_price - price_mean) / price_std

        # 2. Volume Normalization (Is institutional money stepping in?)
        vol_series = pd.Series(historical_volumes)
        vol_mean = vol_series.mean()
        
        # Calculate volume ratio (e.g., 1.5 means 50% higher volume than the 30-day average)
        vol_ratio = current_volume / vol_mean if vol_mean > 0 else 0.0

        return {
            "price_z_score": round(float(price_z_score), 2),
            "volume_ratio": round(float(vol_ratio), 2),
            "historical_volatility": round(float(price_std / price_mean * 100), 2) # Raw % volatility for reference
        }
        
    except Exception as e:
        # Graceful degradation if data is malformed
        return {"price_z_score": 0.0, "volume_ratio": 0.0, "historical_volatility": 0.0}