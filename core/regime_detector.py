# core/regime_detector.py

def detect_market_regime(vix_score: float) -> dict:
    """
    Classifies the market environment to adjust risk and strategy.
    Returns the regime label and weight adjustments for the scoring engine.
    """
    if vix_score < 15.0:
        return {
            "regime": "LOW_VOLATILITY_BULL",
            "description": "Calm market, favorable for trend following.",
            "weights": {"price": 0.4, "sentiment": 0.3, "macro": 0.1, "onchain": 0.2, "risk": 0.0}
        }
    elif 15.0 <= vix_score <= 25.0:
        return {
            "regime": "NORMAL_CHOP",
            "description": "Standard market conditions. Balanced weighting.",
            "weights": {"price": 0.2, "sentiment": 0.2, "macro": 0.2, "onchain": 0.2, "risk": 0.2}
        }
    else:
        return {
            "regime": "HIGH_VOLATILITY_PANIC",
            "description": "High stress. Heavy reliance on Risk and Macro agents.",
            "weights": {"price": 0.05, "sentiment": 0.05, "macro": 0.4, "onchain": 0.1, "risk": 0.4}
        }