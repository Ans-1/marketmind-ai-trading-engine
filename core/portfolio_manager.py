# core/portfolio_manager.py

def size_position(conviction_score: float, current_capital: float, risk_profile: str = "moderate") -> dict:
    """
    Determines if a trade should happen, and calculates position sizing.
    """
    # Threshold: Only trade if the score is strongly bullish (>0.4) or bearish (<-0.4)
    if abs(conviction_score) < 0.4:
        return {"action": "HOLD", "position_size_usd": 0.0, "reason": "Conviction too low."}
    
    action = "BUY" if conviction_score > 0 else "SELL_SHORT"
    
    # Risk parameters based on profile
    risk_fractions = {"conservative": 0.01, "moderate": 0.03, "aggressive": 0.05}
    base_risk = risk_fractions.get(risk_profile, 0.02)
    
    # Scale position size based on how confident the score is
    # e.g., A score of 0.5 gets half the max risk allocation. A score of 1.0 gets max.
    confidence_multiplier = abs(conviction_score)
    allocated_risk = base_risk * confidence_multiplier
    
    position_size_usd = current_capital * allocated_risk
    
    return {
        "action": action,
        "position_size_usd": round(position_size_usd, 2),
        "reason": f"Conviction of {conviction_score} triggered {action}."
    }