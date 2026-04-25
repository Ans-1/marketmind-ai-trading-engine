# core/scoring_engine.py

def calculate_conviction_score(agent_signals: list, regime_weights: dict) -> float:
    """
    Translates textual signals into a weighted mathematical score.
    Returns a float between -1.0 (Max Bearish) and 1.0 (Max Bullish).
    """
    signal_map = {"BULLISH": 1.0, "NEUTRAL": 0.0, "BEARISH": -1.0}
    total_score = 0.0
    
    for signal in agent_signals:
        agent_name = signal["agent"]
        direction = signal_map.get(signal["signal"].upper(), 0.0)
        confidence = signal["confidence"]
        
        # Apply the dynamic weight based on the current market regime
        weight = regime_weights.get(agent_name, 0.2) 
        
        # Add to total score
        total_score += (direction * confidence * weight)
        
    return round(total_score, 2)