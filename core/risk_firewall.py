# core/risk_firewall.py

def run_pre_trade_checks(proposed_trade: dict, current_vix: float, account_daily_drawdown: float) -> tuple[bool, str]:
    """
    Deterministic layer that blocks dangerous trades regardless of AI conviction.
    Returns (True/False, "Reasoning String")
    """
    if proposed_trade["action"] == "HOLD":
        return False, "Signal is HOLD. No execution required."
        
    # Check 1: Max Daily Drawdown Stop
    if account_daily_drawdown < -0.05: # -5% loss on the day
        return False, "FIREWALL BLOCK: Daily drawdown exceeded -5%. Trading halted."
        
    # Check 2: VIX Override
    if current_vix > 30.0 and proposed_trade["action"] == "BUY":
        return False, "FIREWALL BLOCK: VIX > 30. Buying risk assets is disabled."
        
    # Check 3: Max Position Size Cap
    if proposed_trade["position_size_usd"] > 10000: # Hard limit example
        return False, "FIREWALL BLOCK: Requested position size exceeds hardcoded safety cap."

    return True, "FIREWALL PASS: All safety parameters cleared."