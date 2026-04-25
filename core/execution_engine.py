# core/execution_engine.py
import os

def execute_trade(ticker: str, trade_payload: dict, simulated: bool = True):
    """
    Submits the trade to the brokerage API.
    """
    action = trade_payload["action"]
    size = trade_payload["position_size_usd"]
    
    if simulated:
        print("\n" + "="*40)
        print("🟢 SIMULATED EXECUTION ENGINE TRIGGERED")
        print("="*40)
        print(f"BROKER: Local Paper Trading")
        print(f"TICKER: {ticker}")
        print(f"ORDER:  {action}")
        print(f"AMOUNT: ${size}")
        print("STATUS: Order filled locally (Simulation).")
        print("="*40 + "\n")
        return {"status": "success", "filled_price": 0.0, "simulated": True}
        
    else:
        # Future implementation area for Alpaca API
        alpaca_api_key = os.getenv("ALPACA_API_KEY")
        if not alpaca_api_key:
            raise ValueError("Alpaca API keys missing for live execution.")
            
        print(f"🔴 LIVE TRADING NOT YET IMPLEMENTED. Halting {action} order for {ticker}.")
        return {"status": "failed", "reason": "Live execution disabled."}