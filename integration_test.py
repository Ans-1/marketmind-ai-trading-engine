# test_master.py
import time
from core.state import MarketMindState
from core.graph import marketmind_app

def test_execution_pipeline():
    ticker = "BTC-USD"
    
    print("="*60)
    print(f"🚀 INITIATING MARKETMIND EXECUTION PIPELINE FOR {ticker}")
    print("="*60)
    
    initial_state = MarketMindState(
        ticker=ticker,
        asset_type="crypto",
        agent_signals=[],
        final_verdict=None,
        final_confidence=None,
        final_reasoning=None,
        market_regime=None,
        regime_weights=None,
        conviction_score=None,
        proposed_trade=None,
        firewall_passed=None,
        execution_status=None
    )
    
    start_time = time.time()
    
    # 1. Run the entire graph (Oracle -> Scoring -> Firewall -> Execution)
    result = marketmind_app.invoke(initial_state)
    
    execution_time = round(time.time() - start_time, 2)
    
    # 2. Print the Execution Dashboard
    print("\n" + "="*60)
    print("🧠 1. QUANTITATIVE SCORING")
    print("="*60)
    print(f"Market Regime:    {result['market_regime']}")
    print(f"Conviction Score: {result['conviction_score']} (Scale: -1.0 to 1.0)")
    
    print("\n" + "="*60)
    print("🛡️ 2. RISK FIREWALL & PORTFOLIO MANAGER")
    print("="*60)
    trade = result['proposed_trade']
    print(f"Proposed Action:  {trade['action']}")
    print(f"Position Size:    ${trade['position_size_usd']}")
    print(f"Firewall Passed?: {result['firewall_passed']}")
    
    print("\n" + "="*60)
    print("⚡ 3. EXECUTION STATUS")
    print("="*60)
    status = result['execution_status']
    print(f"Status: {status['status'].upper()}")
    if 'reason' in status:
        print(f"Note:   {status['reason']}")
    
    print(f"\n⏱️ Total execution time: {execution_time} seconds")
    print("="*60)

if __name__ == "__main__":
    test_execution_pipeline()