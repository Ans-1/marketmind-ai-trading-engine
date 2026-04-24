# test_master.py
import time
from core.state import MarketMindState
from core.graph import marketmind_app

def test_full_graph():
    ticker = "BTC-USD"
    asset_type = "crypto"
    
    print("="*60)
    print(f"🚀 INITIATING MARKETMIND MULTI-AGENT PIPELINE FOR {ticker}")
    print("="*60)
    
    # 1. Initialize the blank slate
    initial_state = MarketMindState(
        ticker=ticker,
        asset_type=asset_type,
        agent_signals=[],
        final_verdict=None,
        final_confidence=None,
        final_reasoning=None
    )
    
    # Start the timer!
    start_time = time.time()
    
    print("⚡ [FAN-OUT] Firing all 5 specialist agents in parallel...")
    
    # 2. Run the Graph!
    # This single line handles the entire fan-out/fan-in process
    result = marketmind_app.invoke(initial_state)
    
    execution_time = round(time.time() - start_time, 2)
    
    print("📥 [FAN-IN] All agents reported back. Synthesis complete.\n")
    
    # 3. Print the Specialist Breakdowns
    print("-" * 60)
    print("🕵️ SPECIALIST AGENT BREAKDOWN")
    print("-" * 60)
    
    signals = result.get("agent_signals", [])
    for s in signals:
        agent_name = s['agent'].upper()
        # Formatting so the columns align nicely
        print(f"{agent_name:<12} -> {s['signal']:<8} (Conf: {s['confidence']*100:0.0f}%) | {s['summary']}")
        
    # 4. Print the Final Synthesis
    print("\n" + "=" * 60)
    print("👑 THE SYNTHESIS AGENT'S FINAL VERDICT")
    print("=" * 60)
    
    print(f"Verdict:    {result['final_verdict']}")
    print(f"Confidence: {result['final_confidence'] * 100:0.0f}%")
    print(f"Reasoning:  {result['final_reasoning']}\n")
    
    print(f"⏱️ Total parallel execution time: {execution_time} seconds")
    print("=" * 60)

if __name__ == "__main__":
    test_full_graph()