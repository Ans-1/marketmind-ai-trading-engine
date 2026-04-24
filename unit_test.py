# test_agents.py
from core.state import MarketMindState
from core.agents.risk_agent import risk_node

def test_single_agent():
    print("🤖 Booting up the Price Agent workbench...")
    
    # 1. Create a fake "State" to hand to the agent
    mock_state = MarketMindState(
        ticker="BTC-USD",  # Let's test Bitcoin! (You can change this to AAPL or TSLA)
        asset_type="crypto",
        agent_signals=[],
        final_verdict=None,
        final_confidence=None,
        final_reasoning=None
    )
    
    print(f"📈 Fetching live yfinance data for {mock_state['ticker']} and asking Llama 3.3 to analyze...")
    
    # 2. Run the node exactly as LangGraph will
    result = risk_node(mock_state)
    
    # 3. Print the results beautifully
    signals = result.get("agent_signals", [])
    if not signals:
        print("❌ Agent failed to return a signal.")
        return
        
    output = signals[0]
    
    print("\n" + "="*40)
    print("🧠 AGENT VERDICT")
    print("="*40)
    print(f"Agent:      {output['agent'].upper()}")
    print(f"Signal:     {output['signal']}")
    print(f"Confidence: {output['confidence'] * 100}%")
    print(f"Summary:    {output['summary']}")
    
    print("\n📊 RAW MATH")
    print("-" * 20)
    for key, val in output['raw_data'].items():
        print(f"{key.replace('_', ' ').title()}: {val}")
    print("="*40)

if __name__ == "__main__":
    test_single_agent()