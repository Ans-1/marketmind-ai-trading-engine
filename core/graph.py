# core/graph.py
from langgraph.graph import StateGraph, START, END
from core.state import MarketMindState

# Import all 6 of our nodes
from core.agents.price_agent import price_node
from core.agents.sentiment_agent import sentiment_node
from core.agents.macro_agent import macro_node
from core.agents.onchain_agent import onchain_node
from core.agents.risk_agent import risk_node
from core.agents.synthesis_agent import synthesis_node

# Import Execution Engines
from core.regime_detector import detect_market_regime
from core.scoring_engine import calculate_conviction_score
from core.portfolio_manager import size_position
from core.risk_firewall import run_pre_trade_checks
from core.execution_engine import execute_trade

# --- EXECUTION NODE WRAPPERS ---

def quantitative_scoring_node(state: MarketMindState) -> dict:
    """Replaces AI guesswork with math based on the VIX regime."""
    # Extract VIX from the Risk Agent's raw data
    vix = 20.0 
    for sig in state.get("agent_signals", []):
        if sig["agent"] == "risk":
            vix = sig["raw_data"].get("vix_score", 20.0)
            break
            
    regime_data = detect_market_regime(vix)
    score = calculate_conviction_score(state.get("agent_signals", []), regime_data["weights"])
    
    return {
        "market_regime": regime_data["regime"],
        "regime_weights": regime_data["weights"],
        "conviction_score": score
    }

def execution_node(state: MarketMindState) -> dict:
    """Handles position sizing, the risk firewall, and broker execution."""
    # 1. Portfolio Manager
    account_size = 100000.00 # Simulated $100k account
    trade = size_position(state["conviction_score"], account_size, risk_profile="moderate")
    
    # 2. Risk Firewall
    # Extract current max drawdown from Risk Agent
    current_dd = 0.0
    for sig in state.get("agent_signals", []):
        if sig["agent"] == "risk":
            current_dd = sig["raw_data"].get("max_drawdown_pct", 0.0) / 100.0
            break
            
    vix = 20.0 # Standard fallback
    
    passed, reason = run_pre_trade_checks(trade, vix, current_dd)
    
    # 3. Execution
    exec_status = {"status": "blocked", "reason": reason}
    if passed and trade["action"] != "HOLD":
        exec_status = execute_trade(state["ticker"], trade, simulated=True)
        
    return {
        "proposed_trade": trade,
        "firewall_passed": passed,
        "execution_status": exec_status
    }


# 1. Initialize the Graph with our global State
builder = StateGraph(MarketMindState)

# 2. Register all the agent nodes (the "rooms" in our office)
builder.add_node("price_agent", price_node)
builder.add_node("sentiment_agent", sentiment_node)
builder.add_node("macro_agent", macro_node)
builder.add_node("onchain_agent", onchain_node)
builder.add_node("risk_agent", risk_node)
builder.add_node("synthesis_agent", synthesis_node)

# Register Execution Nodes
builder.add_node("quant_scoring", quantitative_scoring_node)
builder.add_node("live_execution", execution_node)

# 3. THE FAN-OUT (Parallel Execution)
# By connecting START to all 5 agents simultaneously, LangGraph knows 
# to fire them all at the exact same time in parallel.
builder.add_edge(START, "price_agent")
builder.add_edge(START, "sentiment_agent")
builder.add_edge(START, "macro_agent")
builder.add_edge(START, "onchain_agent")
builder.add_edge(START, "risk_agent")

# 4. THE FAN-IN (Synthesis)
# Connect all 5 specialists to the Synthesis Boss. 
# LangGraph will automatically wait until ALL 5 are finished before running Synthesis.
builder.add_edge("price_agent", "synthesis_agent")
builder.add_edge("sentiment_agent", "synthesis_agent")
builder.add_edge("macro_agent", "synthesis_agent")
builder.add_edge("onchain_agent", "synthesis_agent")
builder.add_edge("risk_agent", "synthesis_agent")

# 5. Finish the process
builder.add_edge("synthesis_agent", END)

# Route to Execution Pipeline
builder.add_edge("synthesis_agent", "quant_scoring")
builder.add_edge("quant_scoring", "live_execution")
builder.add_edge("live_execution", END)

# 6. Compile the graph into an executable application
marketmind_app = builder.compile()