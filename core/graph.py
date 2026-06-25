# core/graph.py
import time
import mlflow
from langgraph.graph import StateGraph, START, END
from core.state import MarketMindState

# Import all 6 agent nodes
from core.agents.price_agent import price_node
from core.agents.sentiment_agent import sentiment_node
from core.agents.macro_agent import macro_node
from core.agents.onchain_agent import onchain_node
from core.agents.risk_agent import risk_node
from core.agents.synthesis_agent import synthesis_node

# Import execution pipeline
from core.regime_detector import detect_market_regime
from core.scoring_engine import calculate_conviction_score
from core.portfolio_manager import size_position
from core.risk_firewall import run_pre_trade_checks
from core.execution_engine import execute_trade

# ---------------------------------------------------------------------------
# EXECUTION NODE WRAPPERS
# These wrap pure functions into LangGraph-compatible nodes that read from
# and write back to MarketMindState.
# ---------------------------------------------------------------------------

def quantitative_scoring_node(state: MarketMindState) -> dict:
    """
    The Great Divide scoring node.
    
    Reads VIX from the Risk Agent's raw data, classifies the market regime,
    then blends all 5 agent signals using regime-adjusted weights into a
    single Conviction Score between -1.0 (max bearish) and 1.0 (max bullish).
    
    Why this exists as a separate node: it enforces the separation between
    the AI opinion layer (agents) and the deterministic math layer (scoring).
    The score is reproducible given the same inputs — the agents are not.
    """
    # Extract VIX from Risk Agent's raw_data.
    # Fallback to 20.0 (neutral) if risk agent failed or returned no data.
    vix = 20.0
    for sig in state.get("agent_signals", []):
        if sig["agent"] == "risk" and sig.get("raw_data"):
            vix = sig["raw_data"].get("vix_score", 20.0)
            break

    regime_data = detect_market_regime(vix)
    score = calculate_conviction_score(
        state.get("agent_signals", []),
        regime_data["weights"]
    )

    # Log to MLflow if an active run exists
    try:
        mlflow.log_metrics({
            "conviction_score": score,
            "vix": vix,
        })
        mlflow.log_param("market_regime", regime_data["regime"])
    except Exception:
        pass  # MLflow is optional — never crash the pipeline for observability

    return {
        "market_regime": regime_data["regime"],
        "regime_weights": regime_data["weights"],
        "conviction_score": score,
    }


def execution_node(state: MarketMindState) -> dict:
    """
    Position sizing → Risk Firewall → Broker execution.
    
    Three-stage gate:
    1. Portfolio Manager: converts conviction score to a dollar position size
    2. Risk Firewall: deterministic hard checks (VIX > 30, drawdown > 5%, 
       position cap) that override AI conviction entirely
    3. Execution Engine: routes to paper or live broker
    
    The firewall is intentionally placed AFTER sizing so the blocked reason
    message can reference the actual proposed trade amount.
    """
    account_size = 100_000.00  # Simulated $100k account

    # 1. Size the position
    trade = size_position(
        state["conviction_score"],
        account_size,
        risk_profile="moderate"
    )

    # 2. Extract risk metrics from Risk Agent for the firewall
    current_dd = 0.0
    vix = 20.0
    for sig in state.get("agent_signals", []):
        if sig["agent"] == "risk" and sig.get("raw_data"):
            current_dd = sig["raw_data"].get("max_drawdown_pct", 0.0) / 100.0
            vix = sig["raw_data"].get("vix_score", 20.0)
            break

    # 3. Run the firewall
    passed, reason = run_pre_trade_checks(trade, vix, current_dd)

    # 4. Execute if cleared
    exec_status = {"status": "blocked", "reason": reason}
    if passed and trade["action"] != "HOLD":
        exec_status = execute_trade(state["ticker"], trade, simulated=True)

    # Log execution outcome to MLflow
    try:
        mlflow.log_params({
            "trade_action": trade["action"],
            "firewall_passed": str(passed),
            "execution_status": exec_status.get("status", "unknown"),
        })
        mlflow.log_metric("position_size_usd", trade.get("position_size_usd", 0))
    except Exception:
        pass

    return {
        "proposed_trade": trade,
        "firewall_passed": passed,
        "execution_status": exec_status,
    }


# ---------------------------------------------------------------------------
# GRAPH DEFINITION
# ---------------------------------------------------------------------------

builder = StateGraph(MarketMindState)

# --- Register nodes ---
builder.add_node("price_agent",     price_node)
builder.add_node("sentiment_agent", sentiment_node)
builder.add_node("macro_agent",     macro_node)
builder.add_node("onchain_agent",   onchain_node)
builder.add_node("risk_agent",      risk_node)
builder.add_node("synthesis_agent", synthesis_node)
builder.add_node("quant_scoring",   quantitative_scoring_node)
builder.add_node("live_execution",  execution_node)

# --- FAN-OUT: START fires all 5 specialist agents simultaneously ---
# LangGraph's Pregel engine spawns each as a concurrent task.
# The operator.add reducer in state.py handles the concurrent writes safely.
builder.add_edge(START, "price_agent")
builder.add_edge(START, "sentiment_agent")
builder.add_edge(START, "macro_agent")
builder.add_edge(START, "onchain_agent")
builder.add_edge(START, "risk_agent")

# --- FAN-IN: All 5 agents must complete before synthesis runs ---
# LangGraph automatically waits for all incoming edges to resolve.
builder.add_edge("price_agent",     "synthesis_agent")
builder.add_edge("sentiment_agent", "synthesis_agent")
builder.add_edge("macro_agent",     "synthesis_agent")
builder.add_edge("onchain_agent",   "synthesis_agent")
builder.add_edge("risk_agent",      "synthesis_agent")

# --- LINEAR PIPELINE: synthesis → scoring → execution → END ---
# BUG FIX: The original code wired synthesis_agent to BOTH END and
# quant_scoring. LangGraph processed the END edge first and terminated,
# meaning the scoring and execution pipeline never ran.
# Fix: remove the premature END edge. Only live_execution connects to END.
builder.add_edge("synthesis_agent", "quant_scoring")
builder.add_edge("quant_scoring",   "live_execution")
builder.add_edge("live_execution",  END)

# Compile into an executable application
marketmind_app = builder.compile()
