# core/state.py
import operator
from typing import Annotated, Optional
from typing_extensions import TypedDict


class AgentSignal(TypedDict):
    """
    Structured output from a single specialist agent.
    
    Every agent in the swarm returns exactly this shape.
    The raw_data field carries the underlying numbers so the
    scoring engine and MLflow can log what the math actually said,
    not just the LLM's interpretation of it.
    """
    agent:      str    # "price" | "sentiment" | "macro" | "onchain" | "risk"
    signal:     str    # "BULLISH" | "BEARISH" | "NEUTRAL"
    confidence: float  # 0.0 to 1.0
    summary:    str    # One-sentence human-readable explanation
    raw_data:   dict   # The numbers backing up the call (RSI, VIX, etc.)


class MarketMindState(TypedDict):
    """
    Shared state that flows through the entire LangGraph pipeline.
    
    ARCHITECTURE NOTE — the Annotated[list, operator.add] pattern:
    When 5 agents write to agent_signals simultaneously during the fan-out
    phase, LangGraph needs to know how to merge 5 concurrent writes into one
    list without any agent overwriting another. operator.add is the merge
    function: it appends lists rather than replacing them. This is what makes
    true parallel execution safe without locks.
    
    All other fields use last-write-wins (LangGraph default), which is safe
    because only one node writes each field.
    """

    # --- Core inputs ---
    ticker:     str   # e.g. "AAPL", "BTC-USD", "EURUSD=X"
    asset_type: str   # "equities" | "crypto" | "forex" | "commodities"

    # --- Agent swarm output (parallel-safe via operator.add reducer) ---
    agent_signals: Annotated[list[AgentSignal], operator.add]

    # --- Synthesis agent output ---
    final_verdict:   Optional[str]    # "BUY" | "HOLD" | "SELL"
    final_confidence: Optional[float]
    final_reasoning:  Optional[str]

    # --- Quantitative scoring layer ---
    market_regime:    Optional[str]   # "LOW_VOLATILITY_BULL" | "NORMAL_CHOP" | "HIGH_VOLATILITY_PANIC"
    regime_weights:   Optional[dict]  # e.g. {"price": 0.4, "sentiment": 0.3, ...}
    conviction_score: Optional[float] # -1.0 to 1.0

    # --- Execution layer ---
    proposed_trade:   Optional[dict]  # {action, position_size_usd, reason}
    firewall_passed:  Optional[bool]
    execution_status: Optional[dict]  # {status, filled_price, simulated}

    # --- Observability & memory ---
    # mlflow_run_id is set by the API/CLI before invoking the graph so that
    # all nodes can log to the same MLflow run.
    mlflow_run_id:    Optional[str]

    # historical_context is populated by ChromaDB memory layer before the
    # graph runs. It contains past conviction scores for this ticker so the
    # synthesis agent can factor in historical behavior.
    historical_context: Optional[str]

    # Execution timing for latency measurement
    run_start_time:   Optional[float]
