# core/state.py
import operator
from typing import Annotated, Optional
from typing_extensions import TypedDict

class AgentSignal(TypedDict):
    """Output from a single specialist agent."""
    agent: str      # e.g., "price", "sentiment"
    signal: str     # "BULLISH", "BEARISH", or "NEUTRAL"
    confidence: float # 0.0 to 1.0
    summary: str    # One-line explanation
    raw_data: dict  # The numbers backing up the call

class MarketMindState(TypedDict):
    """Shared state that flows through the entire graph."""
    ticker: str
    asset_type: str # "stock" or "crypto"

    # The Reducer: operator.add ensures that when 5 agents write to this list
    # at the exact same millisecond, they append their results instead of 
    # overwriting each other.
    agent_signals: Annotated[list[AgentSignal], operator.add]

    # Populated by the Synthesis Agent at the very end
    final_verdict: Optional[str]
    final_confidence: Optional[float]
    final_reasoning: Optional[str]

    # Execution Layer (New)
    market_regime: str | None
    regime_weights: dict | None
    conviction_score: float | None
    proposed_trade: dict | None
    firewall_passed: bool | None
    execution_status: dict | None