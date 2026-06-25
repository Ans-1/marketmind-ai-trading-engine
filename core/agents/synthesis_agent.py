# core/agents/synthesis_agent.py
"""
The Synthesis Agent — the final decision-maker of the swarm.

ROLE IN THE ARCHITECTURE:
After the 5 specialist agents (price, sentiment, macro, onchain, risk) complete
in parallel, their signals converge here. The synthesis agent's job is to
resolve conflicts and produce one final trading verdict.

WHY AN LLM FOR SYNTHESIS INSTEAD OF PURE MATH:
The conviction score (from scoring_engine.py) is a weighted average of signals.
That works well when agents agree. But when price says BULLISH and macro says
BEARISH, a weighted average just averages the disagreement without understanding
why there's a disagreement or which signal should dominate given the context.
The synthesis agent provides that qualitative judgment — it can say "yes, price
momentum is strong, but a VIX of 35 means we don't trust it right now."

CHAIN-OF-THOUGHT PROMPTING:
We require the LLM to fill a 'thought_process' field before its final signal.
This forces deliberate reasoning rather than pattern matching, produces more
consistent outputs, and makes the decision auditable.

MEMORY INTEGRATION:
If historical_context is present in state (populated by the ChromaDB memory
layer before the graph runs), it's injected into the prompt. This gives the
synthesis agent access to past decisions — turning a stateless analysis into
one that learns from its own track record.
"""

from core.agents import llm, safe_parse_json
from core.state import MarketMindState


def synthesis_node(state: MarketMindState) -> dict:
    """The LangGraph node that reviews all agent signals and makes a final call."""
    ticker  = state["ticker"]
    signals = state.get("agent_signals", [])

    # Format the 5 agent reports into a clean readable block for the LLM
    reports_text = ""
    for s in signals:
        reports_text += (
            f"[{s['agent'].upper()}] "
            f"Signal: {s['signal']} ({s['confidence'] * 100:.0f}% confidence)\n"
            f"Summary: {s['summary']}\n\n"
        )

    # Inject historical memory if available (populated by ChromaDB layer)
    memory_context = state.get("historical_context") or ""

    prompt = f"""
You are the Lead Portfolio Manager for {ticker}.
Your 5 specialist agents have submitted their independent analyses:

{reports_text}
{memory_context}

Your job: synthesize these signals into one final trading decision.

DECISION FRAMEWORK:
- If Risk and Macro are both BEARISH, capital preservation overrides short-term technicals.
- If all 5 agents agree, high confidence is warranted.
- If signals are split 3:2, weigh the Risk agent's view most heavily.
- If historical memory shows this ticker had a bad outcome in similar conditions, note it.

---
EXAMPLES OF EXCELLENT SYNTHESIS:

Scenario: Price (Bullish), Sentiment (Bullish), Macro (Bearish), Risk (Bearish), Onchain (Neutral)
Output:
{{
    "thought_process": "Two of five agents are bearish, but they are the two that matter most for capital preservation — Risk and Macro. Strong price momentum exists, but deploying capital when VIX is elevated and the macro environment is restrictive violates the portfolio management principle of asymmetric risk. The smart move is to wait for macro conditions to improve before acting on the technical setup.",
    "signal": "HOLD",
    "confidence": 0.72,
    "reasoning": "Technically bullish but macro and risk environments are hostile; capital preservation takes priority until conditions improve."
}}

Scenario: All agents BULLISH
Output:
{{
    "thought_process": "Rare alignment across all 5 agents. Price technicals confirm the trend, sentiment confirms the narrative, macro confirms the liquidity environment, risk confirms controlled volatility, and onchain confirms active participation. When specialists unanimously agree, the portfolio manager's job is to act decisively.",
    "signal": "BUY",
    "confidence": 0.91,
    "reasoning": "Full cross-agent alignment across technical, sentiment, macro, and risk dimensions presents an unusually high-conviction long setup."
}}
---

Now, analyze the reports for {ticker}.
You MUST reason in 'thought_process' BEFORE committing to a signal.

Respond with a JSON object matching this EXACT schema:
{{
    "thought_process": "Step-by-step reasoning resolving conflicts between agents...",
    "signal": "BUY" | "HOLD" | "SELL",
    "confidence": float between 0.0 and 1.0,
    "reasoning": "2-3 sentence synthesis of the final decision."
}}
"""

    response = llm.invoke(prompt)
    parsed   = safe_parse_json(response.content)

    return {
        "final_verdict":    parsed.get("signal", "HOLD"),
        "final_confidence": float(parsed.get("confidence", 0.0)),
        "final_reasoning":  parsed.get("reasoning", "Failed to synthesize signals."),
    }
