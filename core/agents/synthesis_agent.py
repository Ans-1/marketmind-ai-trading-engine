# core/agents/synthesis_agent.py
from core.agents import llm, safe_parse_json
from core.state import MarketMindState

def synthesis_node(state: MarketMindState) -> dict:
    """The final node that reviews all 5 agent signals and makes a final call."""
    ticker = state["ticker"]
    signals = state.get("agent_signals", [])
    
    # Format the 5 reports into a clean text block for the LLM to read
    reports_text = ""
    for s in signals:
        reports_text += f"[{s['agent'].upper()}] Signal: {s['signal']} ({s['confidence']*100}% confidence)\n"
        reports_text += f"Summary: {s['summary']}\n\n"

    # # Build the prompt for Groq
    # prompt = f"""
    # You are the Lead Portfolio Manager for {ticker}.
    # Your 5 specialist agents have just submitted their independent analyses:
    
    # {reports_text}
    
    # Your job is to synthesize these conflicting signals into one final trading decision. 
    # - If Price and Sentiment are bullish, but Risk and Macro are screaming bearish, you might lean NEUTRAL or HOLD to protect capital.
    # - Weigh Risk and Macro heavily in your decision.
    
    # Provide your final verdict as a JSON object matching this EXACT schema:
    # {{
    #     "signal": "BUY" | "HOLD" | "SELL",
    #     "confidence": float between 0.0 and 1.0,
    #     "reasoning": "A 2-3 sentence explanation synthesizing the bull/bear cases."
    # }}
    # """

    # Build the engineered prompt for Groq
    prompt = f"""
    You are the Lead Portfolio Manager for {ticker}.
    Your 5 specialist agents have just submitted their independent analyses:
    
    {reports_text}
    
    Your job is to synthesize these conflicting signals into one final trading decision. 

    ---
    EXAMPLES OF EXCELLENT ANALYSIS:
    
    Input Reports: Price (Bullish), Sentiment (Bullish), Macro (Bearish), Risk (Bearish).
    Output:
    {{
        "thought_process": "While short-term price momentum and news sentiment are currently bullish, the Macro and Risk agents are flashing major warning signs. In portfolio management, capital preservation (Risk/Macro) always supersedes short-term technicals. I cannot authorize a BUY when systemic risk is elevated. I will wait for a better macro setup.",
        "signal": "HOLD",
        "confidence": 0.70,
        "reasoning": "Strong technicals are currently negated by dangerous macro and risk environments; prioritizing capital preservation."
    }}
    ---
    
    You MUST think step-by-step in the 'thought_process' field BEFORE providing your final signal. Weigh Risk and Macro heavily in your decision.
    
    Provide your final verdict as a JSON object matching this EXACT schema:
    {{
        "thought_process": "Step-by-step reasoning synthesizing the bull/bear cases and resolving conflicts...",
        "signal": "BUY" | "HOLD" | "SELL",
        "confidence": float between 0.0 and 1.0,
        "reasoning": "A 2-3 sentence explanation synthesizing the final call."
    }}
    """
    
    # Call the LLM and parse
    response = llm.invoke(prompt)
    parsed = safe_parse_json(response.content)
    
    # Notice we return the final keys, NOT appending to agent_signals
    return {
        "final_verdict": parsed.get("signal", "HOLD"),
        "final_confidence": float(parsed.get("confidence", 0.0)),
        "final_reasoning": parsed.get("reasoning", "Failed to synthesize signals.")
    }