# core/agents/macro_agent.py
import os
import requests
from fredapi import Fred
from core.agents import llm, safe_parse_json
from core.state import MarketMindState, AgentSignal

def macro_node(state: MarketMindState) -> dict:
    """The LangGraph node that analyzes macroeconomic conditions."""
    ticker = state["ticker"]
    api_key = os.getenv("FRED_API_KEY")
    
    if not api_key or api_key == "your_fred_api_key_here":
        return {"agent_signals": [AgentSignal(
            agent="macro", signal="NEUTRAL", confidence=0.0,
            summary="FRED API key missing.", raw_data={}
        )]}

    try:
        # 1. Fetch 10-Year Treasury Yield from the Federal Reserve
        fred = Fred(api_key=api_key)
        # 'DGS10' is the official ticker for the 10-Year Treasury Rate
        t10_data = fred.get_series('DGS10').dropna() 
        current_t10_yield = round(t10_data.iloc[-1], 2)
        
        # 2. Fetch the Crypto Fear & Greed Index (Free public API)
        fng_response = requests.get("https://api.alternative.me/fng/?limit=1")
        fng_data = fng_response.json()['data'][0]
        fng_value = int(fng_data['value'])
        fng_sentiment = fng_data['value_classification']

        raw_data = {
            "10_yr_treasury_yield": current_t10_yield,
            "fear_and_greed_score": fng_value,
            "fear_and_greed_label": fng_sentiment
        }
        
    except Exception as e:
        return {"agent_signals": [AgentSignal(
            agent="macro", signal="NEUTRAL", confidence=0.0,
            summary=f"Macro data fetch error: {str(e)}", raw_data={}
        )]}

    # # Build the prompt for Groq
    # prompt = f"""
    # You are an expert macroeconomist analyzing the environment for {ticker} ({state['asset_type']}).
    
    # Current Macro Indicators:
    # - 10-Year Treasury Yield: {current_t10_yield}% (Note: Rising/High yields are generally BEARISH for risk assets like tech stocks and crypto).
    # - Market Fear & Greed Index: {fng_value}/100 ({fng_sentiment}). (Note: Extreme fear can be a buying opportunity, extreme greed implies an overbought market).
    
    # Based on these macroeconomic factors, what is the optimal stance for this asset?
    
    # Provide your verdict as a JSON object matching this EXACT schema:
    # {{
    #     "signal": "BULLISH" | "BEARISH" | "NEUTRAL",
    #     "confidence": float between 0.0 and 1.0,
    #     "summary": "One short sentence explaining the macro environment's impact on the asset."
    # }}
    # """

    # Build the engineered prompt for Groq
    prompt = f"""
    You are an expert macroeconomist analyzing the environment for {ticker} ({state['asset_type']}).
    
    Current Macro Indicators:
    - 10-Year Treasury Yield: {current_t10_yield}%
    - Market Fear & Greed Index: {fng_value}/100 ({fng_sentiment})

    ---
    EXAMPLES OF EXCELLENT ANALYSIS:

    Input: 10-Year Yield: 4.8%, Fear & Greed: 25 (Extreme Fear)
    Output:
    {{
        "thought_process": "A high treasury yield of 4.8% pulls capital out of risk-on assets (like tech and crypto) into safe government bonds. Combined with an Extreme Fear score of 25, the broader macroeconomic environment is hostile to risk.",
        "signal": "BEARISH",
        "confidence": 0.85,
        "summary": "High treasury yields and extreme market fear create a highly restrictive environment for risk assets."
    }}

    Input: 10-Year Yield: 3.2%, Fear & Greed: 75 (Greed)
    Output:
    {{
        "thought_process": "A low treasury yield (3.2%) forces investors to seek returns in riskier assets. The market is greedy (75) but not yet at 'Extreme Greed' bubble levels. This is a classic risk-on, accommodating macro environment.",
        "signal": "BULLISH",
        "confidence": 0.80,
        "summary": "Accommodative bond yields and healthy market greed support a strong risk-on environment."
    }}
    ---

    Now, analyze the Current Macro Indicators for {ticker}.
    You MUST think step-by-step in the 'thought_process' field BEFORE providing your final signal.

    Provide your verdict as a JSON object matching this EXACT schema:
    {{
        "thought_process": "Step-by-step reasoning explaining the macro environment's impact on liquidity...",
        "signal": "BULLISH" | "BEARISH" | "NEUTRAL",
        "confidence": float between 0.0 and 1.0,
        "summary": "One short sentence explaining the macro environment's impact on the asset."
    }}
    """
    
    # Call the LLM and parse
    response = llm.invoke(prompt)
    parsed = safe_parse_json(response.content)
    
    final_signal = AgentSignal(
        agent="macro",
        signal=parsed.get("signal", "NEUTRAL"),
        confidence=float(parsed.get("confidence", 0.0)),
        summary=parsed.get("summary", "Failed to analyze macro environment."),
        raw_data=raw_data
    )
    
    return {"agent_signals": [final_signal]}