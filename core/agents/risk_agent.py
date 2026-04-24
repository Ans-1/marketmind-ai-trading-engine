# core/agents/risk_agent.py
import yfinance as yf
import pandas as pd
import numpy as np
from core.agents import llm, safe_parse_json
from core.state import MarketMindState, AgentSignal

def compute_drawdown(prices: pd.Series) -> float:
    """Calculates the maximum peak-to-trough drop."""
    rolling_max = prices.cummax()
    drawdown = prices / rolling_max - 1.0
    max_drawdown = drawdown.min()
    return round(float(max_drawdown * 100), 2)

def risk_node(state: MarketMindState) -> dict:
    """The LangGraph node that analyzes volatility and downside risk."""
    ticker = state["ticker"]
    asset_type = state["asset_type"].lower()
    
    # Trading days: Crypto trades 365 days a year, stocks trade ~252
    trading_days = 365 if asset_type == "crypto" else 252

    try:
        # Fetch 180 days of historical data for a solid risk baseline
        hist = yf.Ticker(ticker).history(period="180d")
        if hist.empty:
            raise ValueError(f"No price data found for {ticker}")
            
        prices = hist['Close']
        
        # 1. Calculate Daily Returns & Volatility
        daily_returns = prices.pct_change().dropna()
        # Annualized Volatility formula: std_dev * sqrt(trading_days)
        ann_volatility = round(float(daily_returns.std() * np.sqrt(trading_days) * 100), 2)
        
        # 2. Calculate Max Drawdown over this period
        max_dd = compute_drawdown(prices)
        
        # 3. Fetch the VIX (Wall Street's Fear Gauge)
        # yfinance uses ^VIX for the index
        vix_hist = yf.Ticker('^VIX').history(period="5d")
        current_vix = round(float(vix_hist['Close'].iloc[-1]), 2) if not vix_hist.empty else 20.0
        
        raw_data = {
            "annualized_volatility_pct": ann_volatility,
            "max_drawdown_pct": max_dd,
            "vix_score": current_vix
        }
        
    except Exception as e:
        return {"agent_signals": [AgentSignal(
            agent="risk", signal="NEUTRAL", confidence=0.0,
            summary=f"Risk data fetch error: {str(e)}", raw_data={}
        )]}

    # # Build the prompt for Groq
    # prompt = f"""
    # You are an expert Chief Risk Officer analyzing {ticker}.
    
    # Current Risk Metrics:
    # - Annualized Volatility: {ann_volatility}% (Note: >50% is extremely high risk)
    # - 180-Day Max Drawdown: {max_dd}% (How far it fell from its recent peak)
    # - Market VIX (Fear Gauge): {current_vix} (Note: <15 is calm, 15-20 is moderate, >20 is high stress)
    
    # Based on these metrics, evaluate the downside risk of holding this asset right now. If volatility and VIX are dangerously high, lean BEARISH or NEUTRAL to protect capital.
    
    # Provide your verdict as a JSON object matching this EXACT schema:
    # {{
    #     "signal": "BULLISH" | "BEARISH" | "NEUTRAL",
    #     "confidence": float between 0.0 and 1.0,
    #     "summary": "One short sentence explaining the risk profile."
    # }}
    # """

    # Build the engineered prompt for Groq
    prompt = f"""
    You are an expert Chief Risk Officer analyzing {ticker}.
    
    Current Risk Metrics:
    - Annualized Volatility: {ann_volatility}%
    - 180-Day Max Drawdown: {max_dd}%
    - Market VIX (Fear Gauge): {current_vix}

    ---
    EXAMPLES OF EXCELLENT ANALYSIS:

    Input: Volatility: 85%, Drawdown: -60%, VIX: 28
    Output:
    {{
        "thought_process": "The VIX is highly elevated (28), showing systemic market stress. The asset itself is wildly volatile (85%) with a history of catastrophic drawdowns (-60%). This is a capital-destruction environment. We must reduce exposure.",
        "signal": "BEARISH",
        "confidence": 0.90,
        "summary": "Extreme asset volatility combined with high systemic VIX signals severe downside risk."
    }}

    Input: Volatility: 25%, Drawdown: -10%, VIX: 13
    Output:
    {{
        "thought_process": "The broader market is incredibly calm (VIX 13). The asset's volatility is controlled (25%), and its recent maximum drawdown was a shallow -10% correction. This is a highly favorable, asymmetrical risk environment for holding long positions.",
        "signal": "BULLISH",
        "confidence": 0.85,
        "summary": "Low systemic stress and shallow historical drawdowns create a very safe environment for capital."
    }}
    ---

    Now, analyze the Current Risk Metrics for {ticker}.
    You MUST think step-by-step in the 'thought_process' field BEFORE providing your final signal.

    Provide your verdict as a JSON object matching this EXACT schema:
    {{
        "thought_process": "Step-by-step reasoning explaining the downside risk and volatility profile...",
        "signal": "BULLISH" | "BEARISH" | "NEUTRAL",
        "confidence": float between 0.0 and 1.0,
        "summary": "One short sentence explaining the risk profile."
    }}
    """
    
    # Call the LLM and parse
    response = llm.invoke(prompt)
    parsed = safe_parse_json(response.content)
    
    final_signal = AgentSignal(
        agent="risk",
        signal=parsed.get("signal", "NEUTRAL"),
        confidence=float(parsed.get("confidence", 0.0)),
        summary=parsed.get("summary", "Failed to analyze risk profile."),
        raw_data=raw_data
    )
    
    return {"agent_signals": [final_signal]}