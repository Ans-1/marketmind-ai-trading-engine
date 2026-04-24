# core/agents/price_agent.py
import yfinance as yf
import pandas as pd
from core.agents import llm, safe_parse_json
from core.state import MarketMindState, AgentSignal

# --- 1. The Math / Indicator Functions ---

def compute_rsi(prices: pd.Series, period: int = 14) -> float:
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / loss
    return round(float((100 - (100 / (1 + rs))).iloc[-1]), 2)

def compute_macd(prices: pd.Series):
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return round(float(macd_line.iloc[-1]), 4), round(float(signal_line.iloc[-1]), 4)

def compute_bollinger(prices: pd.Series, period: int = 20) -> float:
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper_band = sma + (std * 2)
    lower_band = sma - (std * 2)
    current_price = prices.iloc[-1]
    
    # Price position within Bollinger Bands: 0 = lower band, 1 = upper band.
    position = (current_price - lower_band.iloc[-1]) / (upper_band.iloc[-1] - lower_band.iloc[-1])
    return round(float(position), 2)


# --- 2. The LangGraph Node Function (The Missing Link) ---

def price_node(state: MarketMindState) -> dict:
    """The LangGraph node that executes the Price Agent logic."""
    ticker = state["ticker"]
    
    try:
        # Fetch live data
        hist = yf.Ticker(ticker).history(period="90d")
        if hist.empty:
            raise ValueError(f"No price data found for {ticker}")
            
        prices = hist['Close']
        current_price = round(float(prices.iloc[-1]), 2)
        
        # Calculate indicators
        rsi = compute_rsi(prices)
        macd, macd_signal = compute_macd(prices)
        bb_position = compute_bollinger(prices)
        
        raw_data = {
            "current_price": current_price,
            "rsi_14": rsi,
            "macd": macd,
            "macd_signal": macd_signal,
            "bollinger_position": bb_position
        }
        
    except Exception as e:
        # If yfinance fails, degrade gracefully instead of crashing the whole app
        error_signal = AgentSignal(
            agent="price", signal="NEUTRAL", confidence=0.0,
            summary=f"Data fetch error: {str(e)}", raw_data={}
        )
        return {"agent_signals": [error_signal]}

    # Build the engineered prompt for Groq
    prompt = f"""
    You are an elite quantitative technical analyst.

    Task: Analyze the following technical indicators for {ticker} and determine the immediate price momentum.

    Current Data:
    - Current Price: ${current_price}
    - RSI (14): {rsi}
    - MACD: {macd}, Signal: {macd_signal}
    - Bollinger Band Position: {bb_position} (0 = lower band, 1 = upper band)

    ---
    EXAMPLES OF EXCELLENT ANALYSIS:

    Input: RSI: 82, MACD: 150, Signal: 100, Bollinger: 0.95
    Output:
    {{
        "thought_process": "RSI is severely overbought at 82. Price is riding the extreme upper Bollinger Band (0.95). While MACD shows bullish momentum, the asset is highly overextended and vulnerable to a mean-reversion pullback. The risk of a top is high.",
        "signal": "BEARISH",
        "confidence": 0.75,
        "summary": "Severely overbought RSI and extreme Bollinger extension suggest an imminent pullback despite MACD momentum."
    }}

    Input: RSI: 45, MACD: 5, Signal: -2, Bollinger: 0.40
    Output:
    {{
        "thought_process": "RSI is neutral at 45, leaving plenty of room for upside. MACD has just crossed above the signal line (5 > -2), indicating a fresh bullish momentum shift. Price is near the middle of the Bollinger Bands, showing no overextension.",
        "signal": "BULLISH",
        "confidence": 0.85,
        "summary": "Fresh MACD bullish crossover with neutral RSI indicates strong room for upward momentum."
    }}
    ---

    Now, analyze the Current Data for {ticker}.
    You MUST think step-by-step in the 'thought_process' field BEFORE providing your final signal.

    Provide your verdict as a JSON object matching this EXACT schema:
    {{
        "thought_process": "Step-by-step reasoning explaining how the indicators interact...",
        "signal": "BULLISH" | "BEARISH" | "NEUTRAL",
        "confidence": float between 0.0 and 1.0,
        "summary": "One short sentence summarizing the final call."
    }}
    """
    
    # Call the LLM and parse the response
    response = llm.invoke(prompt)
    parsed = safe_parse_json(response.content)
    
    # Package into our TypedDict
    final_signal = AgentSignal(
        agent="price",
        signal=parsed.get("signal", "NEUTRAL"),
        confidence=float(parsed.get("confidence", 0.0)),
        summary=parsed.get("summary", "Failed to analyze technicals."),
        raw_data=raw_data
    )
    
    # By returning a dict with the `agent_signals` key, LangGraph knows to route 
    # this through our `operator.add` reducer in state.py and append it to the list.
    return {"agent_signals": [final_signal]}