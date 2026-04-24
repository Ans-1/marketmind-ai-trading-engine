# core/agents/onchain_agent.py
from pycoingecko import CoinGeckoAPI
from core.agents import llm, safe_parse_json
from core.state import MarketMindState, AgentSignal

def onchain_node(state: MarketMindState) -> dict:
    """The LangGraph node that analyzes blockchain/crypto specific metrics."""
    ticker = state["ticker"]
    asset_type = state["asset_type"].lower()
    
    # If the user asks for Apple (AAPL), skip this agent. On-chain only applies to crypto.
    if asset_type != "crypto":
        return {"agent_signals": [AgentSignal(
            agent="onchain", signal="NEUTRAL", confidence=0.0,
            summary="N/A: Not a crypto asset.", raw_data={}
        )]}

    cg = CoinGeckoAPI()
    
    # Clean up the ticker (e.g., 'BTC-USD' -> 'bitcoin')
    # CoinGecko uses full names, so we need a tiny translation map for top coins
    ticker_clean = ticker.split("-")[0].lower()
    cg_id_map = {
        "btc": "bitcoin", "eth": "ethereum", "sol": "solana",
        "ada": "cardano", "xrp": "ripple", "doge": "dogecoin"
    }
    
    # Default to passing the ticker directly if it's not in our map
    cg_id = cg_id_map.get(ticker_clean, ticker_clean)

    try:
        # Fetch current market data
        data = cg.get_coin_by_id(id=cg_id, localization=False, tickers=False, market_data=True, community_data=False, developer_data=False, sparkline=False)
        
        market_data = data['market_data']
        rank = market_data['market_cap_rank']
        price_change_24h = round(market_data['price_change_percentage_24h'], 2)
        
        # Calculate Volume to Market Cap ratio (liquidity indicator)
        vol = market_data['total_volume']['usd']
        mcap = market_data['market_cap']['usd']
        vol_mcap_ratio = round((vol / mcap) * 100, 2) if mcap > 0 else 0
        
        raw_data = {
            "market_cap_rank": rank,
            "price_change_24h_pct": price_change_24h,
            "volume_to_mcap_ratio_pct": vol_mcap_ratio
        }
        
    except Exception as e:
        return {"agent_signals": [AgentSignal(
            agent="onchain", signal="NEUTRAL", confidence=0.0,
            summary=f"CoinGecko fetch error: {str(e)}", raw_data={}
        )]}

    # # Build the prompt for Groq
    # prompt = f"""
    # You are an expert On-Chain Crypto Analyst looking at {ticker} (Rank #{rank}).
    
    # Current On-Chain Metrics:
    # - 24 Hour Price Change: {price_change_24h}%
    # - 24H Volume / Market Cap Ratio: {vol_mcap_ratio}% (Note: >10% usually indicates high liquidity/interest, <2% indicates stagnation).
    
    # Analyze these metrics to determine the on-chain momentum. Is smart money actively trading this, or is interest dying down?
    
    # Provide your verdict as a JSON object matching this EXACT schema:
    # {{
    #     "signal": "BULLISH" | "BEARISH" | "NEUTRAL",
    #     "confidence": float between 0.0 and 1.0,
    #     "summary": "One short sentence explaining the on-chain network activity."
    # }}
    # """

    # Build the engineered prompt for Groq
    prompt = f"""
    You are an expert On-Chain Crypto Analyst looking at {ticker} (Rank #{rank}).
    
    Current On-Chain Metrics:
    - 24 Hour Price Change: {price_change_24h}%
    - 24H Volume / Market Cap Ratio: {vol_mcap_ratio}%

    ---
    EXAMPLES OF EXCELLENT ANALYSIS:

    Input: 24H Change: +5%, Vol/Mcap Ratio: 15%
    Output:
    {{
        "thought_process": "A Volume-to-Market-Cap ratio of 15% is extremely healthy, indicating massive liquidity and active participation from smart money. Combined with positive price action, the network is highly active and accumulating.",
        "signal": "BULLISH",
        "confidence": 0.85,
        "summary": "High relative trading volume indicates strong network activity and active accumulation."
    }}

    Input: 24H Change: -1%, Vol/Mcap Ratio: 1.5%
    Output:
    {{
        "thought_process": "A Vol/Mcap ratio of 1.5% is dangerously low. It shows the network is a ghost town. Smart money is not trading this asset right now, meaning any sudden sell-off could cause a liquidity cascade. I must lean bearish on the lack of interest.",
        "signal": "BEARISH",
        "confidence": 0.70,
        "summary": "Severe lack of trading volume relative to market cap suggests network stagnation and low liquidity."
    }}
    ---

    Now, analyze the Current Metrics for {ticker}.
    You MUST think step-by-step in the 'thought_process' field BEFORE providing your final signal.

    Provide your verdict as a JSON object matching this EXACT schema:
    {{
        "thought_process": "Step-by-step reasoning explaining network activity and liquidity...",
        "signal": "BULLISH" | "BEARISH" | "NEUTRAL",
        "confidence": float between 0.0 and 1.0,
        "summary": "One short sentence explaining the on-chain network activity."
    }}
    """
    
    # Call the LLM and parse
    response = llm.invoke(prompt)
    parsed = safe_parse_json(response.content)
    
    final_signal = AgentSignal(
        agent="onchain",
        signal=parsed.get("signal", "NEUTRAL"),
        confidence=float(parsed.get("confidence", 0.0)),
        summary=parsed.get("summary", "Failed to analyze on-chain data."),
        raw_data=raw_data
    )
    
    return {"agent_signals": [final_signal]}