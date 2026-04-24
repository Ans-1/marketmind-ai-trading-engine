# core/agents/sentiment_agent.py
import os
from newsapi import NewsApiClient
from core.agents import llm, safe_parse_json
from core.state import MarketMindState, AgentSignal

def sentiment_node(state: MarketMindState) -> dict:
    """The LangGraph node that reads the news."""
    ticker = state["ticker"]
    api_key = os.getenv("NEWS_API_KEY")
    
    # Safety check if you forgot your API key
    if not api_key or api_key == "your_newsapi_key_here":
        error_signal = AgentSignal(
            agent="sentiment", signal="NEUTRAL", confidence=0.0,
            summary="News API key missing from .env file.", raw_data={}
        )
        return {"agent_signals": [error_signal]}

    try:
        newsapi = NewsApiClient(api_key=api_key)
        
        # Clean up the ticker for the news search (e.g., 'BTC-USD' -> 'BTC')
        query = ticker.split("-")[0]
        
        # Fetch the top 10 most recent articles containing our ticker
        articles = newsapi.get_everything(
            q=query, 
            language='en', 
            sort_by='publishedAt', 
            page_size=10
        )
        
        # Extract just the headlines
        headlines = [article['title'] for article in articles.get('articles', []) if article.get('title')]
        
        if not headlines:
            empty_signal = AgentSignal(
                agent="sentiment", signal="NEUTRAL", confidence=0.0,
                summary=f"No recent news headlines found for {query}.", raw_data={}
            )
            return {"agent_signals": [empty_signal]}
            
        headlines_text = "\n".join([f"- {h}" for h in headlines])
        raw_data = {"article_count": len(headlines), "headlines": headlines}
        
    except Exception as e:
        # Graceful degradation if the API fails
        error_signal = AgentSignal(
            agent="sentiment", signal="NEUTRAL", confidence=0.0,
            summary=f"News API fetch error: {str(e)}", raw_data={}
        )
        return {"agent_signals": [error_signal]}

    # # Build the prompt for Groq
    # prompt = f"""
    # You are an expert financial sentiment analyst. Analyze these recent news headlines for {ticker}:
    
    # {headlines_text}
    
    # Determine the overall market sentiment from these headlines. Are people panicking, euphoric, or neutral?
    
    # Provide your verdict as a JSON object matching this EXACT schema:
    # {{
    #     "signal": "BULLISH" | "BEARISH" | "NEUTRAL",
    #     "confidence": float between 0.0 and 1.0,
    #     "summary": "One short sentence explaining the sentiment driven by the news."
    # }}
    # """

    # Build the engineered prompt for Groq
    prompt = f"""
    You are an expert financial sentiment analyst. 
    
    Task: Analyze these recent news headlines for {ticker} and determine the overall market sentiment.

    Current Headlines:
    {headlines_text}

    ---
    EXAMPLES OF EXCELLENT ANALYSIS:

    Input Headlines: 
    - SEC approves new ETF filing.
    - Institutional inflows hit record highs.
    - CEO announces massive stock buyback program.
    Output:
    {{
        "thought_process": "All headlines point to massive institutional adoption and positive corporate action (buybacks). There is zero mention of regulatory headwinds or hacks. The market is in a state of high euphoria and structural buying.",
        "signal": "BULLISH",
        "confidence": 0.90,
        "summary": "Overwhelmingly positive news regarding institutional inflows and corporate buybacks."
    }}

    Input Headlines:
    - Department of Justice announces investigation into accounting practices.
    - Major exchange halts withdrawals amid liquidity fears.
    - Founder steps down following internal scandal.
    Output:
    {{
        "thought_process": "These headlines represent existential threats. DOJ investigations and frozen withdrawals instantly trigger panic selling. The sentiment is dominated by extreme fear and loss of trust.",
        "signal": "BEARISH",
        "confidence": 0.95,
        "summary": "Severe regulatory threats and liquidity fears dominate the news cycle, driving panic."
    }}
    ---

    Now, analyze the Current Headlines for {ticker}.
    You MUST think step-by-step in the 'thought_process' field BEFORE providing your final signal.

    Provide your verdict as a JSON object matching this EXACT schema:
    {{
        "thought_process": "Step-by-step reasoning weighing the positive vs negative headlines...",
        "signal": "BULLISH" | "BEARISH" | "NEUTRAL",
        "confidence": float between 0.0 and 1.0,
        "summary": "One short sentence summarizing the sentiment driven by the news."
    }}
    """
    
    # Call the LLM and parse
    response = llm.invoke(prompt)
    parsed = safe_parse_json(response.content)
    
    final_signal = AgentSignal(
        agent="sentiment",
        signal=parsed.get("signal", "NEUTRAL"),
        confidence=float(parsed.get("confidence", 0.0)),
        summary=parsed.get("summary", "Failed to analyze sentiment."),
        raw_data=raw_data
    )
    
    return {"agent_signals": [final_signal]}