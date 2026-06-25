# api.py
"""
MarketMind FastAPI layer.

WHY FASTAPI OVER FLASK:
FastAPI is async-native, which matters here because a full MarketMind analysis
run takes 10–30 seconds (5 LLM calls in parallel + data fetches). With Flask
(synchronous), one running analysis blocks the server from handling any other
requests. With FastAPI + async, the server stays responsive during long-running
analyses.

FastAPI also auto-generates OpenAPI documentation at /docs — you can demo the
API in a browser without writing a single line of frontend code.

ENDPOINTS:
  POST /analyze          - Run a full multi-agent analysis on a ticker
  GET  /health           - System health check (API + ChromaDB + MLflow status)
  GET  /history/{ticker} - Retrieve past analyses from ChromaDB memory

DESIGN DECISIONS:
- We use BackgroundTasks for MLflow run teardown so the response returns
  immediately after the analysis without waiting for logging to complete.
- Pydantic models enforce the request/response contract — any client sending
  a malformed request gets a clear 422 error, not a confusing Python traceback.
- The /analyze endpoint returns the full structured result so any consumer
  (Streamlit UI, a mobile app, another service) gets everything in one call.
"""

import time
import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core.state import MarketMindState
from core.graph import marketmind_app
from core.memory import retrieve_context, store_analysis, get_memory_stats
from core import tracking


# ---------------------------------------------------------------------------
# PYDANTIC MODELS — request and response contracts
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    """
    What the caller must send to POST /analyze.
    Pydantic validates this automatically — wrong types return HTTP 422.
    """
    ticker:       str  = Field(..., example="AAPL",     description="Asset ticker symbol")
    asset_type:   str  = Field(..., example="equities", description="equities | crypto | forex | commodities")
    account_size: float = Field(100_000.0, description="Simulated account size in USD")

    class Config:
        json_schema_extra = {
            "example": {
                "ticker":     "AAPL",
                "asset_type": "equities",
                "account_size": 100000.0,
            }
        }


class AgentSignalResponse(BaseModel):
    agent:      str
    signal:     str
    confidence: float
    summary:    str


class TradeTicketResponse(BaseModel):
    action:           str
    position_size_usd: float
    reason:           str


class AnalyzeResponse(BaseModel):
    """
    Full structured response from a completed analysis.
    Every field is Optional so partial failures still return useful data.
    """
    ticker:           str
    asset_type:       str

    # Agent layer
    agent_signals:    list[AgentSignalResponse]

    # Synthesis layer
    final_verdict:    Optional[str]
    final_confidence: Optional[float]
    final_reasoning:  Optional[str]

    # Scoring layer
    market_regime:    Optional[str]
    conviction_score: Optional[float]

    # Execution layer
    proposed_trade:   Optional[TradeTicketResponse]
    firewall_passed:  Optional[bool]
    execution_status: Optional[dict]

    # Metadata
    run_duration_seconds: float
    mlflow_run_id:        Optional[str]


class HealthResponse(BaseModel):
    status:       str
    memory:       dict
    mlflow_ui:    str
    version:      str = "1.0.0"


# ---------------------------------------------------------------------------
# APP SETUP
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once on startup and once on shutdown."""
    print("🧠 MarketMind API starting up...")
    print("📊 MLflow UI: http://localhost:5000")
    print("📚 API docs:  http://localhost:8000/docs")
    yield
    print("👋 MarketMind API shutting down.")


app = FastAPI(
    title="MarketMind AI Trading Engine",
    description=(
        "Multi-agent AI trading analysis system. "
        "Combines quantitative math (RSI, MACD, ATR, VIX) with LLM agents "
        "for sentiment, macro, and synthesis. Returns structured conviction "
        "scores and trade tickets."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow the Streamlit frontend (running on port 8501) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    System health check.
    Returns the status of the API, ChromaDB memory, and MLflow tracking.
    Use this endpoint to verify all infrastructure components are running.
    """
    return HealthResponse(
        status="healthy",
        memory=get_memory_stats(),
        mlflow_ui="http://localhost:5000",
    )


@app.post("/analyze", response_model=AnalyzeResponse, tags=["Analysis"])
async def analyze(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    """
    Run a full multi-agent analysis on a ticker.
    
    This is the core endpoint. It:
    1. Retrieves historical context from ChromaDB memory
    2. Starts an MLflow tracking run  
    3. Invokes the full LangGraph pipeline (5 parallel agents → synthesis → scoring → execution)
    4. Stores the result in ChromaDB for future memory retrieval
    5. Returns the complete structured analysis
    
    Expected latency: 10–30 seconds (dominated by 5 parallel LLM calls via Groq).
    """
    run_start = time.time()
    ticker     = request.ticker.upper()
    asset_type = request.asset_type.lower()

    # --- 1. Retrieve historical memory for this ticker ---
    # We don't know the regime yet (the graph determines it), so we use
    # "NORMAL_CHOP" as a neutral starting query. The synthesis agent
    # will use its own judgment about how relevant the history is.
    historical_context = retrieve_context(
        ticker=ticker,
        current_regime="NORMAL_CHOP",
        n_results=3,
    )

    # --- 2. Start MLflow run ---
    run_id = tracking.start_run(ticker, asset_type)

    # --- 3. Build initial state ---
    initial_state = MarketMindState(
        ticker=ticker,
        asset_type=asset_type,
        agent_signals=[],
        final_verdict=None,
        final_confidence=None,
        final_reasoning=None,
        market_regime=None,
        regime_weights=None,
        conviction_score=None,
        proposed_trade=None,
        firewall_passed=None,
        execution_status=None,
        mlflow_run_id=run_id,
        historical_context=historical_context or None,
        run_start_time=run_start,
    )

    # --- 4. Run the LangGraph pipeline ---
    # We run this in an executor to avoid blocking the async event loop
    # during the synchronous LangGraph invoke call.
    try:
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: marketmind_app.invoke(initial_state)
        )
    except Exception as e:
        tracking.end_run()
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline execution failed: {str(e)}"
        )

    # --- 5. Log results to MLflow in background (non-blocking) ---
    def _log_and_store():
        try:
            tracking.log_agent_signals(result.get("agent_signals", []))
            tracking.log_final_result(result)
        finally:
            tracking.end_run()

        # Store in ChromaDB memory for future runs
        if result.get("conviction_score") is not None:
            store_analysis(
                ticker=ticker,
                asset_type=asset_type,
                conviction_score=result.get("conviction_score", 0.0),
                final_verdict=result.get("final_verdict", "HOLD"),
                market_regime=result.get("market_regime", "UNKNOWN"),
                agent_signals=result.get("agent_signals", []),
                final_reasoning=result.get("final_reasoning", ""),
                run_id=run_id,
            )

    background_tasks.add_task(_log_and_store)

    # --- 6. Build and return response ---
    run_duration = round(time.time() - run_start, 2)

    agent_responses = [
        AgentSignalResponse(
            agent=sig["agent"],
            signal=sig["signal"],
            confidence=sig["confidence"],
            summary=sig["summary"],
        )
        for sig in result.get("agent_signals", [])
    ]

    proposed_trade = result.get("proposed_trade")
    trade_response = None
    if proposed_trade:
        trade_response = TradeTicketResponse(
            action=proposed_trade.get("action", "HOLD"),
            position_size_usd=proposed_trade.get("position_size_usd", 0.0),
            reason=proposed_trade.get("reason", ""),
        )

    return AnalyzeResponse(
        ticker=ticker,
        asset_type=asset_type,
        agent_signals=agent_responses,
        final_verdict=result.get("final_verdict"),
        final_confidence=result.get("final_confidence"),
        final_reasoning=result.get("final_reasoning"),
        market_regime=result.get("market_regime"),
        conviction_score=result.get("conviction_score"),
        proposed_trade=trade_response,
        firewall_passed=result.get("firewall_passed"),
        execution_status=result.get("execution_status"),
        run_duration_seconds=run_duration,
        mlflow_run_id=run_id,
    )


@app.get("/history/{ticker}", tags=["Memory"])
async def get_history(ticker: str, n_results: int = 5):
    """
    Retrieve past analyses for a ticker from ChromaDB memory.
    
    Returns the most recent analyses stored in long-term memory.
    Useful for reviewing the system's historical conviction on an asset.
    """
    from core.memory import _get_collection
    collection = _get_collection()

    if collection is None:
        raise HTTPException(status_code=503, detail="Memory system unavailable.")

    if collection.count() == 0:
        return {"ticker": ticker.upper(), "records": [], "total": 0}

    try:
        results = collection.query(
            query_texts=[f"Analysis of {ticker.upper()}"],
            n_results=min(n_results, collection.count()),
            where={"ticker": {"$eq": ticker.upper()}},
        )

        records = []
        for i, doc in enumerate(results["documents"][0]):
            records.append({
                "document": doc,
                "metadata": results["metadatas"][0][i],
            })

        return {
            "ticker":  ticker.upper(),
            "records": records,
            "total":   len(records),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
