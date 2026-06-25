# main.py
"""
MarketMind entry point.

USAGE:
    # Single ticker analysis (CLI):
    python main.py --ticker AAPL --type equities

    # Start the FastAPI backend:
    uvicorn api:app --reload --port 8000

    # Start the Streamlit GUI:
    streamlit run app.py

    # Start MLflow tracking UI:
    mlflow ui --port 5000
"""

import argparse
import time
import sys

from core.state import MarketMindState
from core.graph import marketmind_app
from core.memory import retrieve_context, store_analysis
from core import tracking


def run_analysis(ticker: str, asset_type: str, account_size: float = 100_000.0):
    """
    Full pipeline: memory retrieval → LangGraph → MLflow logging → memory store.
    Returns the final state dict.
    """
    print(f"\n{'='*60}")
    print(f"  🧠 MarketMind Analysis: {ticker} ({asset_type})")
    print(f"{'='*60}\n")

    run_start = time.time()

    # 1. Retrieve historical context
    print("📚 Retrieving historical memory...")
    historical_context = retrieve_context(ticker, "NORMAL_CHOP", n_results=3)
    if historical_context:
        print(f"   Found past analyses for {ticker}")
    else:
        print(f"   No prior memory for {ticker} — first run")

    # 2. Start MLflow run
    run_id = tracking.start_run(ticker, asset_type)
    print(f"📊 MLflow run started: {run_id[:8]}...")

    try:
        # 3. Build state and invoke graph
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

        print("\n🚀 Deploying agent swarm...\n")
        result = marketmind_app.invoke(initial_state)

        # 4. Log to MLflow
        tracking.log_agent_signals(result.get("agent_signals", []))
        tracking.log_final_result(result)

        # 5. Print results
        _print_results(result, time.time() - run_start)

        # 6. Store in ChromaDB
        if result.get("conviction_score") is not None:
            stored = store_analysis(
                ticker=ticker,
                asset_type=asset_type,
                conviction_score=result.get("conviction_score", 0.0),
                final_verdict=result.get("final_verdict", "HOLD"),
                market_regime=result.get("market_regime", "UNKNOWN"),
                agent_signals=result.get("agent_signals", []),
                final_reasoning=result.get("final_reasoning", ""),
                run_id=run_id,
            )
            if stored:
                print("\n💾 Analysis stored in memory (ChromaDB)")

        return result

    finally:
        tracking.end_run()


def _print_results(result: dict, elapsed: float):
    """Pretty-print the analysis results to console."""
    print(f"\n{'='*60}")
    print("  📋 ANALYSIS RESULTS")
    print(f"{'='*60}")

    # Agent signals
    print("\n🤖 Agent Swarm:")
    signal_emoji = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "🟡"}
    for sig in result.get("agent_signals", []):
        emoji = signal_emoji.get(sig["signal"], "⚪")
        print(
            f"   {emoji} {sig['agent'].upper():<12} "
            f"{sig['signal']:<8} "
            f"({sig['confidence']*100:.0f}% confidence) — {sig['summary']}"
        )

    # Conviction score
    score = result.get("conviction_score")
    regime = result.get("market_regime", "UNKNOWN")
    if score is not None:
        bar_len = int(abs(score) * 20)
        direction = ">" * bar_len if score > 0 else "<" * bar_len
        print(f"\n📊 Regime: {regime}")
        print(f"   Conviction Score: {score:+.2f}  [{direction:^40}]")

    # Final verdict
    verdict = result.get("final_verdict", "HOLD")
    confidence = result.get("final_confidence", 0.0)
    reasoning = result.get("final_reasoning", "")
    verdict_symbol = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(verdict, "⚪")
    print(f"\n{verdict_symbol} FINAL VERDICT: {verdict} ({confidence*100:.0f}% confidence)")
    print(f"   {reasoning}")

    # Trade ticket
    trade = result.get("proposed_trade", {})
    firewall = result.get("firewall_passed")
    if trade:
        print(f"\n💼 Trade Ticket:")
        print(f"   Action:   {trade.get('action', 'HOLD')}")
        print(f"   Size:     ${trade.get('position_size_usd', 0):,.2f}")
        print(f"   Firewall: {'✅ PASSED' if firewall else '❌ BLOCKED'}")
        if not firewall:
            reason = result.get("execution_status", {}).get("reason", "")
            print(f"   Reason:   {reason}")

    print(f"\n⏱️  Completed in {elapsed:.1f}s")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="MarketMind — Multi-Agent AI Trading Analysis"
    )
    parser.add_argument("--ticker",   type=str, required=True,
                        help="Asset ticker (e.g. AAPL, BTC-USD)")
    parser.add_argument("--type",     type=str, default="equities",
                        choices=["equities", "crypto", "forex", "commodities"],
                        help="Asset class")
    parser.add_argument("--account",  type=float, default=100_000.0,
                        help="Simulated account size in USD")
    args = parser.parse_args()

    run_analysis(args.ticker.upper(), args.type, args.account)


if __name__ == "__main__":
    main()
