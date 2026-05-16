# integration_testing.py
import argparse
import time
from core.screener.omni_screener import run_omni_screener
from core.screener.fetcher import get_asset_data
from core.screener.normalizer import calculate_z_scores
from core.state import MarketMindState
from core.graph import marketmind_app

def pre_flight_check(ticker: str, asset_type: str) -> bool:
    """
    Cheap mathematical check to ensure user-provided tickers are actually moving
    before wasting expensive AI API tokens on them.
    """
    print(f"🔍 Running pre-flight momentum check on {ticker}...")
    raw_data = get_asset_data(ticker, asset_type)
    
    if not raw_data or not raw_data.get("historical_prices"):
        print(f"❌ ERROR: Failed to fetch data. Check if {ticker} is a valid symbol.")
        return False
        
    metrics = calculate_z_scores(
        historical_prices=raw_data["historical_prices"],
        current_price=raw_data["current_price"],
        historical_volumes=raw_data["historical_volumes"],
        current_volume=raw_data["current_volume"]
    )
    
    # Early-Kill Threshold: Must have at least moderate momentum (Z > 1.0)
    if abs(metrics["price_z_score"]) < 1.0:
        print(f"🛑 EARLY KILL: {ticker} has low momentum (Z-Score: {metrics['price_z_score']}). AI aborted.")
        return False
        
    print(f"✅ Pre-flight passed. {ticker} is active (Z-Score: {metrics['price_z_score']}).")
    return True

def deploy_ai_swarm(ticker: str, asset_type: str):
    """The wrapper to initialize state and run the LangGraph pipeline."""
    print("\n" + "="*60)
    print(f"🚀 DEPLOYING AI SWARM FOR {ticker}")
    print("="*60)
    
    # Initialize the blank state for the AI graph
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
        execution_status=None
    )
    
    start_time = time.time()
    
    # Run the graph
    result = marketmind_app.invoke(initial_state)
    
    execution_time = round(time.time() - start_time, 2)
    status = result.get('execution_status', {}).get('status', 'UNKNOWN')
    
    print(f"\n⚡ FINAL SYSTEM VERDICT: {status}")
    print(f"⏱️ Total execution time: {execution_time} seconds\n")

def main():
    # 1. Set up the terminal argument parser
    parser = argparse.ArgumentParser(description="MarketMind Execution Engine")
    parser.add_argument("--ticker", type=str, help="Run directly on a specific ticker (e.g., AAPL)")
    parser.add_argument("--type", type=str, default="equities", help="Asset type if using --ticker (e.g., crypto, equities, forex)")
    parser.add_argument("--category", type=str, help="Run screener on a specific category (e.g., crypto)")
    
    args = parser.parse_args()
    
    # 2. Route the execution based on user input
    if args.ticker:
        # EXPLICIT MODE (Sniper)
        ticker = args.ticker.upper()
        if pre_flight_check(ticker, args.type):
            deploy_ai_swarm(ticker, args.type)
            
    elif args.category:
        # CATEGORY MODE (Sector Scan)
        print(f"🌍 Initiating Sector Scan for: {args.category.upper()}...")
        top_targets = run_omni_screener(top_n=5)
        
        # Filter screener results to only include the user's requested category
        filtered_targets = [t for t in top_targets if t["asset_type"].lower() == args.category.lower()]
        
        if not filtered_targets:
            print(f"🛑 No high-momentum setups found in {args.category} today.")
            return
            
        best_target = filtered_targets[0] # Take the top 1 from that category
        deploy_ai_swarm(best_target["ticker"], best_target["asset_type"])
        
    else:
        # OMNI MODE (Global Radar)
        print("🌍 Initiating Global Omni-Scan...")
        top_targets = run_omni_screener(top_n=3)
        
        for target in top_targets:
            deploy_ai_swarm(target["ticker"], target["asset_type"])

if __name__ == "__main__":
    main()