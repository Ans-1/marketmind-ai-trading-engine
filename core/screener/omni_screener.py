# core/screener/omni_screener.py
from core.screener.universe import ASSET_UNIVERSE
from core.screener.fetcher import get_asset_data
from core.screener.normalizer import calculate_z_scores

def run_omni_screener(top_n: int = 3) -> list:
    """
    Scans the global asset universe, normalizes the data,
    and returns the top mathematical setups based on momentum.
    """
    results = []
    
    total_assets = sum(len(tickers) for tickers in ASSET_UNIVERSE.values())
    print(f"🌍 [OMNI-SCREENER] Initiating scan across {total_assets} global assets...")

    for asset_type, tickers in ASSET_UNIVERSE.items():
        for ticker in tickers:
            # 1. Fetch raw data
            raw_data = get_asset_data(ticker, asset_type)
            
            if not raw_data or not raw_data.get("historical_prices"):
                continue
                
            # 2. Normalize into Z-Scores
            metrics = calculate_z_scores(
                historical_prices=raw_data["historical_prices"],
                current_price=raw_data["current_price"],
                historical_volumes=raw_data["historical_volumes"],
                current_volume=raw_data["current_volume"]
            )
            
            # Skip dead assets or API errors
            if metrics["volume_ratio"] == 0.0 and metrics["price_z_score"] == 0.0:
                continue

            # 3. Calculate a unified "Momentum Score"
            # We use absolute Z-score because massive drops (short opportunities) 
            # are just as valuable as massive pumps.
            momentum_score = abs(metrics["price_z_score"]) * metrics["volume_ratio"]
            
            results.append({
                "ticker": ticker,
                "asset_type": asset_type,
                "metrics": metrics,
                "momentum_score": round(momentum_score, 2)
            })

    # 4. Sort by Momentum Score (Highest to Lowest)
    sorted_results = sorted(results, key=lambda x: x["momentum_score"], reverse=True)
    
    # 5. Return the top N setups
    top_setups = sorted_results[:top_n]
    
    print(f"🎯 [OMNI-SCREENER] Scan complete. Found {len(top_setups)} high-conviction setups.")
    return top_setups