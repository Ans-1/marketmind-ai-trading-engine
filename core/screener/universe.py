# core/screener/universe.py

# The Master Watchlist
ASSET_UNIVERSE = {
    "crypto": [
        "bitcoin",    # CoinGecko uses names/IDs, not tickers
        "ethereum", 
        "solana"
    ],
    "equities": [
        "AAPL", "NVDA", "MSFT", "TSLA", "META"
    ],
    "forex": [
        "EURUSD=X", "GBPUSD=X", "JPY=X"
    ],
    "commodities": [
        "GC=F",   # Gold
        "CL=F",   # Crude Oil
        "SI=F"    # Silver
    ]
}