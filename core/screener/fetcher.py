# core/screener/fetcher.py
import yfinance as yf
from pycoingecko import CoinGeckoAPI
import time

cg = CoinGeckoAPI()

def fetch_yfinance_data(ticker: str) -> dict:
    """Pulls 30-day history for Stocks, Forex, and Commodities."""
    try:
        data = yf.Ticker(ticker).history(period="1mo")
        if data.empty:
            return None
            
        prices = data['Close'].tolist()
        volumes = data['Volume'].tolist()
        
        return {
            "historical_prices": prices[:-1], # Everything except today
            "current_price": prices[-1],      # Today's close/current price
            "historical_volumes": volumes[:-1],
            "current_volume": volumes[-1]
        }
    except Exception:
        return None

def fetch_coingecko_data(coin_id: str) -> dict:
    """Pulls 30-day history for Cryptocurrencies."""
    try:
        # Fetch market chart data (prices and total volumes)
        data = cg.get_coin_market_chart_by_id(id=coin_id, vs_currency='usd', days='30')
        
        # CoinGecko returns lists of [timestamp, value]
        prices = [item[1] for item in data['prices']]
        volumes = [item[1] for item in data['total_volumes']]
        
        return {
            "historical_prices": prices[:-1],
            "current_price": prices[-1],
            "historical_volumes": volumes[:-1],
            "current_volume": volumes[-1]
        }
    except Exception:
        return None

def get_asset_data(asset_id: str, asset_type: str) -> dict:
    """Master router function."""
    if asset_type == "crypto":
        # CoinGecko has strict rate limits, adding a tiny sleep prevents IP bans
        time.sleep(0.2) 
        return fetch_coingecko_data(asset_id)
    else:
        return fetch_yfinance_data(asset_id)