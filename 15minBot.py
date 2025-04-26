import ccxt
import pandas as pd
import ta
import time
from datetime import datetime
import os
import requests

# === CONFIG ===
API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')
POSITION_SIZE = 1000
PROFIT_TARGET = 0.01  # 1%
STOP_LOSS = 0.005     # 0.5%
SCAN_INTERVAL = 60    # Every 1 minute
PAIR_LIMIT = 100
TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = '-1002549924266'

# Init
exchange = ccxt.binance({
    'apiKey': os.getenv('APIKEY'),
    'secret': os.getenv('SECRETKEY'),
    'enableRateLimit': True
})
open_trades = {}

def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, json={'chat_id': TELEGRAM_CHAT_ID, 'text': msg})
    except:
        pass

def get_top_pairs():
    tickers = exchange.fetch_tickers()
    usdt_pairs = [s for s in tickers if s.endswith('/USDT') and tickers[s]['quoteVolume'] > 0]
    sorted_pairs = sorted(usdt_pairs, key=lambda s: tickers[s]['quoteVolume'], reverse=True)
    return sorted_pairs[:PAIR_LIMIT]

def fetch_ohlcv(symbol):
    data = exchange.fetch_ohlcv(symbol, timeframe='15m', limit=50)
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['ema20'] = ta.trend.ema_indicator(df['close'], window=20)
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    return df

def check_entry(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    bullish_cross = prev['close'] < prev['ema20'] and last['close'] > last['ema20']
    rsi_ok = 45 < last['rsi'] < 65
    return bullish_cross and rsi_ok

def place_trade(symbol, price):
    qty = POSITION_SIZE / price
    open_trades[symbol] = {
        'entry': price,
        'target': price * (1 + PROFIT_TARGET),
        'stop': price * (1 - STOP_LOSS),
        'qty': qty,
        'time': datetime.now()
    }
    send_telegram(f"🛒 Bought {symbol} at {price:.4f}")

def monitor_trades():
    if not open_trades:
        return

    prices = exchange.fetch_tickers([s for s in open_trades])
    to_close = []

    for symbol, trade in open_trades.items():
        price = prices[symbol]['bid']
        if price >= trade['target']:
            send_telegram(f"🎯 Sold {symbol} at Target {price:.4f} (+1%)")
            to_close.append(symbol)
        elif price <= trade['stop']:
            send_telegram(f"🛑 Sold {symbol} at Stop {price:.4f} (-0.5%)")
            to_close.append(symbol)

    for symbol in to_close:
        del open_trades[symbol]

def main_loop():
    while True:
        try:
            pairs = get_top_pairs()
            monitor_trades()

            for symbol in pairs:
                if symbol in open_trades:
                    continue

                df = fetch_ohlcv(symbol)
                if check_entry(df):
                    price = df['close'].iloc[-1]
                    place_trade(symbol, price)

            print(f"Scan done at {datetime.now().strftime('%H:%M:%S')}")
        except Exception as e:
            print(f"Error: {e}")

        time.sleep(SCAN_INTERVAL)

if __name__ == "__main__":
    main_loop()
