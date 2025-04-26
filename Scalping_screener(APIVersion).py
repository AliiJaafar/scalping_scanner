import ccxt
import pandas as pd
import ta
from datetime import datetime, timedelta
import time
import os
from dotenv import load_dotenv
import logging
import requests

# Configuration
load_dotenv()
API_KEY = os.getenv('BINANCE_API_KEY')
API_SECRET = os.getenv('BINANCE_API_SECRET')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID_CHANNEL')
POSITION_SIZE = 100
MAX_OPEN_TRADES = 5
SCAN_INTERVAL = 60
RSI_MIN, RSI_MAX = 40, 60
VOLUME_MULTIPLIER = 1.5
PROFIT_TARGET = 0.01
STOP_LOSS = 0.01  # Increased to 1%
COOLDOWN_MINUTES = 15
LOG_FILE = 'trades.log'

# Setup logging
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(message)s')

# Initialize Binance exchange
exchange = ccxt.binance({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True
})

open_trades = []
closed_trades = []  # Track closed trades for cooldown

def send_telegram_message(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': message}
        response = requests.post(url, json=payload)
        if response.status_code != 200:
            logging.error(f"Telegram error: {response.json()}")
    except Exception as e:
        logging.error(f"Telegram notification error: {e}")

def get_decimal_places(price):
    if price < 0.01: return 6
    if price < 1: return 4
    if price < 100: return 2
    return 0

def fetch_klines(symbol, timeframe='15m', limit=100):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        # Check data freshness
        if (datetime.now() - df['timestamp'].iloc[-1]).total_seconds() > 900:  # 15 minutes
            logging.warning(f"Stale data for {symbol}")
            return None
        return df
    except Exception as e:
        logging.error(f"Fetch {symbol} error: {e}")
        return None

def calculate_indicators(df):
    df['ema20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
    df['ema50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df['volume_ma20'] = df['volume'].rolling(window=20).mean()
    return df

def get_top_usdt_pairs(limit=50):
    try:
        tickers = exchange.fetch_tickers()
        usdt_pairs = [s for s in tickers if s.endswith('USDT') and tickers[s].get('quoteVolume', 0) > 0]
        return sorted(usdt_pairs, key=lambda x: tickers[x].get('quoteVolume', 0), reverse=True)[:limit]
    except Exception as e:
        logging.error(f"Fetch pairs error: {e}")
        return []

def is_symbol_in_cooldown(symbol):
    """Check if symbol is in cooldown period."""
    for trade in closed_trades:
        if trade['symbol'] == symbol:
            exit_time = datetime.strptime(trade['exit_time'], '%Y-%m-%d %H:%M:%S')
            if (datetime.now() - exit_time).total_seconds() < COOLDOWN_MINUTES * 60:
                return True
    return False

def check_entry(symbol):
    if len(open_trades) >= MAX_OPEN_TRADES:
        return None
    # Check if symbol is already in open trades or in cooldown
    if any(trade['symbol'] == symbol for trade in open_trades) or is_symbol_in_cooldown(symbol):
        return None

    df = fetch_klines(symbol)
    if df is None or len(df) < 50:
        return None

    df = calculate_indicators(df)
    latest = df.iloc[-1]
    previous = df.iloc[-2]

    price_cross = previous['close'] < previous['ema20'] and latest['close'] > latest['ema20']
    trend = latest['ema20'] > latest['ema50']
    rsi_condition = RSI_MIN < latest['rsi'] < RSI_MAX
    volume_condition = latest['volume'] > VOLUME_MULTIPLIER * latest['volume_ma20']

    if price_cross and trend and rsi_condition and volume_condition:
        entry_price = latest['close']
        decimals = get_decimal_places(entry_price)
        trade = {
            'symbol': symbol,
            'entry_price': round(entry_price, decimals),
            'stop_loss': round(entry_price * (1 - STOP_LOSS), decimals),
            'profit_target': round(entry_price * (1 + PROFIT_TARGET), decimals),
            'entry_time': datetime.now(),
            'quantity': POSITION_SIZE / entry_price
        }
        return trade
    return None

def monitor_trades():
    trades_to_close = []
    for trade in open_trades:
        try:
            ticker = exchange.fetch_ticker(trade['symbol'])
            current_price = ticker['bid']
            decimals = get_decimal_places(trade['entry_price'])

            if current_price >= trade['profit_target']:
                outcome = 'Win'
                exit_price = trade['profit_target']
            elif current_price <= trade['stop_loss']:
                outcome = 'Loss'
                exit_price = trade['stop_loss']
            else:
                continue

            profit_loss = trade['quantity'] * (exit_price - trade['entry_price'])
            trades_to_close.append(trade)
            closed_trade = {
                'symbol': trade['symbol'],
                'entry_price': trade['entry_price'],
                'exit_price': exit_price,
                'outcome': outcome,
                'profit_loss': profit_loss,
                'entry_time': trade['entry_time'].strftime('%Y-%m-%d %H:%M:%S'),
                'exit_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            closed_trades.append(closed_trade)
            logging.info(f"Closed {trade['symbol']} | Outcome: {outcome} | P/L: {profit_loss:.2f} USDT")
            print(f"Closed {trade['symbol']} | {outcome} | P/L: {profit_loss:.2f} USDT")
            send_telegram_message(
                f"📊 *Trade Closed*\n"
                f"Symbol: {trade['symbol']}\n"
                f"Outcome: {outcome}\n"
                f"P/L: ${profit_loss:.2f}\n"
                f"Entry: ${trade['entry_price']}\n"
                f"Exit: ${exit_price}"
            )

        except Exception as e:
            logging.error(f"Monitor {trade['symbol']} error: {e}")

    for trade in trades_to_close:
        open_trades.remove(trade)

def scan_market():
    print(f"Scanning at {datetime.now()}")
    pairs = get_top_usdt_pairs()
    for symbol in pairs:
        trade = check_entry(symbol)
        if trade:
            open_trades.append(trade)
            logging.info(f"Opened {symbol} | Entry: {trade['entry_price']} | SL: {trade['stop_loss']} | TP: {trade['profit_target']}")
            print(f"Opened {symbol} | Entry: {trade['entry_price']} | SL: {trade['stop_loss']} | TP: {trade['profit_target']}")
            send_telegram_message(
                f"📈 *New Trade*\n"
                f"Symbol: {symbol}\n"
                f"Entry: ${trade['entry_price']}\n"
                f"Stop-Loss: ${trade['stop_loss']}\n"
                f"Target: ${trade['profit_target']}"
            )
        time.sleep(0.1)

    monitor_trades()
    print(f"Open trades: {len(open_trades)}")

if __name__ == "__main__":
    while True:
        scan_market()
        time.sleep(SCAN_INTERVAL)