import ccxt
import pandas as pd
import ta
import numpy as np
from datetime import datetime, timedelta
import time
import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from dotenv import load_dotenv
import logging

# === CONFIGURATION ===
load_dotenv()
POSITION_SIZE = 1000
SCAN_INTERVAL = 300
PAIR_LIMIT = 300
MAX_THREADS = 10
TRADE_COOLDOWN = 15
MAX_OPEN_TRADES = 10
TRADE_TIMEOUT = 15
RSI_MIN, RSI_MAX = 40, 70
VOLUME_MULTIPLIER = 1.0
USE_DYNAMIC_EXITS = True
MIN_RISK_REWARD = 1.5  # Minimum risk-reward ratio
MIN_PRICE_DIFF = 0.002  # Minimum 0.2% difference for stop-loss/target
CSV_FILE = 'trade_history.csv'
ERROR_LOG = 'errors.log'
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# Setup logging
logging.basicConfig(filename=ERROR_LOG, level=logging.ERROR, format='%(asctime)s - %(message)s')

# Initialize Binance exchange
exchange = ccxt.binance({'enableRateLimit': True})

# Initialize trade lists
open_trades = []
closed_trades = []


# Determine decimal places based on price
def get_decimal_places(price):
    if price < 1:
        return 4
    elif price < 100:
        return 2
    return 0


# Send Telegram notification
def send_telegram_notification(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': 'Markdown'
        }
        response = requests.post(url, json=payload)
        if response.status_code != 200:
            logging.error(f"Telegram API error: {response.json()}")
    except Exception as e:
        logging.error(f"Telegram notification error: {e}")


# Create CSV file
def init_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(
                ['Symbol', 'Entry_Price', 'Exit_Price', 'Outcome', 'Profit_Loss_USDT', 'Entry_Time', 'Exit_Time'])


# Fetch top USDT pairs
def get_top_usdt_pairs(limit=PAIR_LIMIT):
    for attempt in range(3):
        try:
            tickers = exchange.fetch_tickers()
            usdt_pairs = [
                symbol for symbol in tickers
                if symbol.endswith('USDT') and tickers[symbol].get('quoteVolume', 0) > 0
            ]
            sorted_pairs = sorted(
                usdt_pairs,
                key=lambda x: tickers[x].get('quoteVolume', 0),
                reverse=True
            )
            return sorted_pairs[:limit]
        except Exception as e:
            logging.error(f"Fetch pairs error (attempt {attempt + 1}): {e}")
            time.sleep(2)
    return []


# Fetch candlestick data
DATA_CACHE = {}


def fetch_klines(symbol, timeframe, limit=100):
    cache_key = f"{symbol}_{timeframe}"
    current_time = int(time.time() * 1000)
    cache_duration = 300000 if timeframe == '5m' else 900000

    if cache_key in DATA_CACHE:
        cached_data, timestamp = DATA_CACHE[cache_key]
        if current_time - timestamp < cache_duration:
            return cached_data

    for attempt in range(3):
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(
                float)
            DATA_CACHE[cache_key] = (df, current_time)
            return df
        except Exception as e:
            logging.error(f"Fetch {symbol} {timeframe} error (attempt {attempt + 1}): {e}")
            time.sleep(2)
    return None


# Clear cache
def clear_cache():
    global DATA_CACHE
    current_time = int(time.time() * 1000)
    DATA_CACHE = {k: v for k, v in DATA_CACHE.items() if current_time - v[1] < 3600000}


# Calculate indicators
def calculate_indicators(df):
    df['ema20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
    df['ema200'] = ta.trend.EMAIndicator(df['close'], window=200).ema_indicator()
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df['volume_ma20'] = df['volume'].rolling(window=20).mean()
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    df['atr_ma20'] = df['atr'].rolling(window=20).mean()
    return df


# Check bullish candlestick
def is_bullish_candle(row):
    return row['close'] > row['open']


# Check if trade is allowed
def can_open_trade(symbol):
    for trade in open_trades:
        if trade['symbol'] == symbol:
            return False
    now = datetime.now()
    for trade in closed_trades:
        if trade['symbol'] == symbol:
            exit_time = datetime.strptime(trade['exit_time'], '%Y-%m-%d %H:%M:%S')
            if now - exit_time < timedelta(minutes=TRADE_COOLDOWN):
                return False
    return True


# Process a single pair
def process_pair(symbol):
    if len(open_trades) >= MAX_OPEN_TRADES or not can_open_trade(symbol):
        return None

    try:
        df_15m = fetch_klines(symbol, '15m', limit=200)
        if df_15m is None or len(df_15m) < 200:
            return None
        df_15m = calculate_indicators(df_15m)

        latest_15m = df_15m.iloc[-1]
        if pd.isna(latest_15m['ema200']) or latest_15m['ema20'] <= latest_15m['ema200']:
            return None

        df_5m = fetch_klines(symbol, '5m', limit=100)
        if df_5m is None or len(df_5m) < 100:
            return None
        df_5m = calculate_indicators(df_5m)

        latest = df_5m.iloc[-1]
        previous = df_5m.iloc[-2]

        price_cross = previous['close'] < previous['ema20'] and latest['close'] > latest['ema20']
        rsi_condition = RSI_MIN < latest['rsi'] < RSI_MAX and latest['rsi'] > previous['rsi']
        volume_condition = latest['volume'] > VOLUME_MULTIPLIER * latest['volume_ma20']
        atr_condition = latest['atr'] > latest['atr_ma20']
        bullish_candle = is_bullish_candle(latest)

        if price_cross and rsi_condition and volume_condition and atr_condition and bullish_candle:
            entry_price = latest['close']
            decimals = get_decimal_places(entry_price)

            if USE_DYNAMIC_EXITS:
                stop_loss = previous['low']
                # Fallback to fixed if stop_loss equals entry_price
                if abs(entry_price - stop_loss) < entry_price * MIN_PRICE_DIFF:
                    stop_loss = entry_price * (1 - 0.004)
                    logging.error(f"Invalid dynamic stop-loss for {symbol}: using fixed 0.4%")
                profit_target = entry_price + 2 * (entry_price - stop_loss)
            else:
                stop_loss = entry_price * (1 - 0.004)
                profit_target = entry_price * (1 + 0.008)

            # Validate risk-reward and price differences
            risk = entry_price - stop_loss
            reward = profit_target - entry_price
            if risk <= 0 or reward <= 0 or reward / risk < MIN_RISK_REWARD or \
                    abs(entry_price - stop_loss) < entry_price * MIN_PRICE_DIFF or \
                    abs(profit_target - entry_price) < entry_price * MIN_PRICE_DIFF:
                logging.error(
                    f"Invalid trade for {symbol}: entry=${entry_price}, stop=${stop_loss}, target=${profit_target}")
                return None

            return {
                'symbol': symbol,
                'entry_price': round(entry_price, decimals),
                'stop_loss': round(stop_loss, decimals),
                'profit_target': round(profit_target, decimals),
                'rsi': round(latest['rsi'], 2),
                'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'current_price': round(entry_price, decimals),
                'unrealized_pl': 0.0
            }
        return None
    except Exception as e:
        logging.error(f"Process {symbol} error: {e}")
        return None


# Monitor open trades
def monitor_trades():
    global open_trades, closed_trades
    trades_to_remove = []
    now = datetime.now()

    if not open_trades:
        return

    try:
        symbols = [trade['symbol'] for trade in open_trades]
        tickers = exchange.fetch_tickers(symbols)

        for trade in open_trades:
            symbol = trade['symbol']
            if symbol not in tickers:
                continue

            current_price = tickers[symbol].get('bid')
            if current_price is None:
                continue

            entry_time = datetime.strptime(trade['time'], '%Y-%m-%d %H:%M:%S')
            decimals = get_decimal_places(trade['entry_price'])

            if now - entry_time > timedelta(minutes=TRADE_TIMEOUT):
                outcome = 'Timeout'
                exit_price = current_price
            elif current_price <= trade['stop_loss']:
                outcome = 'Loss'
                exit_price = trade['stop_loss']
            elif current_price >= trade['profit_target']:
                outcome = 'Win'
                exit_price = trade['profit_target']
            else:
                qty = POSITION_SIZE / trade['entry_price']
                unrealized_pl = qty * (current_price - trade['entry_price'])
                trade['current_price'] = round(current_price, decimals)
                trade['unrealized_pl'] = round(unrealized_pl, 2)
                continue

            qty = POSITION_SIZE / trade['entry_price']
            profit_loss = qty * (exit_price - trade['entry_price'])
            closed_trade = {
                'symbol': symbol,
                'entry_price': trade['entry_price'],
                'exit_price': round(exit_price, decimals),
                'outcome': outcome,
                'profit_loss': round(profit_loss, 2),
                'entry_time': trade['time'],
                'exit_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            closed_trades.append(closed_trade)
            trades_to_remove.append(trade)

            emoji = '✅' if outcome == 'Win' else '❌' if outcome == 'Loss' else '⏰'
            message = (
                f"{emoji} *Trade Closed*\n"
                f"Symbol: {symbol}\n"
                f"Outcome: {outcome}\n"
                f"Profit/Loss: ${profit_loss:.2f}\n"
                f"Entry: ${trade['entry_price']}\n"
                f"Exit: ${exit_price}\n"
                f"Time: {closed_trade['exit_time']}"
            )
            send_telegram_notification(message)

    except Exception as e:
        logging.error(f"Monitor trades error: {e}")

    open_trades[:] = [t for t in open_trades if t not in trades_to_remove]

    if trades_to_remove:
        with open(CSV_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            for trade in closed_trades[-len(trades_to_remove):]:
                writer.writerow([
                    trade['symbol'],
                    trade['entry_price'],
                    trade['exit_price'],
                    trade['outcome'],
                    trade['profit_loss'],
                    trade['entry_time'],
                    trade['exit_time']
                ])


# Scan for opportunities
def scan_opportunities():
    print(f"\n📈 [SCAN] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    usdt_pairs = get_top_usdt_pairs(limit=PAIR_LIMIT)

    if not usdt_pairs:
        print("⚠️ No USDT pairs available.")
        return

    print(f"Scanning {len(usdt_pairs)} pairs...")
    new_opportunities = []

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        future_to_symbol = {executor.submit(process_pair, symbol): symbol for symbol in usdt_pairs}
        for future in as_completed(future_to_symbol):
            result = future.result()
            if result:
                open_trades.append(result)
                new_opportunities.append(result)

                message = (
                    f"📈 *New Trading Opportunity*\n"
                    f"Symbol: {result['symbol']}\n"
                    f"Entry Price: ${result['entry_price']}\n"
                    f"Stop-Loss: ${result['stop_loss']}\n"
                    f"Profit Target: ${result['profit_target']}\n"
                    f"RSI: {result['rsi']}\n"
                    f"Time: {result['time']}"
                )
                send_telegram_notification(message)

    monitor_trades()

    if new_opportunities:
        print("\n=== New Trading Opportunities ===")
        for opp in new_opportunities:
            print(f"Symbol: {opp['symbol']}")
            print(f"Entry Price: ${opp['entry_price']}")
            print(f"Stop-Loss: ${opp['stop_loss']}")
            print(f"Profit Target: ${opp['profit_target']}")
            print(f"RSI: {opp['rsi']}")
            print(f"Time: {opp['time']}")
            print("-" * 40)

    if open_trades:
        print("\n=== Open Trades ===")
        for trade in open_trades:
            print(f"Symbol: {trade['symbol']}")
            print(f"Entry Price: ${trade['entry_price']}")
            print(f"Current Price: ${trade['current_price']}")
            print(f"Unrealized P/L: ${trade['unrealized_pl']}")
            print(f"Stop-Loss: ${trade['stop_loss']}")
            print(f"Profit Target: ${trade['profit_target']}")
            print(f"Entry Time: {trade['time']}")
            print("-" * 40)

    if closed_trades:
        print("\n=== Recently Closed Trades ===")
        for trade in closed_trades[-5:]:
            print(f"Symbol: {trade['symbol']}")
            print(f"Entry Price: ${trade['entry_price']}")
            print(f"Exit Price: ${trade['exit_price']}")
            print(f"Outcome: {trade['outcome']}")
            print(f"Profit/Loss: ${trade['profit_loss']}")
            print(f"Entry Time: {trade['entry_time']}")
            print(f"Exit Time: {trade['exit_time']}")
            print("-" * 40)

    if not (new_opportunities or open_trades or closed_trades):
        print("No new opportunities or active trades.")

    clear_cache()


# Run scanner
if __name__ == "__main__":
    init_csv()
    while True:
        scan_opportunities()
        print(f"⏳ Waiting for next scan ({SCAN_INTERVAL // 60} minutes)...")
        time.sleep(SCAN_INTERVAL)