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
SCAN_INTERVAL = 300  # 5 minutes
PAIR_LIMIT = 300
MAX_THREADS = 10
TRADE_COOLDOWN = 15  # minutes
MAX_OPEN_TRADES = 10
MIN_TRADE_DURATION = 60  # Seconds to prevent immediate stop-loss triggering
RSI_MIN, RSI_MAX = 45, 65  # RSI range for 15m chart
VOLUME_MULTIPLIER = 1.5  # Volume must be 1.5x its 20-period MA
MIN_ATR_RATIO = 0.005  # ATR > 0.5% of price on 15m chart
FIXED_PROFIT_TARGET_PERCENT = 0.01  # 1% fixed profit target
FIXED_STOP_LOSS_PERCENT = 0.004 # Fixed 0.4% stop loss (can adjust if needed)
MIN_PRICE_DIFF = 0.003  # 0.3% minimum difference between entry and SL/TP

CSV_FILE = 'trade_history.csv'
ERROR_LOG = 'errors.log'
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID_CHANNEL')

# Setup logging
logging.basicConfig(filename=ERROR_LOG, level=logging.ERROR, format='%(asctime)s - %(message)s')

# Initialize Binance exchange
exchange = ccxt.binance({'enableRateLimit': True})

# Initialize trade lists
open_trades = []
closed_trades = []


# Get decimal places based on price
def get_decimal_places(price):
    if price < 0.01:
        return 6
    elif price < 1:
        return 4
    elif price < 100:
        return 2
    return 0


# Send Telegram notification
def send_telegram_notification(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        # print("Telegram credentials not set. Skipping notification.")
        return
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


# Fetch top USDT pairs (Modified to check for active spot markets)
def get_top_usdt_pairs(limit=PAIR_LIMIT):
    print("Fetching and filtering markets...")
    all_markets = None
    active_spot_usdt_symbols = []

    for attempt in range(3): # Retry fetching markets
        try:
            # Load markets (uses caching if available)
            exchange.load_markets()
            all_markets = exchange.markets
            if all_markets:
                break # Success
            else:
                 print(f"Attempt {attempt + 1}: No markets loaded, retrying...")
                 time.sleep(3)
        except Exception as e:
            logging.error(f"Fetch markets error (attempt {attempt + 1}): {e}")
            print(f"Attempt {attempt + 1}: Error fetching markets, retrying...")
            time.sleep(3)

    if not all_markets:
        logging.error("Failed to load markets after multiple attempts.")
        print("⚠️ Failed to load markets after multiple attempts.")
        return []

    # Filter for active, spot, USDT pairs that are currently trading
    for symbol, market_data in all_markets.items():
        if (market_data.get('spot', False) and # Check if it's a spot market
            market_data.get('active', False) and # Check if market is marked active by ccxt
            symbol.endswith('USDT') and        # Check if it's a USDT pair
            market_data.get('info') and        # Ensure 'info' field exists
            market_data['info'].get('status') == 'TRADING'): # **Crucial: Check Binance specific status**
            active_spot_usdt_symbols.append(symbol)

    if not active_spot_usdt_symbols:
         print("⚠️ No actively trading USDT spot pairs found after filtering.")
         return []

    print(f"Found {len(active_spot_usdt_symbols)} active USDT spot pairs. Fetching ticker data...")

    # Now fetch tickers only for these active pairs to get volume data
    for attempt in range(3): # Retry fetching tickers
        try:
            # Important: Fetch tickers for the *filtered list*
            tickers = exchange.fetch_tickers(symbols=active_spot_usdt_symbols)

            # Filter out any tickers that failed to fetch or have zero volume
            valid_tickers = {
                symbol: data for symbol, data in tickers.items()
                if data and data.get('quoteVolume', 0) > 0
            }

            if not valid_tickers:
                print(f"Attempt {attempt + 1}: No valid tickers with volume found for active pairs, retrying...")
                time.sleep(2)
                continue # Retry fetching tickers

            # Sort the valid tickers by quoteVolume
            sorted_symbols = sorted(
                valid_tickers.keys(),
                key=lambda symbol: valid_tickers[symbol].get('quoteVolume', 0),
                reverse=True
            )
            print(f"Sorted {len(sorted_symbols)} pairs by volume.")
            return sorted_symbols[:limit] # Return the top pairs by volume

        except Exception as e:
            logging.error(f"Fetch tickers error (attempt {attempt + 1}): {e}")
            print(f"Attempt {attempt + 1}: Error fetching tickers, retrying...")
            time.sleep(2)

    logging.error("Failed to fetch valid ticker data after multiple attempts.")
    print("⚠️ Failed to fetch valid ticker data after multiple attempts.")
    return []


# Fetch candlestick data
DATA_CACHE = {}


def fetch_klines(symbol, timeframe, limit=100):
    cache_key = f"{symbol}_{timeframe}"
    current_time = int(time.time() * 1000)
    # Use 15 min cache duration (900000 ms)
    cache_duration = 900000 # 15 minutes in milliseconds

    if cache_key in DATA_CACHE:
        cached_data, timestamp = DATA_CACHE[cache_key]
        if current_time - timestamp < cache_duration:
            # print(f"Using cached data for {symbol} {timeframe}") # Optional debug print
            return cached_data.copy() # Return copy to prevent modification issues

    for attempt in range(3):
        try:
            # print(f"Fetching fresh data for {symbol} {timeframe}") # Optional debug print
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not ohlcv: # Handle case where API returns empty list
                 return None
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(
                float)

            # Ensure volume is not zero to avoid division errors later
            df = df[df['volume'] > 0]
            if df.empty:
                return None

            DATA_CACHE[cache_key] = (df.copy(), current_time) # Store copy in cache
            return df
        except ccxt.NetworkError as e:
            logging.error(f"Fetch {symbol} {timeframe} NetworkError (attempt {attempt + 1}): {e}")
            time.sleep(5) # Longer sleep for network errors
        except ccxt.ExchangeError as e:
             logging.error(f"Fetch {symbol} {timeframe} ExchangeError (attempt {attempt + 1}): {e}")
             # Don't retry on certain exchange errors like invalid symbol
             if 'invalid symbol' in str(e).lower():
                  return None
             time.sleep(2)
        except Exception as e:
            logging.error(f"Fetch {symbol} {timeframe} general error (attempt {attempt + 1}): {e}")
            time.sleep(2)
    return None


# Clear cache periodically
def clear_cache():
    global DATA_CACHE
    current_time = int(time.time() * 1000)
    # Keep cache for 1 hour (3600000 ms)
    DATA_CACHE = {k: v for k, v in DATA_CACHE.items() if current_time - v[1] < 3600000}


# Calculate indicators
def calculate_indicators(df):
    # Check if DataFrame is valid and has enough rows
    if df is None or df.empty or len(df) < 20: # Need at least 20 for volume MA
        return None
    try:
        df['ema20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
        df['ema200'] = ta.trend.EMAIndicator(df['close'], window=200).ema_indicator()
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        # Calculate volume MA only where volume is positive to avoid issues
        df['volume_ma20'] = df['volume'].rolling(window=20).mean()
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
        df['atr_ma20'] = df['atr'].rolling(window=20).mean()
        # Drop rows with NaN values created by indicators
        df.dropna(inplace=True)
        return df
    except Exception as e:
         logging.error(f"Indicator calculation error: {e} on df length {len(df) if df is not None else 'None'}")
         return None


# Check bullish candlestick
def is_bullish_candle(row):
    # Ensure 'close' and 'open' exist and are not NaN
    if 'close' in row and 'open' in row and pd.notna(row['close']) and pd.notna(row['open']):
        return row['close'] > row['open']
    return False

# Check if trade is allowed
def can_open_trade(symbol):
    # Check if already in an open trade for this symbol
    for trade in open_trades:
        if trade['symbol'] == symbol:
            return False

    # Check cooldown period from last closed trade for this symbol
    now = datetime.now()
    for trade in closed_trades:
        if trade['symbol'] == symbol:
            try:
                # Use timezone-aware comparison if possible, otherwise naive
                exit_time_str = trade.get('exit_time')
                if exit_time_str:
                     exit_time = datetime.strptime(exit_time_str, '%Y-%m-%d %H:%M:%S')
                     if now - exit_time < timedelta(minutes=TRADE_COOLDOWN):
                            return False
            except Exception as e:
                 logging.error(f"Error parsing exit time for cooldown check: {e}")
                 # Fallback: if parsing fails, perhaps disallow trade to be safe
                 # return False
    return True


# Process a single pair based ONLY on 15m timeframe
def process_pair(symbol):
    if len(open_trades) >= MAX_OPEN_TRADES or not can_open_trade(symbol):
        return None

    try:
        # --- Fetch and Analyse ONLY 15m Data ---
        df_15m = fetch_klines(symbol, '15m', limit=220) # Fetch slightly more for indicator NaNs
        df_15m = calculate_indicators(df_15m)

        # Ensure we have data and indicators after calculations
        if df_15m is None or df_15m.empty or len(df_15m) < 2:
            # print(f"Skipping {symbol}: Insufficient 15m data or indicators.") # Optional debug
            return None

        latest_15m = df_15m.iloc[-1]
        previous_15m = df_15m.iloc[-2]

        # --- Apply Entry Conditions on 15m Data ---

        # 1. Trend Filter: Price > EMA20 and EMA20 > EMA200 (Uptrend)
        trend_condition = (latest_15m['close'] > latest_15m['ema20'] and
                           latest_15m['ema20'] > latest_15m['ema200'])
        if not trend_condition:
            return None

        # 2. Price crossing EMA20 upwards
        price_cross = previous_15m['close'] < previous_15m['ema20'] and latest_15m['close'] > latest_15m['ema20']

        # 3. RSI Condition
        rsi_condition = RSI_MIN < latest_15m['rsi'] < RSI_MAX and latest_15m['rsi'] > previous_15m['rsi']

        # 4. Volume Condition
        volume_condition = latest_15m['volume'] > VOLUME_MULTIPLIER * latest_15m['volume_ma20']

        # 5. ATR Condition (Volatility Check)
        atr_condition = latest_15m['atr'] > latest_15m['atr_ma20'] and latest_15m['atr'] > MIN_ATR_RATIO * latest_15m['close']

        # 6. Bullish Candle
        bullish_candle = is_bullish_candle(latest_15m)

        # Check if ALL conditions are met
        if price_cross and rsi_condition and volume_condition and atr_condition and bullish_candle:
            entry_price = latest_15m['close']
            if entry_price < 0.0001:  # Skip extremely low-price assets to avoid precision issues
                return None
            decimals = get_decimal_places(entry_price)

            # --- Define Fixed Exit Levels ---
            stop_loss = entry_price * (1 - FIXED_STOP_LOSS_PERCENT)
            profit_target = entry_price * (1 + FIXED_PROFIT_TARGET_PERCENT)

            # Validate trade exit levels
            if entry_price <= stop_loss or profit_target <= entry_price or \
               abs(entry_price - stop_loss) < entry_price * MIN_PRICE_DIFF or \
               abs(profit_target - entry_price) < entry_price * MIN_PRICE_DIFF:
                logging.warning(
                    f"Invalid SL/TP for {symbol}: entry=${entry_price:.{decimals}f}, stop=${stop_loss:.{decimals}f}, target=${profit_target:.{decimals}f}. Skipping trade."
                )
                return None

            return {
                'symbol': symbol,
                'entry_price': round(entry_price, decimals),
                'stop_loss': round(stop_loss, decimals),
                'profit_target': round(profit_target, decimals),
                # Trailing stop related fields removed
                'rsi': round(latest_15m['rsi'], 2),
                'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'current_price': round(entry_price, decimals), # Initial value
                'unrealized_pl': 0.0 # Initial value
            }
        return None
    except KeyError as e:
         logging.error(f"Process {symbol} KeyError: {e}. Check indicator calculation or DataFrame columns.")
         return None
    except IndexError as e:
         logging.error(f"Process {symbol} IndexError: {e}. Likely insufficient data after indicator calculation.")
         return None
    except Exception as e:
        logging.error(f"Process {symbol} unexpected error: {e}")
        # Optionally print traceback for debugging
        # import traceback
        # logging.error(traceback.format_exc())
        return None


# Monitor open trades
def monitor_trades():
    global open_trades, closed_trades
    trades_to_remove = []
    now = datetime.now()

    if not open_trades:
        return

    try:
        symbols = list(set([trade['symbol'] for trade in open_trades])) # Use set for unique symbols
        if not symbols:
            return
        tickers = exchange.fetch_tickers(symbols)

        for i in range(len(open_trades) -1, -1, -1): # Iterate backwards for safe removal
            trade = open_trades[i]
            symbol = trade['symbol']
            if symbol not in tickers or tickers[symbol] is None:
                logging.warning(f"Could not fetch ticker data for open trade: {symbol}. Skipping check.")
                continue

            # Use 'last' price as a fallback if 'bid' (for longs) is not available
            current_price = tickers[symbol].get('bid') or tickers[symbol].get('last')
            if current_price is None or current_price <= 0: # Ensure valid price
                 logging.warning(f"Invalid current price ({current_price}) for {symbol}. Skipping check.")
                 continue

            entry_time = datetime.strptime(trade['time'], '%Y-%m-%d %H:%M:%S')
            decimals = get_decimal_places(trade['entry_price'])

            # Prevent immediate stop-loss check right after entry
            time_elapsed = (now - entry_time).total_seconds()
            allow_stop_loss_check = time_elapsed >= MIN_TRADE_DURATION

            # --- Check Exit Conditions (NO TRAILING STOP) ---
            exit_price = None
            outcome = None

            # 1. Check Stop Loss
            if allow_stop_loss_check and current_price <= trade['stop_loss']:
                outcome = 'Loss'
                exit_price = trade['stop_loss'] # Exit at defined stop-loss level
            # 2. Check Profit Target
            elif current_price >= trade['profit_target']:
                outcome = 'Win'
                exit_price = trade['profit_target'] # Exit at defined profit-target level

            # If no exit condition met, update P/L and continue
            if outcome is None:
                qty = POSITION_SIZE / trade['entry_price']
                unrealized_pl = qty * (current_price - trade['entry_price'])
                trade['current_price'] = round(current_price, decimals)
                trade['unrealized_pl'] = round(unrealized_pl, 2)
                continue

            # --- Process Trade Closure ---
            qty = POSITION_SIZE / trade['entry_price']
            profit_loss = qty * (exit_price - trade['entry_price'])
            exit_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            closed_trade = {
                'symbol': symbol,
                'entry_price': trade['entry_price'],
                'exit_price': round(exit_price, decimals),
                'outcome': outcome,
                'profit_loss': round(profit_loss, 2),
                'entry_time': trade['time'],
                'exit_time': exit_time_str
            }
            closed_trades.append(closed_trade)
            trades_to_remove.append(trade) # Mark for removal

            # Log to CSV immediately after closing
            try:
                 with open(CSV_FILE, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        closed_trade['symbol'],
                        closed_trade['entry_price'],
                        closed_trade['exit_price'],
                        closed_trade['outcome'],
                        closed_trade['profit_loss'],
                        closed_trade['entry_time'],
                        closed_trade['exit_time']
                    ])
            except IOError as e:
                 logging.error(f"Error writing closed trade to CSV: {e}")


            emoji = '✅' if outcome == 'Win' else '❌'
            message = (
                f"{emoji} *Trade Closed*\n"
                f"Symbol: {symbol}\n"
                f"Outcome: {outcome}\n"
                f"Profit/Loss: ${profit_loss:.2f}\n"
                f"Entry: ${trade['entry_price']}\n"
                f"Exit: ${exit_price:.{decimals}f}\n" # Use decimals for exit price formatting
                f"Time: {exit_time_str}"
            )
            send_telegram_notification(message)

            # Remove the trade from the open_trades list immediately
            del open_trades[i]


    except ccxt.NetworkError as e:
         logging.error(f"Monitor trades NetworkError: {e}")
    except ccxt.ExchangeError as e:
         logging.error(f"Monitor trades ExchangeError: {e}")
    except Exception as e:
        logging.error(f"Monitor trades unexpected error: {e}")
        # import traceback
        # logging.error(traceback.format_exc())

    # Redundant removal logic removed as it's handled during iteration now
    # open_trades[:] = [t for t in open_trades if t not in trades_to_remove]


# Scan for opportunities
def scan_opportunities():
    print(f"\n{'='*10} SCANNING {'='*10} [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
    print(f"Current Open Trades: {len(open_trades)} / {MAX_OPEN_TRADES}")
    usdt_pairs = get_top_usdt_pairs(limit=PAIR_LIMIT)

    if not usdt_pairs:
        print("⚠️ No USDT pairs found or error fetching pairs.")
        return

    print(f"Scanning {len(usdt_pairs)} potential pairs...")
    new_opportunities_count = 0

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        future_to_symbol = {executor.submit(process_pair, symbol): symbol for symbol in usdt_pairs}
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                result = future.result()
                if result:
                    # Double-check we are not exceeding max trades before adding
                    if len(open_trades) < MAX_OPEN_TRADES:
                        open_trades.append(result)
                        new_opportunities_count += 1
                        print(f"✅ Found Opportunity: {result['symbol']} @ ${result['entry_price']}")

                        message = (
                            f"📈 *New Trading Opportunity*\n"
                            f"Symbol: {result['symbol']}\n"
                            f"Entry Price: ${result['entry_price']}\n"
                            f"Stop-Loss: ${result['stop_loss']}\n"
                            f"Profit Target (1%): ${result['profit_target']}\n" # Clarified TP
                            f"15m RSI: {result['rsi']}\n" # Clarified timeframe
                            f"Time: {result['time']}"
                        )
                        send_telegram_notification(message)
                    else:
                         print(f"⚠️ Max open trades ({MAX_OPEN_TRADES}) reached. Skipping opportunity for {result['symbol']}.")
                         # Optional: Log skipped opportunity
                         logging.info(f"Skipped opportunity for {result['symbol']} due to max open trades limit.")


            except Exception as e:
                 logging.error(f"Error processing result for {symbol}: {e}")
                 # import traceback
                 # logging.error(traceback.format_exc())


    print(f"Scan Complete. Found {new_opportunities_count} new opportunities.")

    # --- Monitoring is done separately in the main loop now ---
    # monitor_trades() # Moved to main loop for consistent checks


# Display Status (can be called periodically)
def display_status():
    print(f"\n{'='*10} STATUS UPDATE {'='*10} [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
    if open_trades:
        print(f"\n--- Open Trades ({len(open_trades)}) ---")
        for trade in open_trades:
             # Calculate current P/L %
            pnl_percent = ((trade['current_price'] - trade['entry_price']) / trade['entry_price']) * 100 if trade['entry_price'] != 0 else 0
            print(f"Symbol: {trade['symbol']}")
            print(f"  Entry: ${trade['entry_price']} ({trade['time']})")
            print(f"  Current: ${trade['current_price']} (P/L: ${trade['unrealized_pl']:.2f} / {pnl_percent:.2f}%)")
            print(f"  SL: ${trade['stop_loss']} | TP: ${trade['profit_target']}")
            print("-" * 20)
    else:
        print("\n--- No Open Trades ---")

    if closed_trades:
        print(f"\n--- Recently Closed Trades (Last 5) ---")
        # Calculate summary stats for closed trades if needed (e.g., win rate, total P/L)
        for trade in closed_trades[-5:]:
            print(f"Symbol: {trade['symbol']} ({trade['outcome']})")
            print(f"  Entry: ${trade['entry_price']} ({trade['entry_time']})")
            print(f"  Exit: ${trade['exit_price']} ({trade['exit_time']})")
            print(f"  P/L: ${trade['profit_loss']:.2f}")
            print("-" * 20)


# Run scanner
if __name__ == "__main__":
    init_csv()
    print("Bot Starting...")
    print(f"Target Profit: {FIXED_PROFIT_TARGET_PERCENT*100}%")
    print(f"Stop Loss: {FIXED_STOP_LOSS_PERCENT*100}%")
    print(f"Max Open Trades: {MAX_OPEN_TRADES}")
    send_telegram_notification("🚀 Trading Bot Started")

    last_scan_time = 0
    last_monitor_time = 0
    MONITOR_INTERVAL = 10 # Check open trades every 10 seconds

    while True:
        current_time = time.time()

        # Scan for new opportunities periodically
        if current_time - last_scan_time >= SCAN_INTERVAL:
             try:
                  scan_opportunities()
                  last_scan_time = current_time
                  display_status() # Display status after scanning
                  clear_cache() # Clear cache after a scan cycle
             except Exception as e:
                  logging.error(f"Error in main scan loop: {e}")
                  # import traceback
                  # logging.error(traceback.format_exc())
             print(f"\n⏳ Next scan in ~{SCAN_INTERVAL // 60} minutes...")


        # Monitor existing trades more frequently
        if current_time - last_monitor_time >= MONITOR_INTERVAL:
            if open_trades: # Only monitor if there are trades
                try:
                    # print(f"DEBUG: Monitoring {len(open_trades)} trades...") # Optional debug
                    monitor_trades()
                    last_monitor_time = current_time
                except Exception as e:
                    logging.error(f"Error in main monitor loop: {e}")
                    # import traceback
                    # logging.error(traceback.format_exc())


        # Prevent busy-waiting
        sleep_time = min(SCAN_INTERVAL - (current_time - last_scan_time),
                         MONITOR_INTERVAL - (current_time - last_monitor_time),
                         1) # Sleep for at least 1 second
        time.sleep(max(1, sleep_time)) # Ensure sleep time is positive