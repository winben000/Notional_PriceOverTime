import asyncio
import ccxt.pro as ccxtpro
import os
import json
import argparse
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import aiohttp
import time
import math

# Load environment variables
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_HYPER_BOT_TOKEN')
TELEGRAM_CHAT_ID = ""

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Track logged timestamps to avoid duplicate alerts
logged_timestamps = set()
all_trades = []  # Store all trades
time_windows = []  # For EMA
buy_volumes = []
sell_volumes = []
ratio_values = []
ema_values = []
last_alert_ema = None
current_ema_range = None  # Track the current EMA range

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Trade alert monitor')
    parser.add_argument('--config', '-c', type=str, default="config/trade_alert.json",
                        help='Path to configuration file (default: config/trade_alert.json)')
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from a JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise

async def send_telegram_message(message):
    """Send a message to Telegram."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': message}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            result = await response.json()
            if response.status != 200:
                logger.error(f"Telegram API error: {result}")
            else:
                logger.info(f"Message sent successfully: {result}")

def setup_exchange(exchange_id, symbol=None):
    """Set up exchange based on exchange ID from config."""
    exchange_class = getattr(ccxtpro, exchange_id)
    
    # Determine if this is a perpetual contract by checking symbol format
    is_perpetual = symbol and ':USDT' in symbol
    market_type = 'swap' if is_perpetual else 'spot'
    
    exchange_config = {
        'options': {
            'defaultType': market_type,
        }
    }

    return exchange_class(exchange_config)

def calculate_ema(series, window):
    """Calculate Exponential Moving Average."""
    return series.ewm(span=window, adjust=False).mean()

async def filter_large_trades(grouped_trades, symbol, min_trade_amount, exchange_name, market_type_indicator=""):
    """Filter and alert for large trades, including mean trade amount of large trades in the past 60 minutes."""
    global all_trades
    
    for (timestamp, side), group in grouped_trades:
        total_amount = group['amount'].sum()
        if total_amount >= min_trade_amount:
            avg_price = group['price'].mean()
            if (timestamp, side) not in logged_timestamps:
                icon = '🟢' if side.lower() == 'buy' else '🔴'
                
                # Calculate mean of large trades in the past 60 minutes
                current_time = int(time.time())
                lookback_period = 3600  # 60 minutes in seconds
                
                trades_df = pd.DataFrame(all_trades)
                mean_large_trades = 0
                if not trades_df.empty:
                    trades_df['timestamp_sec'] = trades_df['timestamp'] // 1000
                    recent_trades = trades_df[trades_df['timestamp_sec'] >= (current_time - lookback_period)]
                    
                    # Filter only large trades
                    large_trades = recent_trades[recent_trades['amount'] >= min_trade_amount]
                    if not large_trades.empty:
                        mean_large_trades = large_trades['amount'].mean()
                
                warning = "⚠️ LARGE TRADE ALERT! ⚠️" if total_amount >= 90000 else "LARGE TRADE ALERT!"
                message = (
                    f"{warning}\n"
                    f"📊 {exchange_name.upper()} {market_type_indicator}\n"
                    f"{icon} {side.upper()} {total_amount/1000:.2f}k@{avg_price:.4f} [mean large trades 60m: {mean_large_trades/1000:.2f}k]\n"
                    f"-----------------------------------------------------"
                )
                logger.info(message)
                await send_telegram_message(message)
                logged_timestamps.add((timestamp, side))

async def process_ema_ratio(symbol, exchange_name, market_type_indicator="", timeframe=300, ema_window=15):
    """Calculate EMA ratio every 5 minutes based on the previous 5-minute trades, including 60-minute mean amounts."""
    global time_windows, buy_volumes, sell_volumes, ratio_values, ema_values, last_alert_ema, all_trades, current_ema_range
    retention_period = 7200  # 2 hours in seconds
    lookback_period = 3600  # 60 minutes in seconds
    
    logger.info(f"Starting EMA ratio calculation for {symbol} with window {ema_window}")
    
    while True:
        current_time = int(time.time())
        # Wait until the next 5-minute mark
        next_snapshot = (current_time // timeframe + 1) * timeframe
        await asyncio.sleep(next_snapshot - current_time)
        
        # Calculate mean trade amounts for buy and sell over the last 60 minutes
        trades_df_all = pd.DataFrame(all_trades)
        mean_amounts = 0
        if not trades_df_all.empty:
            trades_df_all['timestamp_sec'] = trades_df_all['timestamp'] // 1000
            recent_trades = trades_df_all[trades_df_all['timestamp_sec'] >= (next_snapshot - lookback_period)]
            mean_amounts = recent_trades['amount'].mean()
        
        # Filter trades from the last 5 minutes
        snapshot_start = next_snapshot - timeframe
        trades_df = pd.DataFrame(all_trades)
        if not trades_df.empty:
            trades_df['timestamp_sec'] = trades_df['timestamp'] // 1000
            snapshot_trades = trades_df[
                (trades_df['timestamp_sec'] >= snapshot_start) & 
                (trades_df['timestamp_sec'] < next_snapshot)
            ]
            
            # Calculate buy/sell volumes
            window_data = snapshot_trades.groupby('side')['amount'].sum().to_dict()
            buy_vol = window_data.get('buy', 0)
            sell_vol = window_data.get('sell', 0)
            
            # Calculate ratio
            if sell_vol > 0:
                ratio = buy_vol / sell_vol
            elif buy_vol > 0:
                ratio = 2.0
            else:
                ratio = 1.0
                
            time_windows.append(next_snapshot)
            buy_volumes.append(buy_vol)
            sell_volumes.append(sell_vol)
            ratio_values.append(ratio)
            
            if len(time_windows) <= ema_window:
                window_time = datetime.fromtimestamp(next_snapshot)
                logger.info(f"Building EMA data: Window {len(time_windows)}/{ema_window} - "
                           f"Time: {window_time.strftime('%H:%M:%S')}, "
                           f"Buy: {buy_vol/1000:.2f}k, Sell: {sell_vol/1000:.2f}k, Ratio: {ratio:.4f}")
            
            if len(time_windows) >= ema_window:
                ratios = pd.Series(ratio_values[-ema_window:])
                current_ema = calculate_ema(ratios, ema_window).iloc[-1]
                ema_values.append(current_ema)
                
                if len(ema_values) == 1:
                    logger.info(f"First EMA{ema_window} value calculated: {current_ema:.4f}")
                
                # Determine the current EMA range
                new_ema_range = None
                if current_ema <= 0.8:
                    new_ema_range = 1  # Range 1: <= 0.8
                elif 0.8 < current_ema < 1.2:
                    new_ema_range = 2  # Range 2: 0.8-1.2
                else:  # current_ema >= 1.2
                    new_ema_range = 3  # Range 3: >= 1.2
                
                # Check if the EMA has moved to a new range
                if current_ema_range is not None and new_ema_range != current_ema_range:
                    range_desc = {
                        1: "Bearish (≤0.8)",
                        2: "Neutral (0.8-1.2)",
                        3: "Bullish (≥1.2)"
                    }
                    window_time = datetime.fromtimestamp(next_snapshot)
                    alert_msg = (
                        f"⚠️ EMA RANGE CHANGE ⚠️ | {exchange_name.upper()} {market_type_indicator}\n"
                        f"EMA{ema_window}: {current_ema:.4f}\n"
                        f"Window Buy: {buy_vol/1000:.2f}k, Sell: {sell_vol/1000:.2f}k | Ratio: {ratio:.4f}\n"
                        f"Mean (60min): {mean_amounts/1000:.2f}k\n"
                        f"-----------------------------------------------------"
                    )
                    logger.info(alert_msg)
                    await send_telegram_message(alert_msg)
                
                # Update the current range and last_alert_ema
                current_ema_range = new_ema_range
                last_alert_ema = current_ema
                
                window_time = datetime.fromtimestamp(next_snapshot)
                logger.info(f"Snapshot: {window_time.strftime('%H:%M:%S')}, "
                           f"Buy: {buy_vol/1000:.2f}k, Sell: {sell_vol/1000:.2f}k, Ratio: {ratio:.4f} | "
                           f"EMA{ema_window}: {current_ema:.4f} (Range {new_ema_range})")
            
            # Keep only the last 2 * ema_window snapshots
            if len(time_windows) > ema_window * 2:
                time_windows = time_windows[-ema_window*2:]
                buy_volumes = buy_volumes[-ema_window*2:]
                sell_volumes = sell_volumes[-ema_window*2:]
                ratio_values = ratio_values[-ema_window*2:]
                ema_values = ema_values[-ema_window*2:]
                
        # Clear trades older than 2 hours
        all_trades[:] = [t for t in all_trades if (t['timestamp'] // 1000) >= (next_snapshot - retention_period)]

async def watch_trades(exchange, symbol, config):
    """Watch trades in real-time and collect them."""
    exchange_name = exchange.id
    min_trade_amount = float(config.get("min_trade_amount", 10000))
    enable_large_trades = config.get("enable_large_trades", True)
    
    # Determine if this is a perpetual contract
    is_perpetual = ':USDT' in symbol
    market_type_indicator = "PERP" if is_perpetual else ""
    
    logger.info(f"Starting to watch trades for {symbol}...")
    
    while True:
        try:
            trades = await exchange.watch_trades(symbol)
            if trades:
                trades_df = pd.DataFrame(trades)
                trades_df['amount'] = trades_df['amount'].astype(float)
                trades_df['price'] = trades_df['price'].astype(float)
                
                # If large trade monitoring is enabled
                if enable_large_trades:
                    # Group by timestamp and side
                    grouped_trades = trades_df.groupby(['timestamp', 'side'])
                    
                    # Filter large trades and alert
                    await filter_large_trades(grouped_trades, symbol, min_trade_amount, exchange_name, market_type_indicator)
                
                # Save all trades to array
                all_trades.extend(trades_df.to_dict('records'))
                
        except Exception as e:
            logger.error(f"Error watching trades: {str(e)}")
            await asyncio.sleep(10)

async def monitor_depth(exchange, symbol, config):
    """Monitor order book depth and send alerts when conditions are met."""
    exchange_name = exchange.id.upper()
    alert_interval = config.get("depth_alert_interval", 15 * 60)  # Default: 15 minutes
    depth_threshold = config.get("min_depth_notional", 25000)  # Default threshold
    
    # Determine if this is a perpetual contract
    is_perpetual = ':USDT' in symbol
    market_type_indicator = "PERP" if is_perpetual else ""
    
    logger.info(f"Starting depth monitoring for {symbol} with {alert_interval/60} minute intervals")
    
    while True:
        try:
            # Fetch order book
            order_book = await exchange.watch_order_book(symbol)
            mid_price = (order_book['bids'][0][0] + order_book['asks'][0][0]) / 2

            # Calculate 50 bps price range from mid-price (0.5%)
            bps_range = mid_price * 0.005

            # Calculate rounded prices for -50 bps and +50 bps from mid-price
            minus_50bps_px = math.floor((mid_price - bps_range) * 10000) / 10000  # Round down to 4 decimal places
            plus_50bps_px = math.ceil((mid_price + bps_range) * 10000) / 10000    # Round up to 4 decimal places

            # Calculate total notional within 50 bps range for both bids and asks after rounding prices
            total_bid_notional_50bps = sum(math.floor(p * 10000) / 10000 * q for p, q in order_book['bids'] if p >= minus_50bps_px)
            total_ask_notional_50bps = sum(math.ceil(p * 10000) / 10000 * q for p, q in order_book['asks'] if p <= plus_50bps_px)
            
            # Combined total notional
            total_notional_50bps = total_bid_notional_50bps + total_ask_notional_50bps

            # Calculate cumulative quantity for top 5 levels
            cum_qty_bid_top5 = sum(q for _, q in order_book['bids'][:5]) / 1000  # Convert to 'k'
            cum_qty_ask_top5 = sum(q for _, q in order_book['asks'][:5]) / 1000  # Convert to 'k'

            # Calculate spread in bps
            best_bid = order_book['bids'][0][0]
            best_ask = order_book['asks'][0][0]
            spread_bps = (best_ask - best_bid) / mid_price * 10000  # Convert to basis points

            # Check if sell quantity is more than twice the buy quantity
            sell_vs_buy_warning = "❌" if cum_qty_ask_top5 > 2 * cum_qty_bid_top5 else ""

            # Fetch OHLCV data for 15m timeframe to calculate EMAs and Bollinger Bands
            ohlcv_15m = await exchange.fetch_ohlcv(symbol, timeframe='15m', limit=60)
            df_15m = pd.DataFrame(ohlcv_15m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            # Fetch the latest price
            current_price = df_15m['close'].iloc[-1]

            # Calculate EMA and Bollinger Band levels
            ema_13 = df_15m['close'].ewm(span=13, adjust=False).mean().iloc[-1]
            ema_25 = df_15m['close'].ewm(span=25, adjust=False).mean().iloc[-1]
            ema_99 = df_15m['close'].ewm(span=99, adjust=False).mean().iloc[-1]
            sma_20 = df_15m['close'].rolling(window=20).mean().iloc[-1]
            lower_band = sma_20 - 2 * df_15m['close'].rolling(window=20).std().iloc[-1]

            # Check EMA conditions
            if current_price < ema_13:
                ema_message = "⚠️ Px below EMA 13"
            elif current_price < ema_13 and current_price < ema_25 and current_price < ema_99:
                ema_message = "💔Px below EMAs"
            else:
                ema_message = "✅ Px above all EMAs"

            # Check Bollinger Band conditions
            if current_price < lower_band:
                bb_message = "💔Px below lower BB"
            elif sma_20 > current_price >= lower_band:
                bb_message = "⚠️ Px in lower BB range"
            else:
                bb_message = "✅ Px above lower BB"

            # Format bid and ask quantities in k format
            bid_qty_k = int(total_bid_notional_50bps / 1000)
            ask_qty_k = int(total_ask_notional_50bps / 1000)
            
            # Warning indicator
            liquidity_warning = "❌" if total_notional_50bps < depth_threshold * 2 else "⚠️"
            
            # Update message format to include PERP indicator when applicable
            message = (
                f"📊 {exchange_name} {market_type_indicator}\n"
                f"{liquidity_warning} Total depth within 50bps from mid {mid_price:.4f} [{minus_50bps_px:.4f}@{bid_qty_k}k - {plus_50bps_px:.4f}@{ask_qty_k}k] = ${int(total_notional_50bps / 1000)}k.\n"
                f"Spread: {spread_bps:.2f} bps\n"
                f"{ema_message}\n"
                f"{bb_message}\n"
                f"{sell_vs_buy_warning} Cumulative Quantity (Top 5 Levels) - Buy: {cum_qty_bid_top5:.2f}k, Sell: {cum_qty_ask_top5:.2f}k"
            )
            logger.info(message)
            if total_notional_50bps < depth_threshold*2:
                await send_telegram_message(message)

            # Sleep before the next check
            await asyncio.sleep(alert_interval)

        except Exception as e:
            logger.error(f"Error in depth monitoring: {e}")
            await asyncio.sleep(60)  # Wait a minute before retrying

async def main(config_path):
    """Main function to set up exchange and watch trades."""
    global TELEGRAM_CHAT_ID
    config = load_config(config_path)
    exchange_id = config.get("exchange", "bybit")
    symbol = config.get("symbol", "A8/USDT")
    TELEGRAM_CHAT_ID = config.get("group_id", "")
    
    # Feature flags from config
    enable_depth_monitor = config.get("enable_depth_monitor", False)
    enable_ema_ratio = config.get("enable_ema_ratio", True)
    ema_window = config.get("ema_window", 15)
    ema_timeframe = config.get("ema_timeframe", 300)
    
    logger.info(f"Setting up {exchange_id} exchange for {symbol}")
    exchange = setup_exchange(exchange_id, symbol)
    
    # Determine if this is a perpetual contract
    is_perpetual = ':USDT' in symbol
    market_type_indicator = "PERP" if is_perpetual else ""
    
    try:
        tasks = []
        
        # Start trade watching task (always enabled as it collects trades)
        trade_task = asyncio.create_task(watch_trades(exchange, symbol, config))
        tasks.append(trade_task)
        
        # Start EMA ratio calculation if enabled
        if enable_ema_ratio:
            ema_task = asyncio.create_task(
                process_ema_ratio(symbol, exchange.id, market_type_indicator, ema_timeframe, ema_window)
            )
            tasks.append(ema_task)
            
        # Start depth monitoring if enabled
        if enable_depth_monitor:
            depth_task = asyncio.create_task(monitor_depth(exchange, symbol, config))
            tasks.append(depth_task)
            
        # Wait for all tasks to complete (they run indefinitely)
        await asyncio.gather(*tasks)
        
    except Exception as e:
        logger.error(f"Unhandled exception in main: {e}")
    finally:
        await exchange.close()

async def run_with_retry(config_path):
    """Run main function with retry logic."""
    max_retries = 5
    retry_count = 0
    while True:
        try:
            await main(config_path)
        except Exception as e:
            retry_count += 1
            backoff_time = min(2 ** retry_count, 60)
            logger.error(f"Error in main function: {e}. Retrying in {backoff_time} seconds (Retry {retry_count}/{max_retries})...")
            await asyncio.sleep(backoff_time)
            if retry_count >= max_retries:
                logger.error("Max retries reached. Exiting.")
                break
        else:
            retry_count = 0

if __name__ == "__main__":
    args = parse_arguments()
    asyncio.run(run_with_retry(args.config))