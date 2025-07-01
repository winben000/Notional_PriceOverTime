import asyncio
import ccxt.pro as ccxt
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
import matplotlib.pyplot as plt
import seaborn as sns

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Track logged timestamps to avoid duplicate alerts
logged_timestamps = set()
all_trades = []  # Store all trades
buy_volumes = []
sell_volumes = []
ratio_values = []

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

def setup_exchange(exchange_id, symbol=None):
    """Set up exchange based on exchange ID from config."""
    exchange_class = getattr(ccxt, exchange_id)
    
    # Determine if this is a perpetual contract by checking symbol format
    is_spot = symbol and ':USDT' not in symbol
    market_type = 'spot' if is_spot else 'swap'
    
    exchange_config = {
        'options': {
            'defaultType': market_type,
        }
    }
    return exchange_class(exchange_config)

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
                    if isinstance(large_trades, pd.DataFrame):
                        if not large_trades.empty:
                            mean_large_trades = large_trades['amount'].mean()
                    else:
                        if len(large_trades) > 0:
                            mean_large_trades = np.mean([trade['amount'] for trade in large_trades])
                
                warning = "⚠️ LARGE TRADE ALERT! ⚠️" if total_amount >= 90000 else "LARGE TRADE ALERT!"
                message = (
                    f"{warning}\n"
                    f"📊 {exchange_name.upper()} {market_type_indicator}\n"
                    f"{icon} {side.upper()} {total_amount/1000:.2f}k@{avg_price:.4f} [mean large trades 60m: {mean_large_trades/1000:.2f}k]\n"
                    f"-----------------------------------------------------"
                )
                logger.info(message)
                # await send_telegram_message(message)
                logged_timestamps.add((timestamp, side))
                
# Function to watch recent trades
async def watch_trades(exchange, symbol, config):
    exchange_name     = exchange.id
    min_trade_amount  = float(config.get("min_trade_amount", 10000))
    enable_large_trades = config.get("enable_large_trades", True)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"trades_{run_timestamp}.csv"

    logger.info(f"Starting to watch trades for {symbol}...")

    while True:
        try:
            trades = await exchange.watch_trades(symbol)
            if trades:
                trades_df = pd.DataFrame(trades)
                trades_df['amount'] = trades_df['amount'].astype(float)
                trades_df['price']  = trades_df['price'].astype(float)

                # alert large trades
                if enable_large_trades:
                    grouped_trades = trades_df.groupby(['timestamp', 'side'])
                    await filter_large_trades(grouped_trades, symbol,
                                              min_trade_amount,
                                              exchange_name,
                                              "PERP" if ':USDT' in symbol else "")

                # mở rộng all_trades
                all_trades.extend(trades_df.to_dict('records'))

                # —— phần ghi CSV ở đây —— 
                trades_df.to_csv(csv_file, index=False, mode='w', header=True)

        except Exception as e:
            logger.error(f"Error watching trades: {e}")
            await asyncio.sleep(10)


async def main(config_path):
    """Main function to setup exchange and watch trades"""
    config = load_config(config_path)
    exchange_id = config.get("exchange", "bybit")
    symbol = config.get("symbol", "A8/USDT")
    
    logger.info(f"Setting up {exchange_id} exchange for {symbol}")
    exchange = setup_exchange(exchange_id, symbol)
    
    # Determine if this is a perpetual contract
    is_spot = ':USDT' not in symbol
    market_type_indicator = "SPOT" if is_spot else "PERP"
    
    try:
        tasks = []
        trade_task = asyncio.create_task(watch_trades(exchange, symbol, config))
        tasks.append(trade_task)

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
    asyncio.run(main(args.config))
