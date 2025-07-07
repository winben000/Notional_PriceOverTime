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

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Store all trades
all_trades = []

# Trade amount categories
TRADE_CATEGORIES = {
    'small': (0, 100),
    'medium': (100, 1000),
    'large': (1000, float('inf'))
}

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

def categorize_trade_amount(amount):
    """Categorize trade amount into small, medium, or large."""
    for category, (min_amount, max_amount) in TRADE_CATEGORIES.items():
        if min_amount <= amount < max_amount:
            return category
    return 'large'  # Default to large if amount is very high

async def log_all_trades(grouped_trades, symbol, exchange_name, market_type_indicator=""):
    """Log all trades with basic information and categorize by amount."""
    global all_trades
    
    for (timestamp, side), group in grouped_trades:
        total_amount = group['amount'].sum()
        avg_price = group['price'].mean()
        trade_count = len(group)
        category = categorize_trade_amount(total_amount)
        
        icon = '🟢' if side.lower() == 'buy' else '🔴'
        category_emoji = {'small': '🔵', 'medium': '🟡', 'large': '🔴'}
        
        message = (
            f"📊 {exchange_name.upper()} {market_type_indicator}\n"
            f"{icon} {side.upper()} {category_emoji[category]} {category.upper()} {total_amount:.6f}@{avg_price:.6f} ({trade_count} trades)\n"
            f"⏰ {datetime.fromtimestamp(timestamp/1000).strftime('%H:%M:%S')}\n"
            f"-----------------------------------------------------"
        )
        logger.info(message)
                
# Function to watch recent trades
async def watch_trades(exchange, symbol, config):
    exchange_name = exchange.id
    csv_file = f"trades_{symbol}.csv"
    
    # Track if we've written the header yet
    header_written = False

    logger.info(f"Starting to watch trades for {symbol}...")
    logger.info(f"CSV file: {csv_file}")

    while True:
        try:
            trades = await exchange.watch_trades(symbol)
            if trades:
                trades_df = pd.DataFrame(trades)
                trades_df['amount'] = trades_df['amount'].astype(float)
                trades_df['price']  = trades_df['price'].astype(float)

                # log all trades
                grouped_trades = trades_df.groupby(['timestamp', 'side'])
                await log_all_trades(grouped_trades, symbol,
                                    exchange_name,
                                    "PERP" if ':USDT' in symbol else "")

                # mở rộng all_trades
                all_trades.extend(trades_df.to_dict('records'))

                # Add category column and sort by amount categories
                trades_df['category'] = trades_df['amount'].apply(categorize_trade_amount)
                
                # Sort by category priority (large, medium, small) and then by timestamp
                category_order = {'large': 0, 'medium': 1, 'small': 2}
                trades_df['category_order'] = [category_order.get(cat, 2) for cat in trades_df['category']]
                trades_df = trades_df.sort_values(['category_order', 'timestamp'], ascending=[True, False]).reset_index(drop=True)
                trades_df = trades_df.drop('category_order', axis=1)

                # —— phần ghi CSV ở đây —— 
                # Write header only once, then append data
                trades_df.to_csv(csv_file, index=False, mode='a' if header_written else 'w', header=not header_written)
                if not header_written:
                    header_written = True
                    logger.info(f"CSV header written to {csv_file}")
                
                logger.info(f"Appended {len(trades_df)} trades to {csv_file}")

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
