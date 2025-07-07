#!/usr/bin/env python3
"""
Fetch all trades for LA/USDT on Binance and write to CSV.
"""

import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta
import pytz
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def fetch_all_trades(symbol='LA/USDT', limit=1000, output_file='la_trades.csv', since_time=None):
    # 1) Initialize exchange with Binance API credentials
    api_key = os.getenv('binance_api_key')
    api_secret = os.getenv('binance_api_secret')
    
    if api_key and api_secret:
        exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
        })
    else:
        exchange = ccxt.binance({
            'enableRateLimit': True,
        })

    all_trades = []
    # If since_time is provided, use it; otherwise None
    since = since_time
    
    # Set a maximum time limit to prevent infinite loops (24 hours from start)
    max_time = None
    if since_time:
        max_time = since_time + (24 * 60 * 60 * 1000)  # 24 hours in milliseconds
    
    batch_count = 0
    print(f"Starting to fetch trades from {datetime.fromtimestamp(since_time/1000) if since_time else 'beginning'}")

    while True:
        batch_count += 1
        print(f"Fetching batch {batch_count}...")
        
        # 2) Fetch a batch of trades
        try:
            trades = exchange.fetch_trades(symbol, since=since, limit=limit)
        except Exception as e:
            print(f"Error fetching trades batch {batch_count}: {e}")
            break
            
        if not trades:
            print(f"No more trades found in batch {batch_count}")
            break

        print(f"Received {len(trades)} trades in batch {batch_count}")
        all_trades.extend(trades)

        # 3) Determine the timestamp of the last trade and increment by 1ms to avoid duplication
        last_ts = trades[-1]['timestamp']
        if last_ts is not None:
            since = last_ts + 1
            
            # Check if we've exceeded the maximum time limit
            if max_time and since >= max_time:
                print(f"Reached maximum time limit (24 hours from start)")
                break
        else:
            print("Warning: Received trade with None timestamp")
            break

        # If fewer than limit trades were returned, we've reached the end
        if len(trades) < limit:
            print(f"Received fewer trades than limit ({len(trades)} < {limit}), reached end of available data")
            break

        # Be polite with the API
        time.sleep(exchange.rateLimit / 1000)

    print(f"Total batches fetched: {batch_count}")
    print(f"Total trades collected: {len(all_trades)}")

    if not all_trades:
        print("No trades found!")
        return

    # 4) Normalize into a DataFrame
    df = pd.DataFrame(all_trades)

    # Convert timestamp to human-readable datetime
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Convert UTC to GMT+7 for display
    gmt7 = pytz.timezone('Asia/Bangkok')
    df['datetime_gmt7'] = df['datetime'].dt.tz_localize('UTC').dt.tz_convert(gmt7)

    # Optional: select and reorder columns
    df = df[
        ['datetime_gmt7', 'id', 'order', 'side', 'price', 'amount', 'cost']
    ]

    # 5) Write to CSV
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df)} trades to {output_file}")
    
    # Show time range of collected data
    if len(df) > 0:
        print(f"First trade: {df.iloc[0]['datetime_gmt7']}")
        print(f"Last trade: {df.iloc[-1]['datetime_gmt7']}")

if __name__ == '__main__':
    # Check if Binance API credentials are available
    if not os.getenv('binance_api_key') or not os.getenv('binance_api_secret'):
        print("Error: binance_api_key and binance_api_secret must be set in environment variables or .env file")
        print("Please add them to your .env file:")
        print("binance_api_key=your_api_key_here")
        print("binance_api_secret=your_api_secret_here")
        exit(1)
    
    # Fetch trades from 4:00 PM GMT+7, July 2nd, 2025
    gmt7 = pytz.timezone('Asia/Bangkok')  # GMT+7 timezone
    
    # 4:00 PM GMT+7, July 2nd, 2025
    start_gmt7 = gmt7.localize(datetime(2025, 7, 2, 16, 0, 0))  # 4:00 PM GMT+7, July 2, 2025
    
    # Convert to UTC for API calls
    start_utc = start_gmt7.astimezone(pytz.UTC)
    start_ms = int(start_utc.timestamp() * 1000)
    
    print(f"Using Binance API with credentials from environment variables")
    print(f"Fetching trades from: {start_gmt7.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"UTC start time: {start_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    # Fetch trades from 4:00 PM onwards
    print("\n=== Fetching Trades from 4:00 PM GMT+7 ===")
    fetch_all_trades(since_time=start_ms)
