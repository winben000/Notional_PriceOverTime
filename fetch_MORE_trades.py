#!/usr/bin/env python3
"""
Fetch recent trades for MORE/USDT on Gate.io and write to CSV.
"""

import ccxt
import pandas as pd
import time

def fetch_all_trades(symbol='MORE/USDT', limit=1000, output_file='more_trades.csv'):
    # 1) Initialize exchange
    exchange = ccxt.gateio({
        'enableRateLimit': True,   # Respect rate limits
    })

    all_trades = []
    since = None

    while True:
        # 2) Fetch a batch of trades
        try:
            trades = exchange.fetch_trades(symbol, since=since, limit=limit)
        except Exception as e:
            print(f"Error fetching trades: {e}")
            break
            
        if not trades:
            break

        all_trades.extend(trades)

        # 3) Determine the timestamp of the last trade and increment by 1ms to avoid duplication
        last_ts = trades[-1]['timestamp']
        if last_ts is not None:
            since = last_ts + 1
        else:
            print("Warning: Received trade with None timestamp")
            break

        # If fewer than limit trades were returned, we've reached the end
        if len(trades) < limit:
            break

        # Be polite with the API
        time.sleep(exchange.rateLimit / 1000)

    # 4) Normalize into a DataFrame
    df = pd.DataFrame(all_trades)

    # Convert timestamp to human-readable datetime
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Optional: select and reorder columns
    df = df[
        ['datetime', 'id', 'order', 'side', 'price', 'amount', 'cost']
    ]

    # 5) Write to CSV
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df)} trades to {output_file}")

if __name__ == '__main__':
    fetch_all_trades()
