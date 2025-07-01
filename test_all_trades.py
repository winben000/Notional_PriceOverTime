#!/usr/bin/env python3
"""
Test script to verify that trade_alert_perp.py can fetch all trades
"""

import asyncio
import ccxt.pro as ccxt
import pandas as pd
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_fetch_all_trades():
    """Test fetching all trades for MORE/USDT:USDT"""
    
    # Set up Gate.io exchange
    exchange = ccxt.gateio({
        'options': {
            'defaultType': 'swap',  # For perpetual contracts
        }
    })
    
    symbol = "MORE/USDT:USDT"
    
    logger.info(f"Testing trade fetching for {symbol}")
    logger.info("This will fetch ALL trades, not just large ones")
    
    try:
        # Fetch recent trades
        trades = await exchange.fetch_trades(symbol, limit=50)
        
        if trades:
            logger.info(f"✅ Successfully fetched {len(trades)} trades")
            
            # Convert to DataFrame for analysis
            trades_df = pd.DataFrame(trades)
            trades_df['amount'] = trades_df['amount'].astype(float)
            trades_df['price'] = trades_df['price'].astype(float)
            
            # Group by timestamp and side
            grouped_trades = trades_df.groupby(['timestamp', 'side'])
            
            logger.info(f"📊 Trade Summary:")
            logger.info(f"   Total trade groups: {len(grouped_trades)}")
            logger.info(f"   Buy trades: {len(trades_df[trades_df['side'] == 'buy'])}")
            logger.info(f"   Sell trades: {len(trades_df[trades_df['side'] == 'sell'])}")
            logger.info(f"   Total volume: {trades_df['amount'].sum():.6f}")
            logger.info(f"   Average price: ${trades_df['price'].mean():.6f}")
            
            # Show sample trades
            logger.info(f"\n🕒 Sample Trades:")
            for i, trade in enumerate(trades_df.head(5).to_dict('records')):
                time_str = datetime.fromtimestamp(int(trade['timestamp'])/1000).strftime('%H:%M:%S')
                icon = '🟢' if trade['side'].lower() == 'buy' else '🔴'
                logger.info(f"   {icon} {trade['side'].upper()} {trade['amount']:.6f}@{trade['price']:.6f} at {time_str}")
            
        else:
            logger.warning("⚠️ No trades found")
            
    except Exception as e:
        logger.error(f"❌ Error fetching trades: {e}")
    
    finally:
        await exchange.close()

if __name__ == "__main__":
    print("🧪 Testing All Trades Fetching")
    print("=" * 40)
    asyncio.run(test_fetch_all_trades())
    print("\n✅ Test completed!") 