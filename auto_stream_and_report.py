import asyncio
import ccxt.pro as ccxt
import os
import json
import logging
from datetime import datetime
import pandas as pd
import time
import subprocess
import sys
from dotenv import load_dotenv
import requests
import aiohttp
import io
import matplotlib.pyplot as plt
import argparse

# Load environment variables
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID =  os.getenv("TELEGRAM_CHAT_ID")
TOPIC_ID = os.getenv("TOPIC_ID")

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default="gateio_more.json", help="Path to config JSON")
args = parser.parse_args()
CONFIG_PATH = args.config

# Config
CSV_FILE = "auto_trades.csv"      # Output CSV file
PLOT_SCRIPT = "plot.py"           # Path to plot.py
PLOT_PREFIX = "AUTO"
ANALYZE_INTERVAL = 15 * 60         # 15 minutes in seconds

# --- Telegram send function ---
async def send_telegram_photo(photo_path, caption=None):
    """
    Send a photo to Telegram using the bot API.
    Uses aiohttp for async HTTP requests.
    Includes message_thread_id if TOPIC_ID is set (for forum topics).
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("Telegram credentials not set in environment variables.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption or "", "parse_mode": "HTML"}
    if TOPIC_ID:
        data["message_thread_id"] = TOPIC_ID
    try:
        with open(photo_path, "rb") as photo:
            form = aiohttp.FormData()
            for k, v in data.items():
                form.add_field(k, str(v))
            form.add_field('photo', photo, filename=os.path.basename(photo_path), content_type='image/png')
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=form) as response:
                    result = await response.json()
                    if response.status != 200:
                        logger.error(f"Telegram API error: {result}")
                    else:
                        logger.info(f"Telegram photo sent successfully: {photo_path}")
    except Exception as e:
        logger.error(f"Error sending Telegram photo: {e}")

async def send_telegram_photo_bytes(image_bytes, caption=None):
    """
    Send a photo to Telegram using the bot API, from in-memory bytes.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("Telegram credentials not set in environment variables.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption or "", "parse_mode": "HTML"}
    if TOPIC_ID:
        data["message_thread_id"] = TOPIC_ID
    try:
        form = aiohttp.FormData()
        for k, v in data.items():
            form.add_field(k, str(v))
        form.add_field('photo', image_bytes, filename="plot.png", content_type='image/png')
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=form) as response:
                result = await response.json()
                if response.status != 200:
                    logger.error(f"Telegram API error: {result}")
                else:
                    logger.info(f"Telegram photo sent successfully (in-memory)")
    except Exception as e:
        logger.error(f"Error sending Telegram photo: {e}")

# --- Streaming trades ---
async def stream_trades(exchange_id, symbol):
    logger.info(f"Starting trade stream for {exchange_id} {symbol}")
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({"enableRateLimit": True})
    header_written = os.path.exists(CSV_FILE)
    while True:
        try:
            trades = await exchange.watch_trades(symbol)
            if trades:
                df = pd.DataFrame(trades)
                df['amount'] = df['amount'].astype(float)
                df['price'] = df['price'].astype(float)
                # Add category column
                def categorize(amount):
                    if amount < 100: return 'small'
                    elif amount < 1000: return 'medium'
                    else: return 'large'
                df['category'] = df['amount'].apply(categorize)
                # Append to CSV
                df.to_csv(CSV_FILE, index=False, mode='a' if header_written else 'w', header=not header_written)
                header_written = True
                logger.info(f"Appended {len(df)} trades to {CSV_FILE}")
        except Exception as e:
            logger.error(f"Stream error: {e}")
            await asyncio.sleep(10)

# --- Analysis and reporting ---
def analyze_notional_stats(csv_file):
    df = pd.read_csv(csv_file)
    if 'amount' not in df or 'price' not in df:
        raise ValueError('CSV must have amount and price columns')
    df['notional'] = df['amount'] * df['price']
    mean_notional = df['notional'].mean()
    std_notional = df['notional'].std()
    threshold = mean_notional + 3 * std_notional
    anomalies = df[df['notional'] > threshold]
    n_anomalies = len(anomalies)
    # Market impact: notional / sum(notional)
    total_notional = df['notional'].sum()
    if n_anomalies > 0 and isinstance(anomalies, pd.DataFrame):
        anomalies = anomalies.copy()
        anomalies['market_impact'] = anomalies['notional'] / total_notional
        table = anomalies[['datetime', 'amount', 'price', 'notional', 'market_impact']].to_string(index=False, float_format='%.2e')
    else:
        table = 'No anomalies detected.'
    return mean_notional, std_notional, threshold, n_anomalies, table

# --- Plotting helper ---
def plot_notional_over_time(csv_file):
    df = pd.read_csv(csv_file)
    if 'datetime' in df:
        df['datetime'] = pd.to_datetime(df['datetime'])
    else:
        raise ValueError('CSV must have datetime column')
    df['notional'] = df['amount'] * df['price']
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df['datetime'], df['notional'], label='Notional')
    ax.set_title('Notional Over Time')
    ax.set_xlabel('Datetime')
    ax.set_ylabel('Notional')
    ax.legend()
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return buf

# --- Scheduled analysis and reporting ---
async def analyze_and_report():
    while True:
        try:
            # --- Analysis ---
            mean_notional, std_notional, threshold, n_anomalies, table = analyze_notional_stats(CSV_FILE)
            # --- Send stats to Telegram ---
            stats_msg = (
                f"<b>Notional Analysis Report</b>\n"
                f"Mean Notional: <b>{mean_notional:,.2f}</b>\n"
                f"Std Notional: <b>{std_notional:,.2f}</b>\n"
                f"Anomaly Threshold: <b>{threshold:,.2f}</b>\n"
                f"Number of Anomalies: <b>{n_anomalies}</b>"
            )
            # Send stats message
            async with aiohttp.ClientSession() as session:
                url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
                data = {"chat_id": TELEGRAM_CHAT_ID, "text": stats_msg, "parse_mode": "HTML"}
                if TOPIC_ID:
                    data["message_thread_id"] = TOPIC_ID
                async with session.post(url, data=data) as response:
                    result = await response.json()
                    if response.status != 200:
                        logger.error(f"Telegram API error: {result}")
                    else:
                        logger.info(f"Telegram stats message sent successfully")
            # --- Plot and send plot to Telegram (in-memory) ---
            plot_buf = plot_notional_over_time(CSV_FILE)
            await send_telegram_photo_bytes(plot_buf, caption="Notional Over Time")
        except Exception as e:
            logger.error(f"Analysis/report error: {e}")
        logger.info(f"Sleeping {ANALYZE_INTERVAL//60} minutes before next analysis...")
        await asyncio.sleep(ANALYZE_INTERVAL)

# --- Main entry point ---
async def main():
    # Load config
    try:
        with open(CONFIG_PATH) as f:
            config = json.load(f)
        exchange_id = config.get("exchange", "gateio")
        symbol = config.get("symbol", "MORE/USDT")
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return
    # Start streaming and reporting concurrently
    await asyncio.gather(
        stream_trades(exchange_id, symbol),
        analyze_and_report()
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down...") 