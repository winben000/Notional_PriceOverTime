#!/usr/bin/env python3

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib import rcParams
import time
import subprocess
from datetime import datetime
import os

# Set up beautiful styling
plt.style.use('seaborn-v0_8-darkgrid')
rcParams['font.size'] = 12
rcParams['axes.titlesize'] = 16
rcParams['axes.labelsize'] = 14
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
rcParams['legend.fontsize'] = 12
rcParams['figure.titlesize'] = 18

# Set color palette
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#8B5A3C']
sns.set_palette(colors)

def load_data(filepath: str, datetime_col: str = None) -> pd.DataFrame:
    """Load CSV into DataFrame, optionally parsing a datetime column."""
    parse_dates = [datetime_col] if datetime_col else None
    df = pd.read_csv(filepath, parse_dates=parse_dates)
    return df

def calculate_notional(df: pd.DataFrame, price_col: str, amount_col: str, notional_col: str) -> pd.DataFrame:
    """Add a notional column = price * amount."""
    df[notional_col] = df[price_col] * df[amount_col]
    return df

def plot_price_over_time(df: pd.DataFrame, datetime_col: str, price_col: str, args):
    # Parse your datetime column
    df = df.copy()
    
    # Check if the column is already a datetime type
    if pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
        # Already a datetime, just ensure it's timezone-naive for plotting
        df[datetime_col] = df[datetime_col].dt.tz_localize(None)
    elif np.issubdtype(df[datetime_col].dtype, np.number):
        # assume integer ms‐since‐epoch
        df[datetime_col] = pd.to_datetime(df[datetime_col], unit='ms', errors='coerce')
    else:
        df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
    
    df.dropna(subset=[datetime_col], inplace=True)

    # Set datetime column as index
    df.set_index(datetime_col, inplace=True)

    # Create beautiful plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot with enhanced styling
    ax.plot(df.index, df[price_col], 
            marker='o', markersize=4, linestyle='-', linewidth=2, 
            color='#2E86AB', alpha=0.8, markerfacecolor='#2E86AB',
            markeredgecolor='white', markeredgewidth=1)
    
    # Enhanced styling
    ax.set_title("Trade Price Over Time", fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel("Time", fontsize=14, fontweight='bold')
    ax.set_ylabel("Price (USDT)", fontsize=14, fontweight='bold')
    
    # Format the x-axis ticks
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    fig.autofmt_xdate()
    
    # Enhanced grid and styling
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # Add statistics text
    price_mean = df[price_col].mean()
    price_std = df[price_col].std()
    price_min = df[price_col].min()
    price_max = df[price_col].max()
    
    stats_text = f'Statistics:\nMean: ${price_mean:.6f}\nStd: ${price_std:.6f}\nRange: ${price_min:.6f} - ${price_max:.6f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    if args.save_prefix:
        plt.savefig(f"{args.save_prefix}_TRADEPRICEOVERTIME.png")
        plt.close()
    else:
        plt.show()

def plot_notional_over_time(df: pd.DataFrame, datetime_col: str, notional_col: str, args):
    # Set datetime column as datetime
    df = df.copy()
    
    # Check if the column is already a datetime type
    if pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
        # Already a datetime, just ensure it's timezone-naive for plotting
        df[datetime_col] = df[datetime_col].dt.tz_localize(None)
    else:
        df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
    
    df = df.dropna(subset=[datetime_col])

    # Set datetime column as index
    df = df.set_index(datetime_col)

    # Create beautiful plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot with enhanced styling
    ax.plot(df.index, df[notional_col], 
            marker='s', markersize=4, linestyle='-', linewidth=2, 
            color='#A23B72', alpha=0.8, markerfacecolor='#A23B72',
            markeredgecolor='white', markeredgewidth=1)
    
    # Add gradient fill
    ax.fill_between(df.index, df[notional_col], alpha=0.3, color='#A23B72')
    
    # Enhanced styling
    ax.set_title("Trade Notional Over Time", fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel("Time", fontsize=14, fontweight='bold')
    ax.set_ylabel("Notional Value (USDT)", fontsize=14, fontweight='bold')
    
    # Format the x-axis ticks
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    fig.autofmt_xdate()
    
    # Enhanced grid and styling
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # Add statistics text
    notional_mean = df[notional_col].mean()
    notional_std = df[notional_col].std()
    notional_min = df[notional_col].min()
    notional_max = df[notional_col].max()
    total_volume = df[notional_col].sum()
    
    stats_text = f'Statistics:\nMean: ${notional_mean:.2f}\nStd: ${notional_std:.2f}\nRange: ${notional_min:.2f} - ${notional_max:.2f}\nTotal Volume: ${total_volume:.2f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Highlight anomalies
    threshold = notional_mean + 2 * notional_std
    anomalies = df[df[notional_col] > threshold]
    if len(anomalies) > 0:
        ax.scatter(anomalies.index, anomalies[notional_col], 
                  color='#C73E1D', s=100, alpha=0.8, zorder=5,
                  label=f'Anomalies (>${threshold:.2f})')
        ax.legend(loc='upper right', framealpha=0.9)
    
    plt.tight_layout()
    if args.save_prefix:
        plt.savefig(f"{args.save_prefix}_NOTIONALOVERTIME.png")
        plt.close()
    else:
        plt.show()


def calculate_anomalies(df: pd.DataFrame, notional_col: str, std_multiplier: float):
    """Identify rows where notional > mean + std_multiplier * std."""
    mean = df[notional_col].mean()
    std  = df[notional_col].std()
    threshold = mean + std_multiplier * std
    anomalies = df[df[notional_col] > threshold].copy()
    return anomalies, mean, std, threshold

def calculate_market_impact(anomalies: pd.DataFrame, amount_col: str, price_col: str) -> pd.DataFrame:
    """Compute a simple market impact metric for anomalies."""
    price_vol = anomalies[price_col].std()
    avg_vol   = anomalies[amount_col].sum()
    anomalies['market_impact'] = (anomalies[amount_col] / avg_vol) * price_vol
    return anomalies

def main():
    parser = argparse.ArgumentParser(description="Analyze trades and detect anomalies.")
    parser.add_argument("file", help="Path to CSV file containing trade data")
    parser.add_argument("--datetime",    help="Name of datetime column to parse and set as index")
    parser.add_argument("--price-col",   default="price",   help="Column name for trade price")
    parser.add_argument("--amount-col",  default="amount",  help="Column name for trade amount")
    parser.add_argument("--notional-col",default="notional",help="Output column name for notional")
    parser.add_argument("--bins",        type=int, default=100,    help="Number of bins for histogram (more detail)")
    parser.add_argument("--kde",         action="store_true",     help="Overlay KDE on histogram")
    parser.add_argument("--std-mult",    type=float, default=3.0,  help="Std multiplier for anomaly threshold")
    parser.add_argument("--save-prefix", help="Prefix for saved plot files (if set, saves instead of showing)")
    args = parser.parse_args()

    # Load & preview
    df = load_data(args.file, args.datetime)
    if args.datetime:
        df.set_index(args.datetime, inplace=False)
    
    # Compute notional
    df = calculate_notional(df, args.price_col, args.amount_col, args.notional_col)

    # Detect anomalies
    anomalies, mean, std, threshold = calculate_anomalies(df, args.notional_col, args.std_mult)
    print(f"Mean {args.notional_col}: {mean:.2f}")
    print(f"Std  {args.notional_col}: {std:.2f}")
    print(f"Anomaly threshold (> mean + {args.std_mult}σ): {threshold:.2f}")
    print(f"Number of anomalies: {len(anomalies)}\n")

    # Calculate market impact
    impacted = calculate_market_impact(anomalies, args.amount_col, args.price_col)
    print("Anomalies with market impact:")
    print(impacted[[args.price_col, args.amount_col, args.notional_col, 'market_impact']])

    # Plot price over time and histogram
    if args.datetime:
        plot_price_over_time(df, 'datetime', 'price', args)
        plot_notional_over_time(df, 'datetime', 'notional', args)

if __name__ == "__main__":
    main()
