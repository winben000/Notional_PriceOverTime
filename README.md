# Notional_PriceOverTime

This folder contains tools for fetching, analyzing, and visualizing crypto trade data over time, with a focus on notional value and price action for specific tokens and exchanges.

## 📦 Folder Purpose
- **Fetch and store trade data** from exchanges (e.g., Gate.io, Bybit)
- **Stream and log live trades**
- **Analyze and visualize** price and notional value trends
- **Detect anomalies** in trading activity

---

## 🗂️ Main Scripts

### 1. `fetch_MORE_trades.py`
Fetches recent trades for the MORE/USDT pair on Gate.io and writes them to a CSV file.
- **Usage:**
  ```bash
  python3 fetch_MORE_trades.py
  ```
- **Output:** `more_trades.csv` (with trade details and notional calculation)

### 2. `stream_all_trades.py`
Streams live trades for a given symbol and exchange, logs all trades, and appends them to a CSV file in real time.
- **Usage:**
  ```bash
  python3 stream_all_trades.py --config gateio_more.json
  ```
- **Config:** JSON file specifying exchange and symbol (e.g., `gateio_more.json`)
- **Output:** `trades_YYYYMMDD_HHMMSS.csv`

### 3. `plot.py`
Visualizes trade price and notional value over time from a CSV file. Highlights anomalies and provides summary statistics.
- **Usage:**
  ```bash
  python3 plot.py more_trades.csv --datetime datetime
  ```
- **Options:**
  - `--datetime <col>`: Name of datetime column (required for time plots)
  - `--price-col <col>`: Price column (default: `price`)
  - `--amount-col <col>`: Amount column (default: `amount`)
  - `--notional-col <col>`: Notional column (default: `notional`)
  - `--std-mult <float>`: Std multiplier for anomaly detection (default: 3.0)

### 4. `test_all_trades.py`
Test script for fetching and displaying all trades for a given symbol/exchange. Useful for debugging and quick checks.
- **Usage:**
  ```bash
  python3 test_all_trades.py
  ```

---

## 📊 Output Files
- `more_trades.csv`, `trades_YYYYMMDD_HHMMSS.csv`: Raw trade data
- `*_TRADEPRICEOVERTIME.png`, `*_NOTIONALOVERTIME.png`: Saved plots

---

## 🚀 Quick Start
1. **Install dependencies:**
   ```bash
   pip3 install -r requirements.txt
   # or, if requirements.txt is missing:
   pip3 install pandas matplotlib numpy seaborn ccxt python-dotenv
   ```
2. **Fetch trades:**
   ```bash
   python3 fetch_MORE_trades.py
   ```
3. **Plot results:**
   ```bash
   python3 plot.py more_trades.csv --datetime datetime
   ```

---

## ⚙️ Dependencies
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `ccxt`
- `python-dotenv`

---

## 🛠️ Troubleshooting
- **No module named ...**: Install missing packages with `pip3 install ...`
- **Plot not showing:** Make sure you are running in a desktop environment with GUI support.
- **CSV not updating:** Check script logs and file permissions.
- **Timezone issues:** The scripts try to handle both naive and timezone-aware datetimes, but check your CSV for consistency.

---

## 📬 Contact
For questions or improvements, open an issue or contact the maintainer. 