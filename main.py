import os
import requests
import time
import numpy as np

from flask import Flask
import threading
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import ThreadedConnectionPool
from contextlib import contextmanager
import random
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import json
from urllib.parse import quote


# --- DATABASE SETUP ---

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise Exception("DATABASE_URL not set. Please add it to environment variables.")

# Connection pool — প্রতি call-এ নতুন connection খোলার overhead নেই
_pool = ThreadedConnectionPool(minconn=2, maxconn=10, dsn=DATABASE_URL)

@contextmanager
def db_conn():
    conn = _pool.getconn()
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        _pool.putconn(conn)


def init_db():
    try:
        with db_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS trading_stats (
                    id SERIAL PRIMARY KEY,
                    total_trades INTEGER DEFAULT 0,
                    wins INTEGER DEFAULT 0,
                    losses INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS active_trade (
                    id INTEGER PRIMARY KEY DEFAULT 1,
                    symbol TEXT,
                    entry_price FLOAT,
                    side TEXT,
                    start_time FLOAT,
                    expiry_time FLOAT,
                    rec_time INTEGER,
                    is_active BOOLEAN DEFAULT TRUE,
                    indicators_status JSONB,
                    CONSTRAINT single_row CHECK (id = 1)
                );

                CREATE TABLE IF NOT EXISTS indicator_weights (
                    symbol TEXT,
                    name TEXT,
                    weight FLOAT,
                    accuracy_factor FLOAT DEFAULT 1.0,
                    PRIMARY KEY (symbol, name)
                );

                DO $$
                DECLARE
                    sym TEXT;
                    symbols TEXT[] := ARRAY['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'USD/CAD', 'USD/CHF', 'NZD/USD', 'EUR/JPY', 'GBP/JPY', 'EUR/GBP', 'BTC/USD', 'ETH/USD', 'LTC/USD', 'XRP/USD', 'SOL/USD', 'BNB/USD', 'ADA/USD', 'DOT/USD', 'DOGE/USD', 'AVAX/USD', 'XAU/USD', 'XAG/USD', 'WTI/USD', 'BRENT/USD', 'GBP/AUD', 'EUR/AUD', 'AUD/JPY', 'CAD/JPY', 'CHF/JPY', 'NZD/JPY', 'GBP/CAD', 'EUR/CAD', 'GBP/CHF', 'USD/TRY'];
                BEGIN
                    FOREACH sym IN ARRAY symbols
                    LOOP
                        INSERT INTO indicator_weights (symbol, name, weight) VALUES
                            (sym, 'trend', 15), (sym, 'rsi', 10), (sym, 'bb', 10), (sym, 'vol', 10),
                            (sym, 'adx', 5), (sym, 'ema_crossover', 10), (sym, 'macd', 15),
                            (sym, 'atr', 10), (sym, 'vwap', 10), (sym, 'candle', 5)
                        ON CONFLICT (symbol, name) DO NOTHING;
                    END LOOP;
                END $$;
            """)
            cur.execute("SELECT COUNT(*) FROM trading_stats")
            row = cur.fetchone()
            if row and row[0] == 0:
                cur.execute("INSERT INTO trading_stats (total_trades, wins, losses) VALUES (0, 0, 0)")
            conn.commit()
            cur.close()
    except Exception as e:
        print(f"Database Init Error: {e}")


def get_stats():
    try:
        with db_conn() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute("SELECT total_trades, wins, losses FROM trading_stats LIMIT 1")
            row = cur.fetchone()
            cur.close()
            return row if row else {"total_trades": 0, "wins": 0, "losses": 0}
    except Exception as e:
        print(f"Get Stats Error: {e}")
        return {"total_trades": 0, "wins": 0, "losses": 0}


def update_stats(win=True):
    try:
        with db_conn() as conn:
            cur = conn.cursor()
            if win:
                cur.execute("UPDATE trading_stats SET total_trades = total_trades + 1, wins = wins + 1")
            else:
                cur.execute("UPDATE trading_stats SET total_trades = total_trades + 1, losses = losses + 1")
            conn.commit()
            cur.close()
    except Exception as e:
        print(f"Update Stats Error: {e}")


def get_weights(symbol):
    try:
        with db_conn() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute("SELECT name, weight, accuracy_factor FROM indicator_weights WHERE symbol = %s", (symbol,))
            rows = cur.fetchall()
            cur.close()
            if not rows:
                return {'trend': 15, 'rsi': 10, 'bb': 10, 'vol': 10, 'adx': 5, 'ema_crossover': 10, 'macd': 15, 'atr': 10, 'vwap': 10, 'candle': 5}
            return {r['name']: r['weight'] * r['accuracy_factor'] for r in rows}
    except:
        return {'trend': 15, 'rsi': 10, 'bb': 10, 'vol': 10, 'adx': 5, 'ema_crossover': 10, 'macd': 15, 'atr': 10, 'vwap': 10, 'candle': 5}


def update_weights(symbol, indicators_status, win):
    try:
        with db_conn() as conn:
            cur = conn.cursor()
            factor_change = 0.05 if win else -0.05
            for name, active in indicators_status.items():
                if active:
                    cur.execute("""
                        UPDATE indicator_weights
                        SET accuracy_factor = LEAST(GREATEST(accuracy_factor + %s, 0.5), 1.5)
                        WHERE symbol = %s AND name = %s
                    """, (factor_change, symbol, name))
            conn.commit()
            cur.close()
    except Exception as e:
        print(f"Weight Update Error: {e}")


# --- Twelve Data API Settings ---

API_KEYS = [
    os.getenv("TWELVE_KEY_1"),
    os.getenv("TWELVE_KEY_2"),
    os.getenv("TWELVE_KEY_3"),
    os.getenv("TWELVE_KEY_4"),
    os.getenv("TWELVE_KEY_5"),
    os.getenv("TWELVE_KEY_6"),
    os.getenv("TWELVE_KEY_7"),
]

API_KEYS = [k for k in API_KEYS if k]

if not API_KEYS:
    raise Exception("No TwelveData API key found")

TELEGRAM_TOKEN = os.getenv("BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("CHAT_ID")

SYMBOLS = [
    "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "USD/CHF", "NZD/USD",
    "EUR/JPY", "GBP/JPY", "EUR/GBP", "BTC/USD", "ETH/USD", "LTC/USD", "XRP/USD",
    "SOL/USD", "ADA/USD", "XAU/USD", "XAG/USD", "GBP/AUD", "EUR/AUD", "AUD/JPY",
]

app = Flask(__name__)

@app.route("/")
def home():
    return "Trading AI Bot is running!"

def run_web():
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

# প্রতি মিনিটে ৭টি key × ৩টি symbol = ২১টি market scan করা হয়

current_trade = None
last_scan_minute = -1
_last_sent = {"text": "", "ts": 0}
_result_sent_for = None  # কোন trade-এর result ইতিমধ্যে পাঠানো হয়েছে তা track করে


def send_telegram_msg(message):
    global _last_sent
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    # একই message ৬০ সেকেন্দের মধ্যে দুইবার পাঠাবে না
    if message == _last_sent["text"] and time.time() - _last_sent["ts"] < 60:
        return
    _last_sent = {"text": message, "ts": time.time()}
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        requests.post(url, json=payload, timeout=10)
    except:
        pass


def calculate_ema_signal(df):
    if df is None or len(df) < 20:
        return "HOLD"
    df["EMA_FAST"] = df["close"].ewm(span=5,  adjust=False).mean()
    df["EMA_SLOW"] = df["close"].ewm(span=13, adjust=False).mean()
    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]
    if last_row["EMA_FAST"] > last_row["EMA_SLOW"] and prev_row["EMA_FAST"] <= prev_row["EMA_SLOW"]:
        return "BUY"
    elif last_row["EMA_FAST"] < last_row["EMA_SLOW"] and prev_row["EMA_FAST"] >= prev_row["EMA_SLOW"]:
        return "SELL"
    return "HOLD"


def calculate_indicators(values, symbol="EUR/USD"):
    if not values:
        return None

    df = pd.DataFrame(values)
    df.symbol_name = symbol

    cols = {"close": "close", "high": "high", "low": "low", "open": "open", "volume": "volume"}
    for k in cols:
        if k not in df.columns:
            for col in df.columns:
                if col.lower() == k:
                    df[k] = df[col]
                    break

    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["high"]  = pd.to_numeric(df["high"],  errors="coerce")
    df["low"]   = pd.to_numeric(df["low"],   errors="coerce")
    df["open"]  = pd.to_numeric(df["open"],  errors="coerce")
    # volume নাও থাকতে পারে (forex market এ), তাই default 0 দেওয়া হচ্ছে
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)

    df.dropna(inplace=True)
    if len(df) < 50:
        return None

    df = df[::-1].reset_index(drop=True)
    if len(df) < 50:
        return None

    ema_sig = calculate_ema_signal(df)

    prices_series = df["close"]
    prices = prices_series.tolist()

    deltas = np.diff(prices)
    gains  = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    # Wilder's Smoothed Moving Average (classic RSI)
    avg_gain = np.mean(gains[:14])
    avg_loss = np.mean(losses[:14])
    for i in range(14, len(gains)):
        avg_gain = (avg_gain * 13 + gains[i]) / 14
        avg_loss = (avg_loss * 13 + losses[i]) / 14
    rsi = 100 - (100 / (1 + (avg_gain / avg_loss))) if avg_loss != 0 else 100

    sma_20 = np.mean(prices[-20:])
    std_20 = np.std(prices[-20:])
    upper, lower = sma_20 + (2 * std_20), sma_20 - (2 * std_20)

    sma_50    = np.mean(prices[-50:])
    trend_up  = prices[-1] > sma_50

    ema_12      = pd.Series(prices_series).ewm(span=12, adjust=False).mean()
    ema_26      = pd.Series(prices_series).ewm(span=26, adjust=False).mean()
    macd_line   = ema_12 - ema_26
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False).mean()
    macd_up     = bool(macd_line.iloc[-1] > signal_line.iloc[-1])

    high_series, low_series = pd.Series(df["high"]), pd.Series(df["low"])
    tr = pd.concat([
        (high_series - low_series),
        (high_series - pd.Series(prices_series).shift()).abs(),
        (low_series  - pd.Series(prices_series).shift()).abs()
    ], axis=1).max(axis=1)
    atr_val       = float(pd.Series(tr).rolling(14).mean().iloc[-1])
    volatility_low = bool(float(pd.Series(tr).iloc[-1]) < (atr_val * 1.5))

    tp             = (high_series + low_series + pd.Series(prices_series)) / 3
    volume_series  = pd.Series(df["volume"])
    vwap_series    = (tp * volume_series).cumsum() / volume_series.replace(0, 1).cumsum()
    vwap_up        = bool(pd.Series(prices_series).iloc[-1] > vwap_series.iloc[-1])

    vol_spike = volume_series.iloc[-1] > volume_series.rolling(20).mean().iloc[-1] * 1.5

    # Real ADX — Wilder's smoothing (14 period)
    high_arr = df["high"].values
    low_arr  = df["low"].values
    tr_arr   = tr.values
    plus_dm  = np.zeros(len(high_arr))
    minus_dm = np.zeros(len(high_arr))
    for i in range(1, len(high_arr)):
        up   = high_arr[i] - high_arr[i - 1]
        down = low_arr[i - 1] - low_arr[i]
        plus_dm[i]  = up   if (up > down and up > 0)   else 0.0
        minus_dm[i] = down if (down > up and down > 0) else 0.0
    s_tr  = float(np.sum(tr_arr[1:15]))
    s_pdm = float(np.sum(plus_dm[1:15]))
    s_mdm = float(np.sum(minus_dm[1:15]))
    dx_vals = []
    for i in range(15, len(tr_arr)):
        s_tr  = s_tr  - s_tr  / 14 + tr_arr[i]
        s_pdm = s_pdm - s_pdm / 14 + plus_dm[i]
        s_mdm = s_mdm - s_mdm / 14 + minus_dm[i]
        pdi    = 100 * s_pdm / s_tr if s_tr else 0.0
        mdi    = 100 * s_mdm / s_tr if s_tr else 0.0
        di_sum = pdi + mdi
        dx_vals.append(100 * abs(pdi - mdi) / di_sum if di_sum else 0.0)
    if len(dx_vals) >= 14:
        adx_val = float(np.mean(dx_vals[:14]))
        for v in dx_vals[14:]:
            adx_val = (adx_val * 13 + v) / 14
    else:
        adx_val = 0.0
    adx_weak = adx_val <= 25   # ADX ≤ 25 → trend দুর্বল

    side_ema = "BUY" if prices[-1] > sma_50 else "SELL"

    def rsi_signal(rsi_val, side):
        return (side == "BUY" and rsi_val < 40) or (side == "SELL" and rsi_val > 60)

    # --- Real Candlestick Pattern Detection ---
    candle_pattern = False
    if len(df) >= 3:
        o1, h1, l1, c1 = df['open'].iloc[-3], df['high'].iloc[-3], df['low'].iloc[-3], df['close'].iloc[-3]
        o2, h2, l2, c2 = df['open'].iloc[-2], df['high'].iloc[-2], df['low'].iloc[-2], df['close'].iloc[-2]
        o3, h3, l3, c3 = df['open'].iloc[-1], df['high'].iloc[-1], df['low'].iloc[-1], df['close'].iloc[-1]
        body3        = abs(c3 - o3)
        range3       = h3 - l3 if (h3 - l3) > 0 else 1e-10
        body2        = abs(c2 - o2)
        upper_wick3  = h3 - max(o3, c3)
        lower_wick3  = min(o3, c3) - l3
        # Doji: body খুব ছোট (range এর ১০% এর কম)
        doji              = (body3 / range3) < 0.1
        # Pin Bar — Hammer (BUY) / Shooting Star (SELL)
        bullish_pin       = (lower_wick3 >= 2 * body3 and upper_wick3 <= body3 and side_ema == "BUY")
        bearish_pin       = (upper_wick3 >= 2 * body3 and lower_wick3 <= body3 and side_ema == "SELL")
        # Engulfing
        bullish_engulfing = (c2 < o2 and c3 > o3 and o3 <= c2 and c3 >= o2 and side_ema == "BUY")
        bearish_engulfing = (c2 > o2 and c3 < o3 and o3 >= c2 and c3 <= o2 and side_ema == "SELL")
        # Morning Star (BUY) / Evening Star (SELL)
        first_body    = abs(c1 - o1)
        morning_star  = (c1 < o1 and body2 < first_body * 0.3 and c3 > o3 and c3 > (o1 + c1) / 2 and side_ema == "BUY")
        evening_star  = (c1 > o1 and body2 < first_body * 0.3 and c3 < o3 and c3 < (o1 + c1) / 2 and side_ema == "SELL")
        candle_pattern = any([doji, bullish_pin, bearish_pin, bullish_engulfing, bearish_engulfing, morning_star, evening_star])

    ind_scores = {
        'trend':        1.0 if (trend_up and prices[-1] > sma_50) or (not trend_up and prices[-1] < sma_50) else 0.0,
        'rsi':          1.0 if rsi_signal(rsi, side_ema) else 0.0,
        'bb':           1.0 if prices[-1] <= lower or prices[-1] >= upper else 0.0,
        'vol':          1.0 if vol_spike else 0.0,
        'adx':          1.0 if not adx_weak else 0.0,
        'ema_crossover':1.0 if ema_sig != "HOLD" else 0.0,
        'macd':         1.0 if (macd_up and side_ema == "BUY") or (not macd_up and side_ema == "SELL") else 0.0,
        'atr':          1.0 if volatility_low else 0.0,
        'vwap':         1.0 if (vwap_up and side_ema == "BUY") or (not vwap_up and side_ema == "SELL") else 0.0,
        'candle':       1.0 if candle_pattern else 0.0
    }

    weights           = get_weights(symbol)
    symbol_confidence = sum(weights.get(k, 0) * v for k, v in ind_scores.items())

    return {
        "rsi": rsi, "upper": upper, "lower": lower, "sma_50": sma_50, "trend_up": trend_up,
        "vol_spike": vol_spike, "adx_low": adx_weak, "curr_p": prices[-1], "ema_sig": ema_sig,
        "macd_up": macd_up, "volatility_low": volatility_low, "vwap_up": vwap_up,
        "candle_signal": candle_pattern, "atr_val": atr_val, "symbol_confidence": symbol_confidence
    }


def get_market_data(symbol, api_key):
    if not api_key:
        return None
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1min&outputsize=75&apikey={api_key.strip()}"
    try:
        r = requests.get(url, timeout=15).json()
        if "values" in r:
            return r["values"]
        if symbol in r and "values" in r[symbol]:
            return r[symbol]["values"]
        return None
    except:
        return None


def get_batch_market_data(symbols_list, api_key):
    if not api_key or not symbols_list:
        return {}
    url = f"https://api.twelvedata.com/time_series?symbol={quote(','.join(symbols_list), safe=',')}&interval=1min&outputsize=75&apikey={api_key.strip()}"
    try:
        r = requests.get(url, timeout=30).json()
        result = {}
        for symbol in symbols_list:
            if symbol in r and "values" in r[symbol]:
                result[symbol] = r[symbol]["values"]
            elif "values" in r and len(symbols_list) == 1:
                result[symbol] = r["values"]
        return result
    except Exception as e:
        print("Batch API Error:", e)
        return {}


def save_active_trade(trade):
    try:
        with db_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO active_trade (id, symbol, entry_price, side, start_time, expiry_time, rec_time, is_active, indicators_status)
                VALUES (1, %s, %s, %s, %s, %s, %s, TRUE, %s)
                ON CONFLICT (id) DO UPDATE SET
                    symbol = EXCLUDED.symbol, entry_price = EXCLUDED.entry_price, side = EXCLUDED.side,
                    start_time = EXCLUDED.start_time, expiry_time = EXCLUDED.expiry_time,
                    rec_time = EXCLUDED.rec_time, is_active = TRUE, indicators_status = EXCLUDED.indicators_status
            """, (
                trade['symbol'], trade['entry_price'], trade['side'],
                trade['start_time'], trade['expiry_time'],
                trade['rec_time'], json.dumps(trade['indicators_status'])
            ))
            conn.commit()
            cur.close()
            return True
    except Exception as e:
        print(f"Save Trade Error: {e}")
        return False


def load_active_trade():
    try:
        with db_conn() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute("SELECT * FROM active_trade WHERE id = 1 AND is_active = TRUE")
            row = cur.fetchone()
            cur.close()
            return row
    except:
        return None


def clear_active_trade():
    try:
        with db_conn() as conn:
            cur = conn.cursor()
            cur.execute("UPDATE active_trade SET is_active = FALSE WHERE id = 1")
            conn.commit()
            cur.close()
    except:
        pass


def get_candle_at_time(symbol, target_time, api_key):
    values = get_market_data(symbol, api_key)
    if not values:
        return None
    # UTC-based exact minute match — timezone mismatch ও wrong candle দুটোই বন্ধ
    target_dt = datetime.utcfromtimestamp(int(target_time)).replace(second=0, microsecond=0)
    for candle in values:
        try:
            candle_dt = datetime.strptime(candle.get('datetime', ''), "%Y-%m-%d %H:%M:%S")
            if candle_dt.replace(second=0, microsecond=0) == target_dt:
                return float(candle['close'])
        except:
            continue
    return None  # exact match না পেলে None — caller fallback handle করবে


def check_result():
    global current_trade, _result_sent_for
    if not current_trade:
        current_trade = load_active_trade()
        if not current_trade:
            return

    now    = time.time()
    expiry = current_trade['expiry_time']

    if now < expiry + 65:
        return

    # এই trade-এর result আগেই পাঠানো হয়ে গেছে কিনা check করো
    trade_key = f"{current_trade['symbol']}_{current_trade['start_time']}"
    if trade_key == _result_sent_for:
        clear_active_trade()
        current_trade = None
        return

    entry_key = os.getenv("RESULT_KEY_ENTRY", API_KEYS[0])
    exit_key  = os.getenv("RESULT_KEY_EXIT", API_KEYS[1] if len(API_KEYS) > 1 else API_KEYS[0])

    entry_price = get_candle_at_time(current_trade['symbol'], current_trade['start_time'], entry_key)
    exit_price  = get_candle_at_time(current_trade['symbol'], expiry, exit_key)

    if entry_price is None or exit_price is None:
        if now < expiry + 120:
            return
        if entry_price is None:
            entry_price = float(current_trade['entry_price'])
        if exit_price is None:
            values      = get_market_data(current_trade['symbol'], exit_key)
            exit_price  = float(values[0]['close']) if values else entry_price

    win = (exit_price > entry_price) if current_trade['side'] == "BUY" else (exit_price < entry_price)
    update_stats(win)

    if current_trade.get('indicators_status'):
        status = current_trade['indicators_status']
        if isinstance(status, str):
            status = json.loads(status)
        update_weights(current_trade['symbol'], status, win)

    s        = get_stats()
    win_rate = (s['wins'] / s['total_trades'] * 100) if s['total_trades'] > 0 else 0
    result_emoji = "✅ WIN" if win else "❌ LOSS"
    msg = (
        f"🏁 TRADE RESULT\n\n"
        f"📊 Asset: {current_trade['symbol']}\n"
        f"🏆 Result: {result_emoji}\n"
        f"🚀 Entry: {entry_price}\n"
        f"🏁 Exit: {exit_price}\n"
        f"📈 Win Rate: {win_rate:.1f}%"
    )
    trade_symbol    = current_trade['symbol']
    _result_sent_for = trade_key  # এই trade process করা হয়ে গেছে, আর পাঠাবে না
    clear_active_trade()          # message পাঠানোর আগে DB clear — restart হলেও duplicate আসবে না
    current_trade   = None
    send_telegram_msg(msg)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Verified Result: {trade_symbol} - {result_emoji} | Entry: {entry_price} | Exit: {exit_price}")


def fetch_and_analyze_batch(symbols_chunk, api_key):
    batch_data = get_batch_market_data(symbols_chunk, api_key)
    results    = []
    for symbol, values in batch_data.items():
        ind = calculate_indicators(values, symbol)
        if not ind:
            continue

        side_ema = "BUY" if ind['curr_p'] > ind['sma_50'] else "SELL"

        def rsi_signal(rsi_val, side):
            return (side == "BUY" and rsi_val < 40) or (side == "SELL" and rsi_val > 60)

        status = {
            'trend':        bool((ind['trend_up'] and ind['curr_p'] > ind['sma_50']) or (not ind['trend_up'] and ind['curr_p'] < ind['sma_50'])),
            'rsi':          bool(rsi_signal(ind['rsi'], side_ema)),
            'bb':           bool(ind['curr_p'] <= ind['lower'] or ind['curr_p'] >= ind['upper']),
            'vol':          bool(ind['vol_spike']),
            'adx':          bool(not ind['adx_low']),
            'ema_crossover':bool(ind['ema_sig'] != "HOLD"),
            'macd':         bool((ind['macd_up'] and side_ema == "BUY") or (not ind['macd_up'] and side_ema == "SELL")),
            'atr':          bool(ind['volatility_low']),
            'vwap':         bool((ind['vwap_up'] and side_ema == "BUY") or (not ind['vwap_up'] and side_ema == "SELL")),
            'candle':       bool(ind['candle_signal'])
        }

        results.append({
            'symbol': symbol,
            'curr_p': ind['curr_p'],
            'side':   side_ema,
            'score':  ind['symbol_confidence'],
            'status': status,
            'ind':    ind
        })
    return results


def run_scanner():
    global current_trade, last_scan_minute

    current_trade = load_active_trade()
    if current_trade:
        return

    now = datetime.now()
    if now.minute == last_scan_minute or not (0 <= now.second <= 10):
        return

    last_scan_minute = now.minute
    print(f"[{now.strftime('%H:%M:%S')}] Scanning markets for signals...")

    current_trade = load_active_trade()
    if current_trade:
        return

    all_results = []
    # প্রতিটি key-এ ৩টি symbol batch — ৭টি key × ৩টি = ২১টি market
    chunks = [SYMBOLS[i:i + 3] for i in range(0, len(SYMBOLS), 3)]

    with ThreadPoolExecutor(max_workers=7) as executor:
        futures = [
            executor.submit(fetch_and_analyze_batch, chunk, API_KEYS[i % len(API_KEYS)])
            for i, chunk in enumerate(chunks)
        ]
        for future in futures:
            res = future.result()
            if res:
                all_results.extend(res)

    print(f"[{now.strftime('%H:%M:%S')}] Analyzed {len(all_results)} symbols")

    if all_results:
        best      = max(all_results, key=lambda x: x['score'])
        weights   = get_weights(best['symbol'])
        max_score = sum(weights.values())
        conf      = round(min((best['score'] / max_score * 100) if max_score > 0 else 0, 99.0), 1)
        action    = "Strong trade" if conf >= 80 else "Normal trade" if conf >= 60 else "Weak / scalp" if conf >= 40 else "Educational"

        # ATR % of price দিয়ে volatility measure করি
        atr_val = best['ind']['atr_val']
        curr_p  = best['curr_p']
        atr_pct = (atr_val / curr_p * 100) if curr_p > 0 else 0.2

        # High volatility → trade দ্রুত resolve হয় → ছোট expiry
        # Low volatility  → market ধীরে move করে → বড় expiry
        if atr_pct > 0.3:       # High volatility
            rec_time = 5 if conf <= 60 else 7
        elif atr_pct > 0.1:     # Medium volatility
            rec_time = 8 if conf <= 60 else 10
        else:                   # Low volatility
            rec_time = 12 if conf <= 60 else 15

        start  = datetime.now().replace(second=0, microsecond=0) + timedelta(minutes=1)
        expiry = start + timedelta(minutes=rec_time)

        current_trade = {
            'symbol':            best['symbol'],
            'entry_price':       best['curr_p'],
            'side':              best['side'],
            'start_time':        start.timestamp(),
            'expiry_time':       expiry.timestamp(),
            'rec_time':          rec_time,
            'indicators_status': best['status']
        }

        saved = save_active_trade(current_trade)

        if saved:  # DB-তে save সফল হলে তবেই message পাঠাবে
            msg = (
                f"SIGNAL ({action})\n\n"
                f"{best['symbol']} -> {best['side']}\n"
                f"Confidence: {conf:.1f}%\n"
                f"Time: {rec_time} Min\n\n"
                f"RSI: {best['ind']['rsi']:.1f} | Vol: {'High' if best['ind']['vol_spike'] else 'Normal'}\n\n"
                f"Place trade on Quotex exactly at the start of next minute (00s)!"
            )
            send_telegram_msg(msg)


if __name__ == "__main__":
    init_db()

    threading.Thread(target=run_web, daemon=True).start()

    print("Bot is running with Adaptive Confidence System...")

    while True:
        try:
            run_scanner()
            check_result()
        except Exception as e:
            print(f"System Error: {e}")
        time.sleep(1)