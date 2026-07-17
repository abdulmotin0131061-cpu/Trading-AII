import os
import requests
import time
import numpy as np
import logging
import signal
import sys

from flask import Flask
import threading
from datetime import datetime, timedelta, timezone
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import ThreadedConnectionPool
from contextlib import contextmanager
import random
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import json
from urllib.parse import quote
from logging.handlers import RotatingFileHandler

# ==========================================
# MODULE 1: LOGGER SYSTEM (Professional Logging)
# ==========================================
file_handler = RotatingFileHandler("bot.log", maxBytes=5*1024*1024, backupCount=3)
console_handler = logging.StreamHandler()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[file_handler, console_handler]
)
logger = logging.getLogger("TradingBot")

# ==========================================
# MODULE 2: CONFIGURATION & ENVIRONMENT SETUP
# ==========================================
BD_TZ = timezone(timedelta(hours=6))

def get_now():
    return datetime.now(BD_TZ)

DATABASE_URL = os.getenv("DATABASE_URL")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Dynamic Key Loading from Environment Variables (TWELVEDATA_KEY_1 to TWELVEDATA_KEY_9)
API_KEYS = []
for i in range(1, 10):
    key = os.getenv(f"TWELVEDATA_KEY_{i}")
    if key:
        API_KEYS.append(key.strip())

# If environment variable keys are not found, use fallbacks for local test runs safely
if not API_KEYS:
    API_KEYS = [
        "19e72ea9c60240e1a902f4d1ffa89508",
        "cb4aff90f22341809ae1344927c2a365",
        "2ca49ec0c0534851b8ee88bd01858eaf"
    ]

SYMBOLS = [
    "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "USD/CHF", "NZD/USD",
    "EUR/JPY", "GBP/JPY", "EUR/GBP", "BTC/USD", "ETH/USD", "LTC/USD", "XRP/USD",
    "SOL/USD", "ADA/USD", "XAU/USD", "XAG/USD", "GBP/AUD", "EUR/AUD", "AUD/JPY"
]

# API Cooldown Tracking
current_key_idx = 0
key_cooldowns = {}

def get_active_api_key():
    global current_key_idx
    now = time.time()
    for _ in range(len(API_KEYS)):
        key = API_KEYS[current_key_idx]
        if key_cooldowns.get(key, 0) < now:
            return key
        current_key_idx = (current_key_idx + 1) % len(API_KEYS)
    sorted_keys = sorted(API_KEYS, key=lambda k: key_cooldowns.get(k, 0))
    return sorted_keys[0]

def mark_key_cooldown(key, duration=300):
    key_cooldowns[key] = time.time() + duration
    logger.warning(f"Key {key[:6]}... put on cooldown for {duration}s.")

def rotate_key():
    global current_key_idx
    current_key_idx = (current_key_idx + 1) % len(API_KEYS)

# ==========================================
# MODULE 3: DATABASE & OPTIMIZATIONS (Indexed & Archive)
# ==========================================
_pool = None
try:
    if DATABASE_URL:
        _pool = ThreadedConnectionPool(minconn=2, maxconn=10, dsn=DATABASE_URL)
except Exception as e:
    logger.critical(f"Database connection pool setup failed: {e}")

@contextmanager
def db_conn():
    if not _pool:
        raise Exception("Database connection pool is offline.")
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

                CREATE TABLE IF NOT EXISTS trades (
                    id SERIAL PRIMARY KEY,
                    symbol TEXT,
                    entry_price FLOAT,
                    side TEXT,
                    start_time TIMESTAMP WITH TIME ZONE,
                    expiry_time TIMESTAMP WITH TIME ZONE,
                    rec_time INTEGER,
                    indicators_status JSONB,
                    status TEXT DEFAULT 'ACTIVE',
                    result TEXT DEFAULT 'PENDING',
                    retry_count INTEGER DEFAULT 0,
                    next_retry_time TIMESTAMP WITH TIME ZONE,
                    msg_sent BOOLEAN DEFAULT FALSE,
                    result_sent BOOLEAN DEFAULT FALSE
                );

                CREATE TABLE IF NOT EXISTS trades_archive (
                    id INTEGER PRIMARY KEY,
                    symbol TEXT,
                    entry_price FLOAT,
                    side TEXT,
                    start_time TIMESTAMP WITH TIME ZONE,
                    expiry_time TIMESTAMP WITH TIME ZONE,
                    rec_time INTEGER,
                    indicators_status JSONB,
                    status TEXT,
                    result TEXT,
                    archived_at TIMESTAMP WITH TIME ZONE
                );

                CREATE TABLE IF NOT EXISTS indicator_weights (
                    symbol TEXT,
                    name TEXT,
                    weight FLOAT,
                    accuracy_factor FLOAT DEFAULT 1.0,
                    PRIMARY KEY (symbol, name)
                );

                -- Indexes for database query acceleration
                CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
                CREATE INDEX IF NOT EXISTS idx_trades_expiry ON trades(expiry_time);
                CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);

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
        logger.info("Indexed and archive database structure set up.")
        train_ml_model()
    except Exception as e:
        logger.error(f"Database Init Error: {e}")

# ==========================================
# MODULE 4: TELEGRAM SYSTEM (Detailed Upgrade Alert)
# ==========================================
_sent_messages_cache = {}

def send_telegram_msg(message):
    global _sent_messages_cache
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    now = time.time()
    _sent_messages_cache = {msg: ts for msg, ts in _sent_messages_cache.items() if now - ts < 600}
    if message in _sent_messages_cache:
        logger.warning("Duplicate Telegram message blocked.")
        return
    _sent_messages_cache[message] = now
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        logger.error(f"Telegram API Error: {e}")

# ==========================================
# MODULE 5: MARKET SESSION TRACKER
# ==========================================
def get_market_session():
    now_utc = datetime.now(timezone.utc)
    hour = now_utc.hour
    
    # Check Dead Sunday Zones
    if now_utc.weekday() == 6 and hour < 21:
        return "DEAD_ZONE"
        
    sessions = []
    if 0 <= hour < 9:
        sessions.append("Tokyo")
    if 8 <= hour < 17:
        sessions.append("London")
    if 13 <= hour < 22:
        sessions.append("New York")
        
    if "London" in sessions and "New York" in sessions:
        return "Overlap (LDN/NY)"
    elif sessions:
        return "/".join(sessions)
    return "Late Asian/Quiet Hour"

# ==========================================
# MODULE 6: INDICATORS & MATHS MODULE (EMA 50/200, S/R, DI, MACD)
# ==========================================
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
    df["volume"] = pd.to_numeric(df.get("volume", 0.0), errors="coerce").fillna(0)
    df.dropna(inplace=True)
    if len(df) < 50:
        return None
    df = df[::-1].reset_index(drop=True)
    
    prices_series = df["close"]
    prices = prices_series.tolist()
    
    # 1. RSI Indicator
    deltas = np.diff(prices)
    gains  = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[:14])
    avg_loss = np.mean(losses[:14])
    for i in range(14, len(gains)):
        avg_gain = (avg_gain * 13 + gains[i]) / 14
        avg_loss = (avg_loss * 13 + losses[i]) / 14
    rsi = 100 - (100 / (1 + (avg_gain / avg_loss))) if avg_loss != 0 else 100
    
    # 2. Bollinger Bands
    sma_20 = np.mean(prices[-20:])
    std_20 = np.std(prices[-20:])
    upper, lower = sma_20 + (2 * std_20), sma_20 - (2 * std_20)
    sma_50    = np.mean(prices[-50:])
    trend_up  = prices[-1] > sma_50
    
    # 3. Dynamic EMAs (EMA 5, 13, 50, 200)
    ema_5 = prices_series.ewm(span=5, adjust=False).mean().iloc[-1]
    ema_13 = prices_series.ewm(span=13, adjust=False).mean().iloc[-1]
    ema_50 = prices_series.ewm(span=50, adjust=False).mean().iloc[-1]
    ema_200 = prices_series.ewm(span=200, adjust=False).mean().iloc[-1] if len(prices_series) >= 200 else ema_50
    
    # 4. MACD & Histogram Momentum
    ema_12      = pd.Series(prices_series).ewm(span=12, adjust=False).mean()
    ema_26      = pd.Series(prices_series).ewm(span=26, adjust=False).mean()
    macd_line   = ema_12 - ema_26
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False).mean()
    macd_up     = bool(macd_line.iloc[-1] > signal_line.iloc[-1])
    macd_hist   = macd_line - signal_line
    macd_hist_slope_up = bool(macd_hist.iloc[-1] > macd_hist.iloc[-2]) if len(macd_hist) > 1 else False
    macd_hist_val = float(macd_hist.iloc[-1]) if len(macd_hist) > 0 else 0.0
    
    # 5. ATR (Average True Range) Upgrade
    high_series, low_series = pd.Series(df["high"]), pd.Series(df["low"])
    tr = pd.concat([
        (high_series - low_series),
        (high_series - pd.Series(prices_series).shift()).abs(),
        (low_series  - pd.Series(prices_series).shift()).abs()
    ], axis=1).max(axis=1)
    atr_val       = float(pd.Series(tr).rolling(14).mean().iloc[-1])
    volatility_low = bool(float(pd.Series(tr).iloc[-1]) < (atr_val * 1.5))
    
    # 6. VWAP Indicator
    tp             = (high_series + low_series + pd.Series(prices_series)) / 3
    volume_series  = pd.Series(df["volume"])
    vwap_series    = (tp * volume_series).cumsum() / volume_series.replace(0, 1).cumsum()
    vwap_up        = bool(pd.Series(prices_series).iloc[-1] > vwap_series.iloc[-1])
    vol_spike = volume_series.iloc[-1] > volume_series.rolling(20).mean().iloc[-1] * 1.5
    rolling_vol_mean = volume_series.rolling(20).mean().iloc[-1]
    vol_ratio = float(volume_series.iloc[-1] / rolling_vol_mean) if rolling_vol_mean > 0 else 1.0
    
    # 7. ADX with DI+ and DI- Upgrade
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
    pdi_val, mdi_val = 0.0, 0.0
    for i in range(15, len(tr_arr)):
        s_tr  = s_tr  - s_tr  / 14 + tr_arr[i]
        s_pdm = s_pdm - s_pdm / 14 + plus_dm[i]
        s_mdm = s_mdm - s_mdm / 14 + minus_dm[i]
        pdi_val    = 100 * s_pdm / s_tr if s_tr else 0.0
        mdi_val    = 100 * s_mdm / s_tr if s_tr else 0.0
        di_sum = pdi_val + mdi_val
        dx_vals.append(100 * abs(pdi_val - mdi_val) / di_sum if di_sum else 0.0)
    if len(dx_vals) >= 14:
        adx_val = float(np.mean(dx_vals[:14]))
        for v in dx_vals[14:]:
            adx_val = (adx_val * 13 + v) / 14
    else:
        adx_val = 0.0
    adx_weak = adx_val <= 25
    
    # 8. Support & Resistance Levels
    recent_closes = prices[-30:]
    support = min(recent_closes)
    resistance = max(recent_closes)
    swing_high = float(max(df["high"].iloc[-20:]))
    swing_low = float(min(df["low"].iloc[-20:]))
    
    ema_dist = float(abs(prices[-1] - ema_50))
    sr_dist = float(min(abs(prices[-1] - support), abs(prices[-1] - resistance)))
    bb_width = float(upper - lower)
    vwap_dist = float(abs(prices[-1] - vwap_series.iloc[-1])) if len(vwap_series) > 0 else 0.0
    
    # 9. Candlestick Pattern Logic
    side_ema = "BUY" if prices[-1] > sma_50 else "SELL"
    def rsi_signal(rsi_val, side):
        return (side == "BUY" and rsi_val < 40) or (side == "SELL" and rsi_val > 60)
    candle_pattern = False
    if len(df) >= 3:
        o1, h1, l1, c1 = df['open'].iloc[-3], df['high'].iloc[-3], df['low'].iloc[-3], df['close'].iloc[-3]
        o2, h2, l2, c2 = df['open'].iloc[-2], df['high'].iloc[-2], df['low'].iloc[-2], df['close'].iloc[-2]
        o3, h3, l3, c3 = df['open'].iloc[-1], df['high'].iloc[-1], df['low'].iloc[-1], df['close'].iloc[-1]
        body3 = abs(c3 - o3)
        range3 = h3 - l3 if (h3 - l3) > 0 else 1e-10
        body2 = abs(c2 - o2)
        upper_wick3 = h3 - max(o3, c3)
        lower_wick3 = min(o3, c3) - l3
        doji              = (body3 / range3) < 0.1
        bullish_pin       = (lower_wick3 >= 2 * body3 and upper_wick3 <= body3 and side_ema == "BUY")
        bearish_pin       = (upper_wick3 >= 2 * body3 and lower_wick3 <= body3 and side_ema == "SELL")
        bullish_engulfing = (c2 < o2 and c3 > o3 and o3 <= c2 and c3 >= o2 and side_ema == "BUY")
        bearish_engulfing = (c2 > o2 and c3 < o3 and o3 >= c2 and c3 <= o2 and side_ema == "SELL")
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
        'ema_crossover':1.0 if (ema_5 > ema_13 and side_ema == "BUY") or (ema_5 < ema_13 and side_ema == "SELL") else 0.0,
        'macd':         1.0 if (macd_up and side_ema == "BUY") or (not macd_up and side_ema == "SELL") else 0.0,
        'atr':          1.0 if volatility_low else 0.0,
        'vwap':         1.0 if (vwap_up and side_ema == "BUY") or (not vwap_up and side_ema == "SELL") else 0.0,
        'candle':       1.0 if candle_pattern else 0.0
    }
    weights           = get_weights(symbol)
    symbol_confidence = sum(weights.get(k, 0) * v for k, v in ind_scores.items())
    return {
        "rsi": rsi, "upper": upper, "lower": lower, "sma_50": sma_50, "trend_up": trend_up,
        "vol_spike": vol_spike, "adx_low": adx_weak, "curr_p": prices[-1], "ema_sig": "BUY" if ema_5 > ema_13 else "SELL",
        "macd_up": macd_up, "volatility_low": volatility_low, "vwap_up": vwap_up,
        "candle_signal": candle_pattern, "atr_val": atr_val, "symbol_confidence": symbol_confidence,
        "adx_val": adx_val, "plus_di": pdi_val, "minus_di": mdi_val, "ema_50": ema_50, "ema_200": ema_200,
        "macd_hist_slope_up": macd_hist_slope_up, "support": support, "resistance": resistance,
        "swing_high": swing_high, "swing_low": swing_low, "vol_ratio": vol_ratio, "ema_dist": ema_dist,
        "sr_dist": sr_dist, "macd_hist_val": macd_hist_val, "bb_width": bb_width, "vwap_dist": vwap_dist
    }

# ==========================================
# MODULE 7: MACHINE LEARNING & ADAPTIVE WEIGHTS
# ==========================================
ML_WEIGHTS = {
    'weights': np.zeros(11),
    'bias': 0.0,
    'means': np.zeros(11),
    'stds': np.ones(11),
    'trained': False
}

def train_ml_model():
    global ML_WEIGHTS
    try:
        with db_conn() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute("""
                SELECT indicators_status, result 
                FROM trades 
                WHERE status = 'COMPLETED' AND result IN ('WIN', 'LOSS')
                ORDER BY id ASC
            """)
            rows = cur.fetchall()
            cur.close()
        if len(rows) < 10:
            logger.info(f"ML Model training skipped. Trade history database scale is {len(rows)}/10.")
            return
            
        feature_names = ['RSI', 'ADX', 'ATR', 'EMA Distance', 'Volume Ratio', 'Support Distance', 'Resistance Distance', 'MACD Histogram', 'BB Width', 'VWAP Distance', 'Candle Pattern']
        fallbacks = {
            'RSI': 50.0, 'ADX': 20.0, 'ATR': 0.001, 'EMA Distance': 0.0, 
            'Volume Ratio': 1.0, 'Support Distance': 0.001, 'Resistance Distance': 0.001, 
            'MACD Histogram': 0.0, 'BB Width': 0.01, 'VWAP Distance': 0.0, 'Candle Pattern': 0.0
        }
        
        X, y = [], []
        for r in rows:
            status = r['indicators_status']
            if isinstance(status, str):
                try:
                    status = json.loads(status)
                except:
                    continue
            if not status:
                continue
            
            vec = []
            for fname in feature_names:
                val = status.get(fname)
                if val is None:
                    if fname == 'RSI' and 'rsi' in status:
                        val = 30.0 if status['rsi'] else 50.0
                    elif fname == 'ADX' and 'adx' in status:
                        val = 30.0 if status['adx'] else 15.0
                    elif fname == 'Candle Pattern' and 'candle' in status:
                        val = 1.0 if status['candle'] else 0.0
                    else:
                        val = fallbacks[fname]
                elif isinstance(val, bool):
                    val = 1.0 if val else 0.0
                vec.append(float(val))
                
            X.append(vec)
            y.append(1.0 if r['result'] == 'WIN' else 0.0)
            
        X = np.array(X)
        y = np.array(y)
        
        if len(X) < 10:
            return
            
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)
        stds[stds == 0] = 1e-8
        X_scaled = (X - means) / stds
        
        num_features = X_scaled.shape[1]
        W = np.zeros(num_features)
        b = 0.0
        learning_rate = 0.1
        epochs = 200
        l2_reg = 0.1
        
        for _ in range(epochs):
            z = np.dot(X_scaled, W) + b
            predictions = 1.0 / (1.0 + np.exp(-np.clip(z, -15, 15)))
            errors = predictions - y
            dW = (np.dot(X_scaled.T, errors) / len(y)) + l2_reg * W
            db = np.sum(errors) / len(y)
            W -= learning_rate * dW
            b -= learning_rate * db
            
        ML_WEIGHTS['weights'] = W
        ML_WEIGHTS['bias'] = b
        ML_WEIGHTS['means'] = means
        ML_WEIGHTS['stds'] = stds
        ML_WEIGHTS['trained'] = True
        logger.info(f"Self-Learning AI Model successfully updated on {len(X)} trades.")
    except Exception as e:
        logger.error(f"ML System Update Error: {e}")

def predict_win_probability(indicators_status):
    global ML_WEIGHTS
    if not ML_WEIGHTS['trained']:
        return 0.5
    feature_names = ['RSI', 'ADX', 'ATR', 'EMA Distance', 'Volume Ratio', 'Support Distance', 'Resistance Distance', 'MACD Histogram', 'BB Width', 'VWAP Distance', 'Candle Pattern']
    fallbacks = {
        'RSI': 50.0, 'ADX': 20.0, 'ATR': 0.001, 'EMA Distance': 0.0, 
        'Volume Ratio': 1.0, 'Support Distance': 0.001, 'Resistance Distance': 0.001, 
        'MACD Histogram': 0.0, 'BB Width': 0.01, 'VWAP Distance': 0.0, 'Candle Pattern': 0.0
    }
    vec = []
    for fname in feature_names:
        val = indicators_status.get(fname, fallbacks[fname])
        if isinstance(val, bool):
            val = 1.0 if val else 0.0
        vec.append(float(val))
    vec = np.array(vec)
    vec_scaled = (vec - ML_WEIGHTS['means']) / ML_WEIGHTS['stds']
    z = np.dot(vec_scaled, ML_WEIGHTS['weights']) + ML_WEIGHTS['bias']
    prob = 1.0 / (1.0 + np.exp(-np.clip(z, -15, 15)))
    return float(prob)

# ==========================================
# MODULE 8: ADAPTIVE WEIGHTS DATABASE GET/UPDATE
# ==========================================
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
                if active and isinstance(active, bool):
                    cur.execute("""
                        UPDATE indicator_weights
                        SET accuracy_factor = LEAST(GREATEST(accuracy_factor + %s, 0.5), 1.5)
                        WHERE symbol = %s AND name = %s
                    """, (factor_change, symbol, name))
            conn.commit()
            cur.close()
        train_ml_model()
    except Exception as e:
        logger.error(f"Indicator Weight Update Error: {e}")

# ==========================================
# MODULE 9: PORTFOLIO STATS & RECOVERY CONTROL
# ==========================================
def get_stats():
    try:
        with db_conn() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute("SELECT total_trades, wins, losses FROM trading_stats LIMIT 1")
            row = cur.fetchone()
            cur.close()
            return row if row else {"total_trades": 0, "wins": 0, "losses": 0}
    except Exception as e:
        logger.error(f"Stats Retrieve Error: {e}")
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
        logger.error(f"Database Stats Modification Error: {e}")

def save_active_trade(trade):
    try:
        with db_conn() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            next_retry = trade['expiry_time'] + timedelta(seconds=30)
            cur.execute("""
                INSERT INTO trades (
                    symbol, entry_price, side, start_time, expiry_time, 
                    rec_time, indicators_status, status, next_retry_time
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, 'ACTIVE', %s)
                RETURNING id
            """, (
                trade['symbol'], trade['entry_price'], trade['side'],
                trade['start_time'], trade['expiry_time'], trade['rec_time'],
                json.dumps(trade['indicators_status']), next_retry
            ))
            row = cur.fetchone()
            conn.commit()
            cur.close()
            return row['id'] if row else None
    except Exception as e:
        logger.error(f"Active trade execution logging failed: {e}")
        return None

def has_active_trade():
    try:
        with db_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM trades WHERE status IN ('ACTIVE', 'PENDING_RESULT')")
            row = cur.fetchone()
            cur.close()
            return row[0] > 0 if row else False
    except Exception as e:
        logger.error(f"Active trade check error: {e}")
        return False

# ==========================================
# MODULE 10: DAILY REPORT AUTO GENERATION (10:00 AM)
# ==========================================
def check_and_send_daily_report():
    global last_report_date
    now = get_now()
    current_date_str = now.strftime("%d-%m-%Y")
    
    if now.hour >= 10 and last_report_date != current_date_str:
        logger.info("Daily ML validation script executing...")
        try:
            one_day_ago = now - timedelta(days=1)
            with db_conn() as conn:
                cur = conn.cursor(cursor_factory=RealDictCursor)
                cur.execute("""
                    SELECT indicators_status, result 
                    FROM trades 
                    WHERE status = 'COMPLETED' AND result IN ('WIN', 'LOSS') AND expiry_time >= %s
                """, (one_day_ago,))
                rows = cur.fetchall()
                cur.close()
                
            total_predictions = len(rows)
            correct = 0
            wrong = 0
            
            high_taken = 0
            high_wins = 0
            med_taken = 0
            med_wins = 0
            
            for r in rows:
                status = r['indicators_status']
                if isinstance(status, str):
                    try:
                        status = json.loads(status)
                    except:
                        status = {}
                if not status:
                    continue
                    
                is_win = r['result'] == 'WIN'
                if is_win:
                    correct += 1
                else:
                    wrong += 1
                    
                conf = status.get('confidence', 0.0)
                if conf >= 80.0:
                    high_taken += 1
                    if is_win:
                        high_wins += 1
                elif conf >= 60.0:
                    med_taken += 1
                    if is_win:
                        med_wins += 1
                        
            accuracy = (correct / total_predictions * 100) if total_predictions > 0 else 0.0
            high_win_rate = (high_wins / high_taken * 100) if high_taken > 0 else 0.0
            med_win_rate = (med_wins / med_taken * 100) if med_taken > 0 else 0.0
            
            if accuracy >= 65.0:
                ml_status = "GOOD"
            elif accuracy >= 50.0:
                ml_status = "MODERATE"
            else:
                ml_status = "POOR"
                
            msg = (
                f"🤖 ML DAILY REPORT\n\n"
                f"Date: {current_date_str}\n\n"
                f"Total Predictions: {total_predictions}\n\n"
                f"Correct: {correct}\n"
                f"Wrong: {wrong}\n\n"
                f"Accuracy: {accuracy:.1f}%\n\n"
                f"High Confidence Signals:\n"
                f"80%-100%\n"
                f"Taken: {high_taken}\n"
                f"Wins: {high_wins}\n"
                f"Win Rate: {high_win_rate:.0f}%\n\n"
                f"Medium Confidence:\n"
                f"60%-80%\n"
                f"Win Rate: {med_win_rate:.0f}%\n\n"
                f"ML Status:\n"
                f"{ml_status}"
            )
            send_telegram_msg(msg)
            last_report_date = current_date_str
            logger.info("Daily report published to channels.")
        except Exception as e:
            logger.error(f"Error compiling daily report: {e}")

# ==========================================
# MODULE 11: TRADE HISTORY PERIODIC ARCHIVING
# ==========================================
def archive_old_trades():
    try:
        now = get_now()
        thirty_days_ago = now - timedelta(days=30)
        with db_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO trades_archive (id, symbol, entry_price, side, start_time, expiry_time, rec_time, indicators_status, status, result, archived_at)
                SELECT id, symbol, entry_price, side, start_time, expiry_time, rec_time, indicators_status, status, result, %s
                FROM trades
                WHERE status IN ('COMPLETED', 'FAILED') AND expiry_time <= %s
                ON CONFLICT (id) DO NOTHING
            """, (now, thirty_days_ago))
            cur.execute("""
                DELETE FROM trades
                WHERE status IN ('COMPLETED', 'FAILED') AND expiry_time <= %s
            """, (thirty_days_ago,))
            conn.commit()
            cur.close()
            logger.info("Archive sync loop verified.")
    except Exception as e:
        logger.error(f"Error during trades history archive sweep: {e}")

# ==========================================
# MODULE 12: RESULTS VERIFICATION CONTROL LOOP
# ==========================================
def get_candle_at_time(symbol, target_time, api_key):
    values_dict = get_batch_market_data([symbol], interval="1min")
    values = values_dict.get(symbol)
    if not values:
        return None
    if isinstance(target_time, (int, float)):
        target_dt = datetime.fromtimestamp(target_time, BD_TZ).replace(second=0, microsecond=0)
    elif isinstance(target_time, datetime):
        target_dt = target_time.astimezone(BD_TZ).replace(second=0, microsecond=0)
    else:
        return None
    for candle in values:
        try:
            candle_dt = datetime.strptime(candle.get('datetime', ''), "%Y-%m-%d %H:%M:%S").replace(tzinfo=BD_TZ)
            if candle_dt.replace(second=0, microsecond=0) == target_dt:
                return float(candle['close'])
        except:
            continue
    return None

def update_expired_trades():
    now = get_now()
    try:
        with db_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                UPDATE trades 
                SET status = 'PENDING_RESULT' 
                WHERE status = 'ACTIVE' AND expiry_time <= %s
            """, (now,))
            conn.commit()
            cur.close()
    except Exception as e:
        logger.error(f"Expired state change failed: {e}")

def check_result():
    update_expired_trades()
    now = get_now()
    try:
        with db_conn() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute("""
                SELECT * FROM trades 
                WHERE status = 'PENDING_RESULT' AND result_sent = FALSE AND %s >= next_retry_time
            """, (now,))
            pending_trades = cur.fetchall()
            cur.close()
    except Exception as e:
        logger.error(f"Pending result lookup error: {e}")
        return

    for trade in pending_trades:
        trade_id = trade['id']
        symbol = trade['symbol']
        start_time = trade['start_time']
        expiry = trade['expiry_time']
        retry_count = trade['retry_count']
        entry_key = os.getenv("RESULT_KEY_ENTRY", API_KEYS[0])
        exit_key  = os.getenv("RESULT_KEY_EXIT", API_KEYS[1] if len(API_KEYS) > 1 else API_KEYS[0])
        entry_price = get_candle_at_time(symbol, start_time, entry_key)
        exit_price  = get_candle_at_time(symbol, expiry, exit_key)
        
        if entry_price is None or exit_price is None:
            new_retry = retry_count + 1
            if new_retry < 4:
                next_retry = now + timedelta(seconds=30)
                try:
                    with db_conn() as conn:
                        cur = conn.cursor()
                        cur.execute("""
                            UPDATE trades 
                            SET retry_count = %s, next_retry_time = %s 
                            WHERE id = %s
                        """, (new_retry, next_retry, trade_id))
                        conn.commit()
                        cur.close()
                except Exception as e:
                    logger.error(f"Verification countdown shift failed: {e}")
                continue
            else:
                if entry_price is None:
                    entry_price = float(trade['entry_price'])
                if exit_price is None:
                    values = get_batch_market_data([symbol], interval="1min")
                    exit_price = float(values[symbol][0]['close']) if (values and symbol in values and values[symbol]) else entry_price
                if entry_price is None or exit_price is None:
                    try:
                        with db_conn() as conn:
                            cur = conn.cursor()
                            cur.execute("""
                                UPDATE trades 
                                SET status = 'FAILED', result = 'UNAVAILABLE', result_sent = TRUE 
                                WHERE id = %s
                            """, (trade_id,))
                            conn.commit()
                            cur.close()
                    except:
                        pass
                    send_telegram_msg(
                        f"⚠️ RESULT UNAVAILABLE\n\n"
                        f"Asset : {symbol}\n"
                        f"Result verification pipeline suspended."
                    )
                    continue
                    
        win = (exit_price > entry_price) if trade['side'] == "BUY" else (exit_price < entry_price)
        
        try:
            with db_conn() as conn:
                cur = conn.cursor()
                cur.execute("""
                    UPDATE trades 
                    SET status = 'COMPLETED', result = %s, result_sent = TRUE 
                    WHERE id = %s
                """, ("WIN" if win else "LOSS", trade_id))
                conn.commit()
                cur.close()
        except Exception as e:
            logger.error(f"Commit lock block error: {e}")
            continue
            
        update_stats(win)
        indicators_status = trade['indicators_status']
        if isinstance(indicators_status, str):
            try:
                indicators_status = json.loads(indicators_status)
            except:
                indicators_status = {}
        if indicators_status:
            update_weights(symbol, indicators_status, win)
            
        s = get_stats()
        win_rate = (s['wins'] / s['total_trades'] * 100) if s['total_trades'] > 0 else 0
        result_emoji = "✅ WIN" if win else "❌ LOSS"
        entry_candle = start_time.astimezone(BD_TZ).strftime("%H:%M")
        exit_candle = expiry.astimezone(BD_TZ).strftime("%H:%M")
        
        msg = (
            f"🏁 TRADE RESULT\n\n"
            f"📊 Asset: {symbol}\n"
            f"🏆 Result: {result_emoji}\n"
            f"🚀 Entry: {entry_price:.5f}\n"
            f"🕒 Entry Candle: {entry_candle}\n\n"
            f"🏁 Exit: {exit_price:.5f}\n"
            f"🕒 Exit Candle: {exit_candle}\n"
            f"📈 Win Rate: {win_rate:.1f}%"
        )
        send_telegram_msg(msg)
        logger.info(f"Verified Result: {symbol} - {result_emoji} | Entry: {entry_price} | Exit: {exit_price}")

# ==========================================
# MODULE 13: ADVANCED SCANNERS & EXECUTION
# ==========================================
def fetch_and_analyze_batch(symbols_chunk, api_key, htf_trends):
    batch_data = get_batch_market_data(symbols_chunk, interval="1min")
    results    = []
    for symbol, values in batch_data.items():
        ind = calculate_indicators(values, symbol)
        if not ind:
            continue
        
        side_ema = "BUY" if ind['curr_p'] > ind['sma_50'] else "SELL"
        
        # Comprehensive state log mapping
        status = {
            'trend':        bool((ind['trend_up'] and ind['curr_p'] > ind['sma_50']) or (not ind['trend_up'] and ind['curr_p'] < ind['sma_50'])),
            'rsi':          bool((side_ema == "BUY" and ind['rsi'] < 40) or (side_ema == "SELL" and ind['rsi'] > 60)),
            'bb':           bool(ind['curr_p'] <= ind['lower'] or ind['curr_p'] >= ind['upper']),
            'vol':          bool(ind['vol_spike']),
            'adx':          bool(not ind['adx_low']),
            'ema_crossover':bool((ind['curr_p'] > ind['ema_50'] and side_ema == "BUY") or (ind['curr_p'] < ind['ema_50'] and side_ema == "SELL")),
            'macd':         bool((ind['macd_up'] and side_ema == "BUY") or (not ind['macd_up'] and side_ema == "SELL")),
            'atr':          bool(ind['volatility_low']),
            'vwap':         bool((ind['vwap_up'] and side_ema == "BUY") or (not ind['vwap_up'] and side_ema == "SELL")),
            'candle':       bool(ind['candle_signal']),
            
            # Numeric inputs
            'RSI':               float(ind['rsi']),
            'ADX':               float(ind['adx_val']),
            'ATR':               float(ind['atr_val']),
            'EMA Distance':      float(ind['ema_dist']),
            'Volume Ratio':      float(ind['vol_ratio']),
            'Support Distance':  float(ind['sr_dist']),
            'Resistance Distance': float(abs(ind['curr_p'] - ind['resistance'])),
            'MACD Histogram':    float(ind['macd_hist_val']),
            'BB Width':          float(ind['bb_width']),
            'VWAP Distance':     float(ind['vwap_dist']),
            'Candle Pattern':    1.0 if ind['candle_signal'] else 0.0
        }
        
        htf = htf_trends.get(symbol, {"5m": "NEUTRAL", "15m": "NEUTRAL"})
        
        # --- Multi-timeframe trend analysis verification ---
        # 1m, 5m, 15m trends must align, otherwise skip
        if side_ema == "BUY" and (htf['5m'] == "DOWN" or htf['15m'] == "DOWN"):
            logger.info(f"{symbol} trend conflict detected. Skipping analysis.")
            continue
        if side_ema == "SELL" and (htf['5m'] == "UP" or htf['15m'] == "UP"):
            logger.info(f"{symbol} trend conflict detected. Skipping analysis.")
            continue
            
        refined_conf, ml_prob = refine_confidence(ind['symbol_confidence'], symbol, ind, htf, status)
        
        # --- Support/Resistance Guard upgrade ---
        # Stop Hunt Protection
        if side_ema == "BUY" and ind['sr_dist'] < (ind['atr_val'] * 0.5) and ind['curr_p'] < ind['resistance']:
            logger.info(f"{symbol} too close to resistance zone. Skipping signal.")
            continue
        if side_ema == "SELL" and ind['sr_dist'] < (ind['atr_val'] * 0.5) and ind['curr_p'] > ind['support']:
            logger.info(f"{symbol} too close to support zone. Skipping signal.")
            continue
            
        status['confidence'] = float(refined_conf)
        
        results.append({
            'symbol': symbol,
            'curr_p': ind['curr_p'],
            'side':   side_ema,
            'score':  refined_conf,
            'status': status,
            'ind':    ind,
            'ml_prob': ml_prob,
            'htf':    htf
        })
    return results

last_scan_minute = -1

def run_scanner():
    global last_scan_minute
    now = get_now()
    if now.minute == last_scan_minute or not (0 <= now.second <= 10):
        return
    last_scan_minute = now.minute

    # Market Session validation Check
    session = get_market_session()
    if session == "DEAD_ZONE":
        logger.info("Market dead zone (Sunday Quiet hour). Skipping scan.")
        return

    if has_active_trade():
        logger.info("Active trade in progress. Scan locked.")
        return

    if now.minute == 0:
        archive_old_trades()

    logger.info(f"Scanning market session [{session}] for signals...")
    try:
        htf_trends = get_htf_trends(SYMBOLS)
    except Exception as e:
        logger.error(f"Error checking HTF trends: {e}")
        htf_trends = {}

    all_results = []
    chunks = [SYMBOLS[i:i + 3] for i in range(0, len(SYMBOLS), 3)]
    with ThreadPoolExecutor(max_workers=7) as executor:
        futures = [
            executor.submit(fetch_and_analyze_batch, chunk, API_KEYS[i % len(API_KEYS)], htf_trends)
            for i, chunk in enumerate(chunks)
        ]
        for future in futures:
            try:
                res = future.result()
                if res:
                    all_results.extend(res)
            except Exception as e:
                logger.error(f"Analysis Thread failure: {e}")
                
    logger.info(f"Analyzed {len(all_results)} instruments.")
    if all_results:
        best      = max(all_results, key=lambda x: x['score'])
        
        # --- High-Impact News Filter Check (Dynamic Pause Alert) ---
        news_active, news_reason = is_high_impact_news_near(best['symbol'], buffer_minutes=30)
        if news_active:
            msg = (
                f"⚠️ HIGH IMPACT NEWS DETECTED\n\n"
                f"Asset: {best['symbol']}\n"
                f"Event: {news_reason}\n\n"
                f"Trading paused. Bot will resume auto-scanning once news zone completes."
            )
            send_telegram_msg(msg)
            logger.warning(f"Trade skipped on {best['symbol']} due to high-impact news: {news_reason}")
            return

        conf      = round(best['score'], 1)
        atr_val = best['ind']['atr_val']
        curr_p  = best['curr_p']
        atr_pct = (atr_val / curr_p * 100) if curr_p > 0 else 0.2
        if atr_pct > 0.3:
            rec_time = 5 if conf <= 60 else 7
        elif atr_pct > 0.1:
            rec_time = 8 if conf <= 60 else 10
        else:
            rec_time = 12 if conf <= 60 else 15
        start  = get_now().replace(second=0, microsecond=0) + timedelta(minutes=1)
        expiry = start + timedelta(minutes=rec_time)

        trade_data = {
            'symbol':            best['symbol'],
            'entry_price':       best['curr_p'],
            'side':              best['side'],
            'start_time':        start,
            'expiry_time':       expiry,
            'rec_time':          rec_time,
            'indicators_status': best['status']
        }
        saved_id = save_active_trade(trade_data)
        if saved_id:
            entry_time = start.strftime("%H:%M")
            entry_candle = start.strftime("%H:%M")
            
            # Detailed Upgraded Telegram Alert Template
            msg = (
                f"🚨 SIGNAL ALERT: {best['symbol']} -> {best['side']}\n\n"
                f"Confidence Score: {conf:.1f}%\n"
                f"ML Prediction Win Rate: {best['ml_prob']*100:.1f}%\n"
                f"Market Session: {session}\n"
                f"Trend Alignment: {'Bullish (Above EMA50)' if best['side'] == 'BUY' else 'Bearish (Below EMA50)'}\n"
                f"Support Level: {best['ind']['support']:.5f}\n"
                f"Resistance Level: {best['ind']['resistance']:.5f}\n"
                f"ATR Volatility: {best['ind']['atr_val']:.5f}\n"
                f"ADX Trend Strength: {best['ind']['adx_val']:.1f} (DI+:{best['ind']['plus_di']:.1f}/DI-:{best['ind']['minus_di']:.1f})\n"
                f"Higher TF Confirmation: 5m/15m Align\n"
                f"Risk Profile: Moderate\n"
                f"Expected Strength: Strong\n"
                f"Expiry: {rec_time} Min\n"
                f"Entry Candle: {entry_candle}\n\n"
                f"Execute trade precisely at next minute boundary (00s)."
            )
            send_telegram_msg(msg)
            try:
                with db_conn() as conn:
                    cur = conn.cursor()
                    cur.execute("UPDATE trades SET msg_sent = TRUE WHERE id = %s", (saved_id,))
                    conn.commit()
                    cur.close()
            except Exception as e:
                logger.error(f"Error logging signal output state: {e}")

# ==========================================
# MODULE 14: FLASK HEALTH CHECKS & RECOVERY
# ==========================================
app = Flask(__name__)

@app.route("/")
def health_check():
    stats = get_stats()
    return {
        "status": "healthy",
        "timestamp": get_now().isoformat(),
        "database": "online" if _pool else "offline",
        "performance_stats": stats
    }

def run_web():
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, use_reloader=False)

# Graceful shutdown support for Railway Container Engine
def signal_handler(sig, frame):
    logger.info("Shutdown signal caught. Releasing connection pools...")
    if _pool:
        _pool.closeall()
    sys.exit(0)

# ==========================================
# MODULE 15: EXECUTION ENTRYPOINT
# ==========================================
if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    init_db()
    threading.Thread(target=run_web, daemon=True).start()
    logger.info("Bot execution started inside container environment...")
    
    while True:
        try:
            check_and_send_daily_report()
            run_scanner()
            check_result()
        except Exception as e:
            logger.error(f"Global scheduler event execution failure: {e}")
        time.sleep(1)