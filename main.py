import os
import json
import time
import random
import threading
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
import psycopg2
from psycopg2 import errors
from psycopg2.extras import RealDictCursor, Json
from psycopg2.pool import ThreadedConnectionPool
from flask import Flask


# ============================================================
# CONFIGURATION
# ============================================================
#
# ARCHITECTURE OVERVIEW
#
#   PostgreSQL long-term history (unlimited, never deleted)
#       -> Fast Historical Filter (indexed SQL, LIMIT-bounded)
#       -> Feature Similarity (weighted distance)
#       -> Top Relevant Trades (rank + cap)
#       -> Recency-weighted Statistics (per-trade age + similarity weight)
#       -> Historical Evidence (own layer, own status)
#
#   ML Prediction (separate, trained only on strictly-past completed
#   trades, own layer, own status)
#
#   Indicator Confidence (existing weighted-vote engine, own layer,
#   own status)
#
#   Final Decision Engine combines the three layers, weighting only the
#   layers that are actually available/sufficient, and can output
#   BUY / SELL / NO_TRADE.
#
# RULES THAT APPLY EVERYWHERE IN THIS FILE:
#
# 1. Only CLOSED 1-minute candles are used for analysis.
# 2. A signal generated during minute X is for minute X+1.
# 3. Indicator Confidence is NOT a probability of winning.
# 4. Historical Evidence is NOT the same statistic as Indicator
#    Confidence, and is NOT the same statistic as an ML probability.
# 5. Historical setups must: belong to the SAME market, SAME timeframe,
#    SAME direction, have sufficiently SIMILAR setup features, and have
#    been COMPLETED strictly before the current setup (no lookahead).
# 6. Indicator weights/learning, and ML models, are MARKET-SPECIFIC.
#    EUR/USD learning never changes GBP/USD weights or models.
# 7. Trade history in PostgreSQL is permanent. Nothing is ever deleted,
#    truncated, or capped for "storage" reasons. Any LIMIT used when
#    reading history is a *processing* limit for a single decision, not
#    a retention limit -- the rest stays in the database untouched.
# 8. Zero relevant historical trades, or an untrained ML model, is a
#    normal, valid state -- never an error, and never a reason to
#    fabricate a win rate or a probability.
#
# ============================================================


BD_TZ = ZoneInfo("Asia/Dhaka")
TD_TZ = "Asia/Dhaka"
INTERVAL = "1min"
TIMEFRAME = INTERVAL  # explicit timeframe dimension, used by history/ML layers


SYMBOLS = [
    "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "USD/CHF",
    "NZD/USD", "EUR/JPY", "GBP/JPY", "EUR/GBP",
    "BTC/USD", "ETH/USD", "LTC/USD", "XRP/USD", "SOL/USD", "ADA/USD",
    "XAU/USD", "XAG/USD",
    "GBP/AUD", "EUR/AUD", "AUD/JPY",
]


# ============================================================
# MARKET-SPECIFIC INDICATOR WEIGHTS
# ============================================================
#
# Every symbol gets its own copy of these in the database. Learning is
# performed with WHERE symbol=<symbol> AND name=<indicator>, so one
# market can never modify another market's weights.
#
# ============================================================

DEFAULT_WEIGHTS = {
    "trend": 15.0, "rsi": 10.0, "bb": 10.0, "volume": 7.0,
    "adx": 8.0, "ema": 10.0, "macd": 15.0, "vwap": 10.0, "candle": 10.0,
}


DATABASE_URL = "postgresql://neondb_owner:npg_xQl3EYGgjvN4@ep-muddy-tooth-a7smuza8-pooler.ap-southeast-2.aws.neon.tech/neondb?sslmode=require&channel_binding=require"


API_KEYS = [
    "76d81f1fba224b9e88015b34fdcc7f76",
    "5d9e7b9a4a014dd38746242410e033e0",
    "769c447e581d4592ad14f7023db745b3",
    "2ca49ec0c0534851b8ee88bd01858eaf",
    "6a5adb88517744aba3a40c948404d0b9",
    "c0bc2922d3c7486f8234dd67cd18b46a",
]

if not API_KEYS:
    raise RuntimeError("At least one Twelve Data API key is required.")


TELEGRAM_TOKEN = "8385011968:AAEP6CjEuUO77Llary88GI0_snxfkrHjrV0"
TELEGRAM_CHAT_ID = "6793328058"

RESULT_KEY_ENTRY = os.getenv("RESULT_KEY_ENTRY", API_KEYS[0]).strip()
RESULT_KEY_EXIT = os.getenv(
    "RESULT_KEY_EXIT", API_KEYS[1] if len(API_KEYS) > 1 else API_KEYS[0]
).strip()


# ============================================================
# SIGNAL FILTERS (Indicator layer)
# ============================================================

MIN_TRADE_CONFIDENCE = float(os.getenv("MIN_TRADE_CONFIDENCE", "60"))
STRONG_CONFIDENCE = float(os.getenv("STRONG_CONFIDENCE", "80"))
MIN_DIRECTIONAL_VOTES = max(1, int(os.getenv("MIN_DIRECTIONAL_VOTES", "4")))
MAX_DIRECTIONAL_CONFLICT = float(os.getenv("MAX_DIRECTIONAL_CONFLICT", "0.45"))


# ============================================================
# API SETTINGS
# ============================================================

API_RETRIES = max(0, int(os.getenv("API_RETRIES", "2")))
RESULT_MAX_RETRIES = max(1, int(os.getenv("RESULT_MAX_RETRIES", "4")))
RESULT_RETRY_SECONDS = max(1, int(os.getenv("RESULT_RETRY_SECONDS", "30")))
API_TIMEOUT = max(1, int(os.getenv("API_TIMEOUT", "15")))
BATCH_API_TIMEOUT = max(1, int(os.getenv("BATCH_API_TIMEOUT", "30")))


# ============================================================
# HISTORICAL EVIDENCE PIPELINE SETTINGS
# ============================================================
#
#   Stage 1/2 - Fast Historical Filter: bounded, index-backed SQL query
#               (HISTORICAL_LOOKBACK candidates at most). This is a
#               *processing* cap, not a retention limit -- everything
#               else stays in PostgreSQL untouched.
#   Stage 3   - Feature Similarity: weighted distance vs current setup.
#   Stage 4   - Top Relevant Trades: closest HISTORICAL_TOP_N kept.
#   Stage 5   - Recency-weighted statistics + effective sample size.
#
# ============================================================

# Minimum number of *effective* samples before Historical Win Rate is
# considered statistically usable (Stage 3 "minimum useful sample").
HISTORICAL_MIN_SAMPLES = max(10, int(os.getenv("HISTORICAL_MIN_SAMPLES", "20")))

# Effective sample size at/above which historical evidence is "STRONG".
HISTORICAL_STRONG_SAMPLE_MIN = max(
    HISTORICAL_MIN_SAMPLES, float(os.getenv("HISTORICAL_STRONG_SAMPLE_MIN", "100"))
)

# Stage 4: how many closest-matching historical trades to actually use.
# This is the "Top 500" processing limit from the architecture spec.
HISTORICAL_TOP_N = max(1, int(os.getenv("HISTORICAL_TOP_N", "500")))

# Stage 1/2: SQL candidate pool size, must be able to hold at least
# HISTORICAL_TOP_N candidates. Bounded so a single decision never needs
# to scan or load the full table, no matter how large it grows.
HISTORICAL_LOOKBACK = max(
    HISTORICAL_TOP_N, int(os.getenv("HISTORICAL_LOOKBACK", "5000"))
)

# Smaller distance = more similar. Below this, a historical setup is
# considered a genuine match at all.
HISTORICAL_DISTANCE_THRESHOLD = float(os.getenv("HISTORICAL_DISTANCE_THRESHOLD", "0.42"))

# Half-life (in days) used for the recency weighting of historical
# trades: a trade exactly this many days old gets half the weight of a
# brand-new trade with identical similarity.
HISTORICAL_RECENCY_HALF_LIFE_DAYS = max(
    0.1, float(os.getenv("HISTORICAL_RECENCY_HALF_LIFE_DAYS", "30"))
)

# Minimum *effective* sample size before the historical win rate is
# allowed to influence expiry sizing (a conservative, non-guaranteeing
# adjustment only -- see _choose_expiry).
HISTORICAL_CONFIDENCE_MIN_SAMPLES = max(
    HISTORICAL_MIN_SAMPLES, int(os.getenv("HISTORICAL_CONFIDENCE_MIN_SAMPLES", "30"))
)


# ============================================================
# ML PREDICTION LAYER SETTINGS
# ============================================================

ML_ENABLED = os.getenv("ML_ENABLED", "true").strip().lower() not in {"0", "false", "no"}
ML_MIN_TRAINING_SAMPLES = max(10, int(os.getenv("ML_MIN_TRAINING_SAMPLES", "50")))
ML_RETRAIN_COOLDOWN_MINUTES = max(1, int(os.getenv("ML_RETRAIN_COOLDOWN_MINUTES", "30")))
ML_L2_REG = max(0.0, float(os.getenv("ML_L2_REG", "0.01")))
ML_LEARNING_RATE = max(1e-4, float(os.getenv("ML_LEARNING_RATE", "0.1")))
ML_TRAIN_ITERATIONS = max(50, int(os.getenv("ML_TRAIN_ITERATIONS", "300")))
ML_TRAIN_HISTORY_LIMIT = max(
    ML_MIN_TRAINING_SAMPLES, int(os.getenv("ML_TRAIN_HISTORY_LIMIT", "20000"))
)


# ============================================================
# FINAL DECISION ENGINE SETTINGS
# ============================================================

FINAL_WEIGHT_INDICATOR = max(0.0, float(os.getenv("FINAL_WEIGHT_INDICATOR", "0.5")))
FINAL_WEIGHT_HISTORICAL = max(0.0, float(os.getenv("FINAL_WEIGHT_HISTORICAL", "0.3")))
FINAL_WEIGHT_ML = max(0.0, float(os.getenv("FINAL_WEIGHT_ML", "0.2")))
FINAL_MIN_COMBINED_CONFIDENCE = float(
    os.getenv("FINAL_MIN_COMBINED_CONFIDENCE", str(MIN_TRADE_CONFIDENCE))
)
# If historical evidence is STRONG and clearly says this setup loses,
# veto the trade regardless of indicator/ML opinion.
FINAL_HISTORICAL_VETO_WIN_RATE = float(os.getenv("FINAL_HISTORICAL_VETO_WIN_RATE", "35"))


# ============================================================
# EXPIRY LIMITS
# ============================================================

MIN_EXPIRY_MINUTES = max(1, int(os.getenv("MIN_EXPIRY_MINUTES", "5")))
MAX_EXPIRY_MINUTES = max(MIN_EXPIRY_MINUTES, int(os.getenv("MAX_EXPIRY_MINUTES", "15")))


# ============================================================
# HISTORICAL SETUP FEATURES
# ============================================================

FEATURE_NAMES = [
    "rsi_norm", "adx_norm", "bb_position", "ema_spread_norm",
    "macd_hist_norm", "vwap_distance_norm", "volume_ratio_norm",
    "atr_pct_norm", "sma20_sma50_norm", "price_sma20_norm",
    "candle_strength", "trend_strength", "directional_confidence_norm",
    "buy_vote_ratio", "sell_vote_ratio",
]


# ============================================================
# SIMILARITY WEIGHTS (feature-similarity only, NOT indicator votes)
# ============================================================

SIMILARITY_WEIGHTS = {
    "rsi_norm": 1.00, "adx_norm": 0.70, "bb_position": 0.90,
    "ema_spread_norm": 1.00, "macd_hist_norm": 1.00,
    "vwap_distance_norm": 0.90, "volume_ratio_norm": 0.50,
    "atr_pct_norm": 0.60, "sma20_sma50_norm": 0.90,
    "price_sma20_norm": 0.70, "candle_strength": 0.50,
    "trend_strength": 0.70, "directional_confidence_norm": 0.80,
    "buy_vote_ratio": 0.70, "sell_vote_ratio": 0.70,
}


# ============================================================
# GENERAL HELPERS
# ============================================================

def get_now():
    return datetime.now(BD_TZ)


def _safe_float(value, default=np.nan):
    try:
        x = float(value)
        return x if np.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def _clamp(value, low=0.0, high=1.0):
    try:
        return float(max(low, min(high, value)))
    except (TypeError, ValueError):
        return low


def _opinion(signal, strength=0.0, reason=""):
    return {
        "signal": signal if signal in {"BUY", "SELL", "HOLD"} else "HOLD",
        "strength": round(_clamp(strength), 4),
        "reason": str(reason),
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if isinstance(value, datetime):
        return value.isoformat()
    return value


# ============================================================
# DATABASE
# ============================================================

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
    with db_conn() as conn:
        cur = conn.cursor()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS trading_stats (
                id SERIAL PRIMARY KEY,
                total_trades INTEGER NOT NULL DEFAULT 0,
                wins INTEGER NOT NULL DEFAULT 0,
                losses INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS trades (
                id SERIAL PRIMARY KEY,
                symbol TEXT NOT NULL,
                entry_price DOUBLE PRECISION,
                side TEXT NOT NULL CHECK (side IN ('BUY','SELL')),
                start_time TIMESTAMPTZ NOT NULL,
                expiry_time TIMESTAMPTZ NOT NULL,
                rec_time INTEGER NOT NULL,
                indicators_status JSONB NOT NULL DEFAULT '{}'::jsonb,
                indicator_opinions JSONB NOT NULL DEFAULT '{}'::jsonb,
                setup_features JSONB NOT NULL DEFAULT '{}'::jsonb,
                indicator_confidence DOUBLE PRECISION,
                historical_count INTEGER NOT NULL DEFAULT 0,
                historical_wins INTEGER NOT NULL DEFAULT 0,
                historical_losses INTEGER NOT NULL DEFAULT 0,
                historical_win_rate DOUBLE PRECISION,
                historical_distance DOUBLE PRECISION,
                status TEXT NOT NULL DEFAULT 'ACTIVE',
                result TEXT NOT NULL DEFAULT 'PENDING',
                retry_count INTEGER NOT NULL DEFAULT 0,
                next_retry_time TIMESTAMPTZ,
                msg_sent BOOLEAN NOT NULL DEFAULT FALSE,
                result_sent BOOLEAN NOT NULL DEFAULT FALSE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS indicator_weights (
                symbol TEXT NOT NULL,
                name TEXT NOT NULL,
                weight DOUBLE PRECISION NOT NULL,
                accuracy_factor DOUBLE PRECISION NOT NULL DEFAULT 1.0,
                PRIMARY KEY (symbol, name)
            );

            -- Market-specific ML models (logistic regression), separate
            -- per symbol AND side, so learning never crosses markets.
            CREATE TABLE IF NOT EXISTS ml_models (
                symbol TEXT NOT NULL,
                side TEXT NOT NULL CHECK (side IN ('BUY','SELL')),
                weights JSONB NOT NULL,
                bias DOUBLE PRECISION NOT NULL,
                feature_names JSONB NOT NULL,
                training_samples INTEGER NOT NULL DEFAULT 0,
                train_accuracy DOUBLE PRECISION,
                trained_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (symbol, side)
            );

            -- Migration-safe additive columns (existing deployments keep
            -- their data; nothing here ever deletes or truncates rows).
            ALTER TABLE trades ADD COLUMN IF NOT EXISTS setup_features JSONB NOT NULL DEFAULT '{}'::jsonb;
            ALTER TABLE trades ADD COLUMN IF NOT EXISTS indicator_confidence DOUBLE PRECISION;
            ALTER TABLE trades ADD COLUMN IF NOT EXISTS historical_count INTEGER NOT NULL DEFAULT 0;
            ALTER TABLE trades ADD COLUMN IF NOT EXISTS historical_wins INTEGER NOT NULL DEFAULT 0;
            ALTER TABLE trades ADD COLUMN IF NOT EXISTS historical_losses INTEGER NOT NULL DEFAULT 0;
            ALTER TABLE trades ADD COLUMN IF NOT EXISTS historical_win_rate DOUBLE PRECISION;
            ALTER TABLE trades ADD COLUMN IF NOT EXISTS historical_distance DOUBLE PRECISION;
            ALTER TABLE trades ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW();
            ALTER TABLE trades ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();
            ALTER TABLE trades ADD COLUMN IF NOT EXISTS result_sent BOOLEAN NOT NULL DEFAULT FALSE;
            ALTER TABLE trades ADD COLUMN IF NOT EXISTS msg_sent BOOLEAN NOT NULL DEFAULT FALSE;
            ALTER TABLE trades ADD COLUMN IF NOT EXISTS retry_count INTEGER NOT NULL DEFAULT 0;
            ALTER TABLE trades ADD COLUMN IF NOT EXISTS next_retry_time TIMESTAMPTZ;
            ALTER TABLE trades ADD COLUMN IF NOT EXISTS indicator_opinions JSONB NOT NULL DEFAULT '{}'::jsonb;

            -- New columns for the layered Historical Evidence / ML / Final
            -- Decision architecture.
            ALTER TABLE trades ADD COLUMN IF NOT EXISTS timeframe TEXT NOT NULL DEFAULT '1min';
            ALTER TABLE trades ADD COLUMN IF NOT EXISTS historical_weighted_win_rate DOUBLE PRECISION;
            ALTER TABLE trades ADD COLUMN IF NOT EXISTS historical_effective_sample_size DOUBLE PRECISION;
            ALTER TABLE trades ADD COLUMN IF NOT EXISTS historical_similarity_strength DOUBLE PRECISION;
            ALTER TABLE trades ADD COLUMN IF NOT EXISTS historical_evidence_status TEXT;
            ALTER TABLE trades ADD COLUMN IF NOT EXISTS ml_probability DOUBLE PRECISION;
            ALTER TABLE trades ADD COLUMN IF NOT EXISTS ml_status TEXT;
            ALTER TABLE trades ADD COLUMN IF NOT EXISTS final_decision_meta JSONB NOT NULL DEFAULT '{}'::jsonb;

            INSERT INTO trading_stats(total_trades, wins, losses)
            SELECT 0, 0, 0
            WHERE NOT EXISTS (SELECT 1 FROM trading_stats);

            CREATE INDEX IF NOT EXISTS idx_trades_result_queue
                ON trades(status, next_retry_time, id);

            CREATE INDEX IF NOT EXISTS idx_trades_expiry
                ON trades(status, expiry_time);

            CREATE INDEX IF NOT EXISTS idx_trades_history
                ON trades(symbol, side, status, result, start_time);

            -- Timeframe-aware composite index backing the Fast Historical
            -- Filter stage; keeps that query index-only even as the table
            -- grows into the millions of rows.
            CREATE INDEX IF NOT EXISTS idx_trades_history_tf
                ON trades(symbol, timeframe, side, status, result, start_time DESC);

            CREATE UNIQUE INDEX IF NOT EXISTS uq_one_live_trade
                ON trades((1))
                WHERE status IN ('ACTIVE', 'PENDING_RESULT', 'RESULT_PROCESSING');
            """
        )

        for symbol in SYMBOLS:
            for name, weight in DEFAULT_WEIGHTS.items():
                cur.execute(
                    """
                    INSERT INTO indicator_weights(symbol, name, weight)
                    VALUES(%s, %s, %s)
                    ON CONFLICT(symbol, name) DO NOTHING
                    """,
                    (symbol, name, weight),
                )

        conn.commit()
        cur.close()


def get_stats():
    try:
        with db_conn() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(
                "SELECT total_trades, wins, losses FROM trading_stats ORDER BY id LIMIT 1"
            )
            row = cur.fetchone()
            cur.close()
            return dict(row) if row else {"total_trades": 0, "wins": 0, "losses": 0}
    except Exception as exc:
        print(f"Get Stats Error: {exc}")
        return {"total_trades": 0, "wins": 0, "losses": 0}


# ============================================================
# MARKET-SPECIFIC WEIGHTS
# ============================================================

def get_weights(symbol):
    try:
        with db_conn() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(
                "SELECT name, weight, accuracy_factor FROM indicator_weights WHERE symbol=%s",
                (symbol,),
            )
            rows = cur.fetchall()
            cur.close()

        weights = {}
        for row in rows:
            name = row["name"]
            weight = _safe_float(row["weight"], DEFAULT_WEIGHTS.get(name, 0.0))
            accuracy = _safe_float(row["accuracy_factor"], 1.0)
            weights[name] = weight * accuracy

        for name, default in DEFAULT_WEIGHTS.items():
            weights.setdefault(name, default)

        return weights
    except Exception as exc:
        print(f"Get Weights Error: {exc}")
        return DEFAULT_WEIGHTS.copy()


# ============================================================
# WEB / TELEGRAM
# ============================================================

app = Flask(__name__)


@app.route("/")
def home():
    return "Trading AI Bot is running!"


def run_web():
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port, threaded=True)


_telegram_lock = threading.Lock()


def send_telegram_msg(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram configuration missing; message not sent.")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

    with _telegram_lock:
        try:
            response = requests.post(
                url,
                json={"chat_id": TELEGRAM_CHAT_ID, "text": message},
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

            if not data.get("ok"):
                print(f"Telegram API Error: {data}")
                return False

            return True
        except Exception as exc:
            print(f"Telegram Send Error: {exc}")
            return False


# ============================================================
# TWELVE DATA API
# ============================================================

def _request_json(url, timeout, retries=API_RETRIES):
    last_error = None

    for attempt in range(retries + 1):
        response = None
        try:
            response = requests.get(url, timeout=timeout)

            if response.status_code == 429:
                if attempt < retries:
                    retry_after = response.headers.get("Retry-After")
                    try:
                        wait = float(retry_after)
                    except (TypeError, ValueError):
                        wait = min(30.0, 2.0 ** attempt)
                    time.sleep(wait)
                    continue
                raise RuntimeError("Twelve Data rate limit (HTTP 429).")

            if response.status_code in {500, 502, 503, 504} and attempt < retries:
                time.sleep(min(10.0, 1.5 ** attempt + random.random()))
                continue

            response.raise_for_status()
            data = response.json()

            if isinstance(data, dict) and data.get("status") == "error":
                code = data.get("code")
                message = data.get("message", data)

                if code in {400, 401, 403, 404}:
                    raise RuntimeError(f"Twelve Data error {code}: {message}")

                if attempt < retries:
                    time.sleep(min(10.0, 1.5 ** attempt + random.random()))
                    continue

                raise RuntimeError(f"Twelve Data error: {message}")

            return data

        except (requests.Timeout, requests.ConnectionError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(10.0, 1.5 ** attempt + random.random()))

        except (requests.HTTPError, ValueError, RuntimeError) as exc:
            last_error = exc
            status = getattr(response, "status_code", None)

            if isinstance(exc, requests.HTTPError) and status not in {429, 500, 502, 503, 504}:
                break

            if "Twelve Data error 4" in str(exc):
                break

            if attempt < retries:
                time.sleep(min(10.0, 1.5 ** attempt + random.random()))

    print(f"Market API Error: {last_error}")
    return None


def _extract_values(data, symbol):
    if not isinstance(data, dict):
        return None

    if isinstance(data.get("values"), list):
        return data["values"]

    symbol_data = data.get(symbol)
    if isinstance(symbol_data, dict):
        if symbol_data.get("status") == "error":
            print(f"Batch symbol error for {symbol}: {symbol_data.get('message', symbol_data)}")
            return None
        if isinstance(symbol_data.get("values"), list):
            return symbol_data["values"]

    return None


def _api_url(symbol, api_key, outputsize=None, start_dt=None, end_dt=None):
    params = [
        f"symbol={quote(symbol, safe='/')}",
        f"interval={INTERVAL}",
        f"timezone={quote(TD_TZ, safe='')}",
        f"apikey={quote(api_key.strip(), safe='')}",
    ]

    if outputsize is not None:
        params.append(f"outputsize={int(outputsize)}")

    if start_dt is not None:
        params.append(
            "start_date=" + quote(start_dt.astimezone(BD_TZ).strftime("%Y-%m-%dT%H:%M:%S"), safe="")
        )

    if end_dt is not None:
        params.append(
            "end_date=" + quote(end_dt.astimezone(BD_TZ).strftime("%Y-%m-%dT%H:%M:%S"), safe="")
        )

    return "https://api.twelvedata.com/time_series?" + "&".join(params)


def get_market_data(symbol, api_key, outputsize=100):
    if not api_key:
        return None
    return _extract_values(_api_url(symbol, api_key, outputsize=outputsize), symbol)


def get_market_data_window(symbol, api_key, start_dt, end_dt):
    if not api_key:
        return None
    return _extract_values(
        _request_json(_api_url(symbol, api_key, start_dt=start_dt, end_dt=end_dt), API_TIMEOUT),
        symbol,
    )


def get_batch_market_data(symbols_list, api_key, outputsize=100):
    if not api_key or not symbols_list:
        return {}

    symbol_param = quote(",".join(symbols_list), safe=",/")
    url = (
        "https://api.twelvedata.com/time_series?"
        f"symbol={symbol_param}&interval={INTERVAL}&outputsize={int(outputsize)}"
        f"&timezone={quote(TD_TZ, safe='')}&apikey={quote(api_key.strip(), safe='')}"
    )

    data = _request_json(url, BATCH_API_TIMEOUT)
    if not isinstance(data, dict):
        return {}

    result = {}
    for symbol in symbols_list:
        values = _extract_values(data, symbol)
        if values:
            result[symbol] = values
    return result


def _parse_candle_datetime(text):
    if not text:
        return None
    try:
        dt = datetime.strptime(str(text), "%Y-%m-%d %H:%M:%S")
        return dt.replace(tzinfo=BD_TZ)
    except (TypeError, ValueError):
        return None


def _sort_values_chronologically(values):
    rows = []
    for item in values or []:
        if not isinstance(item, dict):
            continue
        dt = _parse_candle_datetime(item.get("datetime"))
        if dt is not None:
            rows.append((dt, item))
    rows.sort(key=lambda x: x[0])
    return [item for _, item in rows]


def _closed_candles(values, now=None):
    now = now or get_now()
    current_minute = now.replace(second=0, microsecond=0)
    rows = []
    for item in _sort_values_chronologically(values):
        dt = _parse_candle_datetime(item.get("datetime"))
        if dt is not None and dt < current_minute:
            rows.append(item)
    return rows


# ============================================================
# INDICATOR CALCULATIONS
# ============================================================

def _wilder_rsi_series(prices, period=14):
    deltas = prices.diff()
    gains = deltas.clip(lower=0.0)
    losses = -deltas.clip(upper=0.0)

    avg_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    rsi = rsi.where(~(avg_loss.eq(0) & avg_gain.gt(0)), 100.0)
    rsi = rsi.where(~(avg_loss.eq(0) & avg_gain.eq(0)), 50.0)
    return rsi


def _wilder_adx_components(df, period=14):
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)

    if len(df) < 2 * period + 2:
        return (np.nan, np.nan, np.nan)

    prev_close = close[:-1]
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(np.abs(high[1:] - prev_close), np.abs(low[1:] - prev_close)),
    )

    up = high[1:] - high[:-1]
    down = low[:-1] - low[1:]

    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    atr_sm = float(np.sum(tr[:period]))
    plus_sm = float(np.sum(plus_dm[:period]))
    minus_sm = float(np.sum(minus_dm[:period]))

    dx_values, plus_di_values, minus_di_values = [], [], []

    for i in range(period, len(tr)):
        atr_sm = atr_sm - atr_sm / period + tr[i]
        plus_sm = plus_sm - plus_sm / period + plus_dm[i]
        minus_sm = minus_sm - minus_sm / period + minus_dm[i]

        if atr_sm <= 0:
            continue

        plus_di = 100.0 * plus_sm / atr_sm
        minus_di = 100.0 * minus_sm / atr_sm
        total = plus_di + minus_di
        dx = 100.0 * abs(plus_di - minus_di) / total if total > 0 else 0.0

        dx_values.append(dx)
        plus_di_values.append(plus_di)
        minus_di_values.append(minus_di)

    if len(dx_values) < period:
        return (np.nan, np.nan, np.nan)

    adx = float(np.mean(dx_values[:period]))
    for value in dx_values[period:]:
        adx = ((period - 1) * adx + value) / period

    return (adx, float(plus_di_values[-1]), float(minus_di_values[-1]))


def _atr_series(df, period=14):
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr, tr.rolling(period).mean()


# ------------------------------------------------------------
# TREND
# ------------------------------------------------------------

def indicator_trend(df):
    close = df["close"]
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()

    if any(pd.isna(x) for x in [sma20.iloc[-2], sma20.iloc[-1], sma50.iloc[-2], sma50.iloc[-1]]):
        return _opinion("HOLD", 0.0, "Not enough SMA20/SMA50 data"), {}

    price = float(close.iloc[-1])
    s20, s50 = float(sma20.iloc[-1]), float(sma50.iloc[-1])
    p20, p50 = float(sma20.iloc[-2]), float(sma50.iloc[-2])

    bullish = price > s20 and s20 > s50 and s20 >= p20 and s50 >= p50
    bearish = price < s20 and s20 < s50 and s20 <= p20 and s50 <= p50

    if bullish:
        strength = min(1.0, 0.65 + abs(price - s20) / max(abs(s20), 1e-12) * 20)
        return (
            _opinion("BUY", strength, "Price above rising SMA20; SMA20 above SMA50"),
            {"trend_state": 1.0, "trend_strength": strength},
        )

    if bearish:
        strength = min(1.0, 0.65 + abs(price - s20) / max(abs(s20), 1e-12) * 20)
        return (
            _opinion("SELL", strength, "Price below falling SMA20; SMA20 below SMA50"),
            {"trend_state": -1.0, "trend_strength": strength},
        )

    return (
        _opinion("HOLD", 0.0, "SMA20/SMA50 and price are not fully aligned"),
        {"trend_state": 0.0, "trend_strength": 0.0},
    )


# ------------------------------------------------------------
# RSI
# ------------------------------------------------------------

def indicator_rsi(df):
    series = _wilder_rsi_series(df["close"], 14)
    rsi = _safe_float(series.iloc[-1])
    prev = _safe_float(series.iloc[-2])

    if not np.isfinite(rsi):
        return _opinion("HOLD", 0.0, "RSI unavailable"), np.nan

    if rsi <= 30 and np.isfinite(prev) and rsi >= prev:
        return (
            _opinion("BUY", min(1.0, 0.70 + (30 - rsi) / 60), f"RSI oversold and recovering ({rsi:.1f})"),
            rsi,
        )

    if rsi >= 70 and np.isfinite(prev) and rsi <= prev:
        return (
            _opinion("SELL", min(1.0, 0.70 + (rsi - 70) / 60), f"RSI overbought and falling ({rsi:.1f})"),
            rsi,
        )

    if 50 < rsi < 70 and rsi > prev:
        return _opinion("BUY", 0.60, f"RSI bullish momentum ({rsi:.1f})"), rsi

    if 30 < rsi < 50 and rsi < prev:
        return _opinion("SELL", 0.60, f"RSI bearish momentum ({rsi:.1f})"), rsi

    return _opinion("HOLD", 0.0, f"RSI not decisive ({rsi:.1f})"), rsi


# ------------------------------------------------------------
# BOLLINGER BANDS
# ------------------------------------------------------------

def indicator_bollinger(df):
    close = df["close"]
    mid = close.rolling(20).mean()
    std = close.rolling(20).std(ddof=0)

    if pd.isna(mid.iloc[-1]) or pd.isna(std.iloc[-1]):
        return _opinion("HOLD", 0.0, "Bollinger data unavailable"), np.nan, np.nan, np.nan

    m, s = float(mid.iloc[-1]), float(std.iloc[-1])
    price = float(close.iloc[-1])
    upper, lower = m + 2 * s, m - 2 * s
    width = max(upper - lower, 1e-12)
    position = (price - lower) / width

    prev_price = float(close.iloc[-2])

    if price <= lower and price >= prev_price:
        return _opinion("BUY", 0.82, "Lower Bollinger touch with bullish reaction"), upper, lower, position

    if price >= upper and price <= prev_price:
        return _opinion("SELL", 0.82, "Upper Bollinger touch with bearish reaction"), upper, lower, position

    if position > 0.80 and price > prev_price and price > m:
        return _opinion("BUY", 0.55, "Upper-half Bollinger momentum"), upper, lower, position

    if position < 0.20 and price < prev_price and price < m:
        return _opinion("SELL", 0.55, "Lower-half Bollinger momentum"), upper, lower, position

    return _opinion("HOLD", 0.0, "Bollinger position not decisive"), upper, lower, position


# ------------------------------------------------------------
# VOLUME
# ------------------------------------------------------------

def indicator_volume(df):
    volume = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)

    if len(volume) < 20:
        return _opinion("HOLD", 0.0, "Not enough volume data"), 1.0

    avg = float(volume.rolling(20).mean().iloc[-1])
    current = float(volume.iloc[-1])

    if avg <= 0:
        return _opinion("HOLD", 0.0, "No usable volume"), 1.0

    ratio = current / avg

    if ratio < 1.5:
        return _opinion("HOLD", 0.0, f"No significant volume expansion ({ratio:.2f}x)"), ratio

    o, c = float(df["open"].iloc[-1]), float(df["close"].iloc[-1])
    strength = min(1.0, 0.65 + min(ratio - 1.5, 2) / 5)

    if c > o:
        return _opinion("BUY", strength, f"Volume expansion with bullish candle ({ratio:.2f}x)"), ratio

    if c < o:
        return _opinion("SELL", strength, f"Volume expansion with bearish candle ({ratio:.2f}x)"), ratio

    return _opinion("HOLD", 0.0, "Volume expansion without directional candle"), ratio


# ------------------------------------------------------------
# ADX
# ------------------------------------------------------------

def indicator_adx(df):
    adx, plus_di, minus_di = _wilder_adx_components(df, 14)

    if not all(np.isfinite(x) for x in [adx, plus_di, minus_di]):
        return _opinion("HOLD", 0.0, "ADX unavailable"), np.nan, np.nan, np.nan

    if adx < 20:
        return _opinion("HOLD", 0.0, f"Weak trend (ADX {adx:.1f})"), adx, plus_di, minus_di

    gap = abs(plus_di - minus_di) / max(plus_di + minus_di, 1e-12)

    if adx >= 25 and plus_di > minus_di:
        return _opinion("BUY", min(1.0, 0.60 + gap), f"+DI above -DI; ADX {adx:.1f}"), adx, plus_di, minus_di

    if adx >= 25 and minus_di > plus_di:
        return _opinion("SELL", min(1.0, 0.60 + gap), f"-DI above +DI; ADX {adx:.1f}"), adx, plus_di, minus_di

    return _opinion("HOLD", 0.0, f"ADX direction not decisive ({adx:.1f})"), adx, plus_di, minus_di


# ------------------------------------------------------------
# EMA
# ------------------------------------------------------------

def indicator_ema(df):
    close = df["close"]
    ema5 = close.ewm(span=5, adjust=False).mean()
    ema13 = close.ewm(span=13, adjust=False).mean()

    diff = float(ema5.iloc[-1] - ema13.iloc[-1])
    prev = float(ema5.iloc[-2] - ema13.iloc[-2])

    if diff > 0:
        strength = 1.0 if prev <= 0 else 0.65
        reason = "EMA5 above EMA13" + (" with fresh bullish crossover" if prev <= 0 else "")
        return _opinion("BUY", strength, reason)

    if diff < 0:
        strength = 1.0 if prev >= 0 else 0.65
        reason = "EMA5 below EMA13" + (" with fresh bearish crossover" if prev >= 0 else "")
        return _opinion("SELL", strength, reason)

    return _opinion("HOLD", 0.0, "EMA5 and EMA13 are equal")


# ------------------------------------------------------------
# MACD
# ------------------------------------------------------------

def indicator_macd(df):
    close = df["close"]
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal

    current, previous = float(hist.iloc[-1]), float(hist.iloc[-2])

    if current > 0:
        strength = 1.0 if previous <= 0 else (0.72 if current > previous else 0.62)
        reason = "MACD histogram bullish" + (" with bullish cross" if previous <= 0 else "")
        return _opinion("BUY", strength, reason)

    if current < 0:
        strength = 1.0 if previous >= 0 else (0.72 if current < previous else 0.62)
        reason = "MACD histogram bearish" + (" with bearish cross" if previous >= 0 else "")
        return _opinion("SELL", strength, reason)

    return _opinion("HOLD", 0.0, "MACD histogram neutral")


# ------------------------------------------------------------
# VWAP
# ------------------------------------------------------------

def indicator_vwap(df):
    volume = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    total = float(volume.sum())

    if total <= 0:
        return _opinion("HOLD", 0.0, "No usable volume for VWAP"), np.nan, np.nan

    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    vwap = float((typical * volume).sum() / total)
    price = float(df["close"].iloc[-1])
    distance = (price - vwap) / max(abs(vwap), 1e-12)

    if price > vwap:
        return _opinion("BUY", min(1.0, 0.60 + abs(distance) * 20), "Price above VWAP"), vwap, distance

    if price < vwap:
        return _opinion("SELL", min(1.0, 0.60 + abs(distance) * 20), "Price below VWAP"), vwap, distance

    return _opinion("HOLD", 0.0, "Price at VWAP"), vwap, distance


# ------------------------------------------------------------
# CANDLE PATTERN
# ------------------------------------------------------------

def indicator_candle(df):
    if len(df) < 3:
        return _opinion("HOLD", 0.0, "Not enough candles"), 0.0

    rows = df.iloc[-3:]
    o1, h1, l1, c1 = [float(rows.iloc[0][x]) for x in ("open", "high", "low", "close")]
    o2, h2, l2, c2 = [float(rows.iloc[1][x]) for x in ("open", "high", "low", "close")]
    o3, h3, l3, c3 = [float(rows.iloc[2][x]) for x in ("open", "high", "low", "close")]

    body3 = abs(c3 - o3)
    range3 = max(h3 - l3, 1e-12)
    body2 = abs(c2 - o2)

    upper3 = max(0.0, h3 - max(o3, c3))
    lower3 = max(0.0, min(o3, c3) - l3)

    doji = body3 / range3 < 0.10
    bullish_pin = not doji and lower3 >= 2 * max(body3, 1e-12) and upper3 <= body3
    bearish_pin = not doji and upper3 >= 2 * max(body3, 1e-12) and lower3 <= body3

    bullish_engulfing = c2 < o2 and c3 > o3 and o3 <= c2 and c3 >= o2
    bearish_engulfing = c2 > o2 and c3 < o3 and o3 >= c2 and c3 <= o2

    first_body = abs(c1 - o1)
    morning_star = (
        c1 < o1 and first_body > 0 and body2 < first_body * 0.30
        and c3 > o3 and c3 > (o1 + c1) / 2
    )
    evening_star = (
        c1 > o1 and first_body > 0 and body2 < first_body * 0.30
        and c3 < o3 and c3 < (o1 + c1) / 2
    )

    bullish = bullish_pin or bullish_engulfing or morning_star
    bearish = bearish_pin or bearish_engulfing or evening_star
    strength = min(1.0, 0.65 + body3 / range3 * 0.35)

    if bullish and not bearish:
        names = [n for n, f in (("bullish pin", bullish_pin), ("bullish engulfing", bullish_engulfing), ("morning star", morning_star)) if f]
        return _opinion("BUY", strength, ", ".join(names)), strength

    if bearish and not bullish:
        names = [n for n, f in (("bearish pin", bearish_pin), ("bearish engulfing", bearish_engulfing), ("evening star", evening_star)) if f]
        return _opinion("SELL", strength, ", ".join(names)), strength

    return _opinion("HOLD", 0.0, "No decisive candle pattern"), 0.0


# ------------------------------------------------------------
# ATR CONTEXT
# ------------------------------------------------------------

def indicator_atr_context(df):
    tr, atr = _atr_series(df, 14)
    atr_now = _safe_float(atr.iloc[-1])
    tr_now = _safe_float(tr.iloc[-1])

    if not np.isfinite(atr_now) or atr_now <= 0 or not np.isfinite(tr_now):
        return {"state": "UNKNOWN", "atr": np.nan, "ratio": np.nan, "reason": "ATR unavailable"}

    ratio = tr_now / atr_now

    if ratio > 2.5:
        state, reason = "HIGH", "Current true range is unusually large"
    elif ratio < 0.35:
        state, reason = "LOW", "Current true range is unusually small"
    else:
        state, reason = "NORMAL", "Current volatility is within normal range"

    return {"state": state, "atr": atr_now, "ratio": ratio, "reason": reason}


# ============================================================
# FEATURE NORMALIZATION
# ============================================================

def _normalise_feature(name, value):
    x = _safe_float(value, 0.0)

    if name == "rsi_norm":
        return _clamp((x - 50.0) / 50.0, -1.0, 1.0)
    if name == "adx_norm":
        return _clamp(x / 50.0, 0.0, 2.0)
    if name in {"bb_position", "buy_vote_ratio", "sell_vote_ratio"}:
        return _clamp(x, 0.0, 1.0)
    if name in {"ema_spread_norm", "macd_hist_norm", "vwap_distance_norm", "sma20_sma50_norm", "price_sma20_norm"}:
        return _clamp(x, -1.0, 1.0)
    if name == "volume_ratio_norm":
        return _clamp((x - 1.0) / 2.0, -1.0, 2.0)
    if name == "atr_pct_norm":
        return _clamp(x / 0.5, 0.0, 3.0)
    if name in {"candle_strength", "trend_strength", "directional_confidence_norm"}:
        return _clamp(x, 0.0, 1.0)
    return x


# ============================================================
# BUILD HISTORICAL SETUP FEATURES
# ============================================================

def build_setup_features(df, ind):
    close = df["close"]
    price = float(close.iloc[-1])
    sma20 = float(close.rolling(20).mean().iloc[-1])
    sma50 = float(close.rolling(50).mean().iloc[-1])
    ema5 = float(close.ewm(span=5, adjust=False).mean().iloc[-1])
    ema13 = float(close.ewm(span=13, adjust=False).mean().iloc[-1])
    ema_spread = (ema5 - ema13) / max(abs(price), 1e-12)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    hist_value = float((macd_line - signal).iloc[-1])

    tr, atr = _atr_series(df, 14)
    atr_value = _safe_float(atr.iloc[-1], 0.0)
    atr_pct = atr_value / max(price, 1e-12) * 100.0

    volume_ratio = 1.0
    if "volume" in df.columns and float(df["volume"].sum()) > 0:
        avg_vol = float(df["volume"].rolling(20).mean().iloc[-1])
        if avg_vol > 0:
            volume_ratio = float(df["volume"].iloc[-1]) / avg_vol

    vwap = ind.get("vwap", np.nan)
    vwap_distance = (price - vwap) / max(abs(vwap), 1e-12) if np.isfinite(vwap) else 0.0

    bb_upper, bb_lower = ind.get("upper", np.nan), ind.get("lower", np.nan)
    if np.isfinite(bb_upper) and np.isfinite(bb_lower):
        bb_position = (price - bb_lower) / max(bb_upper - bb_lower, 1e-12)
    else:
        bb_position = 0.5

    votes = ind["directional_votes"]
    total_votes = max(votes["BUY"] + votes["SELL"], 1)

    raw = {
        "rsi_norm": ind.get("rsi", 50.0),
        "adx_norm": ind.get("adx_val", 0.0),
        "bb_position": bb_position,
        "ema_spread_norm": ema_spread,
        "macd_hist_norm": hist_value / max(abs(price), 1e-12) * 100.0,
        "vwap_distance_norm": vwap_distance,
        "volume_ratio_norm": volume_ratio,
        "atr_pct_norm": atr_pct,
        "sma20_sma50_norm": (sma20 - sma50) / max(abs(price), 1e-12),
        "price_sma20_norm": (price - sma20) / max(abs(price), 1e-12),
        "candle_strength": ind.get("candle_strength", 0.0),
        "trend_strength": ind.get("trend_strength", 0.0),
        "directional_confidence_norm": ind["confidence"] / 100.0,
        "buy_vote_ratio": votes["BUY"] / total_votes,
        "sell_vote_ratio": votes["SELL"] / total_votes,
    }

    return {name: _normalise_feature(name, raw.get(name, 0.0)) for name in FEATURE_NAMES}


# ============================================================
# FEATURE DISTANCE
# ============================================================

def _feature_distance(a, b):
    if not isinstance(a, dict) or not isinstance(b, dict):
        return np.inf

    total, weight_sum = 0.0, 0.0

    for name, weight in SIMILARITY_WEIGHTS.items():
        av = _safe_float(a.get(name), np.nan)
        bv = _safe_float(b.get(name), np.nan)

        if not np.isfinite(av) or not np.isfinite(bv):
            continue

        total += weight * abs(av - bv)
        weight_sum += weight

    if weight_sum <= 0:
        return np.inf

    return total / weight_sum


# ============================================================
# HISTORICAL EVIDENCE PIPELINE
# ============================================================
#
#   Stage 1/2: _fast_historical_filter   -- indexed SQL, bounded LIMIT
#   Stage 3:   _rank_by_similarity       -- weighted feature distance
#   Stage 4:   _select_top_relevant      -- keep closest HISTORICAL_TOP_N
#   Stage 5:   _recency_weighted_statistics -- age+similarity weighting
#              _classify_evidence_status -- INSUFFICIENT/WEAK/MODERATE/STRONG
#
#   find_similar_historical_setups() orchestrates all of the above and is
#   the single entry point the rest of the bot uses -- there is exactly
#   one historical-similarity engine in this codebase.
#
# ============================================================

def _fast_historical_filter(symbol, side, timeframe, start_time):
    """
    Stage 1/2 - Fast Historical Filter.

    Uses the idx_trades_history_tf composite index so this stays an
    index range scan even when the trades table holds millions of rows.
    HISTORICAL_LOOKBACK is a *processing* cap on this single query, not a
    retention limit -- every other historical row remains untouched in
    PostgreSQL.
    """
    try:
        with db_conn() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(
                """
                SELECT id, result, setup_features, start_time
                FROM trades
                WHERE symbol=%s
                  AND timeframe=%s
                  AND side=%s
                  AND status='COMPLETED'
                  AND result IN ('WIN','LOSS')
                  AND start_time < %s
                  AND setup_features <> '{}'::jsonb
                ORDER BY start_time DESC
                LIMIT %s
                """,
                (symbol, timeframe, side, start_time, HISTORICAL_LOOKBACK),
            )
            rows = cur.fetchall()
            cur.close()
        return rows
    except Exception as exc:
        print(f"Fast Historical Filter Error: {exc}")
        return []


def _rank_by_similarity(features, candidates):
    """Stage 3 - Feature Similarity. Only trades within the distance
    threshold are considered a genuine match at all."""
    ranked = []

    for row in candidates:
        fs = row.get("setup_features")

        if isinstance(fs, str):
            try:
                fs = json.loads(fs)
            except (TypeError, ValueError):
                continue

        if not isinstance(fs, dict):
            continue

        distance = _feature_distance(features, fs)

        if np.isfinite(distance) and distance <= HISTORICAL_DISTANCE_THRESHOLD:
            ranked.append((distance, row))

    ranked.sort(key=lambda x: x[0])
    return ranked


def _select_top_relevant(ranked):
    """Stage 4 - Top Relevant Trades. A processing cap, not a retention
    limit: everything not selected here simply isn't used for *this*
    decision, it still exists permanently in PostgreSQL."""
    return ranked[:HISTORICAL_TOP_N]


def _recency_weight(age_days):
    return 0.5 ** (max(0.0, age_days) / HISTORICAL_RECENCY_HALF_LIFE_DAYS)


def _similarity_weight(distance):
    return _clamp(1.0 - (distance / max(HISTORICAL_DISTANCE_THRESHOLD, 1e-9)), 0.05, 1.0)


def _recency_weighted_statistics(selected, now):
    """
    Stage 5 - combines recency and similarity into one per-trade weight,
    then produces weighted win/loss statistics plus an effective sample
    size (Kish's ESS = (sum w)^2 / sum(w^2)), so a handful of heavily
    weighted trades cannot masquerade as a large, reliable sample.

    An empty `selected` list is a normal, valid state and returns a
    neutral/insufficient result -- never a fabricated win rate.
    """
    if not selected:
        return {
            "weighted_wins": 0.0,
            "weighted_losses": 0.0,
            "recent_weighted_win_rate": None,
            "effective_sample_size": 0.0,
            "average_similarity": None,
            "best_similarity": None,
        }

    weights, outcomes, similarities = [], [], []

    for distance, row in selected:
        start_time = row.get("start_time")
        age_days = (
            max(0.0, (now - start_time).total_seconds() / 86400.0)
            if isinstance(start_time, datetime)
            else 0.0
        )

        weights.append(_recency_weight(age_days) * _similarity_weight(distance))
        outcomes.append(1.0 if row.get("result") == "WIN" else 0.0)
        similarities.append(1.0 - _clamp(distance / max(HISTORICAL_DISTANCE_THRESHOLD, 1e-9), 0.0, 1.0))

    weights_arr = np.array(weights, dtype=float)
    outcomes_arr = np.array(outcomes, dtype=float)

    total_weight = float(weights_arr.sum())
    weighted_wins = float((weights_arr * outcomes_arr).sum())
    weighted_losses = float(total_weight - weighted_wins)

    recent_weighted_win_rate = weighted_wins / total_weight * 100.0 if total_weight > 0 else None

    sq_sum = float((weights_arr ** 2).sum())
    effective_sample_size = (total_weight ** 2) / sq_sum if sq_sum > 0 else 0.0

    return {
        "weighted_wins": weighted_wins,
        "weighted_losses": weighted_losses,
        "recent_weighted_win_rate": recent_weighted_win_rate,
        "effective_sample_size": effective_sample_size,
        "average_similarity": float(np.mean(similarities)),
        "best_similarity": float(np.max(similarities)),
    }


def _classify_evidence_status(effective_sample_size):
    if effective_sample_size <= 0:
        return "INSUFFICIENT"
    if effective_sample_size < HISTORICAL_MIN_SAMPLES:
        return "WEAK"
    if effective_sample_size < HISTORICAL_STRONG_SAMPLE_MIN:
        return "MODERATE"
    return "STRONG"


def find_similar_historical_setups(symbol, side, features, start_time, timeframe=TIMEFRAME):
    """
    Historical Evidence layer -- the single entry point used everywhere
    else in the bot. Zero relevant historical trades is a normal, valid
    state: it returns evidence_status="INSUFFICIENT" and win rates of
    None, never a fabricated percentage, and never raises.
    """
    result = {
        "count": 0,
        "wins": 0,
        "losses": 0,
        "win_rate": None,
        "best_distance": None,
        "average_distance": None,
        "weighted_wins": 0.0,
        "weighted_losses": 0.0,
        "recent_weighted_win_rate": None,
        "effective_sample_size": 0.0,
        "average_similarity": None,
        "best_similarity": None,
        "status": "INSUFFICIENT_DATA",
        "evidence_status": "INSUFFICIENT",
    }

    try:
        candidates = _fast_historical_filter(symbol, side, timeframe, start_time)
        ranked = _rank_by_similarity(features, candidates)
        selected = _select_top_relevant(ranked)

        result["count"] = len(selected)
        result["wins"] = sum(1 for _, row in selected if row["result"] == "WIN")
        result["losses"] = sum(1 for _, row in selected if row["result"] == "LOSS")

        if selected:
            distances = [float(d) for d, _ in selected]
            result["best_distance"] = distances[0]
            result["average_distance"] = sum(distances) / len(distances)

        if result["count"] >= HISTORICAL_MIN_SAMPLES:
            result["win_rate"] = result["wins"] / result["count"] * 100.0
            result["status"] = "VALID"

        result.update(_recency_weighted_statistics(selected, get_now()))
        result["evidence_status"] = _classify_evidence_status(result["effective_sample_size"])

        return result

    except Exception as exc:
        print(f"Historical Evidence Pipeline Error: {exc}")
        return result


# ============================================================
# ML PREDICTION LAYER
# ============================================================
#
# A separate, independent evidence layer. Trained only on trades that
# are already COMPLETED with start_time strictly before the moment of
# training/prediction -- the eventual result of the *current* trade is
# never available yet, so there is no leakage into its own prediction.
# Models are per (symbol, side), matching the market-isolation rule used
# for indicator weights. An untrained or under-trained model reports
# UNAVAILABLE / INSUFFICIENT_TRAINING_DATA and never fabricates a
# probability.
#
# ============================================================

def _sigmoid(z):
    z = np.clip(z, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def _fit_logistic_regression(X, y, l2=ML_L2_REG, lr=ML_LEARNING_RATE, iterations=ML_TRAIN_ITERATIONS):
    """
    Dependency-free (numpy only) L2-regularised logistic regression via
    batch gradient descent. Features are standardised for training
    stability, then the standardisation is folded back into a single
    raw-feature weight vector + bias so prediction time is a plain dot
    product against the untouched setup_features.
    """
    n_samples, n_features = X.shape

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-9] = 1.0
    X_std = (X - mean) / std

    weights = np.zeros(n_features, dtype=float)
    bias = 0.0

    for _ in range(iterations):
        preds = _sigmoid(X_std @ weights + bias)
        error = preds - y

        grad_w = (X_std.T @ error) / n_samples + l2 * weights
        grad_b = float(error.mean())

        weights -= lr * grad_w
        bias -= lr * grad_b

    preds = _sigmoid(X_std @ weights + bias)
    train_accuracy = float(((preds >= 0.5).astype(float) == y).mean())

    raw_weights = weights / std
    raw_bias = float(bias - float((mean * raw_weights).sum()))

    return raw_weights, raw_bias, train_accuracy


def _get_ml_model(symbol, side):
    try:
        with db_conn() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(
                """
                SELECT weights, bias, feature_names, training_samples,
                       train_accuracy, trained_at
                FROM ml_models
                WHERE symbol=%s AND side=%s
                """,
                (symbol, side),
            )
            row = cur.fetchone()
            cur.close()
        return dict(row) if row else None
    except Exception as exc:
        print(f"Get ML Model Error: {exc}")
        return None


def _save_ml_model(symbol, side, weights, bias, feature_names, n_samples, train_accuracy):
    try:
        with db_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO ml_models(
                    symbol, side, weights, bias, feature_names,
                    training_samples, train_accuracy, trained_at
                )
                VALUES(%s,%s,%s,%s,%s,%s,%s,NOW())
                ON CONFLICT(symbol, side) DO UPDATE SET
                    weights=EXCLUDED.weights,
                    bias=EXCLUDED.bias,
                    feature_names=EXCLUDED.feature_names,
                    training_samples=EXCLUDED.training_samples,
                    train_accuracy=EXCLUDED.train_accuracy,
                    trained_at=NOW()
                """,
                (
                    symbol, side,
                    Json(_json_safe(list(weights))),
                    float(bias),
                    Json(list(feature_names)),
                    int(n_samples),
                    float(train_accuracy),
                ),
            )
            conn.commit()
            cur.close()
        return True
    except Exception as exc:
        print(f"Save ML Model Error: {exc}")
        return False


def _fetch_ml_training_data(symbol, side, now):
    """Only trades completed strictly before `now` -- same no-lookahead
    guarantee as the historical evidence pipeline."""
    try:
        with db_conn() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(
                """
                SELECT setup_features, result
                FROM trades
                WHERE symbol=%s
                  AND timeframe=%s
                  AND side=%s
                  AND status='COMPLETED'
                  AND result IN ('WIN','LOSS')
                  AND start_time < %s
                  AND setup_features <> '{}'::jsonb
                ORDER BY start_time DESC
                LIMIT %s
                """,
                (symbol, TIMEFRAME, side, now, ML_TRAIN_HISTORY_LIMIT),
            )
            rows = cur.fetchall()
            cur.close()
        return rows
    except Exception as exc:
        print(f"Fetch ML Training Data Error: {exc}")
        return []


def _train_ml_model(symbol, side, now):
    rows = _fetch_ml_training_data(symbol, side, now)

    if len(rows) < ML_MIN_TRAINING_SAMPLES:
        return {"status": "INSUFFICIENT_TRAINING_DATA", "samples": len(rows)}

    X_list, y_list = [], []

    for row in rows:
        fs = row.get("setup_features")

        if isinstance(fs, str):
            try:
                fs = json.loads(fs)
            except (TypeError, ValueError):
                continue

        if not isinstance(fs, dict):
            continue

        vector = [_safe_float(fs.get(name), 0.0) for name in FEATURE_NAMES]
        if not all(np.isfinite(v) for v in vector):
            continue

        X_list.append(vector)
        y_list.append(1.0 if row["result"] == "WIN" else 0.0)

    if len(X_list) < ML_MIN_TRAINING_SAMPLES:
        return {"status": "INSUFFICIENT_TRAINING_DATA", "samples": len(X_list)}

    X = np.array(X_list, dtype=float)
    y = np.array(y_list, dtype=float)

    weights, bias, train_accuracy = _fit_logistic_regression(X, y)
    _save_ml_model(symbol, side, weights, bias, FEATURE_NAMES, len(X_list), train_accuracy)

    return {"status": "TRAINED", "samples": len(X_list), "train_accuracy": train_accuracy}


_ml_model_cache = {}
_ml_cache_lock = threading.Lock()
_ml_last_train_attempt = {}
_ml_train_lock = threading.Lock()


def ml_predict(symbol, side, features):
    """
    Separate ML Prediction layer. Never fabricates a probability: an
    absent or under-trained model reports its status explicitly and
    probability=None, which the Final Decision engine treats as "this
    layer contributes nothing" rather than as a value to trust.
    """
    if not ML_ENABLED:
        return {"status": "DISABLED", "probability": None}

    cache_key = (symbol, side)

    with _ml_cache_lock:
        model = _ml_model_cache.get(cache_key)

    if model is None:
        model = _get_ml_model(symbol, side)
        if model is not None:
            with _ml_cache_lock:
                _ml_model_cache[cache_key] = model

    if model is None:
        return {"status": "UNAVAILABLE", "probability": None}

    if int(model.get("training_samples", 0)) < ML_MIN_TRAINING_SAMPLES:
        return {"status": "INSUFFICIENT_TRAINING_DATA", "probability": None}

    weights_raw = model.get("weights")
    if isinstance(weights_raw, str):
        try:
            weights_raw = json.loads(weights_raw)
        except (TypeError, ValueError):
            return {"status": "UNAVAILABLE", "probability": None}

    feature_names = model.get("feature_names") or FEATURE_NAMES
    if isinstance(feature_names, str):
        try:
            feature_names = json.loads(feature_names)
        except (TypeError, ValueError):
            feature_names = FEATURE_NAMES

    try:
        weights = np.array([float(w) for w in weights_raw], dtype=float)
        vector = np.array([_safe_float(features.get(name), 0.0) for name in feature_names], dtype=float)
        z = float(vector @ weights + float(model.get("bias", 0.0)))
        probability = float(_sigmoid(np.array([z]))[0]) * 100.0
    except Exception as exc:
        print(f"ML Predict Error: {exc}")
        return {"status": "UNAVAILABLE", "probability": None}

    return {
        "status": "AVAILABLE",
        "probability": probability,
        "training_samples": int(model.get("training_samples", 0)),
        "train_accuracy": _safe_float(model.get("train_accuracy"), None),
    }


def maybe_retrain_ml_models(now):
    """
    Retrains at most one (symbol, side) model per call, gated by
    ML_RETRAIN_COOLDOWN_MINUTES, so retraining cost is spread out over
    time rather than blocking the scanner or the result-check loop.
    """
    if not ML_ENABLED:
        return

    with _ml_train_lock:
        for symbol in SYMBOLS:
            for side in ("BUY", "SELL"):
                key = (symbol, side)
                last_attempt = _ml_last_train_attempt.get(key)

                if (
                    last_attempt is not None
                    and (now - last_attempt).total_seconds() < ML_RETRAIN_COOLDOWN_MINUTES * 60
                ):
                    continue

                _ml_last_train_attempt[key] = now
                outcome = _train_ml_model(symbol, side, now)

                if outcome.get("status") == "TRAINED":
                    with _ml_cache_lock:
                        _ml_model_cache.pop(key, None)
                    print(
                        f"[ML] Retrained {symbol} {side}: {outcome['samples']} samples, "
                        f"train_accuracy={outcome.get('train_accuracy', 0):.3f}"
                    )

                return  # one model per call keeps this cheap and non-blocking


# ============================================================
# FINAL DECISION ENGINE
# ============================================================
#
# Combines three independent evidence layers -- Indicator Confidence,
# Historical Evidence, ML Prediction -- without ever conflating them.
# A layer that is unavailable/insufficient contributes zero weight (never
# a fabricated value); the remaining weights are renormalised across
# whatever evidence actually exists. Output is always BUY / SELL /
# NO_TRADE, never a crash, regardless of which layers are available.
#
# ============================================================

def make_final_decision(side, indicator_confidence, historical_evidence, ml_result):
    components = [("indicator", _clamp(indicator_confidence, 0.0, 100.0), FINAL_WEIGHT_INDICATOR)]

    hist_status = historical_evidence.get("evidence_status", "INSUFFICIENT")
    hist_rate = historical_evidence.get("recent_weighted_win_rate")

    if hist_status in {"MODERATE", "STRONG"} and hist_rate is not None:
        components.append(("historical", _clamp(hist_rate, 0.0, 100.0), FINAL_WEIGHT_HISTORICAL))
    else:
        components.append(("historical", None, 0.0))

    ml_status = ml_result.get("status")
    ml_probability = ml_result.get("probability")

    if ml_status == "AVAILABLE" and ml_probability is not None:
        components.append(("ml", _clamp(ml_probability, 0.0, 100.0), FINAL_WEIGHT_ML))
    else:
        components.append(("ml", None, 0.0))

    total_weight = sum(w for _, value, w in components if value is not None)

    if total_weight <= 0:
        combined_confidence = _clamp(indicator_confidence, 0.0, 100.0)
    else:
        combined_confidence = sum(value * w for _, value, w in components if value is not None) / total_weight

    veto = hist_status == "STRONG" and hist_rate is not None and hist_rate < FINAL_HISTORICAL_VETO_WIN_RATE

    decision = side if (not veto and combined_confidence >= FINAL_MIN_COMBINED_CONFIDENCE) else "NO_TRADE"

    return {
        "decision": decision,
        "combined_confidence": round(float(combined_confidence), 2),
        "historical_vetoed": bool(veto),
        "components": {name: (round(value, 2) if value is not None else None) for name, value, _ in components},
    }


# ============================================================
# INDICATOR ENGINE
# ============================================================

def calculate_indicators(values, symbol="EUR/USD"):
    if not values:
        return None

    closed_values = _closed_candles(values)
    if len(closed_values) < 80:
        return None

    df = pd.DataFrame(closed_values)
    required = ["open", "high", "low", "close"]

    for col in required:
        if col not in df.columns:
            return None
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
        volume_available = bool(df["volume"].sum() > 0)
    else:
        df["volume"] = 0.0
        volume_available = False

    df.dropna(subset=required, inplace=True)
    df.reset_index(drop=True, inplace=True)

    if len(df) < 80:
        return None

    trend, trend_meta = indicator_trend(df)
    rsi, rsi_value = indicator_rsi(df)
    bb, upper, lower, bb_position = indicator_bollinger(df)

    if volume_available:
        volume, volume_ratio = indicator_volume(df)
    else:
        volume = _opinion("HOLD", 0.0, "Volume unavailable")
        volume_ratio = 1.0

    adx, adx_value, plus_di, minus_di = indicator_adx(df)
    ema = indicator_ema(df)
    macd = indicator_macd(df)

    if volume_available:
        vwap, vwap_value, vwap_distance = indicator_vwap(df)
    else:
        vwap = _opinion("HOLD", 0.0, "Volume unavailable for VWAP")
        vwap_value, vwap_distance = np.nan, 0.0

    candle, candle_strength = indicator_candle(df)
    atr_context = indicator_atr_context(df)

    opinions = {
        "trend": trend, "rsi": rsi, "bb": bb, "volume": volume, "adx": adx,
        "ema": ema, "macd": macd, "vwap": vwap, "candle": candle,
    }

    weights = get_weights(symbol)

    buy_score, sell_score, available_weight = 0.0, 0.0, 0.0

    for name, opinion in opinions.items():
        weight = float(weights.get(name, 0.0))
        if weight <= 0:
            continue

        available_weight += weight
        strength = _clamp(opinion.get("strength", 0.0))

        if opinion["signal"] == "BUY":
            buy_score += weight * strength
        elif opinion["signal"] == "SELL":
            sell_score += weight * strength

    directional_total = buy_score + sell_score

    if directional_total <= 0:
        final_side, directional_confidence = "HOLD", 0.0
    elif buy_score > sell_score:
        final_side = "BUY"
        directional_confidence = buy_score / directional_total * 100.0
    elif sell_score > buy_score:
        final_side = "SELL"
        directional_confidence = sell_score / directional_total * 100.0
    else:
        final_side, directional_confidence = "HOLD", 0.0

    directional_votes = {
        "BUY": sum(1 for x in opinions.values() if x["signal"] == "BUY"),
        "SELL": sum(1 for x in opinions.values() if x["signal"] == "SELL"),
        "HOLD": sum(1 for x in opinions.values() if x["signal"] == "HOLD"),
    }

    total_directional = directional_votes["BUY"] + directional_votes["SELL"]
    conflict_ratio = (
        min(directional_votes["BUY"], directional_votes["SELL"]) / total_directional
        if total_directional else 0.0
    )

    vote_support = directional_votes.get(final_side, 0)
    vote_bonus = min(8.0, max(0, vote_support - 1) * 1.5)

    confidence = max(
        0.0,
        min(99.0, directional_confidence - conflict_ratio * 10.0 + vote_bonus),
    )

    tradable = (
        final_side in {"BUY", "SELL"}
        and vote_support >= MIN_DIRECTIONAL_VOTES
        and confidence >= MIN_TRADE_CONFIDENCE
        and conflict_ratio <= MAX_DIRECTIONAL_CONFLICT
        and atr_context["state"] != "HIGH"
    )

    result = {
        "curr_p": float(df["close"].iloc[-1]),
        "side": final_side,
        "confidence": float(confidence),
        "directional_confidence": float(directional_confidence),
        "buy_score": float(buy_score),
        "sell_score": float(sell_score),
        "available_weight": float(available_weight),
        "directional_votes": directional_votes,
        "conflict_ratio": float(conflict_ratio),
        "tradable": bool(tradable),
        "opinions": opinions,
        "weights": weights,
        "atr_context": atr_context,
        "rsi": float(rsi_value) if np.isfinite(rsi_value) else np.nan,
        "upper": float(upper) if np.isfinite(upper) else np.nan,
        "lower": float(lower) if np.isfinite(lower) else np.nan,
        "bb_position": float(bb_position) if np.isfinite(bb_position) else 0.5,
        "adx_val": float(adx_value) if np.isfinite(adx_value) else np.nan,
        "plus_di": float(plus_di) if np.isfinite(plus_di) else np.nan,
        "minus_di": float(minus_di) if np.isfinite(minus_di) else np.nan,
        "vwap": float(vwap_value) if np.isfinite(vwap_value) else np.nan,
        "vwap_distance": float(vwap_distance) if np.isfinite(vwap_distance) else 0.0,
        "volume_ratio": float(volume_ratio),
        "candle_strength": float(candle_strength),
        "trend_strength": float(trend_meta.get("trend_strength", 0.0)),
        "trend_state": float(trend_meta.get("trend_state", 0.0)),
        "sma_20": float(df["close"].rolling(20).mean().iloc[-1]),
        "sma_50": float(df["close"].rolling(50).mean().iloc[-1]),
        "last_closed_candle_time": _parse_candle_datetime(closed_values[-1].get("datetime")),
    }

    result["setup_features"] = build_setup_features(df, result)
    return result


# ============================================================
# TRADE STORAGE
# ============================================================

def save_active_trade(trade):
    try:
        with db_conn() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)

            cur.execute(
                """
                INSERT INTO trades(
                    symbol, entry_price, side, start_time, expiry_time, rec_time,
                    timeframe,
                    indicators_status, indicator_opinions, setup_features,
                    indicator_confidence,
                    historical_count, historical_wins, historical_losses,
                    historical_win_rate, historical_distance,
                    historical_weighted_win_rate, historical_effective_sample_size,
                    historical_similarity_strength, historical_evidence_status,
                    ml_probability, ml_status, final_decision_meta,
                    status, next_retry_time
                )
                VALUES(
                    %s,%s,%s,%s,%s,%s,
                    %s,
                    %s,%s,%s,
                    %s,
                    %s,%s,%s,
                    %s,%s,
                    %s,%s,
                    %s,%s,
                    %s,%s,%s,
                    'ACTIVE',
                    %s
                )
                RETURNING id
                """,
                (
                    trade["symbol"],
                    trade["entry_price"],
                    trade["side"],
                    trade["start_time"],
                    trade["expiry_time"],
                    trade["rec_time"],
                    trade.get("timeframe", TIMEFRAME),
                    Json(_json_safe(trade["indicators_status"])),
                    Json(_json_safe(trade["indicator_opinions"])),
                    Json(_json_safe(trade["setup_features"])),
                    trade["indicator_confidence"],
                    trade["historical"]["count"],
                    trade["historical"]["wins"],
                    trade["historical"]["losses"],
                    trade["historical"]["win_rate"],
                    trade["historical"]["best_distance"],
                    trade["historical"].get("recent_weighted_win_rate"),
                    trade["historical"].get("effective_sample_size"),
                    trade["historical"].get("average_similarity"),
                    trade["historical"].get("evidence_status"),
                    trade.get("ml_probability"),
                    trade.get("ml_status"),
                    Json(_json_safe(trade.get("final_decision_meta", {}))),
                    trade["expiry_time"] + timedelta(seconds=30),
                ),
            )

            row = cur.fetchone()
            conn.commit()
            cur.close()
            return row["id"] if row else None

    except errors.UniqueViolation:
        print("Trade creation rejected: another live trade exists.")
        return None
    except Exception as exc:
        print(f"Save Trade Error: {exc}")
        return None


def has_active_trade():
    try:
        with db_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT EXISTS(
                    SELECT 1 FROM trades
                    WHERE status IN ('ACTIVE','PENDING_RESULT','RESULT_PROCESSING')
                )
                """
            )
            row = cur.fetchone()
            cur.close()
            return bool(row[0]) if row else True
    except Exception as exc:
        print(f"Active Trade Check Error: {exc}")
        return True


def update_expired_trades():
    now = get_now()
    try:
        with db_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE trades
                SET status='PENDING_RESULT',
                    next_retry_time=COALESCE(next_retry_time, expiry_time),
                    updated_at=NOW()
                WHERE status='ACTIVE' AND expiry_time<=%s
                """,
                (now,),
            )
            count = cur.rowcount
            conn.commit()
            cur.close()
            return count
    except Exception as exc:
        print(f"Expired Trade Update Error: {exc}")
        return 0


def claim_next_pending_trade(now):
    try:
        with db_conn() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(
                """
                SELECT * FROM trades
                WHERE status='PENDING_RESULT'
                  AND COALESCE(next_retry_time, expiry_time)<=%s
                ORDER BY id
                FOR UPDATE SKIP LOCKED
                LIMIT 1
                """,
                (now,),
            )
            trade = cur.fetchone()

            if not trade:
                conn.commit()
                cur.close()
                return None

            cur.execute(
                "UPDATE trades SET status='RESULT_PROCESSING', updated_at=NOW() WHERE id=%s AND status='PENDING_RESULT'",
                (trade["id"],),
            )
            conn.commit()
            cur.close()
            return dict(trade)
    except Exception as exc:
        print(f"Claim Result Error: {exc}")
        return None


# ============================================================
# RESULT VERIFICATION
# ============================================================

def get_candle_at_time(symbol, target_time, api_key, field="open"):
    if field not in {"open", "high", "low", "close"}:
        return None

    if isinstance(target_time, (int, float)):
        target = datetime.fromtimestamp(target_time, BD_TZ)
    elif isinstance(target_time, datetime):
        target = target_time if target_time.tzinfo else target_time.replace(tzinfo=BD_TZ)
        target = target.astimezone(BD_TZ)
    else:
        return None

    target = target.replace(second=0, microsecond=0)
    values = get_market_data_window(symbol, api_key, target, target + timedelta(minutes=1))

    if not values:
        return None

    for candle in _sort_values_chronologically(values):
        dt = _parse_candle_datetime(candle.get("datetime"))
        if dt and dt.replace(second=0, microsecond=0) == target:
            return _safe_float(candle.get(field), None)

    return None


def _mark_trade_retry(trade_id, retry_count, now):
    new_retry = retry_count + 1
    if new_retry >= RESULT_MAX_RETRIES:
        return False

    try:
        with db_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE trades
                SET status='PENDING_RESULT', retry_count=%s, next_retry_time=%s, updated_at=NOW()
                WHERE id=%s AND status='RESULT_PROCESSING'
                """,
                (new_retry, now + timedelta(seconds=RESULT_RETRY_SECONDS), trade_id),
            )
            conn.commit()
            cur.close()
            return True
    except Exception as exc:
        print(f"Retry Update Error: {exc}")
        return False


def _mark_unavailable(trade_id):
    try:
        with db_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE trades SET status='FAILED', result='UNAVAILABLE', updated_at=NOW() WHERE id=%s AND status='RESULT_PROCESSING'",
                (trade_id,),
            )
            conn.commit()
            cur.close()
    except Exception as exc:
        print(f"Unavailable Update Error: {exc}")


# ============================================================
# MARKET-SPECIFIC INDICATOR LEARNING
# ============================================================

def complete_trade_and_learn(trade_id, symbol, win, entry_price, indicator_opinions):
    try:
        with db_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE trades
                SET status='COMPLETED', result=%s, entry_price=%s, updated_at=NOW()
                WHERE id=%s AND status='RESULT_PROCESSING'
                RETURNING side
                """,
                ("WIN" if win else "LOSS", entry_price, trade_id),
            )
            row = cur.fetchone()

            if row is None:
                conn.rollback()
                cur.close()
                return False

            trade_side = row[0]

            if win:
                cur.execute("UPDATE trading_stats SET total_trades=total_trades+1, wins=wins+1")
            else:
                cur.execute("UPDATE trading_stats SET total_trades=total_trades+1, losses=losses+1")

            for name, opinion in (indicator_opinions or {}).items():
                if name not in DEFAULT_WEIGHTS or not isinstance(opinion, dict):
                    continue

                signal = opinion.get("signal")
                if signal not in {"BUY", "SELL"}:
                    continue

                correct = signal == trade_side
                delta = 0.03 if ((win and correct) or (not win and not correct)) else -0.03

                cur.execute(
                    """
                    UPDATE indicator_weights
                    SET accuracy_factor=LEAST(GREATEST(accuracy_factor + %s, 0.50), 1.50)
                    WHERE symbol=%s AND name=%s
                    """,
                    (delta, symbol, name),
                )

            conn.commit()
            cur.close()
            return True
    except Exception as exc:
        print(f"Complete/Learn Transaction Error: {exc}")
        return False


def mark_message_sent(trade_id, message_type):
    column = "msg_sent" if message_type == "entry" else "result_sent"
    if column not in {"msg_sent", "result_sent"}:
        return False

    try:
        with db_conn() as conn:
            cur = conn.cursor()
            cur.execute(f"UPDATE trades SET {column}=TRUE, updated_at=NOW() WHERE id=%s", (trade_id,))
            conn.commit()
            cur.close()
            return True
    except Exception as exc:
        print(f"Message Flag Error: {exc}")
        return False


# ============================================================
# RESULT CHECKER
# ============================================================

def check_result():
    update_expired_trades()

    while True:
        trade = claim_next_pending_trade(get_now())
        if not trade:
            break

        trade_id = trade["id"]
        symbol = trade["symbol"]
        side = trade["side"]
        start_time = trade["start_time"]
        expiry = trade["expiry_time"]
        retry_count = int(trade["retry_count"] or 0)

        entry_price = get_candle_at_time(symbol, start_time, RESULT_KEY_ENTRY, "open")
        exit_price = get_candle_at_time(symbol, expiry, RESULT_KEY_EXIT, "open")

        if entry_price is None or exit_price is None:
            if retry_count + 1 < RESULT_MAX_RETRIES:
                _mark_trade_retry(trade_id, retry_count, get_now())
            else:
                _mark_unavailable(trade_id)
                send_telegram_msg(
                    f"⚠️ RESULT UNAVAILABLE\n\nAsset: {symbol}\n"
                    f"Could not verify result after {RESULT_MAX_RETRIES} attempts."
                )
            continue

        if side == "BUY":
            win = exit_price > entry_price
        elif side == "SELL":
            win = exit_price < entry_price
        else:
            _mark_unavailable(trade_id)
            continue

        indicator_opinions = trade.get("indicator_opinions") or {}
        if isinstance(indicator_opinions, str):
            try:
                indicator_opinions = json.loads(indicator_opinions)
            except (TypeError, ValueError):
                indicator_opinions = {}

        if not complete_trade_and_learn(trade_id, symbol, win, entry_price, indicator_opinions):
            continue

        stats = get_stats()
        win_rate = stats["wins"] / stats["total_trades"] * 100 if stats["total_trades"] else 0.0
        result_emoji = "✅ WIN" if win else "❌ LOSS"

        msg = (
            "🏁 TRADE RESULT\n\n"
            f"📊 Asset: {symbol}\n"
            f"🏆 Result: {result_emoji}\n"
            f"🚀 Entry: {entry_price:.8f}\n"
            f"🕒 Entry Time: {start_time.astimezone(BD_TZ).strftime('%H:%M')}\n\n"
            f"🏁 Exit: {exit_price:.8f}\n"
            f"🕒 Expiry Time: {expiry.astimezone(BD_TZ).strftime('%H:%M')}\n"
            f"📈 Overall Win Rate: {win_rate:.1f}%"
        )

        if send_telegram_msg(msg):
            mark_message_sent(trade_id, "result")

        print(f"[{get_now().strftime('%H:%M:%S')}] Verified Result: {symbol} - {result_emoji}")


# ============================================================
# SCANNER
# ============================================================

def fetch_and_analyze_batch(symbols_chunk, api_key):
    batch = get_batch_market_data(symbols_chunk, api_key, outputsize=100)
    results = []

    for symbol in symbols_chunk:
        values = batch.get(symbol)
        if not values:
            continue

        ind = calculate_indicators(values, symbol)
        if not ind or ind["side"] not in {"BUY", "SELL"}:
            continue

        results.append({"symbol": symbol, **ind})

    return results


_last_scan_slot = None
_scan_lock = threading.Lock()


def _get_scan_slot(now):
    return now.strftime("%Y-%m-%d %H:%M")


def _choose_best_signal(results):
    tradable = [x for x in results if x["tradable"]]
    if not tradable:
        return None

    return max(
        tradable,
        key=lambda x: (
            x["confidence"],
            x["directional_votes"][x["side"]],
            x["buy_score"] if x["side"] == "BUY" else x["sell_score"],
        ),
    )


def _format_opinion_line(name, opinion):
    return f"{name.upper()}: {opinion.get('signal', 'HOLD')} ({opinion.get('reason', '')})"


def _choose_expiry(confidence, atr_pct, historical):
    """Historical evidence only nudges expiry conservatively -- it is
    never treated as a guaranteed outcome. Requires both a MODERATE/STRONG
    evidence_status AND an effective sample size past
    HISTORICAL_CONFIDENCE_MIN_SAMPLES before it is allowed to move
    anything, otherwise a thin sample cannot sway trade duration."""

    if atr_pct > 0.30:
        base = 5 if confidence < STRONG_CONFIDENCE else 7
    elif atr_pct > 0.10:
        base = 8 if confidence < STRONG_CONFIDENCE else 10
    else:
        base = 12 if confidence < STRONG_CONFIDENCE else 15

    if (
        historical.get("evidence_status") in {"MODERATE", "STRONG"}
        and historical.get("effective_sample_size", 0) >= HISTORICAL_CONFIDENCE_MIN_SAMPLES
    ):
        historical_rate = historical.get("recent_weighted_win_rate") or 0

        if historical_rate < 50:
            base = max(MIN_EXPIRY_MINUTES, base - 2)
        elif historical_rate >= 65:
            base = min(MAX_EXPIRY_MINUTES, base + 1)

    return int(max(MIN_EXPIRY_MINUTES, min(MAX_EXPIRY_MINUTES, base)))


# ============================================================
# MAIN SCANNER
# ============================================================

def run_scanner():
    global _last_scan_slot

    now = get_now()

    if not (0 <= now.second <= 15):
        return

    slot = _get_scan_slot(now)

    with _scan_lock:
        if slot == _last_scan_slot:
            return
        _last_scan_slot = slot

    if has_active_trade():
        print(f"[{now.strftime('%H:%M:%S')}] Active trade in progress. Scanning skipped.")
        return

    print(f"[{now.strftime('%H:%M:%S')}] Scanning markets for independent indicator signals...")

    chunks = [SYMBOLS[i:i + 3] for i in range(0, len(SYMBOLS), 3)]
    all_results = []

    with ThreadPoolExecutor(max_workers=min(7, len(chunks))) as executor:
        futures = {
            executor.submit(fetch_and_analyze_batch, chunk, API_KEYS[i % len(API_KEYS)]): chunk
            for i, chunk in enumerate(chunks)
        }

        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    all_results.extend(result)
            except Exception as exc:
                print(f"Thread Error: {exc}")

    print(f"[{get_now().strftime('%H:%M:%S')}] Analyzed {len(all_results)} candidate symbols")

    if not all_results or has_active_trade():
        return

    best = _choose_best_signal(all_results)
    if not best:
        print("No signal passed independent-indicator agreement filters.")
        return

    conf = round(float(best["confidence"]), 1)
    if conf < MIN_TRADE_CONFIDENCE:
        return

    current_start = get_now().replace(second=0, microsecond=0) + timedelta(minutes=1)

    # --- Historical Evidence layer (only queried for the chosen setup) ---
    historical = find_similar_historical_setups(
        symbol=best["symbol"],
        side=best["side"],
        features=best["setup_features"],
        start_time=current_start,
        timeframe=TIMEFRAME,
    )

    # --- ML Prediction layer ---
    ml_result = ml_predict(best["symbol"], best["side"], best["setup_features"])

    # --- Final Decision engine: combines all three layers ---
    final = make_final_decision(best["side"], conf, historical, ml_result)

    if final["decision"] == "NO_TRADE":
        print(
            f"[{get_now().strftime('%H:%M:%S')}] Final decision = NO_TRADE for "
            f"{best['symbol']} {best['side']} (combined_confidence="
            f"{final['combined_confidence']:.1f}%, vetoed={final['historical_vetoed']})"
        )
        return

    price = best["curr_p"]
    atr = best["atr_context"]["atr"]
    atr_pct = atr / price * 100.0 if (price > 0 and np.isfinite(atr)) else 0.0

    rec_time = _choose_expiry(conf, atr_pct, historical)

    start = get_now().replace(second=0, microsecond=0) + timedelta(minutes=1)
    expiry = start + timedelta(minutes=rec_time)

    trade = {
        "symbol": best["symbol"],
        "entry_price": price,
        "side": best["side"],
        "start_time": start,
        "expiry_time": expiry,
        "rec_time": rec_time,
        "timeframe": TIMEFRAME,
        "indicator_confidence": conf,
        "historical": historical,
        "ml_status": ml_result.get("status"),
        "ml_probability": ml_result.get("probability"),
        "final_decision_meta": final,
        "indicators_status": {
            name: opinion["signal"] == best["side"] for name, opinion in best["opinions"].items()
        },
        "indicator_opinions": best["opinions"],
        "setup_features": best["setup_features"],
    }

    saved_id = save_active_trade(trade)
    if not saved_id:
        return

    votes = best["directional_votes"]

    msg_lines = [
        f"🚨 {best['symbol']} → {best['side']}",
        f"Indicator Confidence: {conf:.1f}%",
        f"Independent Votes: BUY {votes['BUY']} | SELL {votes['SELL']} | HOLD {votes['HOLD']}",
        f"Entry Time: {start.strftime('%H:%M')}",
        f"Expiry: {rec_time} Min",
        "",
    ]

    for name, opinion in best["opinions"].items():
        msg_lines.append(_format_opinion_line(name, opinion))

    msg_lines.extend(["", "Market-Specific Indicator Scores:"])
    for name, weight in best["weights"].items():
        msg_lines.append(f"{name.upper()}: {weight:.2f}")

    msg_lines.extend([
        "",
        f"ATR Context: {best['atr_context']['state']} ({best['atr_context']['reason']})",
    ])

    # --- Historical Evidence section (own layer, own status) ---
    msg_lines.extend([
        "",
        f"📚 Historical Evidence: {historical['evidence_status']}",
        f"Relevant Trades: {historical['count']} (WIN {historical['wins']} / LOSS {historical['losses']})",
        f"Effective Sample Size: {historical['effective_sample_size']:.1f}",
    ])

    if historical.get("recent_weighted_win_rate") is not None:
        msg_lines.append(f"Recency-Weighted Win Rate: {historical['recent_weighted_win_rate']:.1f}%")
    else:
        msg_lines.append("Recency-Weighted Win Rate: Insufficient Data")

    # --- ML Prediction section (own layer, own status) ---
    msg_lines.extend(["", f"🤖 ML Prediction: {ml_result.get('status')}"])

    if ml_result.get("status") == "AVAILABLE":
        msg_lines.append(
            f"{best['side']}-side probability: {ml_result['probability']:.1f}% "
            f"(trained on {ml_result.get('training_samples', 0)} samples)"
        )
    else:
        msg_lines.append("No usable ML probability yet.")

    # --- Final Decision section ---
    comps = final["components"]
    comp_str = ", ".join(
        f"{k}={v:.1f}%" if v is not None else f"{k}=n/a" for k, v in comps.items()
    )
    msg_lines.extend([
        "",
        f"🎯 Final Decision: {final['decision']} (combined confidence {final['combined_confidence']:.1f}%)",
        f"Components: {comp_str}",
    ])

    msg_lines.extend([
        "",
        "⚠️ Indicator Confidence, Historical Evidence and ML Prediction are separate statistics.",
        "Historical setups are matched within the same market, timeframe and direction using setup feature similarity, weighted by recency.",
        "Place trade exactly at the start of the next minute (00s).",
    ])

    if send_telegram_msg("\n".join(msg_lines)):
        mark_message_sent(saved_id, "entry")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    init_db()

    threading.Thread(target=run_web, daemon=True).start()

    print("Trading AI Bot is running with:")
    print("- Closed-candle analysis")
    print("- Next-minute signals")
    print("- Market-specific indicator learning")
    print("- Layered Historical Evidence (fast filter -> similarity -> top-N -> recency weighting)")
    print("- Market-specific ML Prediction (logistic regression, trained on strictly-past trades)")
    print("- Final Decision engine combining Indicator / Historical / ML evidence (BUY/SELL/NO_TRADE)")

    while True:
        try:
            check_result()
            run_scanner()
            maybe_retrain_ml_models(get_now())
        except Exception as exc:
            print(f"System Error: {exc}")

        time.sleep(1)

