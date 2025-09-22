
import os
import time
from typing import List, Dict, Any, Optional
from decimal import Decimal
from datetime import datetime, timedelta, date as date_cls

import yfinance as yf
import pandas as pd

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ----------------------------
# App & CORS
# ----------------------------
app = FastAPI(title="Stock API (Railway)", version="1.0.0")

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")
allow_origins = [o.strip() for o in ALLOWED_ORIGINS.split(",")] if ALLOWED_ORIGINS != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Optional Firebase init (safe on Railway)
# ----------------------------
def maybe_init_firebase():
    try:
        if os.getenv("ENABLE_FIREBASE", "0") != "1":
            return
        from firebase_admin import credentials, initialize_app
        FIREBASE_CRED_JSON = os.getenv("FIREBASE_CRED_JSON", "").strip()
        if not FIREBASE_CRED_JSON:
            return
        # Accept either a JSON path or the raw JSON string
        cred = None
        if FIREBASE_CRED_JSON.startswith("{"):
            import tempfile
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
            tf.write(FIREBASE_CRED_JSON.encode("utf-8"))
            tf.flush()
            cred = credentials.Certificate(tf.name)
        elif os.path.exists(FIREBASE_CRED_JSON):
            cred = credentials.Certificate(FIREBASE_CRED_JSON)
        else:
            return
        initialize_app(cred)
    except Exception as e:
        print(f"[firebase] skipped: {e}")

@app.on_event("startup")
def on_startup():
    maybe_init_firebase()

@app.get("/firebase-ping")
def firebase_ping():
    try:
        from firebase_admin import firestore
        db = firestore.client()
        _ = list(db.collections())
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ----------------------------
# MySQL connection (env-driven)
# ----------------------------
import mysql.connector

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "stock_financials"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "autocommit": True,
}

def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)

# ----------------------------
# Utilities
# ----------------------------
stock_cache: Dict[str, Dict[str, Any]] = {}

def _clean_symbol(s: str) -> str:
    return s.strip().lstrip("$")

def _serialize_df_intraday(df: pd.DataFrame, tz: str) -> List[Dict[str, Any]]:
    """
    Serialize OHLCV to list of dicts with keys: t, o, h, l, c, v
    - t: ISO 8601 with offset (tz)
    - o/h/l/c: raw float (no rounding)
    - v: int (0 if NaN)
    """
    out: List[Dict[str, Any]] = []
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df = df.tz_convert(tz)

    for ts, row in df.iterrows():
        o = float(row.get("Open", float("nan")))
        h = float(row.get("High", float("nan")))
        l = float(row.get("Low", float("nan")))
        c = float(row.get("Close", float("nan")))
        v_raw = row.get("Volume", 0)
        try:
            v = 0 if v_raw != v_raw else int(v_raw)
        except Exception:
            v = 0
        out.append({"t": ts.isoformat(), "o": o, "h": h, "l": l, "c": c, "v": v})
    return out

NUM_FIELDS = ("open_price","high_price","low_price","close_price","adj_close")
def _cast_row(row: Dict[str, Any]) -> Dict[str, Any]:
    if not row:
        return row
    for k in NUM_FIELDS:
        v = row.get(k)
        if isinstance(v, Decimal):
            row[k] = float(v)
    vol = row.get("volume")
    if isinstance(vol, Decimal):
        row["volume"] = int(vol)
    dt = row.get("trade_date")
    if hasattr(dt, "isoformat"):
        row["trade_date"] = dt.isoformat()
    return row

def _fetchall_dict(sql: str, params: tuple = ()):
    conn = get_db_connection()
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute(sql, params)
        rows = cur.fetchall()
        cur.close()
        return rows
    finally:
        conn.close()

# ----------------------------
# Health
# ----------------------------
@app.get("/healthz")
def healthz():
    return {"ok": True, "version": "1.0.0"}

# ----------------------------
# Intraday endpoint (parity with LAN JSON)
# ----------------------------
@app.get("/stock/{symbol}")
async def get_stock_intraday(
    symbol: str,
    interval: str = Query("1m", description="Bar interval: 1m, 2m, 5m, 15m, 30m, 60m"),
    tz: str = Query("Asia/Kuala_Lumpur", description="IANA timezone for timestamps"),
    include_prepost: bool = Query(False, description="Include pre/post market if available"),
    ttl_seconds: int = Query(60, ge=0, le=3600, description="Cache TTL in seconds"),
):
    """
    Return latest trading day's intraday OHLCV + latest price.
    Matches LAN JSON: {symbol, interval, tz, latest_price, series_count, series: [{t,o,h,l,c,v}]}
    """
    try:
        symbol = _clean_symbol(symbol)
        cache_key = f"{symbol}|{interval}|{tz}|{include_prepost}"
        now = time.time()

        if cache_key in stock_cache and (now - stock_cache[cache_key]["timestamp"] < ttl_seconds):
            cached = stock_cache[cache_key]
            return {
                "symbol": symbol,
                "interval": cached["interval"],
                "tz": tz,
                "latest_price": cached["latest_price"],
                "series_count": len(cached["series"]),
                "series": cached["series"],
                "cached": True,
            }

        intervals_to_try = []
        if interval not in intervals_to_try:
            intervals_to_try.append(interval)
        for iv in ["1m", "2m", "5m", "15m", "30m", "60m"]:
            if iv not in intervals_to_try:
                intervals_to_try.append(iv)

        ticker = yf.Ticker(symbol)
        used_interval = None
        df_final = None

        for iv in intervals_to_try:
            df = ticker.history(period="1d", interval=iv, prepost=include_prepost, auto_adjust=False)
            if df.empty:
                df = ticker.history(period="5d", interval=iv, prepost=include_prepost, auto_adjust=False)
            if df.empty:
                continue

            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            df_local = df.tz_convert(tz)
            last_trading_date = df_local.index[-1].date()
            df_day = df_local[df_local.index.date == last_trading_date]
            if not df_day.empty:
                used_interval = iv
                df_final = df_day
                break

        if df_final is None or df_final.empty:
            return JSONResponse(
                status_code=404,
                content={"error": f"No intraday data available for {symbol} (tried {intervals_to_try})."}
            )

        series = _serialize_df_intraday(df_final, tz)
        latest_price = float(df_final["Close"].iloc[-1])

        stock_cache[cache_key] = {
            "timestamp": now,
            "interval": used_interval,
            "latest_price": latest_price,
            "series": series,
        }

        return {
            "symbol": symbol,
            "interval": used_interval,
            "tz": tz,
            "latest_price": latest_price,
            "series_count": len(series),
            "series": series,
            "cached": False,
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ----------------------------
# 10 trading days history (parity)
# ----------------------------
@app.get("/stock-history/{symbol}")
async def stock_history_last_10(symbol: str):
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period="14d")  # capture ~10 trading days
        if data.empty:
            return {"symbol": symbol, "history": []}
        trading_days = data.index[-10:]
        prices = data["Close"].iloc[-10:]
        history = [{"date": d.strftime("%d-%m"), "price": round(float(p), 3)} for d, p in zip(trading_days, prices)]
        return {"symbol": symbol, "history": history}
    except Exception as e:
        return {"error": str(e)}

# ----------------------------
# Company + fundamentals from DB
# ----------------------------
@app.get("/company/{ticker}")
async def get_company_data(ticker: str):
    try:
        t = ticker.strip()
        income = _fetchall_dict("SELECT * FROM income_statement WHERE TRIM(ticker)=%s ORDER BY date ASC", (t,))
        cash   = _fetchall_dict("SELECT * FROM cash_flow WHERE TRIM(ticker)=%s ORDER BY date ASC", (t,))
        bs     = _fetchall_dict("SELECT * FROM balance_sheet WHERE TRIM(ticker)=%s ORDER BY date ASC", (t,))
        comp   = _fetchall_dict("SELECT name FROM companies WHERE ticker=%s", (t,))
        divi   = _fetchall_dict("SELECT total_dividend_per_share_2024, shares_outstanding FROM dividend_info WHERE ticker=%s", (t,))
        score  = _fetchall_dict("""
            SELECT final_score, carg, pe_ratio, profit_margin, roe, dividend_yield, sector 
            FROM gdpprc_scores 
            WHERE ticker=%s ORDER BY scoring_date DESC LIMIT 1
        """, (t,))

        company_name = comp[0]["name"] if comp else t
        total_div_ps = (divi[0].get("total_dividend_per_share_2024") if divi else None)
        shares_out   = (divi[0].get("shares_outstanding") if divi else None)

        final_score = score[0].get("final_score") if score else None
        carg        = score[0].get("carg") if score else None
        pe_ratio    = score[0].get("pe_ratio") if score else None
        profit_mrg  = score[0].get("profit_margin") if score else None
        roe         = score[0].get("roe") if score else None
        div_yield   = score[0].get("dividend_yield") if score else None
        sector      = score[0].get("sector") if score else None

        if not income and not cash and not bs:
            return {"error": "No data found for this ticker"}

        return {
            "ticker": t,
            "company_name": company_name,
            "sector": sector,
            "total_dividend_per_share_2024": total_div_ps,
            "shares_outstanding": shares_out,
            "final_score": final_score,
            "carg": carg,
            "pe_ratio": pe_ratio,
            "profit_margin": profit_mrg,
            "roe": roe,
            "dividend_yield": div_yield,
            "income_statement": income,
            "cash_flow": cash,
            "balance_sheet": bs
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/company/name/{symbol}")
async def get_company_name(symbol: str):
    try:
        stock = yf.Ticker(symbol)
        company_name = stock.info.get("longName", symbol)
        return {"companyName": company_name}
    except Exception as e:
        return {"error": str(e)}

# ----------------------------
# DB helpers endpoints
# ----------------------------
@app.get("/db/tickers")
def db_list_tickers(q: Optional[str] = Query(None), limit: int = Query(500, ge=1, le=5000), offset: int = Query(0, ge=0)):
    params = []
    where = ""
    if q:
        where = "WHERE ticker LIKE %s"
        params.append(f"%{q}%")
    sql = f"""
        SELECT DISTINCT ticker
        FROM prices
        {where}
        ORDER BY ticker
        LIMIT %s OFFSET %s
    """
    params.extend([int(limit), int(offset)])
    rows = _fetchall_dict(sql, tuple(params))
    return [r["ticker"] for r in rows]

@app.get("/db/prices/{ticker}")
def db_get_prices(
    ticker: str,
    start: Optional[date_cls] = Query(None, description="inclusive YYYY-MM-DD"),
    end: Optional[date_cls] = Query(None, description="inclusive YYYY-MM-DD"),
    limit: int = Query(10000, ge=1, le=50000),
    offset: int = Query(0, ge=0),
    order: str = Query("asc", pattern="^(asc|desc)$"),
):
    where = ["ticker = %s"]
    params: List[Any] = [ticker]
    if start:
        where.append("trade_date >= %s"); params.append(start)
    if end:
        where.append("trade_date <= %s"); params.append(end)
    order_by = "ASC" if order == "asc" else "DESC"
    sql = f"""
        SELECT ticker, trade_date, open_price, high_price, low_price, close_price, adj_close, volume
        FROM prices
        WHERE {" AND ".join(where)}
        ORDER BY trade_date {order_by}
        LIMIT %s OFFSET %s
    """
    params.extend([int(limit), int(offset)])
    rows = _fetchall_dict(sql, tuple(params))
    return [_cast_row(r) for r in rows]

@app.get("/db/sparkline/{ticker}")
def db_sparkline(ticker: str, days: int = Query(90, ge=1, le=2000)):
    sql = """
        SELECT trade_date, close_price
        FROM prices
        WHERE ticker=%s
        ORDER BY trade_date DESC
        LIMIT %s
    """
    rows = _fetchall_dict(sql, (ticker, int(days)))
    rows.reverse()
    return {
        "ticker": ticker,
        "points": [
            {"date": r["trade_date"].isoformat() if hasattr(r["trade_date"], "isoformat") else str(r["trade_date"]),
             "close": float(r["close_price"]) if r["close_price"] is not None else None}
            for r in rows
        ],
    }

@app.get("/db/latest")
def db_latest(tickers: str = Query(..., description="comma-separated: e.g. 1155.KL,7113.KL")):
    lst = [t.strip() for t in tickers.split(",") if t.strip()]
    if not lst:
        raise HTTPException(status_code=400, detail="No tickers provided")
    placeholders = ",".join(["%s"] * len(lst))
    sql = f"""
        SELECT p.ticker, p.trade_date, p.open_price, p.high_price, p.low_price, p.close_price, p.adj_close, p.volume
        FROM prices p
        JOIN (
          SELECT ticker, MAX(trade_date) AS last_date
          FROM prices
          WHERE ticker IN ({placeholders})
          GROUP BY ticker
        ) x ON x.ticker = p.ticker AND x.last_date = p.trade_date
        ORDER BY p.ticker
    """
    rows = _fetchall_dict(sql, tuple(lst))
    return [_cast_row(r) for r in rows]

# ----------------------------
# Date-range Yahoo daily (handy for charts)
# ----------------------------
@app.get("/stock/history/{symbol}")
async def get_stock_history_range(symbol: str, start: str, end: str):
    try:
        start_date = datetime.strptime(start, "%Y-%m-%d").date()
        end_date = datetime.strptime(end, "%Y-%m-%d").date()
        if start_date >= end_date:
            return {"error": "End date must be after start date."}
        data = yf.download(symbol, start=start_date, end=end_date + timedelta(days=1), interval="1d", auto_adjust=False)
        if data.empty:
            return {"error": f"No data found for {symbol} from {start} to {end}"}
        prices = []
        for idx, row in data.iterrows():
            close_price = row.get("Close")
            if close_price is not None:
                prices.append({"date": idx.strftime("%Y-%m-%d"), "price": round(float(close_price), 3)})
        return {"symbol": symbol, "start": start, "end": end, "prices": prices}
    except ValueError:
        return {"error": "Invalid date format. Use YYYY-MM-DD."}
    except Exception as e:
        return {"error": str(e)}

# ----------------------------
# Uvicorn entrypoint
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
