import firebase_admin
from firebase_admin import credentials, firestore
import requests
import mysql.connector
from sklearn.linear_model import LogisticRegression
from fastapi import FastAPI
import yfinance as yf
import pymysql
from fastapi.middleware.cors import CORSMiddleware
import json
from ta.trend import SMAIndicator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import shap
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from joblib import load
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import MACD
from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV, GridSearchCV
from fastapi import FastAPI, Query
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pandas_datareader import wb
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import date
from pandas.tseries.offsets import BDay
from refactored_python_year_macro_predict import predict_tomorrow
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator
import joblib
import os
from fastapi import Query
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
import time
from fastapi import HTTPException
from pydantic import BaseModel
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .firebase_init import init_firebase
from firebase_admin import firestore


ALLOWED = ["*"] 

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
def healthz():
    return {"ok": True}


MODEL_DIR = r"D:\Downloads\save_api\modelsQ2"

app = FastAPI()

class ScreenerFilters(BaseModel):
    riskMin: Optional[float] = None
    riskMax: Optional[float] = None
    expReturnMin: Optional[float] = None
    expReturnMax: Optional[float] = None
    cagrMin: Optional[float] = None
    cagrMax: Optional[float] = None
    peMin: Optional[float] = None
    peMax: Optional[float] = None
    profitMarginMin: Optional[float] = None
    profitMarginMax: Optional[float] = None
    roeMin: Optional[float] = None
    roeMax: Optional[float] = None

# ✅ MySQL Database Config
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '@Jiabin123',
    'database': 'stock_financials'
}

# ✅ Initialize Firebase Firestore

@app.on_event("startup")
def _startup():
    init_firebase()

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/firebase-ping")
def firebase_ping():
    db = firestore.client()           # now safe: app is initialized
    return {"collections": db.collections() is not None}


def _yf_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        ),
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    })
    return s

def _yahoo_chart_fallback(symbol: str, days: int = 10, session: requests.Session | None = None):
    """Call Yahoo's chart API directly as a fallback to yfinance."""
    s = session or requests.Session()
    params = {
        "range": "2mo",
        "interval": "1d",
        "includeAdjustedClose": "true",
    }
    for host in ("query1", "query2"):
        url = f"https://{host}.finance.yahoo.com/v8/finance/chart/{symbol}"
        r = s.get(url, params=params, timeout=15)
        if r.status_code != 200:
            continue
        j = r.json()
        result = j.get("chart", {}).get("result")
        if not result:
            continue
        res = result[0]
        ts = res.get("timestamp", []) or []
        q = res.get("indicators", {}).get("quote", [{}])[0]
        closes = q.get("close", []) or []

        # Some series store adjusted close separately
        if not closes or all(v is None for v in closes):
            adj = res.get("indicators", {}).get("adjclose", [{}])[0].get("adjclose", []) or []
            closes = adj

        # Build last N points, skipping None/NaN
        pairs = []
        for t, c in zip(ts, closes):
            if c is None:
                continue
            try:
                if isnan(c):  # type: ignore[arg-type]
                    continue
            except Exception:
                pass
            dt = datetime.utcfromtimestamp(int(t))
            pairs.append({"date": dt.strftime("%d-%m"), "price": round(float(c), 3)})

        if pairs:
            return pairs[-days:]  # last N days

    return []  # nothing from both hosts

features = [
    'RSI','RSI_7', 'RSI_21', 'MACD', 'MACD_signal', 'MACD_histogram', 'bb_bbm', 'bb_bbh', 'bb_bbl', 'bb_bbw', 
    'ATR', 'OBV', 'Close_lag1', 'Close_lag2', 'Close_lag5', 'Close_lag10',
    'Rolling_mean_14', 'Rolling_volatility_14', 'Day_of_Week', 
    'Month', 'Quarter', 'RSI_volatility', 'Stochastic'
]

def get_macro_phases_timeline():
    indicators = {
        "NY.GDP.MKTP.CD": "GDP",
        "FP.CPI.TOTL.ZG": "Inflation",
        "SL.UEM.TOTL.ZS": "Unemployment"
    }

    # Step 1: Download macroeconomic data
    data = wb.download(indicator=list(indicators.keys()), country="MY", start=2000, end=2023)
    data.reset_index(inplace=True)

    macro = data.copy()
    macro.set_index("year", inplace=True)
    macro.drop(columns=["country"], inplace=True)
    macro.rename(columns=indicators, inplace=True)

    # Step 2: Download KLCI and compute return
    klci = yf.download("^KLSE", start="2000-01-01", end="2025-12-31", progress=False)
    klci_yearly = klci["Close"].resample("YE").last().pct_change().dropna()
    if isinstance(klci_yearly, pd.Series):
        klci_df = klci_yearly.to_frame(name="KLCI_Return")
    else:
        klci_df = klci_yearly.copy()
        klci_df.rename(columns={klci_df.columns[0]: "KLCI_Return"}, inplace=True)
    klci_df["year"] = klci_df.index.year
    klci_df.set_index("year", inplace=True)

    macro.index = macro.index.astype(int)
    klci_df.index = klci_df.index.astype(int)

    macro_df = macro.join(klci_df, how="inner")
    macro_df["GDP_Growth"] = macro_df["GDP"].pct_change() * 100
    macro_df.dropna(inplace=True)

    # Step 3: Labeling logic
    gdp_mean = macro_df["GDP_Growth"].mean()
    inf_mean = macro_df["Inflation"].mean()
    unemp_mean = macro_df["Unemployment"].mean()

    def label(row):
        if row["GDP_Growth"] > gdp_mean and row["Inflation"] < inf_mean and row["Unemployment"] < unemp_mean:
            return "Expansion"
        elif row["GDP_Growth"] < gdp_mean and row["Unemployment"] > unemp_mean:
            return "Recession"
        elif row["GDP_Growth"] > gdp_mean and row["Inflation"] > inf_mean:
            return "Peak"
        elif row["GDP_Growth"] < gdp_mean and row["Inflation"] < inf_mean:
            return "Trough"
        else:
            return "Recovery"

    macro_df["Phase"] = macro_df.apply(label, axis=1)

    # Step 4: Predict future (2024–2025)
    X = macro_df[["GDP", "Inflation", "Unemployment", "KLCI_Return"]]
    y = macro_df["Phase"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_scaled, y)

    last_row = X.iloc[-1]
    future_input_scaled = scaler.transform([last_row])
    future_phase = model.predict(future_input_scaled)[0]

    # Step 5: Generate timeline
    timeline = []
    current_phase = None
    start_year = None

    for year, row in macro_df.iterrows():
        phase = row["Phase"]
        if phase != current_phase:
            if current_phase is not None:
                timeline.append({
                    "start_year": int(start_year),
                    "end_year": int(year - 1),
                    "phase": current_phase
                })
            start_year = year
            current_phase = phase

    # Append final segment from historical data
    timeline.append({
        "start_year": int(start_year),
        "end_year": int(macro_df.index[-1]),
        "phase": current_phase
    })

    # Append predicted phase
    timeline.append({
        "start_year": 2024,
        "end_year": 2025,
        "phase": future_phase,
        "predicted": True
    })

    return timeline

def get_db_connection():
    """ Establish a database connection """
    return mysql.connector.connect(**db_config)

def fetch_records(query, params):
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(query, params)
        result = cursor.fetchall()
        cursor.close()
        conn.close()
        return result if result else []
    except Exception as e:
        return {'error': str(e)}

def prepare_live_features(ticker):

    macro_df = pd.read_csv("gdp_qtr_nominal.csv", parse_dates=["date"], dayfirst=True)
    macro_df = macro_df[['date', 'gdp', 'u_rate', 'inflation']]
    macro_df.set_index("date", inplace=True)
    macro_df = macro_df.resample("D").ffill()
    macro_df.columns.name = None
    macro_df['gdp_change'] = macro_df['gdp'].diff()
    macro_df['u_rate_change'] = macro_df['u_rate'].diff()
    macro_df['inflation_change'] = macro_df['inflation'].diff()

    df = yf.download(ticker, period="6mo", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.empty or 'Close' not in df.columns or 'Volume' not in df.columns:
        print(f"⚠️ Skipped {ticker}, insufficient data.")
        return pd.DataFrame()

    df = df[['Close', 'Volume']].copy()
    close = df['Close'].squeeze()
    volume = df['Volume'].squeeze()

    df['RSI'] = RSIIndicator(close, window=14).rsi()
    macd = MACD(close)
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_hist'] = macd.macd_diff()
    boll = BollingerBands(close)
    df['BB_high'] = boll.bollinger_hband()
    df['BB_low'] = boll.bollinger_lband()
    df['Return_5d'] = close.pct_change(5)
    df['OBV'] = OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    df['CMF'] = ChaikinMoneyFlowIndicator(high=close, low=close, close=close, volume=volume).chaikin_money_flow()
    df['DayOfWeek'] = df.index.dayofweek
    df['Month'] = df.index.month

    df = df.join(macro_df, how='left')
    df.dropna(inplace=True)
    df = df.iloc[[-1]]  # Most recent row
    return df.drop(columns=["Close"])

# ✅ Cache for stock prices (Dictionary: {ticker: {price, timestamp}})
stock_cache: dict[str, dict] = {}

def _clean_symbol(sym: str) -> str:
    # strip whitespace and a leading '$' if user passes things like "$6888.KL"
    return sym.strip().lstrip("$")

def _serialize_df(df: pd.DataFrame) -> list[dict]:
    # Expect columns: Open, High, Low, Close, Volume; index as tz-aware DatetimeIndex
    out = []
    for ts, row in df.iterrows():
        out.append({
            "t": ts.isoformat(),                    # ISO 8601 with timezone
            "o": float(row.get("Open", 0.0)),
            "h": float(row.get("High", 0.0)),
            "l": float(row.get("Low", 0.0)),
            "c": float(row.get("Close", 0.0)),
            "v": int(row.get("Volume", 0) or 0),
        })
    return out

@app.get("/stock/{symbol}")
async def get_stock_intraday(
    symbol: str,
    interval: str = Query("1m", description="Bar interval: 1m, 2m, 5m, 15m, 30m, 60m"),
    tz: str = Query("Asia/Kuala_Lumpur", description="IANA timezone to convert timestamps to"),
    include_prepost: bool = Query(False, description="Include pre/post market if available"),
    ttl_seconds: int = Query(60, ge=0, le=3600, description="Cache TTL in seconds"),
):
    """
    Return the *latest trading day's* intraday OHLCV series + latest price.
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

        # Try requested interval first, then fallbacks commonly supported by Yahoo
        tried = []
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
            tried.append(iv)
            # First attempt: last 1 day
            df = ticker.history(period="1d", interval=iv, prepost=include_prepost, auto_adjust=False)

            # Some tickers/regions don't return 1d/1m data out of hours; try 5d and slice the last trading day.
            if df.empty:
                df = ticker.history(period="5d", interval=iv, prepost=include_prepost, auto_adjust=False)

            if df.empty:
                continue

            # Ensure timezone-aware index and convert to requested tz
            if df.index.tz is None:
                # yfinance sometimes returns naive timestamps -> treat as UTC
                df.index = df.index.tz_localize("UTC")
            df = df.tz_convert(tz)

            # Keep only the most recent trading day present in df
            last_trading_date = df.index[-1].date()
            df_day = df[df.index.date == last_trading_date]

            # If we got bars for that day, we're good
            if not df_day.empty:
                used_interval = iv
                df_final = df_day
                break

        if df_final is None or df_final.empty:
            return JSONResponse(
                status_code=404,
                content={
                    "error": f"No intraday data available for {symbol} (tried intervals: {tried}). "
                             "The symbol may be invalid or delisted, or intraday data may be unavailable."
                },
            )

        latest_price = round(float(df_final["Close"].iloc[-1]), 4)
        series = _serialize_df(df_final)

        # Update cache
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



# @app.get("/stock/{symbol}")
# async def get_stock_price(symbol: str):
#     try:
#         current_time = time.time()
        
#         # ✅ Check cache (only fetch new data if 1 minute has passed)
#         if symbol in stock_cache and (current_time - stock_cache[symbol]["timestamp"] < 60):
#             return {
#                 "symbol": symbol,
#                 "price": stock_cache[symbol]["price"],
#                 "cached": True
#             }

#         # ✅ Fetch real-time stock price from Yahoo Finance
#         stock = yf.Ticker(symbol)
#         data = stock.history(period="1d", interval="1m")
        
#         if not data.empty:
#             latest_price = round(data["Close"].iloc[-1], 3)

#             # ✅ Update cache
#             stock_cache[symbol] = {"price": latest_price, "timestamp": current_time}

#             return {"symbol": symbol, "price": latest_price, "cached": False}
        
#         return {"error": "Stock not found"}
    
#     except Exception as e:
#         return {"error": str(e)}

@app.get("/stock-history/{symbol}")
async def get_stock_history(symbol: str, days: int = 10):
    try:
        sess = _yf_session()

        # 1) Primary: yfinance (with explicit session & safe options)
        df = yf.download(
            symbol,
            period="2mo",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
            session=sess,   # <- important on PaaS
        )

        if df is not None and not df.empty and "Close" in df.columns:
            tail = df["Close"].tail(days)
            history = [
                {"date": idx.strftime("%d-%m"), "price": round(float(val), 3)}
                for idx, val in tail.items()
                if val == val  # filter NaN
            ]
            if history:
                return {"symbol": symbol, "history": history, "source": "yfinance"}

        # 2) Fallback: direct Yahoo Chart API
        history = _yahoo_chart_fallback(symbol, days=days, session=sess)
        return {"symbol": symbol, "history": history, "source": "yahoo-chart"}

    except Exception as e:
        # Bubble up a clear error
        raise HTTPException(status_code=502, detail=f"price fetch failed: {e}" )

# ✅ Fetch all financial data (income, cash flow, balance sheet) for a company
@app.get("/company/{ticker}")
async def get_company_data(ticker: str):
    try:
        ticker = ticker.strip()
        income_data = fetch_records("SELECT * FROM income_statement WHERE TRIM(ticker) = %s ORDER BY date ASC", (ticker,))
        cash_flow_data = fetch_records("SELECT * FROM cash_flow WHERE TRIM(ticker) = %s ORDER BY date ASC", (ticker,))
        balance_sheet_data = fetch_records("SELECT * FROM balance_sheet WHERE TRIM(ticker) = %s ORDER BY date ASC", (ticker,))
        company_data = fetch_records("SELECT name FROM companies WHERE ticker = %s", (ticker,))
        dividend_data = fetch_records("SELECT total_dividend_per_share_2024, shares_outstanding FROM dividend_info WHERE ticker = %s", (ticker,))
        gdpprc_data = fetch_records("""
            SELECT final_score, carg, pe_ratio, profit_margin, roe, dividend_yield, sector 
            FROM gdpprc_scores 
            WHERE ticker = %s ORDER BY scoring_date DESC LIMIT 1
        """, (ticker,))

        company_name = company_data[0]['name']
        total_dividend_per_share_2024 = dividend_data[0].get('total_dividend_per_share_2024', None)
        shares_outstanding = dividend_data[0].get('shares_outstanding', None)
        final_score = gdpprc_data[0].get('final_score', None)
        carg = gdpprc_data[0].get('carg', None)
        pe_ratio = gdpprc_data[0].get('pe_ratio', None)
        profit_margin = gdpprc_data[0].get('profit_margin', None)
        roe = gdpprc_data[0].get('roe', None)
        dividend_yield = gdpprc_data[0].get('dividend_yield', None)
        sector = gdpprc_data[0].get('sector', None)

        if not income_data and not cash_flow_data and not balance_sheet_data:
            return {"error": "No data found for this ticker"}

        return {
            "ticker": ticker,
            "company_name": company_name,
            "sector": sector,
            "total_dividend_per_share_2024": total_dividend_per_share_2024,
            "shares_outstanding": shares_outstanding,
            "final_score": final_score,
            "carg": carg,
            "pe_ratio": pe_ratio,
            "profit_margin": profit_margin,
            "roe": roe,
            "dividend_yield": dividend_yield,
            "income_statement": income_data,
            "cash_flow": cash_flow_data,
            "balance_sheet": balance_sheet_data
        }

    except Exception as e:
        return {"error": str(e)}

@app.get("/company/{symbol}")
async def get_company_name(symbol: str):
    try:
        stock = yf.Ticker(symbol)
        company_name = stock.info.get("longName", symbol)  # ✅ Get company name, fallback to ticker
        return {"companyName": company_name}
    except Exception as e:
        return {"error": str(e)}
    
@app.get("/top-expected-return")
async def get_top_expected_return(
    limit: int = 50,
    horizon: int | None = None,   # optional (keep None to ignore)
):
    """
    Returns a flat list of the latest day's predictions, sorted by
    regression_ensemble_return (highest first). Includes name + sector.
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor(dictionary=True)

        # 1) Latest prediction date (optionally constrained by horizon)
        cur.execute(
            """
            SELECT MAX(prediction_date) AS latest
            FROM daily_predictions
            WHERE (%s IS NULL OR horizon_days = %s)
            """,
            (horizon, horizon)
        )
        row = cur.fetchone()
        latest = row["latest"]
        if not latest:
            raise HTTPException(status_code=404, detail="No predictions available.")

        # 2) Fetch top N by expected return for that date
        #    - name from companies
        #    - sector from gdpprc_scores (since companies.sector doesn't exist)
        where = ["dp.prediction_date = %s"]
        params = [latest]

        if horizon is not None:
            where.append("dp.horizon_days = %s")
            params.append(horizon)

        query = f"""
            SELECT
                dp.ticker,
                dp.prediction_date,
                dp.horizon_days,
                dp.regression_ensemble_return   AS expected_return,
                dp.classification_ensemble_proba AS proba,
                dp.recommendation,
                c.name,
                gs.sector,
                NULL AS pe_ratio,   -- alias in later if you add a fundamentals table
                NULL AS roe
            FROM daily_predictions dp
            LEFT JOIN companies c
                   ON c.ticker = dp.ticker
            LEFT JOIN (
                SELECT g1.ticker, g1.sector
                FROM gdpprc_scores g1
                JOIN (
                    SELECT ticker, MAX(final_score) AS mx
                    FROM gdpprc_scores
                    GROUP BY ticker
                ) gm ON gm.ticker = g1.ticker AND gm.mx = g1.final_score
            ) gs ON gs.ticker = dp.ticker
            WHERE {" AND ".join(where)}
            ORDER BY IFNULL(dp.regression_ensemble_return, -1e9) DESC
            LIMIT %s
        """
        params.append(limit)

        cur.execute(query, tuple(params))
        rows = cur.fetchall()
        cur.close(); conn.close()

        if not rows:
            raise HTTPException(
                status_code=404,
                detail="No top predictions found for latest date (check horizon or data)."
            )

        return {
            "prediction_date": str(latest),
            "top_picks": rows
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in /top-expected-return: {e}")
        return {"error": str(e)}

@app.get("/top-predictions-per-sector")
async def get_top_predictions_per_sector(
    k: int = 3,
    horizon: int | None = None,
    order: str = "return"
):
    """
    MySQL 8+ only (uses ROW_NUMBER). Returns flat list containing top k per sector.
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor(dictionary=True)

        cur.execute("SELECT MAX(prediction_date) AS latest FROM daily_predictions")
        latest = cur.fetchone()["latest"]
        if not latest:
            raise HTTPException(status_code=404, detail="No predictions available.")

        where_h = "AND dp.horizon_days = %s" if horizon is not None else ""
        params = [latest] + ([horizon] if horizon is not None else []) + [k]

        order_by = "dp.regression_ensemble_return DESC" if order == "return" \
                   else "dp.classification_ensemble_proba DESC"

        query = f"""
            WITH latest AS (
              SELECT %s AS latest_date
            ),
            ranked AS (
              SELECT
                dp.ticker,
                dp.prediction_date,
                dp.horizon_days,
                dp.classification_ensemble_proba   AS score_proba,
                dp.regression_ensemble_return      AS score_return,
                dp.recommendation,
                c.name,
                c.sector,
                c.pe_ratio,
                c.roe,
                ROW_NUMBER() OVER (
                  PARTITION BY c.sector
                  ORDER BY {order_by}
                ) AS rn
              FROM daily_predictions dp
              JOIN latest l ON dp.prediction_date = l.latest_date
              LEFT JOIN companies c ON c.ticker = dp.ticker
              WHERE 1=1 {where_h}
            )
            SELECT *
            FROM ranked
            WHERE rn <= %s
            ORDER BY sector, rn
        """

        cur.execute(query, tuple(params))
        rows = cur.fetchall()
        cur.close()
        conn.close()

        if not rows:
            raise HTTPException(status_code=404, detail="No sector picks found.")

        return {
            "prediction_date": str(latest),
            "top_picks": rows
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in /top-predictions-per-sector: {e}")
        return {"error": str(e)}

@app.get("/stock-history-1y/{ticker}")
async def get_stock_history(ticker: str):
    try:
        # Fetch data from Yahoo Finance
        stock = yf.Ticker(ticker)
        data = stock.history(period="1y")  # Get 1 year of stock data

        # Check if data is available
        if data.empty:
            return {"error": "No data found for this ticker"}

        # Prepare the data in the required format (list of dictionaries)
        history = []
        for date, row in data.iterrows():
            history.append({
                "date": date.strftime("%d-%m-%Y"),  # Format date as day-month-year
                "price": row['Close'],  # Close price for the stock
            })

        # Return the formatted data
        return {"ticker": ticker, "history": history}

    except Exception as e:
        return {"error": str(e)}

@app.get("/macro_trend")
def get_macro_trend_data():
    timeline = get_macro_phases_timeline()
    return timeline

@app.get("/stock_with_klci")
def get_stock_with_klci(ticker: str = Query(..., description="Company ticker symbol like 7113.KL")):
    try:
        # 📉 Company price
        stock = yf.Ticker(ticker)
        company = stock.history(period="10y")[["Close"]]
        company.rename(columns={"Close": "Company_Close"}, inplace=True)

        # 📊 KLCI price
        klci = yf.Ticker("^KLSE")
        klci_data = klci.history(period="10y")[["Close"]]
        klci_data.rename(columns={"Close": "KLCI_Close"}, inplace=True)

        # 🧩 Merge on date
        combined = company.merge(klci_data, left_index=True, right_index=True, how="inner")
        combined.reset_index(inplace=True)
        combined["date"] = combined["Date"].dt.strftime("%Y-%m-%d")
        combined = combined[["date", "Company_Close", "KLCI_Close"]]

        records = combined.to_dict(orient="records")
        cleaned = [{str(k): v for k, v in row.items()} for row in records]
        return cleaned

    except Exception as e:
        return {"error": str(e)}

@app.get("/macro-phase")
def get_macro_phase():
    try:
        # === Step 1: Load World Bank macro data ===
        indicators = {
            "NY.GDP.MKTP.CD": "GDP",
            "FP.CPI.TOTL.ZG": "Inflation",
            "SL.UEM.TOTL.ZS": "Unemployment"
        }
        data = wb.download(indicator=list(indicators.keys()), country="MY", start=2000, end=2023)
        data.reset_index(inplace=True)
        data.set_index("year", inplace=True)
        data.drop(columns=["country"], inplace=True)
        data.rename(columns=indicators, inplace=True)

        # === Step 2: Load KLCI yearly returns ===
        klci = yf.download("^KLSE", start="2000-01-01", end="2023-12-31", progress=False)
        klci_yearly = klci["Close"].resample("YE").last().pct_change().dropna()
        klci_yearly_df = klci_yearly.reset_index()
        klci_yearly_df["year"] = klci_yearly_df["Date"].dt.year
        klci_yearly_df.set_index("year", inplace=True)
        klci_yearly_df.drop(columns=["Date"], inplace=True)
        klci_yearly_df.rename(columns={klci_yearly_df.columns[0]: "KLCI_Return"}, inplace=True)

        # === Step 3: Combine macro + KLCI ===
        data.index = data.index.astype(int)
        klci_yearly_df.index = klci_yearly_df.index.astype(int)
        macro_df = data.join(klci_yearly_df, how="inner")
        macro_df["GDP_Growth"] = macro_df["GDP"].pct_change() * 100
        macro_df.dropna(inplace=True)

        # === Step 4: Label macro phase ===
        gdp_growth_mean = macro_df["GDP_Growth"].mean()
        inflation_mean = macro_df["Inflation"].mean()
        unemployment_mean = macro_df["Unemployment"].mean()

        def label_macro_phase(row):
            if row["GDP_Growth"] > gdp_growth_mean and row["Inflation"] < inflation_mean and row["Unemployment"] < unemployment_mean:
                return "Expansion"
            elif row["GDP_Growth"] < gdp_growth_mean and row["Unemployment"] > unemployment_mean:
                return "Recession"
            elif row["GDP_Growth"] > gdp_growth_mean and row["Inflation"] > inflation_mean:
                return "Peak"
            elif row["GDP_Growth"] < gdp_growth_mean and row["Inflation"] < inflation_mean:
                return "Trough"
            else:
                return "Recovery"

        macro_df["Phase"] = macro_df.apply(label_macro_phase, axis=1)

        # === Step 5: Return latest year macro + prediction ===
        latest_year = macro_df.index.max()
        row = macro_df.loc[latest_year]

        return {
            "year": int(latest_year),
            "GDP": round(row["GDP"], 2),
            "Inflation": round(row["Inflation"], 3),
            "Unemployment": round(row["Unemployment"], 3),
            "GDP_Growth": round(row["GDP_Growth"], 3),
            "macro_phase": row["Phase"]
        }

    except Exception as e:
        return {"error": str(e)}

@app.get("/predict")
def get_prediction(ticker: str = Query(..., description="Stock ticker, e.g. 7113.KL")):
    result = predict_tomorrow(ticker)
    return result

@app.get("/stock/history/{symbol}")
async def get_stock_history(symbol: str, start: str, end: str):
    """
    Returns daily closing prices for a stock between `start` and `end` (YYYY-MM-DD).
    Example: /stock/history/AAPL?start=2024-01-01&end=2025-06-30
    """
    try:
        start_date = datetime.strptime(start, "%Y-%m-%d").date()
        end_date = datetime.strptime(end, "%Y-%m-%d").date()

        # Make sure end > start
        if start_date >= end_date:
            return {"error": "End date must be after start date."}

        # yfinance treats `end` as exclusive, so add one day
        data = yf.download(symbol, start=start_date, end=end_date + timedelta(days=1), interval="1d")

        if data.empty:
            return {"error": f"No data found for {symbol} from {start} to {end}"}

        prices = []
        for idx, row in data.iterrows():
            close_price = row.get("Close")
            if close_price is not None:
                prices.append({
                    "date": idx.strftime("%Y-%m-%d"),
                    "price": round(close_price, 3)
                })

        return {
            "symbol": symbol,
            "start": start,
            "end": end,
            "prices": prices
        }

    except ValueError:
        return {"error": "Invalid date format. Use YYYY-MM-DD."}
    except Exception as e:
        return {"error": str(e)}


# --- DB price helpers----------------- #
from typing import List, Optional, Dict, Any
from fastapi import Query, HTTPException
from datetime import date
from decimal import Decimal

NUM_FIELDS = ("open_price","high_price","low_price","close_price","adj_close")

def _cast_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Convert MySQL Decimals to float for JSON."""
    if not row:
        return row
    for k in NUM_FIELDS:
        v = row.get(k)
        if isinstance(v, Decimal):
            row[k] = float(v)
    vol = row.get("volume")
    if isinstance(vol, Decimal):
        row["volume"] = int(vol)
    return row

def _fetchall_dict(sql: str, params: tuple = ()):
    """Use your existing mysql.connector config + dict cursor."""
    conn = get_db_connection()
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute(sql, params)
        rows = cur.fetchall()
        cur.close()
        return rows
    finally:
        conn.close()

@app.get("/db/tickers")
def db_list_tickers(q: Optional[str] = Query(None), limit: int = Query(500, ge=1, le=5000), offset: int = Query(0, ge=0)):
    """Distinct tickers in the prices table (filtered by substring if q provided)."""
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
    start: Optional[date] = Query(None, description="inclusive YYYY-MM-DD"),
    end: Optional[date] = Query(None, description="inclusive YYYY-MM-DD"),
    limit: int = Query(10000, ge=1, le=50000),
    offset: int = Query(0, ge=0),
    order: str = Query("asc", pattern="^(asc|desc)$"),
):
    """Daily OHLCV from your MySQL prices table."""
    where = ["ticker = %s"]
    params: List[Any] = [ticker]
    if start:
        where.append("trade_date >= %s")
        params.append(start)
    if end:
        where.append("trade_date <= %s")
        params.append(end)
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
    """Compact close-price series for charts."""
    sql = """
        SELECT trade_date, close_price
        FROM prices
        WHERE ticker=%s
        ORDER BY trade_date DESC
        LIMIT %s
    """
    rows = _fetchall_dict(sql, (ticker, int(days)))
    rows.reverse()  # ascending
    return {
        "ticker": ticker,
        "points": [
            {"date": r["trade_date"].isoformat(), "close": float(r["close_price"]) if r["close_price"] is not None else None}
            for r in rows
        ],
    }

@app.get("/db/latest")
def db_latest(tickers: str = Query(..., description="comma-separated: e.g. 1155.KL,7113.KL")):
    """Latest bar per ticker from DB (fast join on MAX(trade_date))."""
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

@app.post("/api/screener")
def screener(filters: ScreenerFilters):
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        sql = """
            SELECT dp.ticker,
                   dp.prediction_date,
                   dp.horizon_days,
                   dp.regression_ensemble_return AS expected_return,
                   dp.classification_ensemble_proba AS risk,
                   dp.recommendation,
                   g.carg,
                   g.pe_ratio,
                   g.profit_margin,
                   g.roe
            FROM daily_predictions dp
            JOIN gdpprc_scores g ON dp.ticker = g.ticker
        """

        where_clauses = []
        params = []

        if filters.riskMin is not None and filters.riskMax is not None:
            where_clauses.append("dp.classification_ensemble_proba BETWEEN %s AND %s")
            params.extend([filters.riskMin, filters.riskMax])

        if filters.expReturnMin is not None and filters.expReturnMax is not None:
            where_clauses.append("dp.regression_ensemble_return BETWEEN %s AND %s")
            params.extend([filters.expReturnMin, filters.expReturnMax])

        if filters.cagrMin is not None and filters.cagrMax is not None:
            where_clauses.append("g.carg BETWEEN %s AND %s")
            params.extend([filters.cagrMin, filters.cagrMax])

        if filters.peMin is not None and filters.peMax is not None:
            where_clauses.append("g.pe_ratio BETWEEN %s AND %s")
            params.extend([filters.peMin, filters.peMax])

        if filters.profitMarginMin is not None and filters.profitMarginMax is not None:
            where_clauses.append("g.profit_margin BETWEEN %s AND %s")
            params.extend([filters.profitMarginMin, filters.profitMarginMax])

        if filters.roeMin is not None and filters.roeMax is not None:
            where_clauses.append("g.roe BETWEEN %s AND %s")
            params.extend([filters.roeMin, filters.roeMax])

        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)

        sql += " ORDER BY dp.prediction_date DESC LIMIT 100"

        cursor.execute(sql, tuple(params))
        results = cursor.fetchall()
        cursor.close()
        conn.close()

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


from fastapi import FastAPI, HTTPException, Query
from datetime import datetime, timedelta, date as date_cls
from typing import Optional
import yfinance as yf
import pandas as pd

@app.get("/stock/history/date/{ticker}")
def get_price_by_date(
    ticker: str,
    date: str = Query(..., description="YYYY-MM-DD"),
    nearest: bool = True,                # if no trading that day, use previous trading day
    field: str = "close"                 # open|high|low|close|adjclose
):
    # ---- validate date ----
    try:
        q_date: date_cls = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

    # ---- normalize requested field to Yahoo column names ----
    field_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "adjclose": "Adj Close",
    }
    col = field_map.get(field.lower())
    if col is None:
        raise HTTPException(status_code=400, detail="field must be one of: open, high, low, close, adjclose")

    # ---- fetch a small window around the date (to handle weekends/holidays) ----
    start = q_date - timedelta(days=10)
    end   = q_date + timedelta(days=1)   # end is exclusive for yfinance
    try:
        hist: pd.DataFrame = yf.Ticker(ticker).history(start=start.isoformat(), end=end.isoformat(), interval="1d", auto_adjust=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"yfinance error: {e}")

    if hist.empty or col not in hist.columns:
        raise HTTPException(status_code=404, detail=f"No data available for {ticker} in the requested window.")

    # index is DatetimeIndex; make it date-only for comparison
    df = hist.copy()
    df["__date"] = pd.to_datetime(df.index).date

    # try exact match first
    exact_rows = df[df["__date"] == q_date]
    if not exact_rows.empty:
        matched_date = q_date
        price_val = exact_rows.iloc[-1][col]
        if pd.isna(price_val):  # fall back to Close if chosen column is NaN (rare)
            price_val = exact_rows.iloc[-1]["Close"]
        return {
            "ticker": ticker,
            "requested_date": q_date.isoformat(),
            "matched_date": matched_date.isoformat(),
            "mode": "exact",
            "field": col,
            "price": {ticker: float(price_val)}
        }

    # otherwise use the most recent prior trading day if allowed
    if nearest:
        prior = df[df["__date"] < q_date].sort_values("__date")
        if not prior.empty:
            matched_date = prior.iloc[-1]["__date"]
            price_val = prior.iloc[-1][col]
            if pd.isna(price_val):
                price_val = prior.iloc[-1]["Close"]
            return {
                "ticker": ticker,
                "requested_date": q_date.isoformat(),
                "matched_date": matched_date.isoformat(),
                "mode": "previous",
                "field": col,
                "price": {ticker: float(price_val)}
            }

    # nothing found
    raise HTTPException(status_code=404, detail=f"No price found for {ticker} on {q_date} (and no previous trading day).")

from fastapi import FastAPI, HTTPException, Query
from typing import Optional, List
from datetime import date


@app.get("/analytics/predictions/latest-date/{ticker}")
def latest_prediction_date(
    ticker: str,
    horizon: Optional[int] = Query(None, description="Optional horizon filter (5,21,63,126)"),
    include_rows: bool = Query(False, description="Include rows for the latest date"),
):
    """
    Returns the latest prediction_date for the given ticker.
    If `horizon` is provided, restrict the search to that horizon.
    Optionally returns the predictions for that date (all horizons or the chosen horizon).
    """
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor(dictionary=True)

        # 1) latest date
        sql_latest = """
            SELECT MAX(prediction_date) AS latest_date
            FROM daily_predictions
            WHERE ticker = %s {horizon_clause}
        """.format(horizon_clause="AND horizon_days = %s" if horizon is not None else "")
        params = [ticker] + ([horizon] if horizon is not None else [])
        cur.execute(sql_latest, params)
        row = cur.fetchone()
        latest = row["latest_date"]

        if latest is None:
            raise HTTPException(
                status_code=404,
                detail=f"No predictions found for ticker={ticker}"
                       + (f" and horizon={horizon}" if horizon is not None else "")
            )

        # 2) horizons available on that latest date
        sql_horizons = """
            SELECT DISTINCT horizon_days
            FROM daily_predictions
            WHERE ticker = %s AND prediction_date = %s
            ORDER BY horizon_days
        """
        cur.execute(sql_horizons, (ticker, latest))
        horizons_available = [r["horizon_days"] for r in cur.fetchall()]

        result = {
            "ticker": ticker,
            "latest_prediction_date": latest.isoformat() if hasattr(latest, "isoformat") else str(latest),
            "horizons_available": horizons_available,
        }

        # 3) optionally include the rows for that latest date
        if include_rows:
            if horizon is None:
                sql_rows = """
                    SELECT ticker,
                           prediction_date,
                           horizon_days,
                           regression_ensemble_return   AS expected_return,
                           classification_ensemble_proba AS confidence,
                           recommendation
                    FROM daily_predictions
                    WHERE ticker = %s AND prediction_date = %s
                    ORDER BY horizon_days
                """
                cur.execute(sql_rows, (ticker, latest))
            else:
                sql_rows = """
                    SELECT ticker,
                           prediction_date,
                           horizon_days,
                           regression_ensemble_return   AS expected_return,
                           classification_ensemble_proba AS confidence,
                           recommendation
                    FROM daily_predictions
                    WHERE ticker = %s AND prediction_date = %s AND horizon_days = %s
                    ORDER BY horizon_days
                """
                cur.execute(sql_rows, (ticker, latest, horizon))
            rows = cur.fetchall()
            # normalize date to string
            for r in rows:
                d = r.get("prediction_date")
                if d is not None and hasattr(d, "isoformat"):
                    r["prediction_date"] = d.isoformat()
            result["rows"] = rows

        cur.close()
        conn.close()
        return result

    except HTTPException:
        if conn: conn.close()
        raise
    except Exception as e:
        if conn: conn.close()
        raise HTTPException(status_code=500, detail=str(e))


from typing import Optional, List, Any
from fastapi import Query, HTTPException
from datetime import date
from decimal import Decimal

def _cast_signal_row(r: dict) -> dict:
    # convert Decimals & dates so FastAPI can JSON-encode cleanly
    for k in ("price_start", "price_end", "actual_return", "prevented_loss"):
        if k in r and isinstance(r[k], Decimal):
            r[k] = float(r[k])
    if "prediction_date" in r and hasattr(r["prediction_date"], "isoformat"):
        r["prediction_date"] = r["prediction_date"].isoformat()
    if "created_at" in r and hasattr(r["created_at"], "isoformat"):
        r["created_at"] = r["created_at"].isoformat()
    return r

_ALLOWED_ORDER_BY = {
    "rank_idx": "rank_idx",
    "actual_return": "actual_return",
    "prevented_loss": "prevented_loss",
    "prediction_date": "prediction_date",
    "horizon_days": "horizon_days",
    "price_start": "price_start",
    "price_end": "price_end",
}

@app.get("/v1/top_model_signals")
def get_top_model_signals(
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
    segment: Optional[str] = Query(None, pattern=r"^(BUY|SELL)$"),
    horizon_days: Optional[int] = Query(None, ge=1, le=3650),
    prediction_date: Optional[date] = None,
    ticker: Optional[str] = None,
    order_by: str = Query("rank_idx"),
    sort: str = Query("asc", pattern=r"^(?i)(asc|desc)$"),
):
    """
    GET /v1/top_model_signals?limit=30&segment=BUY&horizon_days=30&order_by=rank_idx&sort=asc
    Returns a bare array of rows from top_model_signals.
    """
    order_col = _ALLOWED_ORDER_BY.get(order_by)
    if not order_col:
        raise HTTPException(status_code=400, detail=f"Invalid order_by. Allowed: {list(_ALLOWED_ORDER_BY)}")
    sort_sql = "ASC" if sort.lower() == "asc" else "DESC"

    where: List[str] = []
    params: List[Any] = []
    if segment:
        where.append("segment = %s");           params.append(segment)
    if horizon_days is not None:
        where.append("horizon_days = %s");      params.append(horizon_days)
    if prediction_date is not None:
        where.append("prediction_date = %s");   params.append(prediction_date)
    if ticker:
        where.append("ticker = %s");            params.append(ticker)

    sql = """
        SELECT
          ticker, segment, rank_idx, prediction_id, prediction_date,
          horizon_days, price_start, price_end, actual_return, prevented_loss, created_at
        FROM top_model_signals
    """
    if where:
        sql += " WHERE " + " AND ".join(where)
    # deterministic tie-breakers
    sql += f" ORDER BY {order_col} {sort_sql}, ticker ASC, segment ASC, rank_idx ASC"
    sql += " LIMIT %s OFFSET %s"
    params.extend([limit, offset])

    rows = _fetchall_dict(sql, tuple(params))
    return [_cast_signal_row(r) for r in rows]

@app.get("/v1/top_model_signals/{ticker}")
def get_signals_by_ticker(
    ticker: str,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """
    GET /v1/top_model_signals/7113.KL?limit=50
    """
    sql = """
        SELECT
          ticker, segment, rank_idx, prediction_id, prediction_date,
          horizon_days, price_start, price_end, actual_return, prevented_loss, created_at
        FROM top_model_signals
        WHERE ticker = %s
        ORDER BY prediction_date DESC, rank_idx ASC
        LIMIT %s OFFSET %s
    """
    rows = _fetchall_dict(sql, (ticker, limit, offset))
    return [_cast_signal_row(r) for r in rows]





# ✅ Run the FastAPI Server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

