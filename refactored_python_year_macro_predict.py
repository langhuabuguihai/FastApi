import pandas as pd
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volume import OnBalanceVolumeIndicator
from pandas.tseries.offsets import BDay
import numpy as np
from ta.momentum import ROCIndicator, WilliamsRIndicator
from ta.volume import VolumeWeightedAveragePrice
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def add_technical_indicators(df):
    print("DEBUG SHAPES:")
    for col in ['Close', 'High', 'Low', 'Volume']:
        print(f"  {col}: shape = {df[col].shape}, dtype = {df[col].dtypes}")
        df[col] = pd.Series(df[col].values.flatten(), index=df.index)

    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']

    df['RSI'] = RSIIndicator(close=close, window=14).rsi()
    df['MACD'] = MACD(close=close).macd()
    df['EMA_10'] = EMAIndicator(close=close, window=10).ema_indicator()
    df['OBV'] = OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()
    df['ROC_10'] = ROCIndicator(close=close, window=10).roc()
    df['WilliamsR_14'] = WilliamsRIndicator(high=high, low=low, close=close, lbp=14).williams_r()
    df['VWAP'] = VolumeWeightedAveragePrice(high=high, low=low, close=close, volume=volume).vwap

    return df


def predict_tomorrow(ticker: str):
    df = yf.download(ticker, period="3y", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.empty or len(df) < 60:
        return {"error": "Not enough data for prediction."}

    df.reset_index(inplace=True)
    df["daily_return"] = df["Close"].pct_change()
    df["prev_return"] = df["daily_return"].shift(1)
    df["momentum_3d"] = df["Close"].pct_change(3)
    df["volatility_5d"] = df["daily_return"].rolling(5).std()
    df["abs_return"] = df["daily_return"].abs()
    df["direction"] = (df["prev_return"] > 0).astype(int)

    df = add_technical_indicators(df)
    df.dropna(inplace=True)

    features = [
        "prev_return", "momentum_3d", "volatility_5d", "abs_return", "direction",
        "RSI", "MACD", "EMA_10", "OBV", "ROC_10", "WilliamsR_14", "VWAP"
    ]

    X_train = df.iloc[:-1][features]
    y_train = (df.iloc[:-1]["Close"].shift(-1) > df.iloc[:-1]["Close"]).astype(int)
    X_live = df.iloc[-1:][features]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_live_scaled = scaler.transform(X_live)

    model = XGBClassifier(eval_metric='logloss', random_state=42)
    model.fit(X_train_scaled, y_train)
    prob = model.predict_proba(X_live_scaled)[0][1]

    current_year = pd.Timestamp.today().year
    macro_phase = {
        2002: "Expansion", 2003: "Recovery", 2004: "Recovery", 2005: "Peak", 2006: "Peak",
        2007: "Expansion", 2008: "Peak", 2009: "Recession", 2010: "Expansion", 2011: "Peak",
        2012: "Trough", 2013: "Trough", 2014: "Recovery", 2015: "Trough", 2016: "Trough",
        2017: "Recovery", 2018: "Expansion", 2019: "Trough", 2020: "Recession",
        2021: "Peak", 2022: "Peak", 2023: "Recession", 2024: "Recession", 2025: "Recession"
    }.get(current_year, "Unknown")

    phase_weights = {
        "Recession": 1.1, "Recovery": 1.0, "Expansion": 0.95,
        "Trough": 0.85, "Peak": 0.75, "Unknown": 1.0
    }

    adjusted_prob = prob * phase_weights.get(macro_phase, 1.0)

    if adjusted_prob > 0.85:
        recommendation = "Strong Buy"
    elif adjusted_prob > 0.7:
        recommendation = "Buy"
    elif adjusted_prob < 0.25:
        recommendation = "Strong Sell"
    elif adjusted_prob < 0.4:
        recommendation = "Sell"
    else:
        recommendation = "Hold"

    return {
        "ticker": ticker,
        "date": pd.Timestamp.today().strftime("%Y-%m-%d"),
        "macro_phase": macro_phase,
        "raw_probability": round(float(prob), 4),
        "adjusted_probability": round(float(adjusted_prob), 4),
        "recommendation": recommendation
    }

def backtest_single_ticker(ticker: str, start_year=2010, end_year=2023):
    df = yf.download(ticker, start=f"{start_year-2}-01-01", end=f"{end_year+1}-01-01", progress=False)

    # Fix MultiIndex issue
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.empty or len(df) < 100:
        return pd.DataFrame()

    df.reset_index(inplace=True)
    df.rename(columns={"Date": "date"}, inplace=True)

    # Feature engineering
    df["daily_return"] = df["Close"].pct_change()
    df["prev_return"] = df["daily_return"].shift(1)
    df["momentum_3d"] = df["Close"].pct_change(3)
    df["volatility_5d"] = df["daily_return"].rolling(5).std()
    df["abs_return"] = df["daily_return"].abs()
    df["direction"] = (df["prev_return"] > 0).astype(int)
    df["next_day_return"] = df["Close"].shift(-1) / df["Close"] - 1
    df["profitable_next_day"] = (df["next_day_return"] > 0).astype(int)

    df = add_technical_indicators(df)

    # Injected macro_df for macro-phase classification
    macro_df = pd.DataFrame({
        "year": [2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
        "Phase": ["Expansion", "Recovery", "Recovery", "Peak", "Peak", "Expansion", "Peak", "Recession", "Expansion", "Peak", "Trough", "Trough", "Recovery", "Trough", "Trough", "Recovery", "Expansion", "Trough", "Recession", "Peak", "Peak", "Recession", "Recession", "Recession"]
    })

    year_to_phase = macro_df.set_index("year")["Phase"].to_dict()

    df.dropna(inplace=True)

    phase_weights = {
        "Recession": 1.1,
        "Recovery": 1.0,
        "Expansion": 0.95,
        "Trough": 0.85,
        "Peak": 0.75,
        "Unknown": 1.0
    }

    results = []
    features = [
        "prev_return", "momentum_3d", "volatility_5d", "abs_return", "direction",
        "RSI", "MACD", "EMA_10", "OBV", "ROC_10", "WilliamsR_14", "VWAP"
    ]

    # Loop through years and calculate predictions
    for year in range(start_year, end_year + 1):
        train_data = df[df["date"].dt.year < year]
        test_day = df[df["date"].dt.year == year].head(1)

        if len(train_data) < 60 or test_day.empty:
            continue

        X_train = train_data[features]
        y_train = train_data["profitable_next_day"]
        X_test = test_day[features]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = XGBClassifier(eval_metric='logloss', random_state=42)
        model.fit(X_train_scaled, y_train)

        raw_prob = model.predict_proba(X_test_scaled)[0][1]

        raw_prob = model.predict_proba(scaler.transform(X_test))[0][1]
        macro_phase = year_to_phase.get(year, "Unknown")
        adjusted_prob = raw_prob * phase_weights.get(macro_phase, 1.0)

        if adjusted_prob > 0.85:
            rec = "Strong Buy"
        elif adjusted_prob > 0.7:
            rec = "Buy"
        elif adjusted_prob < 0.25:
            rec = "Strong Sell"
        elif adjusted_prob < 0.4:
            rec = "Sell"
        else:
            rec = "Hold"

        results.append({
            "year": year,
            "ticker": ticker,
            "macro_phase": macro_phase,
            "raw_prob": round(raw_prob, 4),
            "adjusted_prob": round(adjusted_prob, 4),
            "recommendation": rec,
            "actual_next_day_return": round(test_day["next_day_return"].values[0], 4)
        })

    # Convert results to DataFrame for easier analysis
    results_df = pd.DataFrame(results)

    # Calculate accuracy and profitability metrics
    correct_predictions = results_df[results_df["recommendation"] == results_df["actual_next_day_return"].apply(lambda x: "Buy" if x > 0 else "Sell")].shape[0]
    accuracy = correct_predictions / results_df.shape[0]

    total_profit = results_df["actual_next_day_return"].sum()
    average_profit = total_profit / results_df.shape[0]

    # Show metrics
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print(f"Average Profit: {average_profit * 100:.2f}%")

    return results_df

if __name__ == "__main__":
    result = backtest_single_ticker("7113.KL")
    print(result)

if __name__ == "__main__":
    live_result = predict_tomorrow("7113.KL")
    print("📈 Tomorrow's Recommendation:")
    print(live_result)


