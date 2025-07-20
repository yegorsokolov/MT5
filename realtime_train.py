import time
from pathlib import Path
import pandas as pd
import joblib
import MetaTrader5 as mt5
from git import Repo

from utils import load_config
from dataset import make_features


def fetch_ticks(symbol: str, n: int = 1000) -> pd.DataFrame:
    """Fetch recent tick data from MetaTrader5."""
    ticks = mt5.copy_ticks_from(symbol, int(time.time()) - n, n, mt5.COPY_TICKS_ALL)
    if ticks is None or len(ticks) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(ticks)
    df["Timestamp"] = pd.to_datetime(df["time"], unit="s")
    df.rename(columns={"bid": "Bid", "ask": "Ask", "volume": "Volume"}, inplace=True)
    df = df[["Timestamp", "Bid", "Ask", "Volume"]]
    df["BidVolume"] = df["Volume"]
    df["AskVolume"] = df["Volume"]
    df.drop(columns=["Volume"], inplace=True)
    return df


def train_realtime():
    cfg = load_config()
    repo_path = Path(__file__).resolve().parent
    data_path = repo_path / "data" / "history.csv"
    model_path = repo_path / "model.joblib"

    repo = Repo(repo_path)

    if not mt5.initialize():
        raise RuntimeError("Failed to initialize MT5")

    symbols = cfg.get("symbols") or [cfg.get("symbol", "EURUSD")]

    while True:
        tick_frames = []
        for sym in symbols:
            ticks = fetch_ticks(sym, 500)
            if not ticks.empty:
                ticks["Symbol"] = sym
                tick_frames.append(ticks)

        if not tick_frames:
            time.sleep(60)
            continue

        new_ticks = pd.concat(tick_frames, ignore_index=True)

        if data_path.exists():
            history = pd.read_csv(data_path)
            history = pd.concat([history, new_ticks]).drop_duplicates(subset=["Timestamp", "Symbol"])
        else:
            history = new_ticks
        history.to_csv(data_path, index=False)

        # compute features on the full history across all symbols so
        # cross symbol relationships like correlation or momentum can
        # be derived when available
        df = make_features(history)
        if "Symbol" in df.columns:
            df["SymbolCode"] = df["Symbol"].astype("category").cat.codes

        # base set of features used for model training
        features = [
            "return",
            "ma_5",
            "ma_10",
            "ma_30",
            "ma_60",
            "volatility_30",
            "spread",
            "rsi_14",
        ]
        # include optional features when present
        for col in ["cross_corr", "cross_momentum", "news_sentiment"]:
            if col in df.columns:
                features.append(col)
        if "volume_ratio" in df.columns:
            features.extend(["volume_ratio", "volume_imbalance"])
        if "SymbolCode" in df.columns:
            features.append("SymbolCode")
        X = df[features]
        y = (df["return"].shift(-1) > 0).astype(int)

        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from lightgbm import LGBMClassifier

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LGBMClassifier(n_estimators=200, random_state=42)),
        ])

        pipe.fit(X, y)
        joblib.dump(pipe, model_path)

        # commit updates
        repo.git.add(data_path.as_posix())
        repo.git.add(model_path.as_posix())
        repo.index.commit("Update model with realtime data")
        try:
            repo.remote().push()
        except Exception as e:
            print("Git push failed:", e)

        time.sleep(300)


if __name__ == "__main__":
    train_realtime()
