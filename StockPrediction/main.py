import os, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

tf.random.set_seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────
# 1.  LOAD & EXPLORE
# ─────────────────────────────────────────────
print("=" * 60)
print("  BTC-USD LSTM Price Forecasting")
print("=" * 60)

DATA_PATH = "/tmp/data/BTC-USD_stock_data.csv"
df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
df.sort_values("Date", inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"\n[Data] Shape: {df.shape}")
print(df.head())
print(f"\nDate range: {df['Date'].min().date()}  →  {df['Date'].max().date()}")
print(f"Missing values:\n{df.isnull().sum()}")

# ─────────────────────────────────────────────
# 2.  FEATURE ENGINEERING
# ─────────────────────────────────────────────
# Use OHLCV as input features; target is next-day Close
FEATURES = ["Open", "High", "Low", "Close", "Volume"]
TARGET    = "Close"

# Add simple technical indicators
df["Return"]    = df["Close"].pct_change()
df["MA7"]       = df["Close"].rolling(7).mean()
df["MA21"]      = df["Close"].rolling(21).mean()
df["Volatility"]= df["Return"].rolling(7).std()

FEATURES_EXT = FEATURES + ["Return", "MA7", "MA21", "Volatility"]
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"\n[Features] {FEATURES_EXT}  ({len(FEATURES_EXT)} total)")
print(f"[Rows after dropna] {len(df)}")

# ─────────────────────────────────────────────
# 3.  SCALING
# ─────────────────────────────────────────────
feature_scaler = MinMaxScaler()
target_scaler  = MinMaxScaler()

feature_data = feature_scaler.fit_transform(df[FEATURES_EXT])
target_data  = target_scaler.fit_transform(df[[TARGET]])

# ─────────────────────────────────────────────
# 4.  SEQUENCE CREATION
# ─────────────────────────────────────────────
SEQ_LEN = 60   # look back 60 trading days (~3 months)

def create_sequences(features, targets, seq_len):
    X, y = [], []
    for i in range(seq_len, len(features)):
        X.append(features[i - seq_len : i])      # (seq_len, n_features)
        y.append(targets[i])                      # scalar
    return np.array(X), np.array(y)

X, y = create_sequences(feature_data, target_data, SEQ_LEN)
print(f"\n[Sequences]  X: {X.shape}   y: {y.shape}")

# ─────────────────────────────────────────────
# 5.  TRAIN / VALIDATION / TEST SPLIT  (70/15/15)
# ─────────────────────────────────────────────
n        = len(X)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)

X_train, y_train = X[:train_end],     y[:train_end]
X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]
X_test,  y_test  = X[val_end:],       y[val_end:]

print(f"\n[Split]  Train: {X_train.shape[0]}  |  Val: {X_val.shape[0]}  |  Test: {X_test.shape[0]}")

# ─────────────────────────────────────────────
# 6.  MODEL DEFINITION
# ─────────────────────────────────────────────
N_FEATURES = X_train.shape[2]

def build_model(seq_len, n_features):
    model = Sequential([
        Input(shape=(seq_len, n_features)),

        # Layer 1 – Bidirectional LSTM (return sequences for stacking)
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.3),

        # Layer 2 – LSTM (return sequences for stacking)
        LSTM(64, return_sequences=True),
        Dropout(0.2),

        # Layer 3 – LSTM (final representation)
        LSTM(32, return_sequences=False),
        Dropout(0.2),

        # Regression head
        Dense(32, activation="relu"),
        Dense(16, activation="relu"),
        Dense(1),                         # single output = next-day Close
    ], name="BTC_LSTM")
    return model

model = build_model(SEQ_LEN, N_FEATURES)
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="huber",           # robust to price outliers
    metrics=["mae"],
)
model.summary()

# ─────────────────────────────────────────────
# 7.  CALLBACKS
# ─────────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6, verbose=1),
    ModelCheckpoint("/tmp/btc_lstm_best.keras", monitor="val_loss", save_best_only=True, verbose=0),
]

# ─────────────────────────────────────────────
# 8.  TRAINING
# ─────────────────────────────────────────────
print("\n[Training] Starting …")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1,
)

# ─────────────────────────────────────────────
# 9.  EVALUATION
# ─────────────────────────────────────────────
y_pred_scaled = model.predict(X_test, verbose=0)
y_pred = target_scaler.inverse_transform(y_pred_scaled)
y_true = target_scaler.inverse_transform(y_test)

mae  = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2   = r2_score(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print("\n" + "=" * 45)
print("  TEST SET METRICS")
print("=" * 45)
print(f"  MAE  : ${mae:>12,.2f}")
print(f"  RMSE : ${rmse:>12,.2f}")
print(f"  MAPE :  {mape:>10.2f} %")
print(f"  R²   :  {r2:>10.4f}")
print("=" * 45)

# ─────────────────────────────────────────────
# 10. PLOTS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("BTC-USD LSTM Price Forecasting", fontsize=16, fontweight="bold")

# --- (a) Training history ---
ax = axes[0, 0]
ax.plot(history.history["loss"],     label="Train Loss")
ax.plot(history.history["val_loss"], label="Val Loss")
ax.set_title("Training / Validation Loss (Huber)")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
ax.legend(); ax.grid(True, alpha=0.3)

# --- (b) MAE history ---
ax = axes[0, 1]
ax.plot(history.history["mae"],     label="Train MAE")
ax.plot(history.history["val_mae"], label="Val MAE")
ax.set_title("Training / Validation MAE")
ax.set_xlabel("Epoch"); ax.set_ylabel("MAE (normalised)")
ax.legend(); ax.grid(True, alpha=0.3)

# --- (c) Predictions vs Actual ---
ax = axes[1, 0]
ax.plot(y_true,  label="Actual Close",    color="royalblue")
ax.plot(y_pred,  label="Predicted Close", color="tomato",  alpha=0.8)
ax.set_title("Actual vs Predicted (Test Set)")
ax.set_xlabel("Time Step"); ax.set_ylabel("BTC Price (USD)")
ax.legend(); ax.grid(True, alpha=0.3)

# --- (d) Residuals ---
ax = axes[1, 1]
residuals = (y_true - y_pred).flatten()
ax.hist(residuals, bins=40, color="steelblue", edgecolor="white", alpha=0.8)
ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
ax.set_title("Residuals Distribution (Test Set)")
ax.set_xlabel("Error (USD)"); ax.set_ylabel("Frequency")
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = "/mnt/user-data/outputs/btc_lstm_results.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\n[Saved] Plot → {out_path}")
plt.close()

print("\n[Done] BTC LSTM training complete.")