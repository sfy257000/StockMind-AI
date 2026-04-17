import sys
import mplcursors
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# =========================
# 0. 股票代號
# =========================
symbol = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
print(f"\n📈 使用股票代號: {symbol}")

# =========================
# 1. 下載資料
# =========================
df = yf.download(symbol, start="2010-01-01")

if df.empty:
    print("❌ 無資料")
    exit()

df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
df = df[['Open','High','Low','Close','Volume']].copy()

for c in df.columns:
    df[c] = pd.Series(df[c].to_numpy().flatten(), index=df.index)

close = df['Close']

# =========================
# 2. Feature Engineering
# =========================

df['return'] = np.log(close / close.shift(1))

# KD（EMA版本更穩）
low_min = df['Low'].rolling(9).min()
high_max = df['High'].rolling(9).max()
rsv = (close - low_min) / (high_max - low_min) * 100
df['K'] = rsv.ewm(alpha=1/3).mean()
df['D'] = df['K'].ewm(alpha=1/3).mean()

# RSI（Wilder）
delta = close.diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.ewm(alpha=1/14).mean()
avg_loss = loss.ewm(alpha=1/14).mean()
rs = avg_gain / avg_loss
df['RSI'] = 100 - (100 / (1 + rs))

# MACD
ema12 = close.ewm(span=12).mean()
ema26 = close.ewm(span=26).mean()
df['MACD'] = ema12 - ema26
df['MACD_signal'] = df['MACD'].ewm(span=9).mean()

df['volatility'] = df['return'].rolling(20).std()
df['MA20'] = close.rolling(20).mean()
df['MA60'] = close.rolling(60).mean()

df = df.dropna()

if len(df) < 200:
    print("❌ 資料太少")
    exit()

features = [
    'return','K','D','RSI','MACD','MACD_signal',
    'volatility','MA20','MA60','Volume'
]

# =========================
# 3. Dataset
# =========================
WINDOW = 60
PRED_DAYS = 30

X, y_reg, y_cls = [], [], []

for i in range(WINDOW, len(df) - PRED_DAYS):
    X.append(df[features].iloc[i-WINDOW:i].values)
    fut = df['return'].iloc[i:i+PRED_DAYS].sum()

    y_reg.append(fut)
    y_cls.append(1 if fut > 0.02 else 0)

X = np.array(X)
y_reg = np.array(y_reg)
y_cls = np.array(y_cls)

# split
n = len(X)
train_end = int(n*0.7)
val_end = int(n*0.85)

X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
y_reg_train, y_reg_val, y_reg_test = y_reg[:train_end], y_reg[train_end:val_end], y_reg[val_end:]
y_cls_train, y_cls_val, y_cls_test = y_cls[:train_end], y_cls[train_end:val_end], y_cls[val_end:]

# =========================
# 4. Scaling
# =========================
scaler = MinMaxScaler()
scaler.fit(X_train.reshape(-1, len(features)))

def scale(x):
    return scaler.transform(x.reshape(-1,len(features))).reshape(x.shape)

X_train, X_val, X_test = scale(X_train), scale(X_val), scale(X_test)

# =========================
# 5. Model
# =========================
print("\n🔧 training regression...")

reg_model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(WINDOW,len(features))),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1)
])

reg_model.compile("adam","mse")

reg_model.fit(
    X_train,y_reg_train,
    validation_data=(X_val,y_reg_val),
    epochs=30,batch_size=32,verbose=0,
    callbacks=[EarlyStopping(patience=5,restore_best_weights=True)]
)

print("🔧 training classifier...")

cls_model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(WINDOW,len(features))),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1,activation='sigmoid')
])

cls_model.compile("adam","binary_crossentropy",metrics=['accuracy'])

cls_model.fit(
    X_train,y_cls_train,
    validation_data=(X_val,y_cls_val),
    epochs=30,batch_size=32,verbose=0,
    callbacks=[EarlyStopping(patience=5,restore_best_weights=True)]
)

# =========================
# 6. Prediction
# =========================
last = df[features].iloc[-WINDOW:].values
last = scale(last).reshape(1,WINDOW,len(features))

price = float(close.iloc[-1])

pred = float(reg_model.predict(last,verbose=0)[0][0])
pred = np.clip(pred,-0.2,0.2)

up_prob = float(cls_model.predict(last,verbose=0)[0][0])

pred_price = price * np.exp(pred)

# risk band
vol = df['return'].std()*np.sqrt(PRED_DAYS)
low = price*np.exp(pred-vol)
high = price*np.exp(pred+vol)

# confidence
conf = abs(up_prob-0.5)*2

signal = "BUY" if up_prob>0.55 else "SELL" if up_prob<0.45 else "HOLD"

# =========================
# 7. Walk-forward (simple)
# =========================
wf = []
for i in range(0,len(X_test),20):
    p = reg_model.predict(X_test[i:i+20],verbose=0).flatten()
    t = y_reg_test[i:i+20]
    wf.append(np.sqrt(np.mean((p-t)**2)))

wf_rmse = np.mean(wf)

# =========================
# 8. Plot
# =========================
future_days = 30
daily = pred / PRED_DAYS

future = []
tmp = price

for _ in range(future_days):
    tmp *= np.exp(daily)
    future.append(tmp)

hist = df['Close'].iloc[-200:]
idx = pd.date_range(df.index[-1], periods=future_days+1,freq='B')[1:]

plt.figure(figsize=(12,6))
plt.plot(hist.index,hist.values,label="history")
plt.plot(idx,future,'--',label="forecast")

plt.fill_between(idx,
                 np.array(future)*np.exp(-vol),
                 np.array(future)*np.exp(vol),
                 alpha=0.2)

plt.title(symbol)
plt.legend()
plt.grid()

cursor = mplcursors.cursor(hover=True)
cursor.connect("add", lambda sel: sel.annotation.set_text(f"{sel.target[1]:.2f}"))

plt.show()

# =========================
# 9. Output
# =========================
print("\n" + "="*50)
print(f"股票代號：{symbol}")
print(f"目前價格：{price:.2f}")
print(f"預測價格：{pred_price:.2f}")
print(f"上漲機率：{up_prob:.2%}")
print(f"信心值：{conf:.2f}")
print(f"風險區間：{low:.2f} ~ {high:.2f}")
print(f"Walk-forward RMSE：{wf_rmse:.4f}")

print("-"*50)

# =========================
# 🎯 交易決策邏輯
# =========================

score = 0

# 趨勢分數
if up_prob > 0.6:
    score += 2
elif up_prob > 0.55:
    score += 1
elif up_prob < 0.4:
    score -= 2
elif up_prob < 0.45:
    score -= 1

# 信心分數
if conf > 0.7:
    score += 1
elif conf < 0.3:
    score -= 1

# 風險分數
risk_ratio = (high - low) / price
if risk_ratio > 0.15:
    score -= 1

# =========================
# 📊 最終建議
# =========================

if score >= 2:
    advice = "🟢 強烈偏多（可考慮進場）"
elif score == 1:
    advice = "🟡 偏多（觀察後進場）"
elif score == 0:
    advice = "⚪ 中性（不建議操作）"
elif score == -1:
    advice = "🟠 偏空（減碼/觀望）"
else:
    advice = "🔴 強烈偏空（避免進場）"

print(f"📌 交易建議：{advice}")

print("="*50)