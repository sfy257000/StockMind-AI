#  StockMind AI

##  Description
**StockMind AI** 是一套量化交易研究系統，核心採用 **LSTM 深度學習模型** 進行股價預測。系統整合了多樣化的技術指標、波動率建模與 **Walk-forward** 驗證機制，旨在產生精準的交易訊號與風險評估。

---

##  功能特色

* **深度學習預測**：採用 LSTM 神經網絡進行股價走勢建模。
* **技術指標集成**：自動計算 RSI, MACD, KD 等核心指標。
* **長效預測**：支援未來 30 天價格走勢推估。
* **風險管理**：提供風險區間估計（Risk Interval Estimation）。
* **自動化建議**：直觀產出交易建議（BUY / HOLD / SELL）。
* **嚴謹驗證**：內建 Walk-forward 模型回測驗證。
* **交互式圖表**：動態歷史與未來價格曲線（支援 Hover 顯示）。

---

##  模型架構

本系統採用雙模型架構，從不同維度分析市場：

* **Regression Model (回歸模型)**：預測未來的對數收益率 (Log Return)。
* **Classification Model (分類模型)**：預測漲跌機率（Sigmoid 輸出）。

---

##  特徵工程 (Features)

模型輸入層整合了以下關鍵因子：
- **Log Return** (對數收益率)
- **RSI** (Wilder's version)
- **MACD / Signal**
- **KD** (K / D)
- **MA20 / MA60**
- **Volatility** (波動率)
- **Volume** (成交量)

---

##  安裝與執行

### 環境需求
- **Python 版本**：3.11

### 快速開始
1. **安裝依賴**：
   第一次執行時，請先執行 run.bat 安裝所需套件。
   > run.bat

2. **執行分析**：
   預設分析標的為 AAPL。若要更改標的，請在指令後方加上代號：
   > python stock.py 8155.TWO
   > python stock.py 2330.TW

---

##  輸出與圖表

### 數據輸出內容
- 目前價格 / 預測價格
- 上漲機率 / 信心值
- 風險區間 / Walk-forward RMSE
- 交易建議 (BUY / HOLD / SELL)

### 圖表可視化
- 歷史與未來價格曲線
- 風險區間（上下界）
- 互動式圖表（滑鼠 Hover 顯示詳細數值）

---

##  注意事項

- 本專案僅供研究用途，不構成投資建議。
- 股價具有高度隨機性與非穩定性，模型不保證獲利。
- 建議搭配風險控管與資金管理。

---

## 📌 License
MIT License
