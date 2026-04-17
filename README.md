#  StockMind AI

## Description
A quantitative trading research system using LSTM neural networks for stock price forecasting, incorporating technical indicators, volatility modeling, and walk-forward validation to generate trading signals and risk estimates.

------------------------------------------------------------

#  功能特色

-  股票價格預測（LSTM 深度學習）
-  技術指標分析（RSI / MACD / KD）
-  未來 30 天價格預測
-  風險區間估計
-  自動交易建議（BUY / HOLD / SELL）
-  Walk-forward 模型驗證
-  歷史 + 未來價格圖表（含 hover）

------------------------------------------------------------

#  模型架構

本系統使用雙模型架構：

【Regression Model】
- 預測未來 log return

【Classification Model】
- 預測漲跌機率（sigmoid）

------------------------------------------------------------

#  特徵工程（Features）

模型輸入包含：

- log return
- RSI（Wilder version）
- MACD / Signal
- KD（K / D）
- MA20 / MA60
- volatility（波動率）
- Volume（成交量）

------------------------------------------------------------

#  安裝環境

Python 版本：
- Python 3.11

------------------------------------------------------------


# 使用方式

第一次執行先執行run.bat安裝所需套件，安裝完後預設分析APPL
要更改分析股票請參考範例
範例：

python stock.py 8155.TWO
python stock.py 2330.TW

------------------------------------------------------------

#  輸出內容

- 目前價格
- 預測價格
- 上漲機率
- 信心值
- 風險區間
- Walk-forward RMSE
- 交易建議（BUY / HOLD / SELL）

------------------------------------------------------------

#  圖表功能

- 歷史價格曲線
- 未來預測價格曲線
- 風險區間（上下界）
- 滑鼠 hover 顯示數值

------------------------------------------------------------

#  注意事項

- 本專案僅供研究用途，不構成投資建議
- 股價具有高度隨機性與非穩定性
- 模型不保證獲利
- 建議搭配風險控管與資金管理

------------------------------------------------------------

# 📌 License

MIT License
