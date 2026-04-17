@echo off
chcp 65001 >nul

echo =====================================
echo  Stock AI 一鍵安裝 + 執行
echo =====================================

:: =========================
:: 1. 檢查 Python
:: =========================
py -3.11 --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo ❌ 找不到 Python 3.11，請先安裝
    pause
    exit /b
)

echo ✔ Python 3.11 OK

:: =========================
:: 2. 建立 venv
:: =========================
IF NOT EXIST venv (
    echo 建立虛擬環境...
    py -3.11 -m venv venv
)

call venv\Scripts\activate

:: =========================
:: 3. 更新 pip
:: =========================
python -m pip install --upgrade pip

:: =========================
:: 4. 安裝套件
:: =========================
echo 安裝必要套件...

pip install yfinance pandas numpy matplotlib scikit-learn mplcursors tensorflow

IF %ERRORLEVEL% NEQ 0 (
    echo ❌ 套件安裝失敗
    pause
    exit /b
)

echo ✔ 套件安裝完成

:: =========================
:: 5. 執行程式
:: =========================
echo =====================================
echo 🚀 執行 stock.py
echo =====================================

python stock.py

pause