# Project SEKAI 自動化戰績分析系統

這是一個專為音樂節奏遊戲 Project SEKAI 開發的分析系統，旨在幫助音遊玩家即時找到各種難發現的錯誤，增加遊玩體驗。

---

## 檔案說明

* 📁 **`example ROI position/`**：分析影片時所需知道框框的範圍
  * `1953fe42-f605-4a6e-82be-5f33c4db72f6.jpg`：首先框的判定文字範圍
  * `download.jpg`：第二步框的判定combo數範圍
* 📁 **`example screenshot result/`**：範例結果
* 📁 **`src/`**：程式原始碼資料夾
  * `test_gui.py`：使用者介面主程式與執行中心
  * `sekai.py`：後端獨立影像辨識、多尺度模板匹配演算法模組
  * `analyze.py`：CSV 數據讀取與 Matplotlib 歷史進步圖表
* 📁 **`templates/`**：存放影像比對所需的關鍵模板圖片
* 📄 **`.gitignore`**：忽略清單
* 📄 **`hope tear.mp4`**：提供範例影片(影片名稱:希望淚 難度:31)
* 📄 **`requirements.txt`**：環境下載套件清單

---

## 環境架設與執行說明

### 1. 安裝環境依賴套件
確保電腦已安裝 Python 環境，並在終端機中執行以下指令，一鍵安裝本專案所需的所有套件：

```bash
pip install -r requirements.txt
```

### 2. 輸入檔名、歌曲名稱和難度

### 3. 開始分析影片
可依照example ROI position建議框出範圍，分析結束後，因硬體設備和遊戲本身狀況，無法檢測到100%準確，固手動輸入真實數量以便追蹤進步狀況

### 4. 開啟紀錄資料夾
裡面有按分析次序編排的失誤截圖和excel總表

### 5. 檢視進步狀況檢視
跳出 Matplotlib 歷史進步圖表
