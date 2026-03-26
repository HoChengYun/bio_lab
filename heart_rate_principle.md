# 離線 ECG 心率推算原理說明

## 1. 什麼是 ECG 訊號？

心電圖（Electrocardiogram，ECG）記錄心臟每次跳動時產生的電位變化。  
一個完整的心跳週期會產生特徵性的波形，其中最明顯的是高聳的 **R 波**，構成所謂的 **QRS 波群**。

```
振幅
  |         R
  |        /\
  |       /  \
  |   Q  /    \  S
  |   \_/      \_____  T
  |  P                \___/
  |_____________________________ 時間
       ←──── 一個心跳週期 ────→
```

心率（Heart Rate）就是每分鐘心跳的次數，單位為 **BPM（Beats Per Minute）**。  
只要找出相鄰兩個 R 波之間的時間間距（**R-R interval**），就可以計算心率：

$$
\text{Heart Rate (BPM)} = \frac{60}{\text{平均 R-R 間距（秒）}}
$$

---

## 2. 整體處理流程

```
原始 ECG 訊號（ADC 數值）
         │
         ▼
 ① 去除直流偏移（DC Removal）
         │
         ▼
 ② 帶通濾波（Bandpass Filter 5~15 Hz）
         │
         ▼
 ③ 訊號平方（Signal Squaring）
         │
         ▼
 ④ R-peak 偵測（Peak Detection）
         │
         ▼
 ⑤ R-R 間距計算 → 心率（BPM）
```

---

## 3. 各步驟詳細說明

### Step 1 — 去除直流偏移（DC Removal）

原始 ECG 訊號常帶有直流偏移（DC offset），即整體波形上移或下移，  
這會干擾後續濾波器的效果。

做法是將整段訊號減去其平均值：

```python
seg -= seg.mean()
```

**效果：** 訊號以 0 為中心上下擺動，方便後續處理。

---

### Step 2 — 帶通濾波（Bandpass Filter）

ECG 訊號中混有各種雜訊：

| 雜訊來源 | 頻率範圍 |
|----------|----------|
| 基線漂移（呼吸、電極移動） | < 1 Hz |
| 工頻干擾（電源雜訊） | 50 / 60 Hz |
| 肌電雜訊（EMG） | > 100 Hz |
| **QRS 波群（心跳主要能量）** | **5 ~ 15 Hz** ← 保留這段 |

使用 **4 階 Butterworth 帶通濾波器**，只保留 5～15 Hz 的頻率成分：

```python
from scipy.signal import butter, filtfilt

nyq = SAMPLE_RATE / 2.0          # Nyquist 頻率
b, a = butter(4, [5.0/nyq, 15.0/nyq], btype='band')
filtered = filtfilt(b, a, seg)   # 零相位雙向濾波，不造成時間延遲
```

> **為什麼用 `filtfilt` 而不是 `lfilter`？**  
> `filtfilt` 對訊號做兩次濾波（正向＋反向），相位延遲為零，  
> R-peak 的位置不會被偏移，偵測結果更準確。

---

### Step 3 — 訊號平方（Signal Squaring）

帶通濾波後，訊號中仍可能有殘留雜訊導致振幅接近 R 波。  
對訊號取平方有兩個效果：

1. **放大 R 波**：R 波振幅最大，平方後差距更懸殊
2. **全部變正值**：消除正負號的影響，讓峰值更容易偵測

```python
squared = filtered ** 2
```

---

### Step 4 — R-peak 偵測（Pan-Tompkins 精簡版）

使用 `scipy.signal.find_peaks` 搭配兩個條件偵測 R 波：

```python
threshold = squared.mean() * 0.4   # 自適應門檻：平均值的 40%
min_dist  = int(SAMPLE_RATE * 0.5) # 最短 R-R 間距：0.5 秒（對應最高 120 BPM）

peaks, _ = find_peaks(squared, height=threshold, distance=min_dist)
```

| 參數 | 說明 |
|------|------|
| `height=threshold` | 峰值必須超過門檻才算，排除低振幅雜訊 |
| `distance=min_dist` | 兩個峰值之間至少間隔 0.5 秒，避免同一個 QRS 波被重複計算 |

> **為何門檻取 0.4 倍均值？**  
> 因為平方後 R 波遠高於其他波形，均值主要由背景雜訊決定，  
> 0.4 倍可在雜訊較低時保持靈敏，雜訊較高時自動提升門檻。

---

### Step 5 — 計算 R-R 間距與心率

偵測到所有 R-peak 的樣本索引後，計算相鄰峰值之間的時間差：

```python
rr = np.diff(peaks) / SAMPLE_RATE        # 單位：秒

# 過濾生理上不合理的數值（30 ~ 180 BPM）
rr_valid = rr[(rr > 0.33) & (rr < 2.0)]

hr_bpm = 60.0 / rr_valid.mean()          # 平均 R-R → BPM
```

過濾條件說明：

| 條件 | 對應心率 | 用途 |
|------|----------|------|
| R-R > 0.33 秒 | < 180 BPM | 排除假峰值（雜訊造成的超高心率） |
| R-R < 2.0 秒 | > 30 BPM | 排除長時間無訊號的異常間距 |

---

## 4. 為什麼叫「離線（Offline）」推算？

本程式的計算方式是：

- **累積 4 秒**的 ECG 資料後，**一次性**對整段資料進行濾波與峰值偵測
- 每 **2 秒**觸發一次計算，使用最新的 4 秒資料（滑動窗口）

這與「即時（Online/Real-time）」處理不同：

| 比較 | 離線（Offline） | 即時（Online） |
|------|----------------|----------------|
| 資料取得 | 先累積一段再算 | 逐樣本即時處理 |
| 濾波方式 | `filtfilt`（零相位） | `lfilter`（有延遲） |
| 計算延遲 | 2 秒更新一次 | 幾乎無延遲 |
| 實作難度 | 簡單 | 需要滑動窗口演算法 |
| 本實驗使用 | ✅ 是 | ❌ 否 |

---

## 5. 參數設定總結

| 參數 | 數值 | 說明 |
|------|------|------|
| 取樣率 | 400 Hz | STM32 韌體設定 |
| 帶通濾波範圍 | 5 ~ 15 Hz | 保留 QRS 波段 |
| 濾波器階數 | 4 階 Butterworth | 夠陡的滾降曲線 |
| 最小 R-R 間距 | 0.5 秒 | 對應最高 120 BPM |
| 自適應門檻 | 均值 × 0.4 | 依訊號強度自動調整 |
| 分析窗口長度 | 4 秒（1600 samples） | 確保包含多個 R-peak |
| 計算觸發間隔 | 每 2 秒 | 800 個新樣本後重算 |
| 有效心率範圍 | 30 ~ 180 BPM | 過濾異常 R-R 間距 |

---

## 6. 參考方法

本實作參考 **Pan-Tompkins 演算法（1985）** 的核心概念：
- J. Pan and W. J. Tompkins, "A real-time QRS detection algorithm," *IEEE Transactions on Biomedical Engineering*, vol. BME-32, no. 3, pp. 230–236, 1985.

原始演算法還包含移動積分窗口（Moving Window Integration）與雙門檻機制，  
本實作為精簡版，適用於乾淨度較高的 ECG 訊號。
