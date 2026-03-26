import sys
import serial
import serial.tools.list_ports
import numpy as np
from collections import deque
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QComboBox, QPushButton, QHBoxLayout
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
import pyqtgraph as pg

# ================== 參數設定 ==================
BAUDRATE = 115200       
SAMPLE_RATE = 400
WINDOW_SECONDS = 5      
BUFFER_SIZE = SAMPLE_RATE * WINDOW_SECONDS
# ===========================================

class SerialReader(QThread):
    data_received = pyqtSignal(int)
    
    def __init__(self, port):
        super().__init__()
        self.port = port
        self.running = True
        self.ser = None

    def run(self):
        try:
            # timeout 設定為 1 秒，避免 readline 卡死
            self.ser = serial.Serial(self.port, BAUDRATE, timeout=1)
            # 開啟時清空緩衝區
            self.ser.reset_input_buffer() 
            print(f"\n[系統] 已連接 {self.port}，開始接收 ECG...\n")
            
            while self.running:
                if self.ser.in_waiting:
                    try:
                        # 讀取一行並解碼
                        line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                        
                        # 檢查是否為空行 (過濾掉單純的換行符號)
                        if not line:
                            continue

                        # ================= [修改重點開始] =================
                        if line.startswith("E:"):
                            # --- 這是 ECG 訊號資料 ---
                            parts = line.split(':')
                            if len(parts) > 1:
                                try:
                                    val = int(parts[1])
                                    self.data_received.emit(val)

                                    # --- 離線心率推算 ---
                                    # Step 1: 建立 offline buffer（只在第一次初始化）
                                    if not hasattr(self, '_hr_buf'):
                                        self._hr_buf = []        # 累積原始 ECG 樣本
                                        self._hr_count = 0       # 距離上次計算的樣本數

                                    self._hr_buf.append(val)
                                    self._hr_count += 1

                                    # 每累積 SAMPLE_RATE * 2 個樣本（2秒）計算一次
                                    HR_CALC_EVERY = SAMPLE_RATE * 2   # 800 samples
                                    HR_MIN_LEN    = SAMPLE_RATE * 4   # 至少 4 秒資料

                                    if self._hr_count >= HR_CALC_EVERY and len(self._hr_buf) >= HR_MIN_LEN:
                                        self._hr_count = 0

                                        # --- Step 2: 取最近 HR_MIN_LEN 個樣本做離線分析 ---
                                        seg = np.array(self._hr_buf[-HR_MIN_LEN:], dtype=float)
                                        seg -= seg.mean()   # 去直流偏移

                                        # --- Step 3: 帶通濾波 5~15 Hz (保留 QRS 波段) ---
                                        from scipy.signal import butter, filtfilt, find_peaks
                                        nyq = SAMPLE_RATE / 2.0
                                        b, a = butter(4, [5.0/nyq, 15.0/nyq], btype='band')
                                        filtered = filtfilt(b, a, seg)

                                        # --- Step 4: 平方放大 QRS 特徵，偵測 R-peak ---
                                        squared = filtered ** 2
                                        threshold = squared.mean() * 0.4
                                        min_dist  = int(SAMPLE_RATE * 0.5)  # 最短 R-R = 0.5s
                                        peaks, _  = find_peaks(squared, height=threshold, distance=min_dist)

                                        # --- Step 5: 由 R-R 間距計算心率 ---
                                        if len(peaks) >= 2:
                                            rr = np.diff(peaks) / SAMPLE_RATE   # 單位：秒
                                            rr_valid = rr[(rr > 0.33) & (rr < 2.0)]  # 30~180 BPM
                                            if len(rr_valid) > 0:
                                                hr_bpm = 60.0 / rr_valid.mean()
                                                print(f"[離線心率] {hr_bpm:.1f} BPM  "
                                                      f"(偵測 {len(peaks)} 個 R-peak，"
                                                      f"平均 R-R = {rr_valid.mean()*1000:.0f} ms)")
                                            else:
                                                print("[離線心率] R-R 間距超出合理範圍，無法計算")
                                        else:
                                            print(f"[離線心率] 僅偵測到 {len(peaks)} 個 R-peak，資料不足")

                                        # 保留最近 HR_MIN_LEN 筆，避免 buffer 無限增長
                                        self._hr_buf = self._hr_buf[-HR_MIN_LEN:]

                                except ValueError:
                                    pass # 數值轉換失敗則忽略
                        else:
                            # --- 這是非訊號資料 (Log, Debug info, Error msg) ---
                            # 直接印在 Terminal 讓你看
                            print(f"[MCU Log]: {line}")
                        # ================= [修改重點結束] =================

                    except Exception as e:
                        print(f"[讀取錯誤]: {e}")

        except Exception as e:
            print(f"\n[串口錯誤]: {e}")
        finally:
            if self.ser and self.ser.is_open:
                self.ser.close()
                print("[系統] 串口已關閉")

    def stop(self):
        self.running = False
        self.wait()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MAX86150 ECG 即時監測")
        self.resize(1200, 600)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # UI 控制列
        top = QHBoxLayout()
        self.combo = QComboBox()
        self.combo.setMinimumWidth(200)
        self.btn = QPushButton("連接")
        self.btn.clicked.connect(self.toggle)
        self.lbl_status = QLabel("未連接")
        self.lbl_status.setStyleSheet("color: red; font-weight: bold;")
        
        top.addWidget(QLabel("串口:"))
        top.addWidget(self.combo)
        top.addWidget(self.btn)
        top.addWidget(QPushButton("重新整理", clicked=self.refresh_ports))
        top.addWidget(self.lbl_status)
        top.addStretch()
        layout.addLayout(top)

        # ECG 波形圖
        self.plot = pg.PlotWidget()
        self.plot.setBackground('#0d1b2a')
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setLabel('left', 'ECG Value')
        self.plot.setLabel('bottom', 'Samples')
        
        self.curve = self.plot.plot(pen=pg.mkPen('#00ff00', width=2))
        layout.addWidget(self.plot)

        self.buffer = deque(maxlen=BUFFER_SIZE)
        
        self.worker = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(30) # 30ms = 33 FPS
        self.refresh_ports()

    def refresh_ports(self):
        self.combo.clear()
        ports = serial.tools.list_ports.comports()
        for p in ports:
            self.combo.addItem(f"{p.device}")

    def toggle(self):
        if self.worker is None:
            if self.combo.count() == 0: return
            port = self.combo.currentText()
            self.buffer.clear() # 連接時清空圖表
            self.worker = SerialReader(port)
            self.worker.data_received.connect(self.on_data)
            self.worker.start()
            self.btn.setText("斷開")
            self.lbl_status.setText("接收中...")
            self.lbl_status.setStyleSheet("color: green;")
            self.combo.setEnabled(False)
        else:
            self.worker.stop()
            self.worker = None
            self.btn.setText("連接")
            self.lbl_status.setText("已斷開")
            self.lbl_status.setStyleSheet("color: red;")
            self.combo.setEnabled(True)

    def on_data(self, val):
        self.buffer.append(val)

    def update_plot(self):
        if len(self.buffer) < 10: return
        
        data = np.array(self.buffer)
        
        # 去除直流偏移
        display_data = data - np.mean(data)
        
        self.curve.setData(display_data)
        
        if len(display_data) > 0:
            min_y, max_y = np.min(display_data), np.max(display_data)
            range_y = max_y - min_y
            if range_y == 0: range_y = 1
            
            # 設定上下 10% 的邊界
            self.plot.setYRange(min_y - range_y * 0.1, max_y + range_y * 0.1, padding=0)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())