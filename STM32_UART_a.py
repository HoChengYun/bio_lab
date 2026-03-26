import sys
import serial
import serial.tools.list_ports
import numpy as np
from collections import deque
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QComboBox, QPushButton, QHBoxLayout, QFileDialog, QMessageBox
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
import pyqtgraph as pg
from scipy.signal import butter, filtfilt, find_peaks
import csv
import os
from datetime import datetime

# ================== 參數設定 ==================
BAUDRATE = 115200       
SAMPLE_RATE = 400
WINDOW_SECONDS = 5      
BUFFER_SIZE = SAMPLE_RATE * WINDOW_SECONDS
# ===========================================

class SerialReader(QThread):
    data_received = pyqtSignal(int)
    hr_calculated = pyqtSignal(float)  # 新增：心率信號
    
    def __init__(self, port):
        super().__init__()
        self.port = port
        self.running = True
        self.ser = None
        
        # ECG訊號處理相關
        self.ecg_buffer = deque(maxlen=SAMPLE_RATE * 15)  # 15秒緩衝
        self.hr_update_count = 0
        self._init_filter()
        
        # 數據存檔相關
        self.recording_buffer = []  # 完整數據記錄
    
    def _init_filter(self):
        """初始化帶通濾波器 (0.5-50 Hz)"""
        nyquist = SAMPLE_RATE / 2
        low, high = 0.5 / nyquist, 50 / nyquist
        self.b, self.a = butter(4, [low, high], btype='band')
    
    def _process_ecg(self, value):
        """訊號處理與心率計算"""
        self.ecg_buffer.append(value)
        self.hr_update_count += 1
        
        # 每3000點計算一次心率 (7.5秒)
        if self.hr_update_count >= 3000 and len(self.ecg_buffer) > 2000:
            self.hr_update_count = 0
            
            # 濾波
            signal = np.array(list(self.ecg_buffer))
            try:
                filtered = filtfilt(self.b, self.a, signal)
                # 標準化
                norm_signal = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-6)
                # 峰值檢測 (R波)
                peaks, _ = find_peaks(norm_signal, height=0.5, distance=SAMPLE_RATE//3)
                
                # 計算心率
                if len(peaks) >= 2:
                    intervals = np.diff(peaks) / SAMPLE_RATE
                    hr = 60.0 / np.mean(intervals)
                    if 30 < hr < 200:  # 合理範圍
                        self.hr_calculated.emit(float(hr))
            except:
                pass

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

                        if line.startswith("E:"):
                            # --- 這是 ECG 訊號資料 ---
                            parts = line.split(':')
                            if len(parts) > 1:
                                try:
                                    val = int(parts[1])
                                    self.data_received.emit(val)
                                    # ===== 離線心率計算 =====
                                    self._process_ecg(val)
                                    # ===== 保存數據 =====
                                    self.recording_buffer.append({
                                        'timestamp': len(self.recording_buffer) / SAMPLE_RATE,
                                        'value': val
                                    })
                                except ValueError:
                                    pass # 數值轉換失敗則忽略
                        else:
                            # --- 這是非訊號資料 (Log, Debug info, Error msg) ---
                            # 直接印在 Terminal 讓你看
                            print(f"[MCU Log]: {line}")

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
        
        # 心率顯示
        self.lbl_hr = QLabel("HR: -- bpm")
        self.lbl_hr.setStyleSheet("color: #FF6B6B; font-weight: bold; font-size: 12px;")
        
        # 導出按鈕
        self.btn_export = QPushButton("導出數據")
        self.btn_export.clicked.connect(self.export_data)
        self.btn_export.setEnabled(False)
        
        top.addWidget(QLabel("串口:"))
        top.addWidget(self.combo)
        top.addWidget(self.btn)
        top.addWidget(QPushButton("重新整理", clicked=self.refresh_ports))
        top.addWidget(self.lbl_hr)
        top.addWidget(self.lbl_status)
        top.addWidget(self.btn_export)
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
        self.filtered_buffer = deque(maxlen=BUFFER_SIZE)  # 存儲濾波數據
        
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
            self.worker.hr_calculated.connect(self.on_hr_update)  # 心率信號
            self.worker.start()
            self.btn.setText("斷開")
            self.lbl_status.setText("接收中...")
            self.lbl_status.setStyleSheet("color: green;")
            self.combo.setEnabled(False)
            self.btn_export.setEnabled(True)
        else:
            self.worker.stop()
            self.worker = None
            self.btn.setText("連接")
            self.lbl_status.setText("已斷開")
            self.lbl_status.setStyleSheet("color: red;")
            self.combo.setEnabled(True)
            self.btn_export.setEnabled(True)  # 保持啟用以導出已收集的數據

    def on_data(self, val):
        self.buffer.append(val)
    
    def on_hr_update(self, hr):
        """更新心率顯示"""
        self.lbl_hr.setText(f"HR: {hr:.1f} bpm")
    
    def export_data(self):
        """導出數據為 CSV 和 NPY 格式（包含 Raw Data）"""
        if self.worker is None or len(self.worker.recording_buffer) == 0:
            QMessageBox.warning(self, "警告", "沒有可導出的數據")
            return
        
        # 選擇保存位置
        save_dir = QFileDialog.getExistingDirectory(self, "選擇保存位置")
        if not save_dir:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # 提取 raw data
            raw_data = np.array([r['value'] for r in self.worker.recording_buffer])
            time_axis = np.array([r['timestamp'] for r in self.worker.recording_buffer])
            
            # ===== 計算濾波數據 =====
            filtered_data = filtfilt(self.worker.b, self.worker.a, raw_data)
            
            # ===== 檢測峰值 =====
            norm_signal = (filtered_data - np.mean(filtered_data)) / (np.std(filtered_data) + 1e-6)
            peaks, _ = find_peaks(norm_signal, height=0.5, distance=SAMPLE_RATE//3)
            
            # 1. 導出完整 CSV (Raw + Filtered + Peak Flag)
            csv_path = os.path.join(save_dir, f"ecg_complete_{timestamp}.csv")
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Index', 'Time (s)', 'Raw ECG', 'Filtered ECG', 'Is Peak'])
                for i in range(len(raw_data)):
                    is_peak = 1 if i in peaks else 0
                    writer.writerow([i, f"{time_axis[i]:.4f}", raw_data[i], 
                                    f"{filtered_data[i]:.4f}", is_peak])
            print(f"✓ 完整 CSV 已保存: {csv_path}")
            
            # 2. 導出 Raw Data NPY (原始未處理數據)
            raw_npy_path = os.path.join(save_dir, f"ecg_raw_{timestamp}.npy")
            np.save(raw_npy_path, raw_data)
            print(f"✓ Raw Data NPY 已保存: {raw_npy_path}")
            
            # 3. 導出 Filtered Data NPY (濾波後數據)
            filtered_npy_path = os.path.join(save_dir, f"ecg_filtered_{timestamp}.npy")
            np.save(filtered_npy_path, filtered_data)
            print(f"✓ 濾波數據 NPY 已保存: {filtered_npy_path}")
            
            # 4. 導出峰值位置
            peaks_path = os.path.join(save_dir, f"ecg_peaks_{timestamp}.csv")
            with open(peaks_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Peak Index', 'Time (s)', 'Raw Value', 'Filtered Value', 'R-R Interval (s)'])
                for i, peak_idx in enumerate(peaks):
                    r_r_interval = 0 if i == 0 else (peaks[i] - peaks[i-1]) / SAMPLE_RATE
                    writer.writerow([peak_idx, f"{time_axis[peak_idx]:.4f}", 
                                    raw_data[peak_idx], f"{filtered_data[peak_idx]:.4f}",
                                    f"{r_r_interval:.4f}"])
            print(f"✓ 峰值信息已保存: {peaks_path}")
            
            # 5. 導出詳細報告 TXT
            info_path = os.path.join(save_dir, f"ecg_report_{timestamp}.txt")
            with open(info_path, 'w', encoding='utf-8') as f:
                f.write(f"ECG 數據導出詳細報告\n")
                f.write(f"="*60 + "\n\n")
                f.write(f"【基本信息】\n")
                f.write(f"-"*60 + "\n")
                f.write(f"導出時間:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"採樣率:          {SAMPLE_RATE} Hz\n")
                f.write(f"採樣點數:        {len(raw_data)}\n")
                f.write(f"採集時間:        {len(raw_data) / SAMPLE_RATE:.2f} 秒\n\n")
                
                f.write(f"【Raw Data 統計】\n")
                f.write(f"-"*60 + "\n")
                f.write(f"最大值:          {np.max(raw_data):.2f}\n")
                f.write(f"最小值:          {np.min(raw_data):.2f}\n")
                f.write(f"平均值:          {np.mean(raw_data):.2f}\n")
                f.write(f"標準差:          {np.std(raw_data):.2f}\n\n")
                
                f.write(f"【濾波後數據統計】\n")
                f.write(f"-"*60 + "\n")
                f.write(f"最大值:          {np.max(filtered_data):.4f}\n")
                f.write(f"最小值:          {np.min(filtered_data):.4f}\n")
                f.write(f"平均值:          {np.mean(filtered_data):.4f}\n")
                f.write(f"標準差:          {np.std(filtered_data):.4f}\n\n")
                
                f.write(f"【R 波檢測結果】\n")
                f.write(f"-"*60 + "\n")
                f.write(f"檢測到的 R 波:   {len(peaks)} 個\n")
                if len(peaks) >= 2:
                    intervals = np.diff(peaks) / SAMPLE_RATE
                    f.write(f"平均 R-R 間隔:   {np.mean(intervals):.4f} 秒\n")
                    f.write(f"標準差:          {np.std(intervals):.4f} 秒\n")
                    hr = 60.0 / np.mean(intervals)
                    f.write(f"計算心率:        {hr:.1f} bpm\n\n")
                
                f.write(f"【生成的文件】\n")
                f.write(f"-"*60 + "\n")
                f.write(f"✓ {os.path.basename(csv_path)}          (完整數據 CSV)\n")
                f.write(f"✓ {os.path.basename(raw_npy_path)}     (Raw Data)\n")
                f.write(f"✓ {os.path.basename(filtered_npy_path)} (濾波數據)\n")
                f.write(f"✓ {os.path.basename(peaks_path)}       (峰值信息)\n")
            print(f"✓ 報告已保存: {info_path}")
            
            QMessageBox.information(
                self, 
                "成功", 
                f"所有數據已成功導出！\n\n"
                f"保存位置: {save_dir}\n\n"
                f"【生成的文件】\n"
                f"✓ ecg_complete_{timestamp}.csv\n"
                f"  └─ 完整數據 (Raw + Filtered + Peak)\n\n"
                f"✓ ecg_raw_{timestamp}.npy\n"
                f"  └─ 原始 Raw Data (NumPy)\n\n"
                f"✓ ecg_filtered_{timestamp}.npy\n"
                f"  └─ 濾波數據 (NumPy)\n\n"
                f"✓ ecg_peaks_{timestamp}.csv\n"
                f"  └─ 所有峰值詳細信息\n\n"
                f"✓ ecg_report_{timestamp}.txt\n"
                f"  └─ 詳細統計報告"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"導出失敗: {e}")
            print(f"✗ 導出失敗: {e}")

    def update_plot(self):
        if len(self.buffer) < 10: return
        
        data = np.array(self.buffer)
        
        # 去除直流偏移
        display_data = data - np.mean(data)
        
        # 計算並保存濾波數據
        try:
            filtered_display = filtfilt(self.worker.b, self.worker.a, display_data) if self.worker else display_data
            self.filtered_buffer = deque(filtered_display[-len(self.buffer):], maxlen=BUFFER_SIZE)
        except:
            self.filtered_buffer = self.buffer
        
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