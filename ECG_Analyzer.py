"""
ECG 数据离线分析工具
用于处理导出的 CSV/NPY 数据并生成可视化报告
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from scipy.fft import fft, fftfreq
import os
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ECGAnalyzer:
    """ECG 数据离线分析器"""
    
    def __init__(self, sample_rate=400):
        self.sample_rate = sample_rate
        self.b, self.a = self._design_filter()
    
    def _design_filter(self):
        """设计带通滤波器"""
        nyquist = self.sample_rate / 2
        low, high = 0.5 / nyquist, 50 / nyquist
        b, a = butter(4, [low, high], btype='band')
        return b, a
    
    def load_csv(self, csv_path):
        """加载 CSV 数据"""
        df = pd.read_csv(csv_path)
        return df['ECG Value'].values if 'ECG Value' in df.columns else df.iloc[:, 1].values
    
    def load_npy(self, npy_path):
        """加载 NPY 数据"""
        return np.load(npy_path)
    
    def filter_signal(self, signal):
        """带通滤波"""
        if len(signal) < 10:
            return signal
        return filtfilt(self.b, self.a, signal)
    
    def normalize_signal(self, signal):
        """信号标准化"""
        mean = np.mean(signal)
        std = np.std(signal)
        if std == 0:
            return signal - mean
        return (signal - mean) / std
    
    def detect_r_waves(self, signal):
        """检测 R 波峰值"""
        norm_signal = self.normalize_signal(signal)
        peaks, properties = find_peaks(
            norm_signal,
            height=0.5,
            distance=self.sample_rate // 3
        )
        return peaks
    
    def calculate_hr_from_peaks(self, peaks):
        """根据峰值计算心率"""
        if len(peaks) < 2:
            return 0
        intervals = np.diff(peaks) / self.sample_rate
        avg_interval = np.mean(intervals)
        hr = 60.0 / avg_interval
        return hr if 30 < hr < 200 else 0
    
    def compute_fft(self, signal):
        """计算频谱"""
        fft_result = fft(signal)
        freqs = fftfreq(len(signal), 1/self.sample_rate)
        idx = freqs > 0
        return freqs[idx], np.abs(fft_result[idx])
    
    def get_statistics(self, signal):
        """计算信号统计特性"""
        return {
            'mean': np.mean(signal),
            'std': np.std(signal),
            'min': np.min(signal),
            'max': np.max(signal),
            'rms': np.sqrt(np.mean(signal**2))
        }
    
    def generate_report(self, signal, output_dir='./ecg_report', title='ECG 信号分析报告'):
        """生成完整的分析报告和可视化"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 计算各种处理结果
        filtered = self.filter_signal(signal)
        peaks = self.detect_r_waves(filtered)
        hr = self.calculate_hr_from_peaks(peaks)
        freqs, magnitude = self.compute_fft(filtered)
        
        # 时间轴
        time_axis = np.arange(len(signal)) / self.sample_rate
        
        # 创建多子图
        fig = plt.figure(figsize=(16, 12))
        
        # 1. 原始信号
        ax1 = plt.subplot(3, 2, 1)
        ax1.plot(time_axis, signal, color='#CCCCCC', linewidth=0.8, label='原始信号')
        ax1.set_xlabel('时间 (秒)')
        ax1.set_ylabel('振幅 (mV)')
        ax1.set_title('1. 原始 ECG 信号波形')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. 滤波信号
        ax2 = plt.subplot(3, 2, 2)
        ax2.plot(time_axis, filtered, color='#00ff00', linewidth=1, label='滤波信号')
        ax2.set_xlabel('时间 (秒)')
        ax2.set_ylabel('振幅')
        ax2.set_title('2. 带通滤波后的信号 (0.5-50 Hz)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. R 波峰值检测
        ax3 = plt.subplot(3, 2, 3)
        ax3.plot(time_axis, filtered, color='#00ff00', linewidth=1, label='滤波信号')
        if len(peaks) > 0:
            peak_times = peaks / self.sample_rate
            peak_values = filtered[peaks]
            ax3.plot(peak_times, peak_values, 'ro', markersize=8, label='R 波峰值')
        ax3.set_xlabel('时间 (秒)')
        ax3.set_ylabel('振幅')
        ax3.set_title(f'3. R 波峰值检测 (共 {len(peaks)} 个)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 4. 频谱图
        ax4 = plt.subplot(3, 2, 4)
        ax4.semilogy(freqs, magnitude, color='#00ccff', linewidth=1.5)
        ax4.set_xlabel('频率 (Hz)')
        ax4.set_ylabel('幅度 (对数)')
        ax4.set_title('4. 频谱分析 (FFT)')
        ax4.set_xlim([0, 60])
        ax4.grid(True, alpha=0.3)
        
        # 5. 标准化信号
        ax5 = plt.subplot(3, 2, 5)
        norm_filtered = self.normalize_signal(filtered)
        ax5.plot(time_axis, norm_filtered, color='#FF6B6B', linewidth=1, label='标准化信号')
        ax5.axhline(y=0.5, color='r', linestyle='--', linewidth=1, alpha=0.5, label='峰值检测阈值')
        ax5.set_xlabel('时间 (秒)')
        ax5.set_ylabel('标准化振幅')
        ax5.set_title('5. 标准化信号与检测阈值')
        ax5.grid(True, alpha=0.3)
        ax5.legend()
        
        # 6. 统计信息文本
        ax6 = plt.subplot(3, 2, 6)
        ax6.axis('off')
        
        stats = self.get_statistics(signal)
        info_text = f"""
信号统计信息
─────────────────────
采样率:        {self.sample_rate} Hz
采集时长:      {len(signal)/self.sample_rate:.2f} 秒
采样点数:      {len(signal)} 点

原始信号统计
─────────────────────
最大值:        {stats['max']:.2f} mV
最小值:        {stats['min']:.2f} mV
平均值:        {stats['mean']:.2f} mV
标准差:        {stats['std']:.2f} mV
均方根:        {stats['rms']:.2f} mV

R 波检测结果
─────────────────────
检测数量:      {len(peaks)} 个

计算心率
─────────────────────
瞬时心率:      {hr:.1f} bpm

生成时间:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        ax6.text(0.1, 0.5, info_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        # 保存报告
        report_path = os.path.join(output_dir, f'ECG_Report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(report_path, dpi=300, bbox_inches='tight')
        print(f"✓ 报告已保存: {report_path}")
        
        # 生成文本总结
        summary_path = os.path.join(output_dir, f'Summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"ECG 信号分析总结 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n")
            f.write("="*60 + "\n\n")
            f.write("1. 信号基本信息\n")
            f.write("-"*60 + "\n")
            f.write(f"采样率: {self.sample_rate} Hz\n")
            f.write(f"采集时长: {len(signal)/self.sample_rate:.2f} 秒\n")
            f.write(f"采样点数: {len(signal)} 点\n\n")
            
            f.write("2. 原始信号统计\n")
            f.write("-"*60 + "\n")
            f.write(f"最大值: {stats['max']:.2f} mV\n")
            f.write(f"最小值: {stats['min']:.2f} mV\n")
            f.write(f"平均值: {stats['mean']:.2f} mV\n")
            f.write(f"标准差: {stats['std']:.2f} mV\n")
            f.write(f"均方根: {stats['rms']:.2f} mV\n\n")
            
            f.write("3. R 波检测结果\n")
            f.write("-"*60 + "\n")
            f.write(f"检测到 R 波个数: {len(peaks)} 个\n")
            if len(peaks) > 0:
                intervals = np.diff(peaks) / self.sample_rate
                f.write(f"平均 R-R 间隔: {np.mean(intervals):.4f} 秒\n")
                f.write(f"R-R 间隔标准差: {np.std(intervals):.4f} 秒\n\n")
            
            f.write("4. 心率计算\n")
            f.write("-"*60 + "\n")
            f.write(f"计算心率: {hr:.1f} bpm\n")
            f.write(f"心率范围: 30-200 bpm\n")
            f.write(f"有效性: {'有效' if 30 < hr < 200 else '无效'}\n\n")
            
            f.write("5. 频谱信息\n")
            f.write("-"*60 + "\n")
            f.write(f"主频率范围: 0.5-50 Hz\n")
            f.write(f"最高能量频率: {freqs[np.argmax(magnitude)]:.2f} Hz\n")
        
        print(f"✓ 总结已保存: {summary_path}")
        plt.show()
        
        return {
            'signal_stats': stats,
            'peaks': peaks,
            'heart_rate': hr,
            'freqs': freqs,
            'magnitude': magnitude
        }


def main():
    """主函数 - 使用示例"""
    import sys
    
    print("="*60)
    print("ECG 信号离线分析工具")
    print("="*60)
    
    # 示例: 分析导出的 CSV 文件
    csv_file = input("\n请输入 CSV 文件路径 (或按 Enter 跳过): ").strip()
    
    if csv_file and os.path.exists(csv_file):
        analyzer = ECGAnalyzer(sample_rate=400)
        
        try:
            signal = analyzer.load_csv(csv_file)
            print(f"\n✓ 已加载 {len(signal)} 个采样点")
            print("正在生成分析报告...")
            
            results = analyzer.generate_report(
                signal,
                output_dir='./ecg_report',
                title='ECG 信号离线分析报告'
            )
            
            print("\n" + "="*60)
            print("分析完成!")
            print("="*60)
            print(f"心率: {results['heart_rate']:.1f} bpm")
            print(f"检测到的 R 波: {len(results['peaks'])} 个")
            
        except Exception as e:
            print(f"✗ 错误: {e}")
    else:
        print("\n未输入有效的文件路径")
        print("\n使用示例:")
        print("  analyzer = ECGAnalyzer()")
        print("  signal = analyzer.load_csv('ecg_data.csv')")
        print("  results = analyzer.generate_report(signal)")


if __name__ == '__main__':
    main()
