import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt

# ==========================================
# 0. 学术图表全局字体与分辨率配置 (新增)
# ==========================================
plt.rcParams.update({
    "font.family": "serif",       # 使用衬线字体 (类似于 Times New Roman，学术感更强)
    "font.size": 10,              # 全局基础字体大小
    "axes.labelsize": 10,         # X轴和Y轴标签的字体大小
    "xtick.labelsize": 9,         # X轴刻度数字大小
    "ytick.labelsize": 9,         # Y轴刻度数字大小
    "legend.fontsize": 9,         # 图例的字体大小
    "figure.dpi": 300             # 强制高分辨率，确保屏幕显示和导出时清晰不模糊
})

# ==========================================
# 1. 定义带通滤波器
# ==========================================
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y_filtered = filtfilt(b, a, data)
    return y_filtered

# ==========================================
# 2. 原始基频提取函数
# ==========================================
def extract_pitch_autocorr(y, sr, fmin, fmax, hop_length=512, frame_length=2048):
    num_frames = 1 + (len(y) - frame_length) // hop_length
    f0 = np.full(num_frames, np.nan)
    times = np.zeros(num_frames)
    
    min_lag = int(np.floor(sr / fmax))
    max_lag = int(np.ceil(sr / fmin))
    if max_lag > frame_length:
        max_lag = frame_length
    hanning_window = np.hanning(frame_length)

    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        frame = y[start:end]
        frame = frame - np.mean(frame) 
        frame = frame * hanning_window
        corr = np.correlate(frame, frame, mode='full')
        corr = corr[len(corr)//2:] 
        valid_corr = corr[min_lag:max_lag]
        if len(valid_corr) > 0 and np.max(valid_corr) > 0:
            best_lag_idx = np.argmax(valid_corr)
            true_lag = min_lag + best_lag_idx
            
            if true_lag > 0: 
                f0[i] = sr / true_lag
        times[i] = (start + frame_length / 2) / sr
    return f0, times

# ==========================================
# 主流程：数据读取与预处理
# ==========================================
y, sr = sf.read("sound.wav")
if y.ndim > 1:
    y = np.mean(y, axis=1)
fmin_target = 1000
fmax_target = 3000

y_filtered = bandpass_filter(y, lowcut=fmin_target, highcut=fmax_target, fs=sr, order=4)

f0, times = extract_pitch_autocorr(y_filtered, sr, fmin=fmin_target, fmax=fmax_target)

# ==========================================
# 提取周期与可视化 (频率轨迹)
# ==========================================
valid_indices = ~np.isnan(f0)
times_valid = times[valid_indices]
f0_valid = f0[valid_indices]

peaks, properties = find_peaks(f0_valid, prominence=10)

if len(peaks) > 1:
    peak_times = times_valid[peaks]
    periods = np.diff(peak_times)
    avg_period = np.mean(periods)
    modulation_freq = 1 / avg_period
    print(f"Average Period: {avg_period:.4f} s")
    print(f"Modulation Freq: {modulation_freq:.2f} Hz")
else:
    print("No prominent periodic frequency modulation detected.")

fmax = np.max(f0_valid)
fmin = np.min(f0_valid)
a = fmax/fmin
v = 340*(a-1)/(1+a)
print(f"speed of speaker:{v:.4f} m/s")

# --- 绘图 1：基频轨迹图 ---
plt.figure(figsize=(3.5, 2.6))
plt.plot(times_valid, f0_valid, label='Pitch Track', color='dodgerblue', alpha=0.8)

if len(peaks) > 1:
    plt.plot(times_valid[peaks], f0_valid[peaks], 'rx', markersize=6, label='Peaks') # 标记适当缩小一点避免拥挤
    
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.legend(loc='lower right') # 调整图例到右下角，避免遮挡波峰
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# ==========================================
# 能量包络计算与可视化
# ==========================================
window_time = 0.02 
window_samples = int(sr * window_time)

squared_signal = y_filtered ** 2
mean_squared_energy = np.convolve(squared_signal, np.ones(window_samples)/window_samples, mode='same')
rms_energy = np.sqrt(mean_squared_energy)
t_energy = np.arange(len(y_filtered)) / sr

# --- 绘图 2：RMS 能量图 ---
plt.figure(figsize=(3.5, 2.6))
plt.fill_between(t_energy, rms_energy, color='darkorange', alpha=0.6, label='RMS Area')
plt.plot(t_energy, rms_energy, color='orangered', linewidth=1.5, label='RMS Line')

plt.xlabel('Time (s)')
plt.ylabel('Energy / Amplitude')
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

r = v * avg_period / (2 * np.pi)
E_max = 0.284
E_min = 0.077
d = (1 + np.sqrt(E_min/E_max)) / (1 - np.sqrt(E_min/E_max)) * r
print(f"d: {d:.4f}, r: {r:.4f}")