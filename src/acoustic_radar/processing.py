"""信号处理链：混频低通、ESPRIT 超分辨频率提取、DSC-FMCW 参数解算。

处理流程（对应论文图 3）::

    发射模板 ─┐
              ├─► 混频器 ⊗ ─► 低通滤波 ─► Z 变换极点提取 ─► DSC-FMCW 补偿
    接收回声 ─┘

核心思想是：先把宽带 Chirp 差频化为低频中频（IF）信号，再用 ESPRIT
在时域直接分离多径分量，最后利用 DSC-FMCW 的对称频率约束把多普勒项
代数抵消，得到与速度无关的绝对距离。
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigvals, pinv, svd
from scipy.signal import butter, filtfilt, hilbert


# ==========================================================================
# 混频与低通滤波
# ==========================================================================
def mix_and_lowpass(
    tx_signal: np.ndarray,
    rx_signal: np.ndarray,
    fs: float,
    cutoff_freq: float = 2000.0,
) -> np.ndarray:
    """将发射模板与接收信号混频，低通滤波提取中频 (IF)。

    时域相乘利用积化和差产生 (f_tx + f_rx) 与 (f_tx - f_rx) 两组分量，
    低通滤波器保留差频项。``filtfilt`` 实现零相移滤波，避免普通 IIR
    滤波器引入的群延迟——这对测距精度至关重要。

    :param tx_signal: 本地发射模板 (1D array)
    :param rx_signal: 麦克风接收信号 (1D array)
    :param fs: 采样率 (Hz)
    :param cutoff_freq: 低通截止频率 (Hz)，需大于最大可能距离产生的差频
    :return: 纯净的中频信号 y_if（已乘 2 补偿 cosA*cosB 的 0.5 系数）
    """
    mixed_signal = tx_signal * rx_signal

    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(4, normal_cutoff, btype="low", analog=False)

    y_if = filtfilt(b, a, mixed_signal)

    # cosA * cosB = 0.5*cos(A-B) + 0.5*cos(A+B)，低通后幅度减半，故乘 2
    return y_if * 2.0


# ==========================================================================
# ESPRIT 超分辨频率提取
# ==========================================================================
def esprit_extract_frequencies(y_if: np.ndarray, fs: float) -> np.ndarray:
    """用 ESPRIT 从时域中频信号中提取全部多径分量的差频集合。

    算法要点：
      1. 解析信号化（Hilbert）得到复序列；
      2. 构造 Hankel 数据矩阵并做 SVD，按奇异值能量划分信号/噪声子空间；
      3. 利用信号子空间的时移旋转不变性求解旋转矩阵 Φ = U1⁻¹U2；
      4. 对 Φ 做特征值分解，由特征值相角得到各分量的差频。

    与传统 FFT 寻峰相比，ESPRIT 不受栅栏效应与分辨率极限约束，
    且对多径造成的频谱畸变鲁棒。

    :return: 各路径的差频（Hz），已取绝对值
    """
    n = len(y_if)
    y_complex = hilbert(y_if)

    # 构造 Hankel/轨迹矩阵：L 取 N//3 在估计精度与计算量之间取得平衡
    l = n // 3
    x = np.zeros((l, n - l + 1), dtype=complex)
    for i in range(l):
        x[i, :] = y_complex[i : i + n - l + 1]

    u, s, _ = svd(x, full_matrices=False)

    # 按奇异值阈值判断多径数量（相对首奇异值 10%）
    threshold = 0.1 * s[0]
    num_sources = max(1, int(np.sum(s > threshold)))

    us = u[:, :num_sources]
    u1 = us[:-1, :]
    u2 = us[1:, :]

    # 旋转矩阵 Phi 的特征值即 Z 平面上的极点 z_k = exp(j*2*pi*f_k*Ts)
    phi = pinv(u1) @ u2
    lambdas = eigvals(phi)

    freqs = np.abs(np.angle(lambdas) * fs / (2 * np.pi))
    return freqs


def esprit_extract_direct_path(y_if: np.ndarray, fs: float) -> float:
    """提取直射波（最短路径）的差频。

   声波沿直线传播距离最短，因此直射波对应差频集合中的绝对最小值。
    系统无需任何环境先验或复杂波形配对，仅凭此物理约束即可锁定直射分量。
    """
    return float(np.min(esprit_extract_frequencies(y_if, fs)))


# ==========================================================================
# DSC-FMCW 多普勒频移补偿
# ==========================================================================
def dsc_fmcw_distance(fp_up: float, fp_down: float, c: float, t: float, b: float) -> float:
    """由升/降频差频的算术平均解算绝对距离（与目标速度无关）。

    DSC-FMCW 强制 downchirp 起始频率比 upchirp 高出 2B。此时

        (f'_p + f_p)/2 = BR/cT + (f0 - f0')v/2c + (B + B')v/2c
                       = BR/cT - Bv/c + Bv/c = BR/cT

    多普勒项在代数上被完全抵消，故

        R = (f'_p + f_p) * c * T / (2B)

    :param fp_up: 升频差频 (Hz)
    :param fp_down: 降频差频 (Hz)
    :return: 绝对距离 (m)
    """
    return (fp_down + fp_up) * c * t / (2 * b)


def dsc_fmcw_radial_velocity(
    fp_up: float,
    fp_down: float,
    c: float,
    f0: float,
    b: float,
) -> float:
    """由升/降频差频之差解算径向速度。

        (f'_p - f_p)/2 = (f0 + B)v / c
      => v_r = (f'_p - f_p) * c / (2(f0 + B))

    正号表示目标正在靠近基站。

    :return: 径向速度 (m/s)
    """
    return (-fp_up + fp_down) * c / (2 * (f0 + b))


# ==========================================================================
# 圆周运动基频追踪（sound.wav 分析用）
# ==========================================================================
def bandpass_filter(
    data: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    """双向巴特沃斯带通滤波，滤除环境低频噪声与高频杂音且无相位失真。"""
    nyquist = 0.5 * fs
    b, a = butter(order, [lowcut / nyquist, highcut / nyquist], btype="band")
    return filtfilt(b, a, data)


def extract_pitch_autocorr(
    y: np.ndarray,
    sr: float,
    fmin: float,
    fmax: float,
    hop_length: int = 512,
    frame_length: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    """基于时域自相关的基频（多普勒频率）轨迹追踪。

    分帧后加 Hanning 窗抑制频谱泄漏，计算每帧自相关函数并搜索主峰延迟，
    由此重构接收频率随时间变化的平滑轨迹 f(t)。

    :return: (f0, times)，无效帧的 f0 为 NaN
    """
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
        frame = y[start : start + frame_length]
        frame = frame - np.mean(frame)          # 去直流
        frame = frame * hanning_window

        corr = np.correlate(frame, frame, mode="full")
        corr = corr[len(corr) // 2 :]           # 保留滞后 >= 0 的部分

        valid_corr = corr[min_lag:max_lag]
        if len(valid_corr) > 0 and np.max(valid_corr) > 0:
            true_lag = min_lag + int(np.argmax(valid_corr))
            if true_lag > 0:
                f0[i] = sr / true_lag

        times[i] = (start + frame_length / 2) / sr

    return f0, times


def rms_envelope(y: np.ndarray, sr: float, window_sec: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    """提取信号的 RMS 能量包络。

    利用声学能量随距离平方反比衰减的模型，由包络极大/极小值之比
    可以反解声源圆周运动的圆心到麦克风的距离。

    :return: (t, rms_energy)
    """
    window_samples = int(sr * window_sec)
    squared_signal = y**2
    mean_squared = np.convolve(squared_signal, np.ones(window_samples) / window_samples, mode="same")
    t = np.arange(len(y)) / sr
    return t, np.sqrt(mean_squared)
