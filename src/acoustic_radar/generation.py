"""发射波形合成：线性调频（Chirp）信号与双基站 TDM/FDM 复用。

物理背景
--------
单个基站同时发射一对对称的扫频信号（Upchirp + Downchirp），二者在空中
叠加为一路复合声波。两个基站按时间轮流占用信道（TDM），因此一个完整的
探测周期为 2T（100 ms）：前 50 ms 由基站 1 发声，后 50 ms 由基站 2 发声。

在立体声 WAV 中，这一时序被映射到左右两个声道，分别驱动两个功放通道，
使得一次播放即可让两个基站严格同步。
"""

from __future__ import annotations

import numpy as np
import scipy.io.wavfile as wav


def generate_fmcw_chirp(
    f_start: float,
    f_end: float,
    duration: float,
    fs: float = 48000,
) -> np.ndarray:
    """生成单向线性调频信号（Chirp）。

    瞬时频率在 ``duration`` 内从 ``f_start`` 线性变化到 ``f_end``，
    通过对瞬时频率积分得到相位，保证相位连续（不会在频带边缘产生跳变）。

    :param f_start: 起始频率 (Hz)
    :param f_end: 终止频率 (Hz)
    :param duration: 信号时长 (s)
    :param fs: 采样率 (Hz)
    :return: 归一化幅度的实数 Chirp 序列
    """
    t = np.arange(0, duration, 1 / fs)
    bandwidth = f_end - f_start
    # 相位积分：2*pi*(f_start*t + 0.5*k*t^2)，扫频斜率 k = B/T
    phase = 2 * np.pi * (f_start * t + (bandwidth / (2 * duration)) * t**2)
    return np.cos(phase)


def build_dual_bs_frame(
    fs: float = 48000,
    duration: float = 0.05,
    f0_up: float = 4000.0,
    bandwidth: float = 4000.0,
) -> np.ndarray:
    """合成单个基站在一个扫频周期内的并发波形（Upchirp + Downchirp）。

    基站 1 与基站 2 共用同一套波形（信道由 TDM 区分），
    因此两个基站可以只生成一次。

    除以 2.0 是为了防止升频与降频波峰叠加时超出声卡动态范围导致削顶爆音。

    :return: 形状为 (duration * fs,) 的复合波形
    """
    f0_down = f0_up + 2 * bandwidth
    chirp_up = generate_fmcw_chirp(f0_up, f0_up + bandwidth, duration, fs)
    chirp_down = generate_fmcw_chirp(f0_down, f0_down - bandwidth, duration, fs)
    return (chirp_up + chirp_down) / 2.0


def build_tdm_stereo(
    fs: float = 48000,
    duration: float = 0.05,
    f0_up: float = 4000.0,
    bandwidth: float = 4000.0,
    repeat_times: int = 100,
) -> np.ndarray:
    """组装双基站 TDM 立体声序列。

    - 左声道：前 50 ms 驱动基站 1，后 50 ms 静音
    - 右声道：前 50 ms 静音，后 50 ms 驱动基站 2

    :param repeat_times: 100 ms 探测周期重复次数
    :return: 形状为 (repeat_times * 2 * duration * fs, 2) 的立体声矩阵
    """
    simultaneous = build_dual_bs_frame(fs, duration, f0_up, bandwidth)
    silence = np.zeros_like(simultaneous)

    left_channel = np.concatenate((simultaneous, silence))
    right_channel = np.concatenate((silence, simultaneous))

    stereo_block = np.column_stack((left_channel, right_channel))
    return np.tile(stereo_block, (repeat_times, 1))


def write_stereo_wav(path, stereo_sequence: np.ndarray, fs: float = 48000) -> None:
    """将浮点立体声序列写为 16-bit PCM WAV。

    乘以 30000 而非 32767，为叠加与重放留出少许动态余量。
    """
    wav_stereo = np.int16(stereo_sequence * 30000)
    wav.write(str(path), int(fs), wav_stereo)
