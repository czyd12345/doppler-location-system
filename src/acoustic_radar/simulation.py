"""全链路声学物理仿真器。

与简单的"给差频加个偏移量"式造假不同，本模块严格从延迟方程出发
生成接收信号，因此天然包含了距离-多普勒耦合、多径相消以及噪声底噪：

    τ(t) = (R - v·t) / c          # 动态物理延迟，v > 0 表示目标在靠近
    t_delayed = t - τ(t)
    s_rx(t) = Σ_k A_k · cos(2π(f₀·t_delayed + B/(2T)·t_delayed²))

把延迟后的时间整体代回相位方程，可精确还原高速运动下的相位历程，
这是验证 DSC-FMCW 能否真正消除多普勒残余误差的关键。
"""

from __future__ import annotations

import numpy as np


def generate_physical_signals(
    t: np.ndarray,
    f_start: float,
    b_sweep: float,
    duration: float,
    r: float,
    v: float,
    c: float,
    multipath_delays=(),
    noise_std: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """生成发射模板与含多径、多普勒频移的接收回声。

    :param t: 时间轴 (s)
    :param f_start: 扫频起始频率 (Hz)
    :param b_sweep: 扫频带宽 (Hz)，取负值即为降频
    :param duration: 扫频周期 T (s)
    :param r: 目标到基站的初始距离 R (m)
    :param v: 径向速度 (m/s)
    :param c: 声速 (m/s)
    :param multipath_delays: 各反射路径的额外路程 (m)
    :param noise_std: 环境底噪标准差
    :param rng: 随机数发生器（传入固定 seed 可保证结果可复现）
    :return: (tx_signal, rx_signal)
    """
    if rng is None:
        rng = np.random.default_rng()

    phase_tx = 2 * np.pi * (f_start * t + (b_sweep / (2 * duration)) * t**2)
    tx_signal = np.cos(phase_tx)

    rx_signal = np.zeros_like(t)

    # 直射波路径（绝对距离 R）与反射波路径（R + 额外路程）
    paths = [(r, 1.0)]
    for dr in multipath_delays:
        paths.append((r + dr, rng.uniform(0.4, 0.8)))

    for dist, amp in paths:
        tau = (dist - v * t) / c
        t_delayed = t - tau
        phase_rx = 2 * np.pi * (f_start * t_delayed + (b_sweep / (2 * duration)) * t_delayed**2)
        rx_signal += amp * np.cos(phase_rx)

    rx_signal += rng.normal(0, noise_std, len(t))

    return tx_signal, rx_signal
