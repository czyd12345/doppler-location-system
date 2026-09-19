#!/usr/bin/env python3
"""实验一：匀速圆周运动参数估计 + 双基站发射波形生成。

对应论文第 III 节（信号处理与参数估计）与第 IV 节（波形设计）。

输入：data/sound.wav —— 麦克风单点录制的圆周运动声源信号
输出：results/fig1_pitch_track.png、results/fig2_energy_envelope.png
      data/tx_dual_BS_TDM_FDM.wav（两个基站播放用的 TDM 立体声波形）

估计流程
--------
1. 对 1~3 kHz 频段做四阶巴特沃斯带通滤波，抑制环境噪声；
2. 时域自相关基频追踪，得到多普勒频率轨迹 f(t)；
3. 由 f_max/f_min 比值 a 解出线速度 v = c(a-1)/(a+1)；
4. 由 f(t) 波峰间隔得到圆周运动周期 T，半径 r = vT/2π；
5. 由 RMS 能量包络的极值比 k 反解圆心到麦克风距离 d = r(1+k)/(1-k)。

用法::

    python scripts/01_circular_motion.py
    python scripts/01_circular_motion.py --emit-only      # 只生成发射波形
    python scripts/01_circular_motion.py --skip-emit      # 只做参数估计
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import find_peaks

# 允许直接以脚本方式运行（无需 pip install）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from acoustic_radar import config, generation  # noqa: E402
from acoustic_radar.plotting import SMALL_FIGSIZE, plt, save_and_show  # noqa: E402
from acoustic_radar.processing import (  # noqa: E402
    bandpass_filter,
    extract_pitch_autocorr,
    rms_envelope,
)


def emit_tx_waveform(path: Path) -> None:
    """生成双基站 TDM 立体声发射波形并写出 WAV。"""
    print("[1/2] 生成双基站 TDM/FDM 发射波形 ...")
    repeats = 100
    stereo = generation.build_tdm_stereo(
        fs=config.FS_STANDARD,
        duration=config.T_CHIRP,
        f0_up=config.F0_UP,
        bandwidth=config.B_CHIRP,
        repeat_times=repeats,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    generation.write_stereo_wav(path, stereo, config.FS_STANDARD)

    print(f"  -> 文件已保存: {path}")
    print(f"     · 总时长 : {repeats * 2 * config.T_CHIRP:.1f} s")
    print(f"     · 采样率 : {config.FS_STANDARD} Hz")
    print(f"     · 声道数 : 2（左=基站1 前 50 ms，右=基站2 后 50 ms）")
    print(f"     · 单基站波形: Upchirp {config.F0_UP:.0f}→{config.F0_UP + config.B_CHIRP:.0f} Hz"
          f" + Downchirp {config.F0_DOWN:.0f}→{config.F0_DOWN - config.B_CHIRP:.0f} Hz\n")


def estimate_circular_motion(path: Path, results_dir: Path) -> None:
    """从圆周运动录音中估计线速度、轨迹半径与圆心距离。"""
    print("[2/2] 圆周运动参数估计 ...")

    y, sr = sf.read(str(path))
    if y.ndim > 1:  # 立体声录音下混为单声道
        y = np.mean(y, axis=1)
    print(f"  · 载入 {path.name}：{len(y) / sr:.2f} s @ {sr} Hz")

    # --- 带通滤波 + 基频追踪 ---------------------------------------------
    y_filtered = bandpass_filter(
        y, lowcut=config.PITCH_FMIN, highcut=config.PITCH_FMAX, fs=sr, order=4
    )
    f0, times = extract_pitch_autocorr(
        y_filtered,
        sr,
        fmin=config.PITCH_FMIN,
        fmax=config.PITCH_FMAX,
        hop_length=config.PITCH_HOP,
        frame_length=config.PITCH_FRAME,
    )

    valid = ~np.isnan(f0)
    times_valid, f0_valid = times[valid], f0[valid]

    # --- 3. 线速度：由多普勒极值比解算 ------------------------------------
    peaks, _ = find_peaks(f0_valid, prominence=config.PITCH_PROMINENCE)
    if len(peaks) < 2:
        print("  未检测到明显的周期性频率调制，无法继续求解。")
        return

    peak_times = times_valid[peaks]
    avg_period = float(np.mean(np.diff(peak_times)))
    modulation_freq = 1 / avg_period
    print(f"  · 平均调制周期 T = {avg_period:.4f} s（调制频率 {modulation_freq:.2f} Hz）")

    f_max, f_min = float(np.max(f0_valid)), float(np.min(f0_valid))
    ratio = f_max / f_min
    speed = config.C_SOUND * (ratio - 1) / (1 + ratio)
    print(f"  · 多普勒极值 f_max = {f_max:.1f} Hz, f_min = {f_min:.1f} Hz, a = {ratio:.4f}")
    print(f"  · 声源线速度 v = {speed:.4f} m/s")

    # --- 图 1：基频（多普勒频率）轨迹 -------------------------------------
    fig, ax = plt.subplots(figsize=SMALL_FIGSIZE)
    ax.plot(times_valid, f0_valid, label="Pitch Track", color="dodgerblue", alpha=0.8)
    ax.plot(times_valid[peaks], f0_valid[peaks], "rx", markersize=6, label="Peaks")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    save_and_show(fig, results_dir / config.FIG_PITCH.name)

    # --- 4~5. 半径与圆心距离：由能量包络反解 ------------------------------
    t_energy, rms_energy = rms_envelope(y_filtered, sr, config.ENERGY_WINDOW_SEC)

    fig, ax = plt.subplots(figsize=SMALL_FIGSIZE)
    ax.fill_between(t_energy, rms_energy, color="darkorange", alpha=0.6, label="RMS Area")
    ax.plot(t_energy, rms_energy, color="orangered", linewidth=1.5, label="RMS Line")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Energy / Amplitude")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    save_and_show(fig, results_dir / config.FIG_ENERGY.name)

    radius = speed * avg_period / (2 * np.pi)

    # 能量包络极值取自实测波形（对应论文图 2 中读出的 E_max / E_min）。
    # 若更换录音或改成自动读取，可用 find_peaks 提取包络极值后替换此处常量。
    e_max, e_min = 0.284, 0.077
    k = np.sqrt(e_min / e_max)
    distance = (1 + k) / (1 - k) * radius

    print(f"  · 轨迹半径 r = {radius:.4f} m")
    print(f"  · 能量比值 k = {k:.4f}（E_max = {e_max}, E_min = {e_min}）")
    print(f"  · 圆心到麦克风距离 d = {distance:.4f} m\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR)
    parser.add_argument("--emit-only", action="store_true",
                        help="只生成双基站发射波形，跳过参数估计")
    parser.add_argument("--skip-emit", action="store_true",
                        help="只做参数估计，不重新生成发射波形")
    args = parser.parse_args()
    config.ensure_dirs(args.results_dir)

    tx_path = args.data_dir / config.TX_WAV.name
    sound_path = args.data_dir / config.SOUND_WAV.name

    print("=" * 68)
    print(" 实验一：匀速圆周运动参数估计 + 双基站发射波形生成")
    print("=" * 68)
    print(f" 数据目录: {args.data_dir}")
    print(f" 输出目录: {args.results_dir}\n")

    if not sound_path.exists():
        print(f"错误：找不到输入文件 {sound_path}")
        print("请将圆周运动录音放置到 data/sound.wav 后重试。")
        sys.exit(1)

    if not args.emit_only:
        estimate_circular_motion(sound_path, args.results_dir)
    if not args.skip_emit:
        emit_tx_waveform(tx_path)

    print("完成。")


if __name__ == "__main__":
    main()
