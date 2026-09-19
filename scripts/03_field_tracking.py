#!/usr/bin/env python3
"""实验三：外场实测数据的连续轨迹与速度解算（论文第 V 节 B 部分）。

输入 data/rx_field_test_raw.wav —— 户外开阔环境下用双基站 + 单麦克风
录制的实测音频（11 s @ 48 kHz）。

按 100 ms 的 TDM 探测周期滑动窗口切帧，逐帧独立完成：
时域解复用 → 混频低通 → ESPRIT 提取直射波差频 → DSC-FMCW 解算距离与
径向速度 → 双圆交点定位 + 方向投影测速，最终绘制带速度矢量的运动轨迹。

用法::

    python scripts/03_field_tracking.py
    python scripts/03_field_tracking.py --points 30      # 解算更多轨迹点
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wav

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from acoustic_radar import config  # noqa: E402
from acoustic_radar.geometry import (  # noqa: E402
    select_valid_solution,
    solve_2d_position_2anchors,
    solve_2d_velocity,
)
from acoustic_radar.plotting import LARGE_FIGSIZE, plt, save_and_show  # noqa: E402
from acoustic_radar.processing import (  # noqa: E402
    dsc_fmcw_distance,
    dsc_fmcw_radial_velocity,
    esprit_extract_direct_path,
    mix_and_lowpass,
)

# 速度箭头缩放系数：数值越小箭头越长（matplotlib quiver 语义）
VELOCITY_ARROW_SCALE = 20


def load_audio(path: Path) -> tuple[int, np.ndarray]:
    """读取 WAV 并归一化到 [-1, 1]。"""
    fs, audio = wav.read(str(path))
    audio = audio.astype(np.float64)
    return fs, audio / np.max(np.abs(audio))


def solve_frame(
    frame: np.ndarray,
    tx_up: np.ndarray,
    tx_down: np.ndarray,
    fs: int,
) -> tuple[float, float, float, float] | None:
    """解算单个 100 ms 探测帧，返回 (R1, v_r1, R2, v_r2)。

    帧结构：前 50 ms 来自基站 1，后 50 ms 来自基站 2（TDM 解复用）。
    两个测距圆无交点时返回 None。

    注意：中频信号只取前 65%（``SAFE_RATIO``）。TDM 切换点附近会残留
    前一基站的尾音，若不截断会污染 ESPRIT 的频谱估计。
    """
    half_len = int(config.T_CHIRP * fs)
    rx_from_bs1 = frame[:half_len]
    rx_from_bs2 = frame[half_len : 2 * half_len]
    safe_length = int(half_len * config.SAFE_RATIO)

    results = []
    for rx in (rx_from_bs1, rx_from_bs2):
        if_up = mix_and_lowpass(tx_up, rx, fs, config.LPF_CUTOFF)
        if_down = mix_and_lowpass(tx_down, rx, fs, config.LPF_CUTOFF)

        fp_up = esprit_extract_direct_path(if_up[:safe_length], fs)
        fp_down = esprit_extract_direct_path(if_down[:safe_length], fs)

        r = dsc_fmcw_distance(fp_up, fp_down, config.C_SOUND, config.T_CHIRP, config.B_CHIRP)
        v_r = dsc_fmcw_radial_velocity(fp_up, fp_down, config.C_SOUND, config.F0_UP, config.B_CHIRP)
        results.extend([r, v_r])

    return tuple(results)  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR)
    parser.add_argument("--points", type=int, default=config.TARGET_POINTS,
                        help=f"目标解算点数 (默认: {config.TARGET_POINTS})")
    args = parser.parse_args()

    config.ensure_dirs(args.results_dir)
    rx_path = args.data_dir / config.RX_FIELD_WAV.name

    print("=" * 68)
    print(" 实验三：外场实测数据连续轨迹解算")
    print("=" * 68)

    if not rx_path.exists():
        print(f"错误：找不到录音文件 {rx_path}")
        print("请将实测音频放置到 data/rx_field_test_raw.wav 后重试。")
        sys.exit(1)

    fs, audio = load_audio(rx_path)
    print(f" 成功加载音频: {rx_path.name} (采样率: {fs} Hz, 时长 {len(audio) / fs:.2f} s)")

    # --- 滑动窗口参数 -----------------------------------------------------
    start_idx = int(config.TRACK_OFFSET_SEC * fs)   # 跳过开头的唤醒静音
    frame_len = int(config.FRAME_SEC * fs)          # 每个探测帧 100 ms
    total_frames = (len(audio) - start_idx) // frame_len
    print(f" 录音包含有效数据约 {total_frames * config.FRAME_SEC:.1f} s，"
          f"共计 {total_frames} 个探测帧\n")

    # --- 本地发射模板 -----------------------------------------------------
    t = np.arange(0, config.T_CHIRP, 1 / fs)
    tx_up = np.cos(2 * np.pi * (config.F0_UP * t + (config.B_CHIRP / (2 * config.T_CHIRP)) * t**2))
    tx_down = np.cos(2 * np.pi * (config.F0_DOWN * t + (-config.B_CHIRP / (2 * config.T_CHIRP)) * t**2))

    anchors = config.ANCHORS
    trajectory_x, trajectory_y = [], []
    trajectory_vx, trajectory_vy = [], []
    valid_times = []

    # --- 逐帧解算 ---------------------------------------------------------
    for i in range(total_frames):
        current = start_idx + i * frame_len
        solved = solve_frame(audio[current : current + frame_len], tx_up, tx_down, fs)
        if solved is None:
            continue
        r1, v_r1, r2, v_r2 = solved

        candidates = solve_2d_position_2anchors(anchors[0], anchors[1], r1, r2)
        final_xy = select_valid_solution(candidates, anchors, y_positive=True)
        if final_xy is None:
            continue

        # 飞点过滤：只保留落在有效场地范围内的解
        if not (config.FIELD_X_LIMIT[0] < final_xy[0] < config.FIELD_X_LIMIT[1]
                and config.FIELD_Y_LIMIT[0] < final_xy[1] < config.FIELD_Y_LIMIT[1]):
            continue

        final_v = solve_2d_velocity(anchors, final_xy, [v_r1, v_r2])
        trajectory_x.append(final_xy[0])
        trajectory_y.append(final_xy[1])
        trajectory_vx.append(final_v[0])
        trajectory_vy.append(final_v[1])
        valid_times.append(config.TRACK_OFFSET_SEC + i * config.FRAME_SEC)

        n = len(trajectory_x)
        print(f" 成功提取第 {n}/{args.points} 个点: "
              f"X={final_xy[0]:.2f}m, Y={final_xy[1]:.2f}m, "
              f"Vx={final_v[0]:.2f}m/s, Vy={final_v[1]:.2f}m/s")

        if n >= args.points:
            print(f" 已收集到所需的 {args.points} 个有效轨迹点，提前结束运算。")
            break

    print("\n 解算完成！开始绘制带速度矢量的轨迹图 ...")

    # --- 图 4：二维空间运动轨迹（含速度场） -------------------------------
    fig, ax = plt.subplots(figsize=LARGE_FIGSIZE)

    # 基站位置
    ax.scatter(anchors[:, 0], anchors[:, 1], c="red", marker="^", s=200,
               label="Base Stations", zorder=5)
    for i, (ax_x, ax_y) in enumerate(anchors):
        ax.text(ax_x, ax_y - 0.15, f"BS{i + 1}", fontsize=12, ha="center", color="red")

    if trajectory_x:
        ax.plot(trajectory_x, trajectory_y, "-o", color="blue", alpha=0.4,
                markersize=4, label="Path")

        # angles='xy', scale_units='xy' 保证箭头方向与物理坐标系严格一致
        ax.quiver(trajectory_x, trajectory_y, trajectory_vx, trajectory_vy,
                  angles="xy", scale_units="xy", scale=VELOCITY_ARROW_SCALE,
                  color="darkorange", width=0.005, headwidth=4, alpha=0.9,
                  label="Velocity Vector")

        ax.scatter(trajectory_x[0], trajectory_y[0], c="green", s=120,
                   label="Start", zorder=4)
        ax.scatter(trajectory_x[-1], trajectory_y[-1], c="purple", s=120,
                   label="End", zorder=4)
    else:
        print("警告：没有解算出有效的轨迹点。")

    ax.set_title("Acoustic Radar Tracking with Velocity Vectors", fontsize=16)
    ax.set_xlabel("X Coordinate (m)", fontsize=14)
    ax.set_ylabel("Y Coordinate (m)", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(loc="upper right")

    if trajectory_x:
        ax.set_xlim(min(min(trajectory_x), anchors[0][0]) - 1,
                    max(max(trajectory_x), anchors[1][0]) + 1)
        ax.set_ylim(min(0, min(trajectory_y)) - 1, max(trajectory_y) + 1)

    # X/Y 等比例，保证轨迹形状不失真（datalim 模式不会顶掉上面的显式限幅）
    ax.set_aspect("equal", adjustable="datalim")

    save_and_show(fig, args.results_dir / config.FIG_TRAJECTORY.name)
    print("完成。")


if __name__ == "__main__":
    main()
