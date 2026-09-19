#!/usr/bin/env python3
"""实验二：全链路物理仿真验证（论文第 V 节 A 部分，仿真实验）。

在 Python 中搭建完整的声学物理层仿真环境：目标以给定的真实坐标与速度
运动，回声信号严格按延迟方程生成，并叠加多径反射与环境底噪。
随后跑通"混频 → 低通 → ESPRIT → DSC-FMCW → 几何求解"的完整算法链，
把解算结果与真值逐项对比。

用法::

    python scripts/02_simulation.py
    python scripts/02_simulation.py --seed 0        # 更换随机种子
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from acoustic_radar import config  # noqa: E402
from acoustic_radar.geometry import (  # noqa: E402
    select_valid_solution,
    solve_2d_position_2anchors,
    solve_2d_velocity,
)
from acoustic_radar.processing import (  # noqa: E402
    dsc_fmcw_distance,
    dsc_fmcw_radial_velocity,
    esprit_extract_direct_path,
    mix_and_lowpass,
)
from acoustic_radar.simulation import generate_physical_signals  # noqa: E402


def make_templates(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """构造本地升频/降频发射模板（接收端混频用）。"""
    up = np.cos(2 * np.pi * (config.F0_UP * t + (config.B_CHIRP / (2 * config.T_CHIRP)) * t**2))
    down = np.cos(2 * np.pi * (config.F0_DOWN * t + (-config.B_CHIRP / (2 * config.T_CHIRP)) * t**2))
    return up, down


def run_anchor(
    t: np.ndarray,
    r_true: float,
    v_true: float,
    tx_up: np.ndarray,
    tx_down: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """对单个基站执行一次完整的收发链路仿真与解算。

    基站前后交替发射升频与降频信号，空气中叠加为一路复合声波；
    接收端将其与本地纯净模板混频，分别提取两个差频。

    :return: (解算距离 R, 解算径向速度 v_r)
    """
    _, rx_up = generate_physical_signals(
        t, config.F0_UP, config.B_CHIRP, config.T_CHIRP, r_true, v_true,
        config.C_SOUND, config.SIM_MULTIPATH_DELAYS, config.SIM_NOISE_STD, rng,
    )
    _, rx_down = generate_physical_signals(
        t, config.F0_DOWN, -config.B_CHIRP, config.T_CHIRP, r_true, v_true,
        config.C_SOUND, config.SIM_MULTIPATH_DELAYS, config.SIM_NOISE_STD, rng,
    )
    rx_air = (rx_up + rx_down) / 2.0  # 基站制造的空气总声波

    if_up = mix_and_lowpass(tx_up, rx_air, config.FS_STANDARD, config.LPF_CUTOFF)
    if_down = mix_and_lowpass(tx_down, rx_air, config.FS_STANDARD, config.LPF_CUTOFF)

    fp_up = esprit_extract_direct_path(if_up, config.FS_STANDARD)
    fp_down = esprit_extract_direct_path(if_down, config.FS_STANDARD)

    r_est = dsc_fmcw_distance(fp_up, fp_down, config.C_SOUND, config.T_CHIRP, config.B_CHIRP)
    v_est = dsc_fmcw_radial_velocity(fp_up, fp_down, config.C_SOUND, config.F0_UP, config.B_CHIRP)
    return r_est, v_est


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument("--seed", type=int, default=config.SIM_SEED,
                        help=f"随机种子，用于复现多径衰减与底噪 (默认: {config.SIM_SEED})")
    args = parser.parse_args()
    config.ensure_dirs(config.RESULTS_DIR)

    print("=" * 68)
    print(" 实验二：全链路物理仿真验证")
    print("=" * 68)
    print(f" 基站布局 : BS1 {tuple(map(float, config.ANCHORS[0]))} / "
          f"BS2 {tuple(map(float, config.ANCHORS[1]))} (m)")
    print(f" 真实坐标 : {tuple(map(float, config.SIM_TRUE_POSITION))} m")
    print(f" 真实速度 : {tuple(map(float, config.SIM_TRUE_VELOCITY))} m/s")
    print(f" 波形参数 : T = {config.T_CHIRP * 1e3:.0f} ms, B = {config.B_CHIRP / 1e3:.0f} kHz, "
          f"f0 = {config.F0_UP / 1e3:.0f}/{config.F0_DOWN / 1e3:.0f} kHz")
    print(f" 随机种子 : {args.seed}\n")

    rng = np.random.default_rng(args.seed)
    t = np.arange(0, config.T_CHIRP, 1 / config.FS_STANDARD)
    tx_up, tx_down = make_templates(t)

    anchors = config.ANCHORS
    true_position = config.SIM_TRUE_POSITION
    true_velocity = config.SIM_TRUE_VELOCITY

    # --- 各基站的真实距离与径向速度（几何真值） ---------------------------
    true_r, true_vr = [], []
    for anchor in anchors:
        d = true_position - anchor
        r = float(np.linalg.norm(d))
        true_r.append(r)
        # 径向速度 = 绝对速度在"目标指向基站"方向上的投影的负值
        true_vr.append(float(-np.dot(true_velocity, d) / r))

    # --- 逐基站解算 -------------------------------------------------------
    r_est, v_est = [], []
    for i in range(len(anchors)):
        r, v = run_anchor(t, true_r[i], true_vr[i], tx_up, tx_down, rng)
        r_est.append(r)
        v_est.append(v)
        print(f"基站 {i + 1} 信号处理完毕:")
        print(f"  -> 真实距离: {true_r[i]:.4f} m | 测算距离: {r:.4f} m")
        print(f"  -> 真实径向: {true_vr[i]:.4f} m/s | 测算径向: {v:.4f} m/s\n")

    # --- 几何融合：双圆交点定位 + 方向投影测速 ----------------------------
    candidates = solve_2d_position_2anchors(anchors[0], anchors[1], r_est[0], r_est[1])
    final_xy = select_valid_solution(candidates, anchors, y_positive=True)
    if final_xy is None:
        print("定位失败：两圆无交点，几何不成立。")
        sys.exit(1)

    print(f"根据场地边界约束，锁定真实坐标为: X = {final_xy[0]:.4f} m, Y = {final_xy[1]:.4f} m")
    print(f"   [真实坐标: X = {true_position[0]:.4f} m, Y = {true_position[1]:.4f} m]\n")

    final_v = solve_2d_velocity(anchors, final_xy, v_est)
    print(f"绝对速度矢量评估: Vx = {final_v[0]:.4f} m/s, Vy = {final_v[1]:.4f} m/s")
    print(f"   [真实速度: Vx = {true_velocity[0]:.4f} m/s, Vy = {true_velocity[1]:.4f} m/s]\n")

    # --- 误差汇总 ---------------------------------------------------------
    dx = final_xy[0] - true_position[0]
    dy = final_xy[1] - true_position[1]
    pos_error = np.hypot(dx, dy)
    true_range = np.linalg.norm(true_position)
    est_range = np.linalg.norm(final_xy)
    speed_error = np.linalg.norm(final_v - true_velocity)

    print("-" * 68)
    print("误差汇总")
    print("-" * 68)
    print(f" 定位绝对误差   : ΔX = {abs(dx) * 100:.2f} cm, ΔY = {abs(dy) * 100:.2f} cm")
    print(f" 综合距离误差   : ΔP = {pos_error:.4f} m "
          f"(相对 {pos_error / true_range * 100:.2f}%)")
    print(f" 合速度大小     : 真实 {np.linalg.norm(true_velocity):.4f} m/s | "
          f"测算 {np.linalg.norm(final_v):.4f} m/s")
    print(f" 绝对速度偏差   : ΔV = {speed_error:.4f} m/s "
          f"(相对 {speed_error / np.linalg.norm(true_velocity) * 100:.2f}%)")
    print("-" * 68)
    print("\n注：接收信号包含随机多径衰减与高斯底噪，"
          "每次运行的数值会有微小浮动（论文报告: ΔX=1.51cm, ΔY=0.89cm, ΔV=0.1517m/s）。")


if __name__ == "__main__":
    main()
