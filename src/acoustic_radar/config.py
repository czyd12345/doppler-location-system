"""全局配置：物理常数、雷达波形参数、几何布局与路径管理。

所有可调参数集中在此处，脚本与算法模块均从这里读取，
修改后无需改动任何算法代码即可复现不同实验场景。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# ==========================================================================
# 路径管理
# ==========================================================================
ROOT = Path(__file__).resolve().parents[2]   # 仓库根目录（config.py 位于 src/acoustic_radar/）
DATA_DIR = ROOT / "data"          # 输入音频
RESULTS_DIR = ROOT / "results"    # 输出音频与图片

# 音频文件
TX_WAV = DATA_DIR / "tx_dual_BS_TDM_FDM.wav"   # 双基站 TDM/FDM 发射波形
SOUND_WAV = DATA_DIR / "sound.wav"             # 圆周运动实测录音
RX_FIELD_WAV = DATA_DIR / "rx_field_test_raw.wav"  # 双基站外场实测录音

# 图片输出
FIG_PITCH = RESULTS_DIR / "fig1_pitch_track.png"
FIG_ENERGY = RESULTS_DIR / "fig2_energy_envelope.png"
FIG_TRAJECTORY = RESULTS_DIR / "fig4_trajectory.png"


# ==========================================================================
# 物理常数
# ==========================================================================
C_SOUND = 340.0        # 声速 (m/s)，20°C 空气中的典型值
FS_STANDARD = 48000    # 商用声卡标准采样率 (Hz)


# ==========================================================================
# DSC-FMCW 波形参数
# --------------------------------------------------------------------------
# 关键设计约束：Downchirp 起始频率必须比 Upchirp 起始频率高出 2B，
# 即 f0_down = f0_up + 2B。此时升/降频差频的算术平均中，
# 多普勒项 (f0 - f0')v/c 与 (B + B')v/c 恰好代数抵消，只剩下距离项。
# 同时 12 kHz 的最高瞬时频率远低于声卡 20 kHz 抗混叠滤波转折点，
# 避免了高频段的幅频/相频失真。
# ==========================================================================
T_CHIRP = 0.05             # 单次扫频周期 (s)，50 ms
B_CHIRP = 4000.0           # 绝对扫频带宽 (Hz)
F0_UP = 4000.0             # Upchirp 起始频率 (Hz)，扫至 f0_up + B
F0_DOWN = F0_UP + 2 * B_CHIRP  # Downchirp 起始频率 (Hz) = 12000，扫至 f0_down - B

# 混频后低通截止频率 (Hz)，需大于最远距离产生的差频
LPF_CUTOFF = 2000.0


# ==========================================================================
# 几何布局（单位：米）
# --------------------------------------------------------------------------
# 两个静态基站（扬声器）沿 X 轴水平放置，间距 0.5 m。
# 目标（麦克风）位于 y > 0 一侧，该先验边界用于在双圆交点中锁定唯一解。
# ==========================================================================
ANCHORS = np.array([[0.0, 0.0], [0.5, 0.0]])

# 仿真场景中目标的真实状态（用于与解算结果对比）
SIM_TRUE_POSITION = np.array([0.21, 0.82])     # 真实二维坐标 (m)
SIM_TRUE_VELOCITY = np.array([0.5, -0.3])      # 真实速度矢量 (m/s)
SIM_MULTIPATH_DELAYS = [0.1, 0.2]              # 多径额外路程 (m)
SIM_NOISE_STD = 0.05                           # 环境底噪标准差
SIM_SEED = 202416010201                        # 随机种子，保证仿真结果可复现


# ==========================================================================
# 圆周运动参数估计（sound.wav）参数
# --------------------------------------------------------------------------
# 声源做匀速圆周运动，麦克风静止于圆心外距离 d 处。
# 对 1~3 kHz 频段的窄带信号做自相关基频追踪，得到多普勒频率轨迹 f(t)。
# ==========================================================================
PITCH_FMIN = 1000.0        # 带通下限 / 搜索基频下限 (Hz)
PITCH_FMAX = 3000.0        # 带通上限 / 搜索基频上限 (Hz)
PITCH_HOP = 512            # 分帧步进 (采样点)
PITCH_FRAME = 2048         # 分帧窗长 (采样点)
PITCH_PROMINENCE = 10.0    # 峰值检测的最小突出度 (Hz)

ENERGY_WINDOW_SEC = 0.02   # RMS 能量包络滑动窗长 (s)


# ==========================================================================
# 外场连续解算（rx_field_test_raw.wav）参数
# ==========================================================================
FRAME_SEC = 2 * T_CHIRP    # 每个 TDM 探测帧时长 (s)，100 ms
TRACK_OFFSET_SEC = 0.5     # 跳过录音开头的唤醒静音 (s)
SAFE_RATIO = 0.65          # 中频截断比例，防止 TDM 切换处的串扰污染频谱
TARGET_POINTS = 10         # 需要解算的有效轨迹点数
FIELD_X_LIMIT = (0.0, 10.0)  # 飞点过滤：有效场地范围 (m)
FIELD_Y_LIMIT = (0.0, 10.0)


# ==========================================================================
# 命令行参数
# ==========================================================================
def parse_args(description: str) -> argparse.Namespace:
    """为入口脚本提供统一的 `--data-dir` / `--results-dir` 参数。"""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--data-dir", type=Path, default=DATA_DIR,
        help=f"输入音频目录 (默认: {DATA_DIR})",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=RESULTS_DIR,
        help=f"输出目录 (默认: {RESULTS_DIR})",
    )
    return parser.parse_args()


def ensure_dirs(results_dir: Path) -> None:
    """确保输出目录存在。"""
    results_dir.mkdir(parents=True, exist_ok=True)
