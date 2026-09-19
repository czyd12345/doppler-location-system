"""绘图样式与保存工具。

统一使用衬线字体与高分辨率导出，符合学术论文插图规范。
设置环境变量 ``ACOUSTIC_RADAR_SHOW=1`` 可在保存后同时弹出交互窗口。
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

if os.environ.get("ACOUSTIC_RADAR_SHOW") != "1":
    matplotlib.use("Agg")  # 无显示环境下静默出图

import matplotlib.pyplot as plt

# 学术图表全局配置（类似 Times New Roman 的衬线体，导出不模糊）
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
})

# 中小尺寸单栏插图（论文图 1、图 2）
SMALL_FIGSIZE = (3.5, 2.6)
# 大尺寸轨迹图（论文图 4）
LARGE_FIGSIZE = (10, 8)


def save_and_show(fig, path: Path) -> None:
    """按 300 dpi 保存图片，并按需弹出窗口。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  -> 图片已保存: {path}")
    if os.environ.get("ACOUSTIC_RADAR_SHOW") == "1":
        plt.show()
    plt.close(fig)
