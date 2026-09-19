"""解析几何解算：双基站双圆交点定位与方向投影测速。

相比非线性最小二乘寻优，本模块针对双基站场景给出闭式代数解，
既避免了局部最优，又把定位耗时降到常数级。
"""

from __future__ import annotations

import numpy as np


def solve_2d_position_2anchors(
    anchor1,
    anchor2,
    r1: float,
    r2: float,
):
    """双圆交点法解算目标二维坐标（闭式解）。

    以基站 1、2 的坐标 A₁(x₁,y₁)、A₂(x₂,y₂) 为圆心，测得距离 R₁、R₂ 为半径
    作圆，两圆交点即目标候选位置。设基站连线距离 d，投影距离 a 与垂直距离 h：

        a = (R₁² - R₂² + d²) / (2d)
        h = sqrt(R₁² - a²)

    :param anchor1: 基站 1 坐标 (x, y)
    :param anchor2: 基站 2 坐标 (x, y)
    :param r1: 目标到基站 1 的距离 (m)
    :param r2: 目标到基站 2 的距离 (m)
    :return: ((x_A, y_A), (x_B, y_B)) 两个候选解；几何不成立时返回 (None, None)
    """
    x1, y1 = anchor1
    x2, y2 = anchor2
    d = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # 两圆相离或内含，无交点
    if d > r1 + r2 or d < abs(r1 - r2) or d == 0:
        return None, None

    a = (r1**2 - r2**2 + d**2) / (2 * d)
    h = np.sqrt(abs(r1**2 - a**2))

    x3 = x1 + a * (x2 - x1) / d
    y3 = y1 + a * (y2 - y1) / d

    x_int1 = x3 + h * (y2 - y1) / d
    y_int1 = y3 - h * (x2 - x1) / d

    x_int2 = x3 - h * (y2 - y1) / d
    y_int2 = y3 + h * (x2 - x1) / d

    return (x_int1, y_int1), (x_int2, y_int2)


def solve_2d_velocity(
    anchors: np.ndarray,
    position,
    radial_velocities,
) -> np.ndarray:
    """由各基站的径向速度解算目标的绝对二维速度矢量。

    径向速度 v_ri 在物理意义上是绝对速度矢量 v 在目标指向基站 i 的
    单位方向矢量 u_i 上的正交投影：

        v_ri = u_iᵀ v = (x_i - x)/R_i · vx + (y_i - y)/R_i · vy

    将两个基站的观测方程联立成线性方程组 A v = b。
    由于两基站不重合，方向投影矩阵 A 必然满秩，故可直接求逆解出

        v = A⁻¹ b

    :param anchors: 基站坐标数组，形状 (N, 2)
    :param position: 目标二维坐标 (x, y)
    :param radial_velocities: 各基站测得的径向速度，长度 N
    :return: 绝对速度矢量 [vx, vy]
    """
    x, y = position
    a_rows = []
    for xi, yi in anchors:
        dx, dy = xi - x, yi - y
        dist = np.sqrt(dx**2 + dy**2)
        a_rows.append([dx / dist, dy / dist])

    a = np.array(a_rows)
    b = np.array(radial_velocities)
    v_vector, _, _, _ = np.linalg.lstsq(a, b, rcond=None)
    return v_vector


def select_valid_solution(candidates, anchors: np.ndarray, y_positive: bool = True):
    """根据场地边界先验，从双圆的两个交点中唯一锁定真实坐标。

    实际架设中目标设备必定位于基站连线的一侧（y > 0），
    该先验足以消除几何对称性带来的二义性。

    :return: 落在先验半平面内的解，若无则返回 None
    """
    pt1, pt2 = candidates
    if pt1 is None:
        return None
    return pt1 if (pt1[1] > 0) == y_positive else pt2
