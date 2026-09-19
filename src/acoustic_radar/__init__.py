"""声学 FMCW 二维定位与测速系统 —— Python 参考实现。

模块划分：
    config      全局参数（物理常数、波形参数、几何布局）
    generation  发射波形合成（Chirp、双基站 TDM/FDM 复用）
    processing  信号处理链（混频低通、ESPRIT 超分辨频率提取、DSC-FMCW 解算）
    geometry    解析几何解算（双圆交点定位、方向投影测速）
"""

__all__ = ["config", "generation", "processing", "geometry"]
