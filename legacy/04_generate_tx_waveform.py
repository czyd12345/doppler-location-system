import numpy as np
import scipy.io.wavfile as wav

def generate_fmcw_chirp(f_start, f_end, T_duration, fs=48000):
    """
    生成单向扫频信号 (Chirp)
    """
    t = np.arange(0, T_duration, 1/fs)
    B = f_end - f_start
    # 相位积分公式：2*pi*(f_start*t + 0.5 * k * t^2)，其中扫频斜率 k = B/T
    phase = 2 * np.pi * (f_start * t + (B / (2 * T_duration)) * t**2)
    return np.cos(phase)

if __name__ == "__main__":
    # ==========================================
    # 1. 物理参数设定 
    # ==========================================
    fs = 48000
    T = 0.05          # 单次扫频周期 50ms
    B_chirp = 4000    # 绝对带宽 4000Hz

    f0_up = 4000      # 升频起点
    f1 = 8000         # 升降频交汇点
    f0_down = 12000   # 降频起点


    # ==========================================
    # 2. 生成单基站的并发信号 
    # ==========================================
    # 分别生成 50ms 的升频和降频
    chirp_up = generate_fmcw_chirp(f0_up, f1, T, fs)
    chirp_down = generate_fmcw_chirp(f0_down, f1, T, fs)

    # 物理融合：在空气中同时发送，除以 2.0 是为了防止两个波峰叠加时超出声卡动态范围导致削顶爆音
    tx_simultaneous = (chirp_up + chirp_down) / 2.0

    # 生成一段等长 (50ms) 的绝对静音数组
    silence = np.zeros_like(tx_simultaneous)

    # ==========================================
    # 3. 组装双基站复用块 (TDM, 100ms 为一个完整周期)
    # ==========================================
    # 左声道：接基站 1 喇叭 (前 50ms 满功率发射，后 50ms 不发声)
    left_channel = np.concatenate((tx_simultaneous, silence))

    # 右声道：接基站 2 喇叭 (前 50ms 不发声，后 50ms 满功率发射)
    right_channel = np.concatenate((silence, tx_simultaneous))

    # 将一维数组合并为标准的立体声二维矩阵 Shape: [N, 2]
    stereo_block = np.column_stack((left_channel, right_channel))

    # ==========================================
    # 4. 循环铺满，导出实战 WAV 文件
    # ==========================================
    # 我们将这个 100ms 的探测周期循环 100 次，总共生成 10 秒的测试音频
    repeat_times = 100
    stereo_sequence = np.tile(stereo_block, (repeat_times, 1))

    # 转换为 16-bit PCM 格式 (WAV 的标准格式)
    # 乘以 30000 是为了留出少许动态余量 (16-bit 最大值是 32767)
    wav_stereo = np.int16(stereo_sequence * 30000)

    # 写入文件
    filename = "tx_dual_BS_TDM_FDM.wav"
    wav.write(filename, fs, wav_stereo)
    
    print(f"生成完毕！文件已保存为: {filename}")
    print(f"   - 总时长: {repeat_times * 0.1:.1f} 秒")
    print(f"   - 采样率: {fs} Hz")
    print(f"   - 声道数: 2 (立体声)")