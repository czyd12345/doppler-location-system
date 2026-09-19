import numpy as np
from scipy.linalg import svd, pinv, eigvals
from scipy.signal import hilbert, butter, filtfilt
import scipy.io.wavfile as wav  # 务必在开头加上这个导入
import matplotlib.pyplot as plt
# ==========================================
# 模块 0：物理混频与低通滤波 (新增核心模块)
# ==========================================
def mix_and_lowpass(tx_signal, rx_signal, fs, cutoff_freq=2000):
    """
    将发送信号与接收信号混频，并进行低通滤波提取中频 (IF)
    :param tx_signal: 本地发送模板 (1D array)
    :param rx_signal: 麦克风接收到的信号 (1D array)
    :param fs: 采样率 (Hz)
    :param cutoff_freq: 低通滤波器截止频率 (Hz)，需大于最大可能距离产生的差频
    :return: 纯净的中频信号 y_if
    """
    # 1. 时域相乘 (混频 Mixer)：根据积化和差，会产生 (f_tx + f_rx) 和 (f_tx - f_rx)
    mixed_signal = tx_signal * rx_signal
    
    # 2. 设计巴特沃斯低通滤波器，滤除极高频的 (f_tx + f_rx) 和频分量
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
    
    # 3. 零相移滤波 (filtfilt 避免普通 filter 带来的额外群延迟，这对测距极其重要)
    y_if = filtfilt(b, a, mixed_signal)
    
    # 乘以 2 恢复原本的幅度 (cosA * cosB = 0.5 * cos(A-B) + ...)
    return y_if * 2.0

# ==========================================
# 模块 0.5：真实声学物理信号发生器 (模拟真实环境)
# ==========================================
def generate_physical_signals(t, f_start, B_sweep, T, R, v, c, multipath_delays=[]):
    """生成真实的 Tx 模板和含有多径、多普勒频移的 Rx 回声"""
    # 1. 生成发送信号 Tx
    phase_tx = 2 * np.pi * (f_start * t + (B_sweep / (2 * T)) * t**2)
    tx_signal = np.cos(phase_tx)
    
    rx_signal = np.zeros_like(t)
    
    # 直射波路径 (绝对距离 R) 和反射波路径 (R + 额外路程)
    paths = [(R, 1.0)] # (distance, amplitude)
    for dR in multipath_delays:
        paths.append((R + dR, np.random.uniform(0.4, 0.8))) # 随机衰减的多径
        
    # 2. 生成接收信号 Rx (应用严格的物理延迟方程)
    for dist, amp in paths:
        # 动态物理延迟: tau(t) = (R - v*t) / c
        # (v > 0 代表目标正在靠近，距离随时间缩短)
        tau = (dist - v * t) / c
        t_delayed = t - tau
        
        # 将延迟后的时间代入相位方程，极其精确地还原多普勒效应与距离耦合
        phase_rx = 2 * np.pi * (f_start * t_delayed + (B_sweep / (2 * T)) * t_delayed**2)
        rx_signal += amp * np.cos(phase_rx)
        
    # 加入环境底噪
    rx_signal += np.random.normal(0, 0.05, len(t))
    
    return tx_signal, rx_signal

# ==========================================
# 模块 1：基于 Z 变换 (ESPRIT) 的抗多径频率提取
# ==========================================
def esprit_extract_direct_path(y_if, fs):
    N = len(y_if)
    y_complex = hilbert(y_if)
    L = N // 3 
    X = np.zeros((L, N - L + 1), dtype=complex)
    for i in range(L):
        X[i, :] = y_complex[i : i + N - L + 1]
        
    U, S, Vh = svd(X, full_matrices=False)
    
    threshold = 0.1 * S[0]
    num_sources = np.sum(S > threshold) 
    num_sources = max(1, num_sources) 
    
    Us = U[:, :num_sources] 
    
    U1 = Us[:-1, :] 
    U2 = Us[1:, :]  
    
    Phi = pinv(U1) @ U2
    lambdas = eigvals(Phi)
    
    angles = np.angle(lambdas)
    freqs = angles * fs / (2 * np.pi)
    freqs = np.abs(freqs) 
    
    f_direct = np.min(freqs)
    return f_direct

# ==========================================
# 模块 2：DSC-FMCW 多普勒补偿与绝对距离计算
# ==========================================
def dsc_fmcw_distance(fp_up, fp_down, c, T, B):
    R = ((fp_down + fp_up) * c * T) / (2 * B)
    return R

def dsc_fmcw_radial_velocity(fp_up, fp_down, c, f0, B):
    v_radial = (-fp_up+fp_down) * c / (2 * (f0 + B))
    return v_radial

# ==========================================
# 模块 3：双基站解析几何求交点 & 速度解算
# ==========================================
def solve_2d_position_2anchors(anchor1, anchor2, r1, r2):
    x1, y1 = anchor1
    x2, y2 = anchor2
    d = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
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

def solve_2d_velocity(anchors, position, radial_velocities):
    x, y = position
    A = []
    for i in range(len(anchors)):
        xi, yi = anchors[i]
        dx = xi - x
        dy = yi - y
        dist = np.sqrt(dx**2 + dy**2)
        A.append([dx / dist, dy / dist])
        
    A = np.array(A)
    b = np.array(radial_velocities)
    v_vector, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    return v_vector

if __name__ == "__main__":
    c_sound = 340.0       
    T_chirp = 0.05        
    B_chirp = 4000        
    
    f0_up = 4000
    f0_down = f0_up + 2 * B_chirp  # 12000 Hz
    
    # 基站坐标
    anchors = np.array([[0.0, 0.0], [0.5, 0.0]])
    
    print("=== 开始读取音频进行连续轨迹与速度解算 ===\n")
    
    RX_FILE = "rx_field_test_raw.wav" 
    try:
        fs, rx_audio_full = wav.read(RX_FILE)
        print(f"成功加载音频: {RX_FILE} (采样率: {fs} Hz)")
    except FileNotFoundError:
        print(f"找不到文件 {RX_FILE}")
        exit()

    rx_audio_full = rx_audio_full.astype(np.float64)
    rx_audio_full = rx_audio_full / np.max(np.abs(rx_audio_full))

    # ==========================================
    # 连续处理参数设置
    # ==========================================
    offset_sec = 0.5  # 跳过开头的唤醒静音时间
    start_idx = int(offset_sec * fs)
    frame_len = int(2 * T_chirp * fs)  # 每次步进 100ms
    
    total_frames = (len(rx_audio_full) - start_idx) // frame_len
    print(f"录音包含有效数据约 {total_frames * 0.1:.1f} 秒，共计 {total_frames} 个探测帧。\n")

    t = np.arange(0, T_chirp, 1/fs)
    tx_up_template = np.cos(2 * np.pi * (f0_up * t + (B_chirp / (2 * T_chirp)) * t**2))
    tx_down_template = np.cos(2 * np.pi * (f0_down * t + (-B_chirp / (2 * T_chirp)) * t**2))


    trajectory_x = []
    trajectory_y = []
    trajectory_vx = [] # 保存 Vx
    trajectory_vy = [] # 保存 Vy
    valid_times = []

    # ==========================================
    # 滑动窗口循环解算
    # ==========================================
    TARGET_POINTS = 10       # 目标计算的点数
    valid_points_count = 0   # 已成功解算的有效点计数器

    for i in range(total_frames):
        current_idx = start_idx + i * frame_len
        rx_100ms_period = rx_audio_full[current_idx : current_idx + frame_len]
        
        half_len = int(T_chirp * fs) 
        rx_from_BS1 = rx_100ms_period[0 : half_len]
        rx_from_BS2 = rx_100ms_period[half_len : 2 * half_len]
        safe_length = int(half_len * 0.65) 

        # --- 解算基站 1 ---
        if_up_dirty_1 = mix_and_lowpass(tx_up_template, rx_from_BS1, fs)
        if_down_dirty_1 = mix_and_lowpass(tx_down_template, rx_from_BS1, fs)
        fp_up_1 = esprit_extract_direct_path(if_up_dirty_1[:safe_length], fs)
        fp_down_1 = esprit_extract_direct_path(if_down_dirty_1[:safe_length], fs)
        R1 = dsc_fmcw_distance(fp_up_1, fp_down_1, c_sound, T_chirp, B_chirp)
        v_r1 = dsc_fmcw_radial_velocity(fp_up_1, fp_down_1, c_sound, f0_up, B_chirp)

        # --- 解算基站 2 ---
        if_up_dirty_2 = mix_and_lowpass(tx_up_template, rx_from_BS2, fs)
        if_down_dirty_2 = mix_and_lowpass(tx_down_template, rx_from_BS2, fs)
        fp_up_2 = esprit_extract_direct_path(if_up_dirty_2[:safe_length], fs)
        fp_down_2 = esprit_extract_direct_path(if_down_dirty_2[:safe_length], fs)
        R2 = dsc_fmcw_distance(fp_up_2, fp_down_2, c_sound, T_chirp, B_chirp)
        v_r2 = dsc_fmcw_radial_velocity(fp_up_2, fp_down_2, c_sound, f0_up, B_chirp)

        # --- 坐标与速度融合 ---
        pt1, pt2 = solve_2d_position_2anchors(anchors[0], anchors[1], R1, R2)
        
        if pt1 is not None:
            final_xy = pt1 if pt1[1] > 0 else pt2
            
            # 过滤飞点保护机制 (假设测试场在 10x10 米内)
            if 0 < final_xy[0] < 10 and 0 < final_xy[1] < 10:
                final_v_xy = solve_2d_velocity(anchors, final_xy, [v_r1, v_r2])
                
                trajectory_x.append(final_xy[0])
                trajectory_y.append(final_xy[1])
                trajectory_vx.append(final_v_xy[0])
                trajectory_vy.append(final_v_xy[1])
                valid_times.append(offset_sec + i * 0.1)

                valid_points_count += 1
                print(f"成功提取第 {valid_points_count}/{TARGET_POINTS} 个点: X={final_xy[0]:.2f}m, Y={final_xy[1]:.2f}m, Vx={final_v_xy[0]:.2f}m/s")
        
        if valid_points_count >= TARGET_POINTS:
            print("已收集到所需的 10 个有效轨迹点，提前结束 DSP 运算！")
            break

    print("\n 解算完成！开始绘制带速度矢量的轨迹图...")

    # ==========================================
    # 绘制二维空间运动轨迹 (含速度场)
    # ==========================================
    plt.figure(figsize=(10, 8))
    
    # 1. 画基站
    plt.scatter(anchors[:, 0], anchors[:, 1], c='red', marker='^', s=200, label='Base Stations', zorder=5)
    plt.text(anchors[0, 0], anchors[0, 1] - 0.15, 'BS1', fontsize=12, ha='center', color='red')
    plt.text(anchors[1, 0], anchors[1, 1] - 0.15, 'BS2', fontsize=12, ha='center', color='red')

    if len(trajectory_x) > 0:
        # 2. 画轨迹连线
        plt.plot(trajectory_x, trajectory_y, '-o', color='blue', alpha=0.4, markersize=4, label='Path')
        
        # ----------------------------------------------------
        # 3：使用 quiver 绘制速度箭头
        # ----------------------------------------------------
        # scale_units='xy', angles='xy' 确保箭头的方向和物理坐标系完全一致
        # scale 是缩放系数：如果觉得箭头太短，把 scale 调小(如 0.5)；如果箭头太长，把 scale 调大(如 2)
        plt.quiver(trajectory_x, trajectory_y, trajectory_vx, trajectory_vy, 
                   angles='xy', scale_units='xy', scale=20, 
                   color='darkorange', width=0.005, headwidth=4, alpha=0.9, label='Velocity Vector')
        
        # 3. 标注起点和终点
        plt.scatter(trajectory_x[0], trajectory_y[0], c='green', s=120, label='Start', zorder=4)
        plt.scatter(trajectory_x[-1], trajectory_y[-1], c='purple', s=120, label='End', zorder=4)
    else:
        print("警告：没有解算出有效的轨迹点。")

    # 样式整理
    plt.title('Acoustic Radar Tracking with Velocity Vectors', fontsize=16)
    plt.xlabel('X Coordinate (m)', fontsize=14)
    plt.ylabel('Y Coordinate (m)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')
    plt.axis('equal')  # 极其重要：确保 X 和 Y 比例 1:1
    
    if len(trajectory_x) > 0:
        plt.xlim(min(min(trajectory_x), anchors[0][0]) - 1, max(max(trajectory_x), anchors[1][0]) + 1)
        plt.ylim(min(0, min(trajectory_y)) - 1, max(trajectory_y) + 1)

    plt.show()