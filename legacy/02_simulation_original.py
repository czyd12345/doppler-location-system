import numpy as np
from scipy.linalg import svd, pinv, eigvals
from scipy.signal import hilbert, butter, filtfilt

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
    num_sources = max(1, num_sources) # 兜底，至少提取 1 个
    
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
    fs = 48000            
    T_chirp = 0.05        
    B_chirp = 4000        
    
    f0_up = 4000
    f0_down = f0_up + 2 * B_chirp  # 12000 Hz
    
    
    anchors = np.array([[0.0, 0.0], [0.5, 0.0]])
    
    true_position = np.array([0.21, 0.82]) 
    true_velocity = np.array([0.5, -0.3]) 
    
    t = np.arange(0, T_chirp, 1/fs)
    

    tx_up_template = np.cos(2 * np.pi * (f0_up * t + (B_chirp / (2 * T_chirp)) * t**2))
    tx_down_template = np.cos(2 * np.pi * (f0_down * t + (-B_chirp / (2 * T_chirp)) * t**2))

    dx1 = true_position[0] - anchors[0][0]
    dy1 = true_position[1] - anchors[0][1]
    true_R1 = np.sqrt(dx1**2 + dy1**2)
    true_vr1 = - (true_velocity[0]*dx1 + true_velocity[1]*dy1) / true_R1
    multipath = [0.1, 0.2] # 模拟两个反射多径
    
    _, rx_up_1 = generate_physical_signals(t, f0_up, B_chirp, T_chirp, true_R1, true_vr1, c_sound, multipath)
    _, rx_down_1 = generate_physical_signals(t, f0_down, -B_chirp, T_chirp, true_R1, true_vr1, c_sound, multipath)
    rx_air_BS1 = (rx_up_1 + rx_down_1) / 2.0 # 基站 1 制造的空气总声波
    
    # 2. 模拟后 50ms：基站 2 在发声
    dx2 = true_position[0] - anchors[1][0]
    dy2 = true_position[1] - anchors[1][1]
    true_R2 = np.sqrt(dx2**2 + dy2**2)
    true_vr2 = - (true_velocity[0]*dx2 + true_velocity[1]*dy2) / true_R2
    
    _, rx_up_2 = generate_physical_signals(t, f0_up, B_chirp, T_chirp, true_R2, true_vr2, c_sound, multipath)
    _, rx_down_2 = generate_physical_signals(t, f0_down, -B_chirp, T_chirp, true_R2, true_vr2, c_sound, multipath)
    rx_air_BS2 = (rx_up_2 + rx_down_2) / 2.0 # 基站 2 制造的空气总声波

    # 3. 麦克风录制到的完整 100ms 音频数据
    rx_100ms_period = np.concatenate((rx_air_BS1, rx_air_BS2))


    half_len = int(T_chirp * fs) 

    # 步骤 1：时域物理切片 (解复用)
    rx_from_BS1 = rx_100ms_period[0 : half_len]
    rx_from_BS2 = rx_100ms_period[half_len : 2 * half_len]

    # 步骤 2：基站 1 独立解算 (截断法防串扰)
    if_up_dirty_1 = mix_and_lowpass(tx_up_template, rx_from_BS1, fs)
    if_down_dirty_1 = mix_and_lowpass(tx_down_template, rx_from_BS1, fs)

    safe_length = int(half_len) 
    fp_up_1 = esprit_extract_direct_path(if_up_dirty_1[:safe_length], fs)
    fp_down_1 = esprit_extract_direct_path(if_down_dirty_1[:safe_length], fs)

    R1 = dsc_fmcw_distance(fp_up_1, fp_down_1, c_sound, T_chirp, B_chirp)
    v_r1 = dsc_fmcw_radial_velocity(fp_up_1, fp_down_1, c_sound, f0_up, B_chirp)

    # 步骤 3：基站 2 独立解算
    if_up_dirty_2 = mix_and_lowpass(tx_up_template, rx_from_BS2, fs)
    if_down_dirty_2 = mix_and_lowpass(tx_down_template, rx_from_BS2, fs)

    fp_up_2 = esprit_extract_direct_path(if_up_dirty_2[:safe_length], fs)
    fp_down_2 = esprit_extract_direct_path(if_down_dirty_2[:safe_length], fs)

    R2 = dsc_fmcw_distance(fp_up_2, fp_down_2, c_sound, T_chirp, B_chirp)
    v_r2 = dsc_fmcw_radial_velocity(fp_up_2, fp_down_2, c_sound, f0_up, B_chirp)

    # 打印单基站指标
    print(f"基站 1 信号处理完毕:")
    print(f"  -> 真实距离: {true_R1:.4f} m | 测算距离: {R1:.4f} m")
    print(f"  -> 真实径向: {true_vr1:.4f} m/s | 测算径向: {v_r1:.4f} m/s\n")
    
    print(f"基站 2 信号处理完毕:")
    print(f"  -> 真实距离: {true_R2:.4f} m | 测算距离: {R2:.4f} m")
    print(f"  -> 真实径向: {true_vr2:.4f} m/s | 测算径向: {v_r2:.4f} m/s\n")

    # 步骤 4：融合解算绝对坐标与速度
    pt1, pt2 = solve_2d_position_2anchors(anchors[0], anchors[1], R1, R2)
    
    if pt1 is not None:
        final_xy = pt1 if pt1[1] > 0 else pt2
        print(f"根据场地边界约束，锁定真实坐标为: X = {final_xy[0]:.4f} m, Y = {final_xy[1]:.4f} m")
        print(f"   [真实坐标: X = {true_position[0]:.4f} m, Y = {true_position[1]:.4f} m]\n")
        
        final_v_xy = solve_2d_velocity(anchors, final_xy, [v_r1, v_r2])
        print(f" 绝对速度矢量评估: Vx = {final_v_xy[0]:.4f} m/s, Vy = {final_v_xy[1]:.4f} m/s")
        print(f"   [真实速度: Vx = {true_velocity[0]:.4f} m/s, Vy = {true_velocity[1]:.4f} m/s]")
    else:
        print(" 定位失败，几何不成立。")