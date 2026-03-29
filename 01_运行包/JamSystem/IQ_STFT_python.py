import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib

# 设置中文字体（可选）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def add_noise_with_fixed_isr(iq_signal, target_isr_db=30.0, rng=None):
    """
    Add complex white Gaussian noise with a fixed input ISR.

    Here ISR is treated as the power ratio between interference/noise and
    useful signal before any restoration:

        ISR(dB) = 10 * log10(P_noise / P_signal)

    So when target_isr_db is 30 dB, the generated noise power is scaled to be
    30 dB above the useful signal power.
    """
    iq_signal = np.asarray(iq_signal, dtype=np.complex64)
    if iq_signal.size == 0:
        return iq_signal

    if rng is None:
        rng = np.random.default_rng()

    signal_power = np.mean(np.abs(iq_signal) ** 2)
    target_noise_power = signal_power * (10 ** (target_isr_db / 10.0))

    noise = (
        rng.standard_normal(iq_signal.shape) +
        1j * rng.standard_normal(iq_signal.shape)
    ).astype(np.complex64)

    raw_noise_power = np.mean(np.abs(noise) ** 2) + 1e-12
    noise *= np.sqrt(target_noise_power / raw_noise_power)
    return iq_signal + noise


class IQSTFTAnalyzer:
    """IQ信号的STFT分析器"""

    def __init__(self, fs, nperseg=1024, noverlap=None, window='hann'):
        """
        初始化STFT分析器

        参数:
        fs: 采样率 (Hz)
        nperseg: 窗长度（点数）
        noverlap: 重叠长度（点数）
        window: 窗函数类型
        """
        self.fs = fs
        self.nperseg = nperseg
        self.noverlap = noverlap if noverlap is not None else nperseg // 2
        self.window = window
        self.hop_length = nperseg - self.noverlap

    def generate_test_iq_signal(self, n_samples=None, duration=None, f0=0, bandwidth=1000, mod_type='cw'):
        """
        生成测试IQ信号

        参数:
        n_samples: 采样点数（优先使用）
        duration: 信号时长 (秒)
        f0: 中心频率偏移 (Hz)
        bandwidth: 信号带宽 (Hz)
        mod_type: 调制类型 ('cw', 'qpsk', 'fsk', 'ofdm')

        返回:
        iq_signal: 复数IQ信号
        """
        # 确定采样点数
        if n_samples is not None:
            n = n_samples
        elif duration is not None:
            n = int(duration * self.fs)
        else:
            n = int(0.01 * self.fs)  # 默认10ms

        t = np.arange(n) / self.fs

        if mod_type == 'cw':
            # 连续波信号
            iq_signal = np.exp(1j * 2 * np.pi * f0 * t)

        elif mod_type == 'qpsk':
            # 简化版QPSK信号
            n_symbols = max(10, n // 100)  # 至少10个符号
            symbols = np.random.choice([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], n_symbols)

            # 脉冲成型
            sps = max(1, n // n_symbols)  # 每个符号的采样点数
            pulse = np.ones(sps)

            # 上采样和成型
            upsampled = np.zeros(n_symbols * sps, dtype=complex)
            upsampled[::sps] = symbols
            iq_signal = np.convolve(upsampled, pulse, mode='same')

            # 调整到指定长度
            if len(iq_signal) > n:
                iq_signal = iq_signal[:n]
            elif len(iq_signal) < n:
                pad_len = n - len(iq_signal)
                iq_signal = np.pad(iq_signal, (0, pad_len))

            # 添加频偏
            iq_signal *= np.exp(1j * 2 * np.pi * f0 * t[:len(iq_signal)])

        elif mod_type == 'fsk':
            # FSK信号
            freq_dev = bandwidth / 4
            n_symbols = max(10, n // 100)
            data = np.random.randint(0, 2, n_symbols)

            # 生成相位
            phase = np.zeros(n)
            symbol_len = n // n_symbols

            for i in range(n_symbols):
                start = i * symbol_len
                end = min(start + symbol_len, n)
                freq = f0 + freq_dev if data[i] == 1 else f0 - freq_dev
                phase[start:end] = 2 * np.pi * freq * t[start:end]

            iq_signal = np.exp(1j * phase)

        elif mod_type == 'ofdm':
            # 简化OFDM信号
            n_subcarriers = 64
            cp_len = 16
            symbol_len = n_subcarriers + cp_len
            n_symbols = max(1, n // symbol_len)

            iq_signal = np.zeros(n, dtype=complex)

            for i in range(n_symbols):
                # 生成随机QPSK符号
                symbols = np.random.choice([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], n_subcarriers)

                # IFFT
                ofdm_symbol = np.fft.ifft(symbols, n_subcarriers)

                # 添加循环前缀
                ofdm_symbol_with_cp = np.concatenate([ofdm_symbol[-cp_len:], ofdm_symbol])

                # 放置到信号中
                start_idx = i * symbol_len
                end_idx = min(start_idx + len(ofdm_symbol_with_cp), n)

                if start_idx < n:
                    copy_len = min(len(ofdm_symbol_with_cp), n - start_idx)
                    iq_signal[start_idx:start_idx + copy_len] = ofdm_symbol_with_cp[:copy_len]

            # 添加频偏
            iq_signal *= np.exp(1j * 2 * np.pi * f0 * t)

        else:
            # 默认：带噪声的复正弦波
            iq_signal = np.exp(1j * 2 * np.pi * f0 * t)

        # 添加固定 ISR 的白噪声，满足“原始加噪信号 ISR 固定为 30 dB”。
        iq_signal = add_noise_with_fixed_isr(iq_signal, target_isr_db=30.0)

        return iq_signal

    def stft_analysis(self, iq_signal, mode='magnitude'):
        """
        对IQ信号进行STFT分析

        参数:
        iq_signal: 复数IQ信号
        mode: 分析模式 ('magnitude', 'phase', 'instant_freq', 'all')

        返回:
        分析结果字典
        """
        # 使用scipy的STFT（支持复数信号）
        f, t_stft, Zxx = signal.stft(iq_signal, self.fs,
                                     window=self.window,
                                     nperseg=self.nperseg,
                                     noverlap=self.noverlap,
                                     return_onesided=False)

        # 重新排列频率轴（从负频率到正频率）
        Zxx = np.fft.fftshift(Zxx, axes=0) #用于将频谱的零频分量从中心移回到数组的起始位置
        f = np.fft.fftshift(f)

        # 计算各种特征
        magnitude = np.abs(Zxx)
        phase = np.angle(Zxx)

        # 计算瞬时频率（通过相位差分）
        if mode == 'all' or mode == 'instant_freq':
            phase_unwrapped = np.unwrap(phase, axis=0) #用于解决相位不连续性问题，使相位数据连续化
            instant_freq = np.diff(phase_unwrapped, axis=0) * self.fs / (2 * np.pi * (f[1] - f[0])) #np.diff沿给定轴计算离散差值，默认n=1，计算一阶差分
        else:
            instant_freq = None

        results = {
            'frequencies': f,
            'times': t_stft,
            'stft_matrix': Zxx,
            'magnitude': magnitude,
            'phase': phase,
            'instant_freq': instant_freq,
            'magnitude_db': 10 * np.log10(magnitude + 1e-10)
        }

        return results

    def plot_iq_stft(self, iq_signal, results, title="IQ信号STFT分析"):
        """
        绘制IQ信号的STFT分析结果

        参数:
        iq_signal: 原始IQ信号
        results: STFT分析结果
        title: 图标题
        """
        f = results['frequencies']
        t_stft = results['times']

        fig = plt.figure(figsize=(16, 10))

        # 1. 原始IQ信号（时域）
        ax1 = plt.subplot(3, 3, 1)
        n_show = min(1000, len(iq_signal))
        ax1.plot(np.real(iq_signal)[:n_show], label='I路', alpha=0.7, linewidth=0.5)
        ax1.plot(np.imag(iq_signal)[:n_show], label='Q路', alpha=0.7, linewidth=0.5)
        ax1.set_title('原始IQ信号（前{}点）'.format(n_show))
        # ax1.set_xlabel('采样点')
        # ax1.set_ylabel('幅度')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # 2. IQ平面（星座图）
        ax2 = plt.subplot(3, 3, 2)
        step = max(1, len(iq_signal) // 1000)  # 最多显示1000个点
        ax2.scatter(np.real(iq_signal)[::step], np.imag(iq_signal)[::step],
                    s=1, alpha=0.5, c='blue')
        ax2.set_title('IQ平面（星座图）')
        ax2.set_xlabel('I分量')
        ax2.set_ylabel('Q分量')
        ax2.axis('equal')
        ax2.grid(True, alpha=0.3)

        # 3. 幅度谱（STFT）
        ax3 = plt.subplot(3, 3, 3)
        magnitude_mean = np.mean(results['magnitude_db'], axis=1)
        ax3.plot(f, magnitude_mean, linewidth=0.5)
        ax3.set_title('平均频谱')
        ax3.set_xlabel('频率 (Hz)')
        ax3.set_ylabel('幅度 (dB)')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim([-self.fs / 2, self.fs / 2])

        # 4. STFT幅度谱图
        ax4 = plt.subplot(3, 2, 3)
        im1 = ax4.pcolormesh(t_stft, f, results['magnitude_db'],
                             shading='gouraud', cmap='viridis',
                             vmin=np.max(results['magnitude_db']) - 80,
                             vmax=np.max(results['magnitude_db']))
        ax4.set_title('STFT幅度谱')
        # ax4.set_xlabel('时间 (s)')
        ax4.set_ylabel('频率 (Hz)')
        plt.colorbar(im1, ax=ax4, label='幅度 (dB)')
        ax4.set_ylim([-self.fs / 4, self.fs / 4])  # 只显示中心频率附近
        # ax4.set_ylim([-500e3, 500e3])  # 只显示中心频率附近

        # 5. STFT相位谱图
        ax5 = plt.subplot(3, 2, 4)
        im2 = ax5.pcolormesh(t_stft, f, results['phase'],
                             shading='gouraud', cmap='hsv')
        ax5.set_title('STFT相位谱')
        ax5.set_xlabel('时间 (s)')
        ax5.set_ylabel('频率 (Hz)')
        plt.colorbar(im2, ax=ax5, label='相位 (rad)')
        ax5.set_ylim([-self.fs / 4, self.fs / 4])

        # 6. 瞬时频率
        ax6 = plt.subplot(3, 2, 5)
        if results['instant_freq'] is not None and results['instant_freq'].size > 0:
            instant_freq_mean = np.mean(results['instant_freq'], axis=1)
            ax6.plot(f[:-1], instant_freq_mean, linewidth=0.5)
            ax6.set_title('平均瞬时频率')
            ax6.set_xlabel('频率 (Hz)')
            ax6.set_ylabel('瞬时频率 (Hz)')
            ax6.grid(True, alpha=0.3)
            ax6.set_xlim([-self.fs / 4, self.fs / 4])
        else:
            ax6.text(0.5, 0.5, '瞬时频率未计算',
                     horizontalalignment='center', verticalalignment='center',
                     transform=ax6.transAxes)
            ax6.set_title('瞬时频率')
            ax6.axis('off')

        # 7. 信号统计信息
        ax7 = plt.subplot(3, 2, 6)
        ax7.axis('off')

        # 计算统计信息
        power = np.mean(np.abs(iq_signal) ** 2)
        mean_i = np.mean(np.real(iq_signal))
        mean_q = np.mean(np.imag(iq_signal))
        iq_balance = np.abs(mean_i - mean_q) / (np.abs(mean_i) + np.abs(mean_q) + 1e-10)

        stats_text = f"""
        信号统计信息:
        ---------------------------
        采样率: {self.fs / 1e6:.2f} MHz
        信号长度: {len(iq_signal)} 点
        时长: {len(iq_signal) / self.fs * 1000:.1f} ms

        平均功率: {10 * np.log10(power + 1e-10):.1f} dB
        I路均值: {mean_i:.4f}
        Q路均值: {mean_q:.4f}
        IQ不平衡: {iq_balance * 100:.2f}%

        STFT参数:
        窗长度: {self.nperseg} 点
        重叠: {self.noverlap} 点
        频率分辨率: {self.fs / self.nperseg:.1f} Hz
        时间分辨率: {self.hop_length / self.fs * 1000:.1f} ms
        """

        ax7.text(0, 1, stats_text, transform=ax7.transAxes,
                 fontsize=8, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle(title, fontsize=14, fontweight='bold')
        # plt.tight_layout()
        plt.show()

        return fig


# =============================
# 使用示例
# =============================

def example_1_basic_iq_stft():
    """示例1: 基本IQ信号STFT分析"""
    print("=" * 60)
    print("示例1: 基本IQ信号STFT分析")
    print("=" * 60)

    # 参数设置
    fs = 10e6  # 10 MHz采样率
    duration = 0.01  # 10 ms信号

    # 创建分析器
    analyzer = IQSTFTAnalyzer(fs=fs, nperseg=1024, noverlap=512)

    # 生成测试IQ信号
    iq_signal = analyzer.generate_test_iq_signal(
        duration=duration,
        f0=100e3,  # 100 kHz中心频率偏移
        bandwidth=1e6,  # 1 MHz带宽
        mod_type='qpsk'
    )

    # 进行STFT分析
    results = analyzer.stft_analysis(iq_signal, mode='all')

    # 绘制结果
    analyzer.plot_iq_stft(iq_signal, results, title="QPSK信号STFT分析")

    return analyzer, iq_signal, results


def example_2_simple_modulation_comparison():
    """示例2: 不同调制信号的STFT对比（简化版）"""
    print("\n" + "=" * 60)
    print("示例2: 不同调制信号的STFT对比")
    print("=" * 60)

    fs = 20e6  # 20 MHz采样率
    duration = 0.005  # 5 ms信号
    n_samples = int(fs * duration)

    mod_types = ['cw', 'qpsk', 'fsk']

    fig, axes = plt.subplots(len(mod_types), 3, figsize=(15, 3 * len(mod_types)))

    for idx, mod_type in enumerate(mod_types):
        # 创建分析器
        analyzer = IQSTFTAnalyzer(fs=fs, nperseg=512, noverlap=256)

        # 生成信号
        iq_signal = analyzer.generate_test_iq_signal(
            n_samples=n_samples,
            f0=200e3,
            bandwidth=2e6,
            mod_type=mod_type
        )

        # STFT分析
        results = analyzer.stft_analysis(iq_signal)

        # 绘制幅度谱
        ax1 = axes[idx, 0]
        im1 = ax1.pcolormesh(results['times'], results['frequencies'],
                             results['magnitude_db'], shading='gouraud',
                             cmap='viridis', vmin=np.max(results['magnitude_db']) - 60)
        ax1.set_title(f'{mod_type.upper()} - 幅度谱')
        ax1.set_ylabel('频率 (Hz)')
        if idx == len(mod_types) - 1:
            ax1.set_xlabel('时间 (s)')
        ax1.set_ylim([-fs / 4, fs / 4])

        # 绘制相位谱
        ax2 = axes[idx, 1]
        im2 = ax2.pcolormesh(results['times'], results['frequencies'],
                             results['phase'], shading='gouraud',
                             cmap='hsv')
        ax2.set_title(f'{mod_type.upper()} - 相位谱')
        ax2.set_ylabel('频率 (Hz)')
        if idx == len(mod_types) - 1:
            ax2.set_xlabel('时间 (s)')
        ax2.set_ylim([-fs / 4, fs / 4])

        # 绘制星座图
        ax3 = axes[idx, 2]
        step = max(1, len(iq_signal) // 500)
        ax3.scatter(np.real(iq_signal)[::step], np.imag(iq_signal)[::step],
                    s=1, alpha=0.5)
        ax3.set_title(f'{mod_type.upper()} - 星座图')
        ax3.set_xlabel('I分量')
        ax3.set_ylabel('Q分量')
        ax3.axis('equal')
        ax3.grid(True, alpha=0.3)

    plt.suptitle('不同调制信号的STFT对比分析', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def example_3_simple_real_time():
    """示例3: 简单实时STFT分析模拟"""
    print("\n" + "=" * 60)
    print("示例3: 简单实时STFT分析模拟")
    print("=" * 60)

    fs = 2e6  # 2 MHz采样率（降低以加快处理速度）
    duration = 0.05  # 50 ms信号
    n_samples = int(fs * duration)

    # 创建分析器
    analyzer = IQSTFTAnalyzer(fs=fs, nperseg=256, noverlap=128)

    # 生成随时间变化的信号
    t = np.arange(n_samples) / fs

    # 创建随时间变化的频率
    freq_profile = np.zeros(n_samples)

    # 第1段: 线性扫频 (0-20ms)
    seg1_end = n_samples // 3
    freq_profile[:seg1_end] = np.linspace(-500e3, 500e3, seg1_end) #linspace 函数用于生成等间隔的数值序列

    # 第2段: 固定频率 (20-40ms)
    seg2_end = 2 * n_samples // 3
    freq_profile[seg1_end:seg2_end] = 300e3

    # 第3段: 正弦调频 (40-50ms)
    freq_profile[seg2_end:] = 100e3 + 200e3 * np.sin(2 * np.pi * 5e3 * t[seg2_end:])

    # 生成IQ信号
    phase = 2 * np.pi * np.cumsum(freq_profile) / fs #numpy.cumsum() 用于计算数组元素在指定轴上的累积和，即依次将元素相加并返回中间结果。
    iq_signal = np.exp(1j * phase)

    # 添加固定 ISR 的白噪声，避免不同样本间噪声强度漂移。
    iq_signal = add_noise_with_fixed_isr(iq_signal, target_isr_db=30.0)

    # 模拟实时处理
    chunk_size = 1024  # 每次处理的点数
    n_chunks = min(10, n_samples // chunk_size)  # 只处理前10个chunk

    print(f"总采样点数: {n_samples}")
    print(f"数据块大小: {chunk_size}")
    print(f"处理数据块数: {n_chunks}")

    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size

        chunk = iq_signal[start_idx:end_idx]
        results = analyzer.stft_analysis(chunk)

        print(f"处理第{i}个数据块, 时间: {start_idx / fs * 1000:.1f}ms")

        # 绘制当前chunk的频谱
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # 时域信号
        t_chunk = np.arange(len(chunk)) / fs * 1000
        axes[0].plot(t_chunk, np.real(chunk), label='I路', alpha=0.7, linewidth=0.5)
        axes[0].plot(t_chunk, np.imag(chunk), label='Q路', alpha=0.7, linewidth=0.5)
        axes[0].set_title(f'Chunk {i} - 时域信号')
        axes[0].set_xlabel('时间 (ms)')
        axes[0].set_ylabel('幅度')
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        # 平均频谱
        axes[1].plot(results['frequencies'] / 1e3, np.mean(results['magnitude_db'], axis=1))
        axes[1].set_title(f'Chunk {i} - 平均频谱')
        axes[1].set_xlabel('频率 (kHz)')
        axes[1].set_ylabel('幅度 (dB)')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim([-fs / 2e3 / 4, fs / 2e3 / 4])

        # 时频谱
        im = axes[2].pcolormesh(results['times'] * 1000, results['frequencies'] / 1e3,
                                results['magnitude_db'], shading='gouraud',
                                cmap='viridis', vmin=np.max(results['magnitude_db']) - 60)
        axes[2].set_title(f'Chunk {i} - 时频谱')
        axes[2].set_xlabel('时间 (ms)')
        axes[2].set_ylabel('频率 (kHz)')
        plt.colorbar(im, ax=axes[2], label='幅度 (dB)')
        axes[2].set_ylim([-fs / 2e3 / 4, fs / 2e3 / 4])

        plt.suptitle(f'实时STFT分析 - 时间: {start_idx / fs * 1000:.1f}ms', fontweight='bold')
        plt.tight_layout()
        plt.show()

    # 对整个信号进行STFT分析
    print("\n对整个信号进行STFT分析...")
    full_results = analyzer.stft_analysis(iq_signal)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 频率随时间变化
    axes[0, 0].plot(t * 1000, freq_profile / 1e3)
    axes[0, 0].set_title('频率随时间变化')
    axes[0, 0].set_xlabel('时间 (ms)')
    axes[0, 0].set_ylabel('频率 (kHz)')
    axes[0, 0].grid(True, alpha=0.3)

    # 完整信号时频谱
    im = axes[0, 1].pcolormesh(full_results['times'] * 1000, full_results['frequencies'] / 1e3,
                               full_results['magnitude_db'], shading='gouraud',
                               cmap='viridis', vmin=np.max(full_results['magnitude_db']) - 60)
    axes[0, 1].set_title('完整信号STFT')
    axes[0, 1].set_xlabel('时间 (ms)')
    axes[0, 1].set_ylabel('频率 (kHz)')
    plt.colorbar(im, ax=axes[0, 1], label='幅度 (dB)')
    axes[0, 1].set_ylim([-fs / 2e3 / 4, fs / 2e3 / 4])

    # 特定时间的频谱
    time_idx = full_results['magnitude_db'].shape[1] // 2
    axes[1, 0].plot(full_results['frequencies'] / 1e3, full_results['magnitude_db'][:, time_idx])
    axes[1, 0].set_title(f'时间 {full_results["times"][time_idx] * 1000:.1f}ms 处的频谱')
    axes[1, 0].set_xlabel('频率 (kHz)')
    axes[1, 0].set_ylabel('幅度 (dB)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim([-fs / 2e3 / 4, fs / 2e3 / 4])

    # 星座图
    axes[1, 1].scatter(np.real(iq_signal)[::10], np.imag(iq_signal)[::10], s=1, alpha=0.5)
    axes[1, 1].set_title('完整信号星座图')
    axes[1, 1].set_xlabel('I分量')
    axes[1, 1].set_ylabel('Q分量')
    axes[1, 1].axis('equal')
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('实时STFT分析 - 完整信号', fontweight='bold')
    plt.tight_layout()
    plt.show()

    print(f"模拟完成，共处理{n_chunks}个数据块")
    return analyzer, iq_signal


def example_4_practical_application():
    """示例4: 实际应用 - 多信号检测"""
    print("\n" + "=" * 60)
    print("示例4: 实际应用 - 多信号检测")
    print("=" * 60)

    fs = 10e6  # 10 MHz采样率
    duration = 0.02  # 20 ms信号
    n_samples = int(fs * duration)

    analyzer = IQSTFTAnalyzer(fs=fs, nperseg=2048, noverlap=1024)

    # 创建多信号场景
    t = np.arange(n_samples) / fs

    # 创建干净的IQ信号（初始化为噪声）
    np.random.seed(42)  # 设置随机种子以便结果可重现
    iq_signal = (np.random.randn(n_samples) + 1j * np.random.randn(n_samples)) * 0.1

    # 添加多个信号
    # 信号1: QPSK信号 (中心在2MHz)
    sig1 = analyzer.generate_test_iq_signal(n_samples=n_samples, f0=2e6, bandwidth=1e6, mod_type='qpsk')
    iq_signal += sig1 * 0.5

    # 信号2: 连续波信号 (中心在-1MHz)
    sig2 = np.exp(1j * 2 * np.pi * (-1e6) * t)
    iq_signal += sig2 * 0.3

    # 信号3: 扫频信号 (从3MHz扫到4MHz)
    chirp_freq = 3e6 + 1e6 * t / duration
    phase_chirp = 2 * np.pi * np.cumsum(chirp_freq) / fs
    sig3 = np.exp(1j * phase_chirp)
    iq_signal += sig3 * 0.2

    # STFT分析
    results = analyzer.stft_analysis(iq_signal)

    # 绘制高级分析结果
    fig = plt.figure(figsize=(16, 10))

    # 1. 完整时频谱
    ax1 = plt.subplot(2, 3, 1)
    im1 = ax1.pcolormesh(results['times'], results['frequencies'] / 1e6,
                         results['magnitude_db'], shading='gouraud',
                         cmap='hot', vmin=np.max(results['magnitude_db']) - 70)
    ax1.set_title('多信号时频谱')
    ax1.set_xlabel('时间 (s)')
    ax1.set_ylabel('频率 (MHz)')
    plt.colorbar(im1, ax=ax1, label='幅度 (dB)')
    ax1.set_ylim([-fs / 2e6 / 2, fs / 2e6 / 2])

    # 2. 特定时间的频谱切片
    ax2 = plt.subplot(2, 3, 2)
    time_idx = results['magnitude_db'].shape[1] // 2
    ax2.plot(results['frequencies'] / 1e6, results['magnitude_db'][:, time_idx])
    ax2.set_title(f'时间 {results["times"][time_idx] * 1000:.1f}ms 处的频谱')
    ax2.set_xlabel('频率 (MHz)')
    ax2.set_ylabel('幅度 (dB)')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([-fs / 2e6 / 2, fs / 2e6 / 2])

    # 3. 信号检测结果
    ax3 = plt.subplot(2, 3, 3)
    ax3.axis('off')

    # 自动检测信号
    threshold_db = np.max(results['magnitude_db']) - 30
    signal_mask = results['magnitude_db'] > threshold_db

    detected_signals = []
    for col in range(signal_mask.shape[1]):
        col_mask = signal_mask[:, col]
        if np.any(col_mask):
            indices = np.where(col_mask)[0]
            # 找到连续的区域
            regions = []
            start_idx = indices[0]
            for i in range(1, len(indices)):
                if indices[i] - indices[i - 1] > 5:  # 不连续，开始新区域
                    regions.append((start_idx, indices[i - 1]))
                    start_idx = indices[i]
            regions.append((start_idx, indices[-1]))

            for start, end in regions:
                freq = np.mean(results['frequencies'][start:end + 1])
                time = results['times'][col]
                power = np.mean(results['magnitude_db'][start:end + 1, col])
                detected_signals.append((time, freq, power))

    # 去重和整理
    if detected_signals:
        # 按频率分组
        freq_groups = {}
        for time, freq, power in detected_signals:
            freq_rounded = round(freq / 1e6, 2)  # 以MHz为单位分组，保留2位小数
            key = f"{freq_rounded:.2f} MHz"
            if key not in freq_groups:
                freq_groups[key] = []
            freq_groups[key].append((time, power))

        info_text = "检测到的信号:\n"
        info_text += "=" * 30 + "\n"

        for freq_mhz, measurements in freq_groups.items():
            times = [m[0] for m in measurements]
            powers = [m[1] for m in measurements]
            info_text += f"\n{freq_mhz}:\n"
            info_text += f"  时间范围: {min(times) * 1000:.1f}-{max(times) * 1000:.1f} ms\n"
            info_text += f"  平均功率: {np.mean(powers):.1f} dB\n"
            info_text += f"  持续时间: {(max(times) - min(times)) * 1000:.1f} ms\n"
    else:
        info_text = "未检测到明显信号"

    ax3.text(0, 1, info_text, transform=ax3.transAxes,
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    # 4. 时域信号包络
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(t[:500] * 1000, np.abs(iq_signal[:500]))
    ax4.set_title('信号包络（前0.5ms）')
    ax4.set_xlabel('时间 (ms)')
    ax4.set_ylabel('幅度')
    ax4.grid(True, alpha=0.3)

    # 5. 相位变化
    ax5 = plt.subplot(2, 3, 5)
    phase = np.unwrap(np.angle(iq_signal[:500]))
    ax5.plot(t[:500] * 1000, phase)
    ax5.set_title('相位变化（前0.5ms）')
    ax5.set_xlabel('时间 (ms)')
    ax5.set_ylabel('相位 (rad)')
    ax5.grid(True, alpha=0.3)

    # 6. 功率谱密度
    ax6 = plt.subplot(2, 3, 6)
    f_psd, psd = signal.welch(iq_signal, fs, nperseg=1024)
    ax6.plot(f_psd / 1e6, 10 * np.log10(psd))
    ax6.set_title('功率谱密度 (Welch方法)')
    ax6.set_xlabel('频率 (MHz)')
    ax6.set_ylabel('PSD (dB/Hz)')
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim([-fs / 2e6 / 2, fs / 2e6 / 2])

    plt.suptitle('无线信号检测与参数估计', fontsize=14, fontweight='bold')
    # plt.tight_layout()
    plt.show()

    print("信号检测完成:")
    if detected_signals:
        print(f"共检测到 {len(freq_groups)} 个不同的信号频率")
        for freq_mhz, measurements in freq_groups.items():
            print(f"  {freq_mhz}: {len(measurements)} 个检测点")
    else:
        print("未检测到明显信号")

    return analyzer, iq_signal, results


# =============================
# 主程序
# =============================
if __name__ == "__main__":
    print("无线通信基带IQ信号的STFT分析")
    print("=" * 60)

    try:
        # 运行示例1
        print("\n运行示例1: 基本IQ信号STFT分析...")
        analyzer1, iq1, results1 = example_1_basic_iq_stft()

        # 运行示例2
        print("\n运行示例2: 不同调制信号的STFT对比...")
        example_2_simple_modulation_comparison()

        # 运行示例3
        print("\n运行示例3: 简单实时STFT分析模拟...")
        analyzer3, iq3 = example_3_simple_real_time()

        # 运行示例4
        print("\n运行示例4: 实际应用 - 多信号检测...")
        analyzer4, iq4, results4 = example_4_practical_application()

        print("\n" + "=" * 60)
        print("所有示例运行完成！")
        print("=" * 60)

    except Exception as e:
        print(f"\n运行时出错: {e}")
        print("尝试运行简化版本...")

        # 运行一个最简单的示例
        print("\n运行最简示例...")
        fs = 1e6  # 1 MHz
        analyzer = IQSTFTAnalyzer(fs=fs, nperseg=256, noverlap=128)

        # 生成简单信号
        t = np.arange(10000) / fs
        iq_signal = np.exp(1j * 2 * np.pi * 100e3 * t)  # 100 kHz信号

        # 添加固定 ISR 的白噪声。
        iq_signal = add_noise_with_fixed_isr(iq_signal, target_isr_db=30.0)

        # STFT分析
        results = analyzer.stft_analysis(iq_signal[:5000])  # 只分析前5000点

        # 简单绘图
        plt.figure(figsize=(10, 6))

        plt.subplot(2, 2, 1)
        plt.plot(t[:200], np.real(iq_signal[:200]), label='I')
        plt.plot(t[:200], np.imag(iq_signal[:200]), label='Q')
        plt.title('时域信号')
        plt.xlabel('时间 (s)')
        plt.ylabel('幅度')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 2)
        plt.scatter(np.real(iq_signal[:1000]), np.imag(iq_signal[:1000]), s=1, alpha=0.5)
        plt.title('星座图')
        plt.xlabel('I')
        plt.ylabel('Q')
        plt.axis('equal')
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 1, 2)
        plt.pcolormesh(results['times'], results['frequencies'], results['magnitude_db'],
                       shading='gouraud', cmap='viridis')
        plt.title('STFT时频谱')
        plt.xlabel('时间 (s)')
        plt.ylabel('频率 (Hz)')
        plt.colorbar(label='幅度 (dB)')

        plt.tight_layout()
        plt.show()

        print("最简示例运行完成！")
