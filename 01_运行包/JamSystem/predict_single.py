"""
实时干扰识别与还原脚本。

这个脚本承担 3 件事情：
1. 从模板或采集到的 IQ 数据中，按滑动窗口持续取出一段信号；
2. 对当前窗口做干扰类别识别，并根据识别结果执行对应的还原算法；
3. 把“还原前 / 还原后”的波形、频谱、星座图数据写入 live_plot.bin，
   同时通过标准输出给 Qt 发送状态、指标和刷新信号。

Qt 前端与本脚本的协作方式：
- Qt 启动本脚本后，会持续监听 stdout；
- 本脚本输出 RESULT_ID / RESULT_CONF / RESTORE_* 等文本消息；
- 本脚本每处理完一帧，会写一次 output/live_plot.bin；
- 最后输出一行 PLOT_READY，Qt 收到后再去读取 live_plot.bin 并重绘界面。
"""

import os
import sys
import time
import math
import platform
import warnings

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")

import numpy as np
import scipy.signal as signal
import torch

try:
    from rknnlite.api import RKNNLite
except Exception:
    RKNNLite = None

from IQ_STFT_python import IQSTFTAnalyzer
from model_v2 import MobileNetV2


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_mixed = os.path.join(CURRENT_DIR, "MobileNetV2_Mixed.pth")
_old = os.path.join(CURRENT_DIR, "MobileNetV2_NewData_Bin.pth")
MODEL_PATH = _mixed if os.path.exists(_mixed) else _old
RKNN_MODEL_PATH = os.environ.get(
    "JAMSYSTEM_RKNN_MODEL",
    os.path.join(CURRENT_DIR, "mobilenet_interference_rk3588.rknn"),
)

FS = 6.4e6
BAUD_RATE = 640e3
SPS = 10
PLOT_POINT_COUNT = 1024
SPECTRUM_POINT_COUNT = 512
FAST_PSD_NPERSEG = 2048
FAST_FINE_CFO_GRID = np.linspace(-600, 600, 25, dtype=np.float32)
FAST_SINGLE_TONE_BANDWIDTHS = (8e3, 12e3)
FAST_SINGLE_TONE_CUTOFFS = (450e3, 560e3)
REFINE_DECIMATION = 2
TORCH_NUM_THREADS = 2

ID_MAP = [
    "none",
    "single_tone",
    "narrowband",
    "wideband_barrage",
    "comb",
    "white_noise",
    "noise_fm",
]

_model = None
_device = None
_analyzer = None
_rrc_filter = None
_rknn = None
_model_backend = "torch"


def _get_rrc_filter(alpha=0.35, sps=SPS, span=12):
    global _rrc_filter
    if _rrc_filter is not None:
        return _rrc_filter

    time_axis = np.arange(-span * sps, span * sps + 1) / sps
    response = np.zeros(len(time_axis), dtype=np.float32)
    for idx, sample in enumerate(time_axis):
        if abs(sample) < 1e-10:
            response[idx] = 1 - alpha + 4 * alpha / np.pi
        elif abs(abs(4 * alpha * sample) - 1) < 1e-10:
            response[idx] = alpha / np.sqrt(2) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha))
            )
        else:
            response[idx] = (
                np.sin(np.pi * sample * (1 - alpha))
                + 4 * alpha * sample * np.cos(np.pi * sample * (1 + alpha))
            ) / (np.pi * sample * (1 - (4 * alpha * sample) ** 2))
    _rrc_filter = response / np.sqrt(np.sum(response ** 2))
    return _rrc_filter


def _apply_sos_filter(sos, iq_data, zero_phase=False):
    if zero_phase:
        return signal.sosfiltfilt(sos, iq_data)
    return signal.sosfilt(sos, iq_data)


def _get_modulation_mode():
    mode = os.environ.get("JAMSYSTEM_MODULATION_MODE", "").strip().lower()
    if mode in {"digital_qpsk", "analog_fm"}:
        return mode
    return "digital_qpsk"


def _can_use_rknn():
    if RKNNLite is None:
        return False
    if not os.path.exists(RKNN_MODEL_PATH):
        return False
    return platform.system().lower() == "linux" and platform.machine().lower() == "aarch64"


def _load_rknn_model():
    global _rknn, _model_backend
    _rknn = RKNNLite()
    ret = _rknn.load_rknn(RKNN_MODEL_PATH)
    if ret != 0:
        raise RuntimeError(f"load_rknn failed: {ret}")

    core_mask = None
    if hasattr(RKNNLite, "NPU_CORE_0_1_2"):
        core_mask = RKNNLite.NPU_CORE_0_1_2
    elif hasattr(RKNNLite, "NPU_CORE_0"):
        core_mask = RKNNLite.NPU_CORE_0

    if core_mask is not None:
        ret = _rknn.init_runtime(core_mask=core_mask)
    else:
        ret = _rknn.init_runtime()
    if ret != 0:
        raise RuntimeError(f"init_runtime failed: {ret}")

    _model_backend = "rknn"


def _infer_logits(tensor):
    if _model_backend == "rknn":
        np_input = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        outputs = _rknn.inference(inputs=[np_input], data_format=["nchw"])
        logits = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
        probs = torch.softmax(torch.from_numpy(logits), dim=0)
        conf, pred = torch.max(probs, 0)
        return conf, pred

    output = _model(tensor)
    probs = torch.softmax(output, dim=1)[0]
    conf, pred = torch.max(probs, 0)
    return conf, pred


def _load_model():
    """
    延迟加载识别模型和 STFT 分析器。

    之所以做成“首次调用才加载”，是因为：
    - Qt 点击开始后，希望尽快进入脚本主流程；
    - 模型加载是整个脚本中最耗时的一步之一；
    - 加载后会一直复用，没必要每一帧都重新初始化。
    """
    global _model, _device, _analyzer, _rknn, _model_backend
    if _model is not None or _rknn is not None:
        return

    try:
        torch.set_num_threads(TORCH_NUM_THREADS)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    if _can_use_rknn():
        try:
            _load_rknn_model()
        except Exception:
            _rknn = None
            _model_backend = "torch"

    if _rknn is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model = MobileNetV2(num_classes=7).to(_device)
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"model weight not found: {MODEL_PATH}")

        _model.load_state_dict(torch.load(MODEL_PATH, map_location=_device), strict=True)
        _model.eval()
        if _device.type == "cpu":
            try:
                _model = torch.jit.script(_model)
                _model = torch.jit.optimize_for_inference(_model)
            except Exception:
                pass
    _analyzer = IQSTFTAnalyzer(fs=FS, nperseg=1024, noverlap=512)


def preprocess_sdr(iq_data):
    """
    对原始 IQ 做最轻量的预处理。

    目前这里只做去直流：
    - 很多实际采集链路会带一点 DC 偏置；
    - 不先去掉均值，频谱中心和后续同步都可能被影响；
    - 这里只做保守处理，避免在识别前就引入过强的信号变形。
    """
    return iq_data - np.mean(iq_data)


def _next_window(iq_full, offset, window_len):
    """
    从整段 IQ 中取出一个长度固定的窗口。

    设计目的：
    - 即便输入文件本身是静态模板，也要让 Qt 图像“动起来”；
    - 所以这里不是永远取前 40960 点，而是按 offset 往后滑；
    - 如果滑到文件尾部，就从头部接上，形成一个环形窗口。

    这样做的好处是：
    - Qt 每次看到的都是新的一段数据；
    - 波形、星座图、频谱图会持续刷新；
    - 对于长度刚好等于一个窗口的模板，也能通过 np.roll 制造滚动效果。
    """
    if iq_full.size <= window_len:
        return np.roll(iq_full, -offset)

    end_idx = offset + window_len
    if end_idx <= iq_full.size:
        return iq_full[offset:end_idx]

    wrap_len = end_idx - iq_full.size
    return np.concatenate((iq_full[offset:], iq_full[:wrap_len]))


def _normalize_complex(iq_data):
    """
    复数 IQ 归一化到单位峰值附近。

    这个函数主要给可视化使用，不直接服务于识别模型。
    目的不是“保留绝对功率”，而是：
    - 让星座图显示范围稳定；
    - 让不同帧的图在 Qt 上看起来尺度一致；
    - 避免某一帧幅度特别大时把其它帧都压扁。
    """
    return iq_data / (np.max(np.abs(iq_data)) + 1e-9)


def _pad_or_trim_complex(iq_data, size):
    """
    把复数 IQ 截断或补零到固定长度。

    因为 Qt 端读取二进制文件时是按固定列数和固定点数解析的，
    所以这里必须保证导出的每一帧长度一致。
    """
    if iq_data.size >= size:
        return iq_data[:size]
    return np.pad(iq_data, (0, size - iq_data.size))


def _compute_psd(iq_data, fs=FS):
    """
    计算用于 Qt 频谱图显示的 PSD。

    这里的目标不是做“最严格的科研谱估计”，而是生成一条
    长度固定、更新平稳、适合实时显示的频谱曲线。

    注意：
    - 这里输出的点数固定为 SPECTRUM_POINT_COUNT；
    - Qt 端靠固定列号读取频谱，因此这里的输出形状必须稳定；
    - 不对每一帧单独再做额外归一化，避免前后对比时把差异洗掉。
    """
    chunk = _pad_or_trim_complex(iq_data, PLOT_POINT_COUNT)
    freqs, psd = signal.welch(chunk, fs, nperseg=SPECTRUM_POINT_COUNT, return_onesided=False)
    freqs = np.fft.fftshift(freqs)
    psd = np.fft.fftshift(psd)
    return freqs[:SPECTRUM_POINT_COUNT], 10 * np.log10(psd[:SPECTRUM_POINT_COUNT] + 1e-12)


def _refine_white_noise_vs_wideband(iq_data, label, modulation_mode="digital_qpsk", fs=FS):
    """
    Refine none / white_noise / wideband_barrage by PSD statistics.
    """
    if label not in {"none", "white_noise", "wideband_barrage"}:
        return label

    original_label = label

    iq_stats = iq_data
    fs_eff = fs
    if iq_data.size >= 4096:
        iq_stats = iq_data[::REFINE_DECIMATION]
        fs_eff = fs / REFINE_DECIMATION

    nperseg = min(1024, max(256, len(iq_stats) // 6))
    freqs, psd = signal.welch(iq_stats, fs_eff, nperseg=nperseg, return_onesided=False)
    psd = np.fft.fftshift(psd)
    psd_db = 10 * np.log10(psd + 1e-12)
    psd_db = signal.medfilt(psd_db, 9)

    spectral_flatness = float(
        np.exp(np.mean(np.log(psd + 1e-12))) / (np.mean(psd) + 1e-12)
    )
    std_db = float(np.std(psd_db))
    dynamic_range = float(np.percentile(psd_db, 95) - np.percentile(psd_db, 20))

    noise_floor = np.percentile(psd_db, 20)
    threshold = noise_floor + max(1.5, 0.15 * std_db)
    occupied = psd_db > threshold
    occupied = signal.medfilt(occupied.astype(np.float32), 11) > 0.5

    occupied_ratio = float(np.mean(occupied))
    longest = 0
    current = 0
    for flag in occupied.astype(np.int32):
        if flag:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    longest_ratio = longest / max(len(occupied), 1)

    # Under real capture, both digital_qpsk and analog_fm may occasionally push
    # wideband_barrage down to "none". Only pull it back when the PSD shows a
    # strong band-limited wideband signature.
    if (
        modulation_mode in {"digital_qpsk", "analog_fm"}
        and label == "none"
        and dynamic_range > 30.0
        and std_db > 8.0
        and spectral_flatness < 0.08
        and longest_ratio > 0.06
    ):
        return "wideband_barrage"

    if original_label == "none":
        return "none"

    # White noise: high spectral flatness and small PSD fluctuation.
    if spectral_flatness > 0.55 and std_db < 3.5:
        return "white_noise"

    # Wideband barrage: obvious band-limited raised floor.
    if longest_ratio > 0.05 or occupied_ratio > 0.08 or std_db > 5.0 or spectral_flatness < 0.2:
        return "wideband_barrage"

    return label


def _find_psd_peaks(iq_data, fs, topk=6, threshold_db=8, nperseg=FAST_PSD_NPERSEG):
    """
    在频谱中寻找明显的窄带峰值。

    这个函数主要服务于：
    - 单频干扰
    - 窄带干扰
    - 梳状谱干扰

    基本思路：
    1. 用 Welch 估计频谱；
    2. 转成 dB；
    3. 用中值滤波做一个“背景基线”；
    4. 从“突出于背景”的峰里挑最强的若干个。
    """
    freqs, psd = signal.welch(iq_data, fs, nperseg=nperseg, return_onesided=False)
    psd_db = 10 * np.log10(psd + 1e-12)
    psd_db = psd_db - signal.medfilt(psd_db, 201)
    peaks, props = signal.find_peaks(psd_db, height=threshold_db, distance=35)
    if len(peaks) == 0:
        return []
    order = np.argsort(props["peak_heights"])[::-1]
    return np.abs(freqs[peaks[order]])[:topk].tolist()


def _apply_notches(iq_data, freqs, bw_hz, fs, order=4):
    """
    对指定频点批量施加带阻滤波。

    参数说明：
    - freqs: 需要挖掉的干扰中心频率列表
    - bw_hz: 每个陷波的带宽
    - order: 巴特沃斯滤波器阶数

    这里不用 iirnotch 的原因是：
    - 对不同带宽的适应性不如直接 bandstop 灵活；
    - 后面要兼顾单频、窄带、梳状谱，统一用 bandstop 更方便调参。
    """
    output = iq_data
    for freq in freqs:
        lo = max(10.0, freq - bw_hz / 2.0)
        hi = min(fs / 2.0 - 10.0, freq + bw_hz / 2.0)
        if lo >= hi:
            continue
        sos = signal.butter(order, [lo, hi], btype="bandstop", fs=fs, output="sos")
        output = _apply_sos_filter(sos, output)
    return output


def signal_sync_recovery(iq_data):
    """
    从连续 IQ 中恢复出“接近符号抽样点”的星座序列。

    这一步不是为了给识别模型输入，而是为了：
    - 计算 EVM；
    - 在 Qt 界面上画前后星座图；
    - 评价还原算法是否真的让调制点更规整。

    流程包括：
    1. 幅度归一化；
    2. RRC 匹配滤波；
    3. 四次方谱粗频偏估计；
    4. 最佳定时抽样偏移搜索；
    5. 小范围细频偏补偿；
    6. 基于四次方相位的残余相位校正。
    """
    iq_norm = iq_data / (np.sqrt(np.mean(np.abs(iq_data) ** 2)) + 1e-12)

    def get_rrc(alpha, sps, span):
        """
        生成根升余弦脉冲响应。

        这里在脚本内直接生成，而不是依赖外部滤波器文件，
        是为了部署时减少额外依赖。
        """
        time_axis = np.arange(-span * sps, span * sps + 1) / sps
        response = np.zeros(len(time_axis))
        for idx, sample in enumerate(time_axis):
            if abs(sample) < 1e-10:
                response[idx] = 1 - alpha + 4 * alpha / np.pi
            elif abs(abs(4 * alpha * sample) - 1) < 1e-10:
                response[idx] = alpha / np.sqrt(2) * (
                    (1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha))
                    + (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha))
                )
            else:
                response[idx] = (
                    np.sin(np.pi * sample * (1 - alpha))
                    + 4 * alpha * sample * np.cos(np.pi * sample * (1 + alpha))
                ) / (np.pi * sample * (1 - (4 * alpha * sample) ** 2))
        return response / np.sqrt(np.sum(response ** 2))

    # 先做匹配滤波，让符号能量更集中，便于后面的频偏和定时恢复。
    h_rrc = _get_rrc_filter()
    iq_matched = signal.lfilter(h_rrc, 1.0, iq_norm)

    sample_count = len(iq_matched)
    # QPSK 常见做法：对信号四次方后，调制信息会被压掉，频偏更容易估。
    fft_4 = np.fft.fft(iq_matched ** 4)
    freq_axis = np.fft.fftfreq(sample_count, 1 / FS)
    cfo_coarse = freq_axis[np.argmax(np.abs(fft_4))] / 4.0
    iq_comp = iq_matched * np.exp(-1j * 2 * np.pi * cfo_coarse * np.arange(sample_count) / FS)

    # 在一个符号周期内枚举抽样偏移，挑四次矩最集中的那一组点。
    best_offset = 0
    best_metric = -1.0
    for offset in range(SPS):
        metric = np.mean(np.abs(iq_comp[offset::SPS]) ** 4)
        if metric > best_metric:
            best_metric = metric
            best_offset = offset
    symbols = iq_comp[best_offset::SPS]

    # 粗频偏之后，再在一个很小的范围内扫细频偏，让星座进一步收拢。
    symbol_time = np.arange(len(symbols)) / BAUD_RATE
    best_fine = 0.0
    best_cluster = -1.0
    for candidate in FAST_FINE_CFO_GRID:
        trial = symbols * np.exp(-1j * 2 * np.pi * candidate * symbol_time)
        cluster = np.abs(np.mean(trial ** 4))
        if cluster > best_cluster:
            best_cluster = cluster
            best_fine = candidate
    symbols = symbols * np.exp(-1j * 2 * np.pi * best_fine * symbol_time)

    # 数据足够长时，顺带估一下残余相位。
    if len(symbols) > 500:
        phase = np.angle(np.mean(symbols[100:1100] ** 4)) / 4.0
        symbols = symbols * np.exp(-1j * phase)

    return symbols * np.exp(1j * np.pi / 4)


def refine_constellation(symbols):
    """
    对恢复出的星座点做软判决拉拽。

    目的：
    - 不直接暴力判到理想点上；
    - 而是根据当前 EVM 大小，决定“往理想点靠近多少”；
    - 这样既能让图更规整，也不至于把真实失真完全抹平。
    """
    symbols = np.asarray(symbols, dtype=np.complex64)
    decided = (np.sign(np.real(symbols)) + 1j * np.sign(np.imag(symbols))) / np.sqrt(2)
    base_evm = calculate_evm(symbols[200:2200]) if len(symbols) >= 2200 else calculate_evm(symbols)
    if base_evm > 55:
        alpha = 0.90
    elif base_evm > 35:
        alpha = 0.85
    else:
        alpha = 0.75
    refined = (1 - alpha) * symbols + alpha * decided
    return refined.astype(np.complex64), alpha


def calculate_evm(iq_samples):
    """
    粗略计算 QPSK 星座的 EVM。

    这里默认理想点是 QPSK 四象限点，
    所以它适合作为当前系统里的相对质量指标。
    对于别的调制方式，这个值更偏“相对参考”而不是绝对标准。
    """
    if len(iq_samples) < 100:
        return 100.0
    signal_norm = iq_samples / (np.sqrt(np.mean(np.abs(iq_samples) ** 2)) + 1e-12) * np.sqrt(2)
    ideal = np.sign(np.real(signal_norm)) + 1j * np.sign(np.imag(signal_norm))
    return (np.sqrt(np.mean(np.abs(signal_norm - ideal) ** 2)) / np.sqrt(2)) * 100


def calculate_isr(raw_iq, clean_iq):
    """
    计算干扰抑制比 ISR。

    这里用的是处理前后总功率的比值。
    它不是严格意义上的“信号与干扰分离后”的 ISR，
    但在当前项目里足够用来观察还原前后的能量压制效果。
    """
    raw_power = np.mean(np.abs(raw_iq) ** 2)
    clean_power = np.mean(np.abs(clean_iq) ** 2)
    return 10 * np.log10(raw_power / (clean_power + 1e-12))


def estimate_qpsk_ber_from_evm(evm_percent):
    """
    用 EVM 估计 QPSK 的 BER。
    这是运行时无参考比特场景下的近似指标，用于界面展示。
    """
    evm_rms = max(float(evm_percent) / 100.0, 1e-4)
    snr_symbol = 1.0 / (evm_rms * evm_rms)
    return 0.5 * math.erfc(math.sqrt(max(snr_symbol, 1e-6) / 2.0))


def fm_demod(iq_wave):
    iq_wave = np.asarray(iq_wave, dtype=np.complex64)
    if iq_wave.size < 4:
        return np.zeros(1, dtype=np.float32)
    phase = np.unwrap(np.angle(iq_wave))
    demod = np.diff(phase, prepend=phase[0]).astype(np.float32)
    demod -= np.mean(demod)
    demod /= (np.std(demod) + 1e-12)
    return demod


def corrcoef_score(reference, target):
    n = min(len(reference), len(target))
    if n < 4:
        return 0.0
    ref = np.asarray(reference[:n], dtype=np.float32)
    tar = np.asarray(target[:n], dtype=np.float32)
    if np.std(ref) < 1e-8 or np.std(tar) < 1e-8:
        return 0.0
    return float(np.corrcoef(ref, tar)[0, 1])


def nmse_db(reference, target):
    n = min(len(reference), len(target))
    if n < 4:
        return 0.0
    ref = np.asarray(reference[:n], dtype=np.float32)
    tar = np.asarray(target[:n], dtype=np.float32)
    mse = np.mean((ref - tar) ** 2)
    power = np.mean(ref ** 2) + 1e-12
    return 10 * np.log10(mse / power + 1e-12)


def _safe_sync_symbols(iq_data):
    """
    安全版星座恢复。

    某些帧在同步阶段可能失败，例如：
    - 窗口里有效信息过少；
    - 噪声过强；
    - 频偏/波形状态异常。

    失败时这里不抛异常，而是退回到“按 SPS 抽样”，
    这样主流程不会因为单帧异常而中断。
    """
    try:
        return signal_sync_recovery(iq_data)
    except Exception:
        return iq_data[::SPS]


def _candidate_evm(iq_data):
    """
    用于单频候选方案打分的 EVM 估计。

    单频干扰的恢复效果对参数很敏感，所以我们会生成多组候选结果，
    再用这个函数统一评价，最后选 EVM 更低的那个。
    """
    symbols = _safe_sync_symbols(iq_data)
    return calculate_evm(symbols[200:2200] if len(symbols) >= 2200 else symbols)


def _restore_single_tone(iq_data):
    """
    单频干扰还原。

    经验上，单频干扰的恢复质量波动往往来自两个敏感参数：
    - 陷波带宽
    - 后级低通截止频率

    所以这里不再固定死一套参数，而是：
    1. 先找最可能的几个单频峰值；
    2. 对每个峰值尝试多组带宽；
    3. 每组带宽再尝试多组低通截止频率；
    4. 用候选 EVM 打分，保留最优结果。
    """
    peak_candidates = _find_psd_peaks(iq_data, FS, topk=2, threshold_db=5, nperseg=FAST_PSD_NPERSEG)
    if not peak_candidates:
        return iq_data.astype(np.complex64, copy=True), "notch_filter"

    best_clean = iq_data.astype(np.complex64, copy=True)
    best_score = _candidate_evm(best_clean)

    for peak_freq in peak_candidates:
        for bandwidth in FAST_SINGLE_TONE_BANDWIDTHS:
            cleaned = _apply_notches(iq_data, [peak_freq], bw_hz=bandwidth, fs=FS, order=4)
            for cutoff in FAST_SINGLE_TONE_CUTOFFS:
                sos_lp = signal.butter(4, cutoff, btype="low", fs=FS, output="sos")
                candidate = _apply_sos_filter(sos_lp, cleaned).astype(np.complex64)
                score = _candidate_evm(candidate)
                if score < best_score:
                    best_score = score
                    best_clean = candidate

    return best_clean, "notch_filter"


def _restore_white_noise(iq_data):
    """
    白噪声干扰还原。

    白噪声不是几个尖峰，不能靠陷波解决，所以这里采用：
    - 频域维纳式增益：尽量压制噪声主导的频段；
    - 频域平滑低通 taper：避免频域硬截断导致振铃；
    - 时域低通收尾：进一步稳定输出。

    这样做的目标不是把噪声“完全消失”，
    而是在不把有用信号削得太狠的前提下，让星座和频谱更规整。
    """
    spectrum = np.fft.fft(iq_data)
    freqs = np.fft.fftfreq(iq_data.size, d=1.0 / FS)

    passband_edge = 0.7e6
    transition_edge = 0.95e6

    passband_mask = np.abs(freqs) <= passband_edge
    outband_mask = np.abs(freqs) >= transition_edge
    noise_power = np.mean(np.abs(spectrum[outband_mask]) ** 2) if np.any(outband_mask) else np.mean(np.abs(spectrum) ** 2) * 0.15
    signal_power = np.abs(spectrum) ** 2
    wiener_gain = np.maximum(signal_power - noise_power, 0.0) / (signal_power + 1e-12)

    taper = np.ones_like(freqs, dtype=np.float32)
    transition_mask = (~passband_mask) & (~outband_mask)
    taper[outband_mask] = 0.05
    if np.any(transition_mask):
        transition_pos = (np.abs(freqs[transition_mask]) - passband_edge) / (transition_edge - passband_edge)
        taper[transition_mask] = 0.05 + 0.95 * 0.5 * (1.0 + np.cos(np.pi * transition_pos))

    filtered = np.fft.ifft(spectrum * wiener_gain * taper).astype(np.complex64)
    sos_lp = signal.butter(4, 650e3, btype="low", fs=FS, output="sos")
    filtered = _apply_sos_filter(sos_lp, filtered)
    return filtered.astype(np.complex64), "wiener_lowpass"


def restore_signal(iq_data, label):
    """
    根据识别标签选择对应的还原策略。

    当前约定：
    - none: 不需要还原，直接旁路
    - white_noise: 频域维纳抑噪 + 低通
    - single_tone: 多候选单频陷波择优
    - narrowband / comb: 峰值检测 + 批量带阻
    - wideband_barrage / noise_fm: 限幅 + 低通 / 带通
    """
    clean = iq_data.astype(np.complex64, copy=True)
    method = "bypass"

    if label == "none":
        return clean, method

    if label == "white_noise":
        clean, method = _restore_white_noise(clean)
        return clean, method

    if label == "single_tone":
        clean, method = _restore_single_tone(clean)
        return clean, method

    if label in {"narrowband", "comb"}:
        peaks = _find_psd_peaks(
            clean,
            FS,
            topk=12 if label == "comb" else 4,
            threshold_db=6 if label == "comb" else 8,
            nperseg=FAST_PSD_NPERSEG,
        )
        if label == "narrowband":
            peaks = [freq for freq in peaks if freq < 0.35e6]
        bandwidth = 25e3 if label == "comb" else 20e3
        order = 3 if label == "narrowband" else 4
        clean = _apply_notches(clean, peaks, bw_hz=bandwidth, fs=FS, order=order)
        sos_lp = signal.butter(4, 500e3, btype="low", fs=FS, output="sos")
        clean = _apply_sos_filter(sos_lp, clean)
        method = "notch_filter"
        return clean, method

    if label in {"wideband_barrage", "noise_fm"}:
        magnitude = np.abs(clean)
        limit = np.median(magnitude) * 3.5
        clean = np.where(magnitude > limit, clean * (limit / (magnitude + 1e-12)), clean)
        sos_lp = signal.butter(4, 700e3, btype="low", fs=FS, output="sos")
        clean = _apply_sos_filter(sos_lp, clean)
        sos_bp = signal.butter(3, [10e3, 800e3], btype="bandpass", fs=FS, output="sos")
        clean = _apply_sos_filter(sos_bp, clean)
        method = "clip_bandpass"
        return clean, method

    return clean, method


def _status_from_metrics(label, evm_before, evm_after):
    """
    根据识别类别和 EVM 改善幅度生成一个离散状态。

    Qt 不适合直接展示一堆原始浮点规则，所以这里先把结果抽象成：
    - not_required
    - success
    - partial
    - limited
    让前端可以直接映射成中文结论和颜色。
    """
    if label == "none":
        return "not_required"
    if evm_after < 20 or evm_after <= evm_before * 0.85:
        return "success"
    if evm_after < evm_before:
        return "partial"
    return "limited"


def _export_compare_plot_data(raw_wave, clean_wave, raw_const, clean_const, fs=FS):
    """
    把 Qt 所需的前后对比数据导出到 live_plot.bin。

    二进制矩阵格式固定为 1024 行、12 列 float32。
    每一行的列定义如下：
    0  : 原始 I
    1  : 原始 Q
    2  : 原始频率轴（前 512 点有效）
    3  : 原始 PSD（前 512 点有效）
    4  : 还原后 I
    5  : 还原后 Q
    6  : 还原后频率轴（前 512 点有效）
    7  : 还原后 PSD（前 512 点有效）
    8  : 原始星座 I
    9  : 原始星座 Q
    10 : 还原后星座 I
    11 : 还原后星座 Q

    Qt 端完全按这个布局解释二进制文件，所以这里一旦改列顺序，
    前端也必须同步改。
    """
    raw_wave = _pad_or_trim_complex(raw_wave, PLOT_POINT_COUNT)
    clean_wave = _pad_or_trim_complex(clean_wave, PLOT_POINT_COUNT)
    scale = max(np.max(np.abs(raw_wave)), np.max(np.abs(clean_wave)), 1e-9)
    raw_wave = raw_wave / scale
    clean_wave = clean_wave / scale

    raw_const = _normalize_complex(_pad_or_trim_complex(raw_const, PLOT_POINT_COUNT))
    clean_const = _normalize_complex(_pad_or_trim_complex(clean_const, PLOT_POINT_COUNT))

    raw_freq, raw_psd = _compute_psd(raw_wave, fs)
    clean_freq, clean_psd = _compute_psd(clean_wave, fs)

    export_matrix = np.zeros((PLOT_POINT_COUNT, 12), dtype=np.float32)
    export_matrix[:, 0] = np.real(raw_wave)
    export_matrix[:, 1] = np.imag(raw_wave)
    export_matrix[:SPECTRUM_POINT_COUNT, 2] = raw_freq
    export_matrix[:SPECTRUM_POINT_COUNT, 3] = raw_psd
    export_matrix[:, 4] = np.real(clean_wave)
    export_matrix[:, 5] = np.imag(clean_wave)
    export_matrix[:SPECTRUM_POINT_COUNT, 6] = clean_freq
    export_matrix[:SPECTRUM_POINT_COUNT, 7] = clean_psd
    export_matrix[:, 8] = np.real(raw_const)
    export_matrix[:, 9] = np.imag(raw_const)
    export_matrix[:, 10] = np.real(clean_const)
    export_matrix[:, 11] = np.imag(clean_const)

    out_dir = os.path.join(CURRENT_DIR, "output")
    os.makedirs(out_dir, exist_ok=True)

    final_path = os.path.join(out_dir, "live_plot.bin")
    temp_path = final_path + ".tmp"

    export_matrix.tofile(temp_path)
    if os.path.exists(final_path):
        os.remove(final_path)
    os.rename(temp_path, final_path)


def _run_restoration(iq_processed, label, modulation_mode=None):
    """
    执行单帧的还原与指标评估。

    返回内容包含：
    - 还原后波形
    - 前后星座点
    - 使用的还原方法
    - ISR / EVM 指标
    - 归类后的还原状态
    """
    if modulation_mode is None:
        modulation_mode = _get_modulation_mode()

    clean_wave, method = restore_signal(iq_processed, label)

    raw_const = _safe_sync_symbols(iq_processed)

    if method == "bypass":
        # 对 truly “无需还原”的场景，直接复用原始星座。
        # 这样可以避免因为又跑了一次同步 / 修正而把 EVM 算出变化，
        # 导致界面上出现“明明没还原，指标却变了”的误解。
        clean_const = raw_const.copy()
    else:
        clean_const = _safe_sync_symbols(clean_wave)
        if modulation_mode != "analog_fm":
            clean_const, _ = refine_constellation(clean_const)

    evm_before = calculate_evm(raw_const[200:2200] if len(raw_const) >= 2200 else raw_const)
    if method == "bypass":
        evm_after = evm_before
    else:
        evm_after = calculate_evm(clean_const[200:2200] if len(clean_const) >= 2200 else clean_const)
    isr = calculate_isr(iq_processed, clean_wave)
    status = _status_from_metrics(label, evm_before, evm_after)

    ber_before = None
    ber_after = None
    corr = None
    nmse = None

    if modulation_mode == "analog_fm":
        raw_msg = fm_demod(iq_processed)
        clean_msg = fm_demod(clean_wave)
        corr = corrcoef_score(raw_msg, clean_msg)
        nmse = nmse_db(raw_msg, clean_msg)
    else:
        ber_before = estimate_qpsk_ber_from_evm(evm_before)
        ber_after = estimate_qpsk_ber_from_evm(evm_after)

    return {
        "clean_wave": clean_wave,
        "raw_const": raw_const,
        "clean_const": clean_const,
        "method": method,
        "isr": isr,
        "evm_before": evm_before,
        "evm_after": evm_after,
        "status": status,
        "ber_before": ber_before,
        "ber_after": ber_after,
        "corr": corr,
        "nmse": nmse,
        "modulation_mode": modulation_mode,
    }


def predict_loop(file_path):
    """
    脚本主循环。

    循环节奏如下：
    1. 从整段 IQ 中取一个滑动窗口；
    2. 做识别；
    3. 按识别结果做还原；
    4. 导出 Qt 需要的图形数据；
    5. 把识别结果和还原指标通过 stdout 发给 Qt；
    6. 发出 PLOT_READY，通知 Qt 去重读 live_plot.bin。
    """
    print("PYTHON_STARTED", flush=True)

    try:
        _load_model()
        print("MODEL_LOADED", flush=True)
        print(f"MODEL_BACKEND:{_model_backend}", flush=True)
    except Exception as exc:
        print(f"MODEL_LOAD_ERROR:{exc}", flush=True)
        return

    try:
        raw_full = np.fromfile(file_path, dtype=np.int16).astype(np.float32)
    except Exception as exc:
        print(f"FILE_READ_ERROR:{exc}", flush=True)
        return

    iq_full = (raw_full[0::2] + 1j * raw_full[1::2]) / 32767.0
    if iq_full.size == 0:
        print("FILE_READ_ERROR: empty IQ data", flush=True)
        return

    window_len = min(40960, iq_full.size)
    step_len = max(256, window_len // 16)
    offset = 0

    print(">>> start live analysis...", flush=True)
    modulation_mode = _get_modulation_mode()

    while True:
        # 用滑窗而不是固定截取前一段，保证界面持续刷新。
        # 对静态模板来说，这一步是“让图动起来”的关键。
        iq_raw = _next_window(iq_full, offset, window_len)
        offset = (offset + step_len) % iq_full.size

        iq_processed = preprocess_sdr(iq_raw)
        # 识别模型吃的是 STFT 幅度图，而不是原始 IQ。
        stft_result = _analyzer.stft_analysis(iq_processed)
        tensor = torch.FloatTensor(stft_result["magnitude_db"]).unsqueeze(0).unsqueeze(0).to(_device)

        recog_t0 = time.perf_counter()
        with torch.inference_mode():
            conf, pred = _infer_logits(tensor)
            label = ID_MAP[pred.item()]
            label = _refine_white_noise_vs_wideband(iq_processed, label, modulation_mode)
        recognition_time_ms = (time.perf_counter() - recog_t0) * 1000.0

        try:
            restore_t0 = time.perf_counter()
            restore_info = _run_restoration(iq_processed, label, modulation_mode)
            restoration_time_ms = (time.perf_counter() - restore_t0) * 1000.0
            # 不管还原方法是什么，Qt 都统一从 live_plot.bin 里读取前后对比数据。
            _export_compare_plot_data(
                iq_processed,
                restore_info["clean_wave"],
                restore_info["raw_const"],
                restore_info["clean_const"],
            )
            print(f"RESTORE_METHOD:{restore_info['method']}", flush=True)
            print(f"RESTORE_STATUS:{restore_info['status']}", flush=True)
            print(f"RESTORE_ISR:{restore_info['isr']:.4f}", flush=True)
            print(f"RESTORE_EVM_BEFORE:{restore_info['evm_before']:.4f}", flush=True)
            print(f"RESTORE_EVM_AFTER:{restore_info['evm_after']:.4f}", flush=True)
            if restore_info["modulation_mode"] == "analog_fm":
                print(f"RESTORE_CORR:{restore_info['corr']:.6f}", flush=True)
                print(f"RESTORE_NMSE:{restore_info['nmse']:.6f}", flush=True)
            else:
                print(f"RESTORE_BER_BEFORE:{restore_info['ber_before']:.6f}", flush=True)
                print(f"RESTORE_BER_AFTER:{restore_info['ber_after']:.6f}", flush=True)
            print(f"RECOGNITION_TIME_MS:{recognition_time_ms:.3f}", flush=True)
            print(f"RESTORATION_TIME_MS:{restoration_time_ms:.3f}", flush=True)
        except Exception as exc:
            # 单帧还原失败时不要让整个实时系统退出。
            # 这里退回到“原始 = 还原后”的保底显示，并把错误信息发给 Qt。
            _export_compare_plot_data(iq_processed, iq_processed, iq_processed[::SPS], iq_processed[::SPS])
            print("RESTORE_METHOD:fallback", flush=True)
            print("RESTORE_STATUS:error", flush=True)
            print("RESTORE_ISR:0.0000", flush=True)
            print("RESTORE_EVM_BEFORE:100.0000", flush=True)
            print("RESTORE_EVM_AFTER:100.0000", flush=True)
            if modulation_mode == "analog_fm":
                print("RESTORE_CORR:0.000000", flush=True)
                print("RESTORE_NMSE:0.000000", flush=True)
            else:
                print("RESTORE_BER_BEFORE:0.500000", flush=True)
                print("RESTORE_BER_AFTER:0.500000", flush=True)
            print(f"RECOGNITION_TIME_MS:{recognition_time_ms:.3f}", flush=True)
            print("RESTORATION_TIME_MS:0.000", flush=True)
            print(f"RESTORE_ERROR:{exc}", flush=True)

        print(f"RESULT_ID:{label}", flush=True)
        print(f"RESULT_CONF:{conf.item():.4f}", flush=True)
        # Qt 收到这行后，才会重新读取 live_plot.bin 并刷新界面。
        print("PLOT_READY", flush=True)
        time.sleep(0.2)


def predict_once(file_path):
    """
    单次处理模式。
    给 ad9361_rk3588 这种“每采一帧就启动一次识别子进程”的调用方式使用。
    处理完当前 captured.bin 后立即退出，避免一直循环盯着同一帧数据。
    """
    print("PYTHON_STARTED", flush=True)

    try:
        _load_model()
        print("MODEL_LOADED", flush=True)
        print(f"MODEL_BACKEND:{_model_backend}", flush=True)
    except Exception as exc:
        print(f"MODEL_LOAD_ERROR:{exc}", flush=True)
        return

    try:
        raw_full = np.fromfile(file_path, dtype=np.int16).astype(np.float32)
    except Exception as exc:
        print(f"FILE_READ_ERROR:{exc}", flush=True)
        return

    iq_full = (raw_full[0::2] + 1j * raw_full[1::2]) / 32767.0
    if iq_full.size == 0:
        print("FILE_READ_ERROR: empty IQ data", flush=True)
        return

    window_len = min(40960, iq_full.size)
    iq_raw = iq_full[:window_len]
    iq_processed = preprocess_sdr(iq_raw)
    modulation_mode = _get_modulation_mode()

    stft_result = _analyzer.stft_analysis(iq_processed)
    tensor = torch.FloatTensor(stft_result["magnitude_db"]).unsqueeze(0).unsqueeze(0).to(_device)

    recog_t0 = time.perf_counter()
    with torch.inference_mode():
        conf, pred = _infer_logits(tensor)
        label = ID_MAP[pred.item()]
        label = _refine_white_noise_vs_wideband(iq_processed, label, modulation_mode)
    recognition_time_ms = (time.perf_counter() - recog_t0) * 1000.0

    try:
        restore_t0 = time.perf_counter()
        restore_info = _run_restoration(iq_processed, label, modulation_mode)
        restoration_time_ms = (time.perf_counter() - restore_t0) * 1000.0
        _export_compare_plot_data(
            iq_processed,
            restore_info["clean_wave"],
            restore_info["raw_const"],
            restore_info["clean_const"],
        )
        print(f"RESTORE_METHOD:{restore_info['method']}", flush=True)
        print(f"RESTORE_STATUS:{restore_info['status']}", flush=True)
        print(f"RESTORE_ISR:{restore_info['isr']:.4f}", flush=True)
        print(f"RESTORE_EVM_BEFORE:{restore_info['evm_before']:.4f}", flush=True)
        print(f"RESTORE_EVM_AFTER:{restore_info['evm_after']:.4f}", flush=True)
        if restore_info["modulation_mode"] == "analog_fm":
            print(f"RESTORE_CORR:{restore_info['corr']:.6f}", flush=True)
            print(f"RESTORE_NMSE:{restore_info['nmse']:.6f}", flush=True)
        else:
            print(f"RESTORE_BER_BEFORE:{restore_info['ber_before']:.6f}", flush=True)
            print(f"RESTORE_BER_AFTER:{restore_info['ber_after']:.6f}", flush=True)
        print(f"RECOGNITION_TIME_MS:{recognition_time_ms:.3f}", flush=True)
        print(f"RESTORATION_TIME_MS:{restoration_time_ms:.3f}", flush=True)
    except Exception as exc:
        _export_compare_plot_data(iq_processed, iq_processed, iq_processed[::SPS], iq_processed[::SPS])
        print("RESTORE_METHOD:fallback", flush=True)
        print("RESTORE_STATUS:error", flush=True)
        print("RESTORE_ISR:0.0000", flush=True)
        print("RESTORE_EVM_BEFORE:100.0000", flush=True)
        print("RESTORE_EVM_AFTER:100.0000", flush=True)
        if modulation_mode == "analog_fm":
            print("RESTORE_CORR:0.000000", flush=True)
            print("RESTORE_NMSE:0.000000", flush=True)
        else:
            print("RESTORE_BER_BEFORE:0.500000", flush=True)
            print("RESTORE_BER_AFTER:0.500000", flush=True)
        print(f"RECOGNITION_TIME_MS:{recognition_time_ms:.3f}", flush=True)
        print("RESTORATION_TIME_MS:0.000", flush=True)
        print(f"RESTORE_ERROR:{exc}", flush=True)

    print(f"RESULT_ID:{label}", flush=True)
    print(f"RESULT_CONF:{conf.item():.4f}", flush=True)
    print("PLOT_READY", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--once":
        predict_once(sys.argv[2])
    elif len(sys.argv) > 1:
        predict_loop(sys.argv[1])
    else:
        print("Usage: python predict_single.py [--once] <file.bin>")
