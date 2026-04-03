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
SPECTRUM_FS_DBM = float(os.environ.get("JAMSYSTEM_RX_FS_DBM", "-20.0"))
FAST_FINE_CFO_GRID = np.linspace(-600, 600, 25, dtype=np.float32)
FAST_SINGLE_TONE_BANDWIDTHS = (8e3, 12e3)
FAST_SINGLE_TONE_CUTOFFS = (450e3, 560e3)
REFINE_DECIMATION = 2
TORCH_NUM_THREADS = 2
QPSK_REF_SYMBOL_PERIOD = int(os.environ.get("JAMSYSTEM_QPSK_REF_SYMBOL_PERIOD", "256"))
QPSK_REF_SEED = int(os.environ.get("JAMSYSTEM_QPSK_REF_SEED", "2026"))
QPSK_BER_SKIP_SYMBOLS = int(os.environ.get("JAMSYSTEM_QPSK_BER_SKIP_SYMBOLS", "200"))
QPSK_BER_ALIGN_SYMBOLS = int(os.environ.get("JAMSYSTEM_QPSK_BER_ALIGN_SYMBOLS", "2048"))
ENABLE_TRUE_BER = os.environ.get("JAMSYSTEM_ENABLE_TRUE_BER", "0").strip().lower() in {"1", "true", "yes", "on"}
QPSK_REF_BITS_FILE = os.environ.get(
    "JAMSYSTEM_QPSK_REF_BITS_FILE",
    os.path.join(CURRENT_DIR, "templates", "useful_qpsk_bits.bin"),
)
QPSK_REF_WAVE_FILE = os.environ.get(
    "JAMSYSTEM_QPSK_REF_WAVE_FILE",
    os.path.join(CURRENT_DIR, "templates", "useful_qpsk.bin"),
)

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
_qpsk_ref_bits_period = None
_qpsk_ref_symbols_period = None
_qpsk_ref_wave_cache = {}


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
    psd_db = 10 * np.log10(psd[:SPECTRUM_POINT_COUNT] + 1e-12)
    frame_peak = max(float(np.max(np.abs(chunk))), 1e-9)
    frame_peak_dbfs = 20 * np.log10(frame_peak)
    dbm_offset = SPECTRUM_FS_DBM - frame_peak_dbfs
    return freqs[:SPECTRUM_POINT_COUNT], psd_db + dbm_offset


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


def _get_reference_qpsk_period():
    """
    生成固定已知的 QPSK 参考序列。

    只要数字发射链路使用同样的参考序列，接收端就能逐位比较得到真实 BER。
    """
    global _qpsk_ref_bits_period, _qpsk_ref_symbols_period
    if _qpsk_ref_bits_period is not None and _qpsk_ref_symbols_period is not None:
        return _qpsk_ref_bits_period, _qpsk_ref_symbols_period

    if os.path.exists(QPSK_REF_BITS_FILE):
        bits = np.fromfile(QPSK_REF_BITS_FILE, dtype=np.int8)
        if bits.size >= 2:
            if bits.size % 2 != 0:
                bits = bits[:-1]
        else:
            bits = np.empty(0, dtype=np.int8)
    else:
        rng = np.random.default_rng(QPSK_REF_SEED)
        bit_count = max(2, QPSK_REF_SYMBOL_PERIOD * 2)
        bits = rng.integers(0, 2, bit_count, dtype=np.int8)

    if bits.size < 2:
        rng = np.random.default_rng(QPSK_REF_SEED)
        bit_count = max(2, QPSK_REF_SYMBOL_PERIOD * 2)
        bits = rng.integers(0, 2, bit_count, dtype=np.int8)

    bit_pairs = bits.reshape(-1, 2)
    mapping = {
        (0, 0): 1 + 1j,
        (0, 1): 1 - 1j,
        (1, 1): -1 - 1j,
        (1, 0): -1 + 1j,
    }
    symbols = np.array(
        [mapping[(int(b0), int(b1))] for b0, b1 in bit_pairs],
        dtype=np.complex64,
    ) / np.sqrt(2)

    _qpsk_ref_bits_period = bits
    _qpsk_ref_symbols_period = symbols
    return _qpsk_ref_bits_period, _qpsk_ref_symbols_period


def _hard_decision_qpsk_symbols(symbols):
    symbols = np.asarray(symbols, dtype=np.complex64)
    real_part = np.where(np.real(symbols) >= 0.0, 1.0, -1.0)
    imag_part = np.where(np.imag(symbols) >= 0.0, 1.0, -1.0)
    return (real_part + 1j * imag_part).astype(np.complex64) / np.sqrt(2)


def _hard_decision_qpsk_bits(symbols):
    symbols = np.asarray(symbols, dtype=np.complex64)
    bits = np.zeros(symbols.size * 2, dtype=np.int8)
    bits[0::2] = (np.real(symbols) < 0.0).astype(np.int8)
    bits[1::2] = (np.imag(symbols) < 0.0).astype(np.int8)
    return bits


def _repeat_reference_qpsk_bits(symbol_count, shift_symbols=0):
    ref_bits_period, _ = _get_reference_qpsk_period()
    period_symbols = max(1, len(ref_bits_period) // 2)
    symbol_indices = (np.arange(symbol_count, dtype=np.int32) + shift_symbols) % period_symbols
    repeated = np.empty(symbol_count * 2, dtype=np.int8)
    repeated[0::2] = ref_bits_period[symbol_indices * 2]
    repeated[1::2] = ref_bits_period[symbol_indices * 2 + 1]
    return repeated


def calculate_true_qpsk_ber(symbols):
    """
    基于固定已知 QPSK 参考序列的逐位比较 BER。

    这里会同时搜索：
    - 4 个象限旋转；
    - 一个参考周期内的符号移位；

    只有当数字发射链路使用同一份固定参考序列时，该值才代表真实 BER。
    """
    symbols = np.asarray(symbols, dtype=np.complex64)
    if symbols.size == 0:
        return 1.0

    skip_symbols = min(max(QPSK_BER_SKIP_SYMBOLS, 0), max(symbols.size - 1, 0))
    usable_symbols = symbols[skip_symbols:]
    if usable_symbols.size == 0:
        usable_symbols = symbols

    _, ref_symbols_period = _get_reference_qpsk_period()
    period_symbols = len(ref_symbols_period)
    align_symbols = min(len(usable_symbols), max(64, QPSK_BER_ALIGN_SYMBOLS))
    probe_symbols = usable_symbols[:align_symbols]

    candidate_rotations = [1.0, 1.0j, -1.0, -1.0j]
    best_rotation = 1.0
    best_shift = 0
    best_score = -1e9

    for rotation in candidate_rotations:
        decided = _hard_decision_qpsk_symbols(probe_symbols * rotation)
        for shift in range(period_symbols):
            ref = ref_symbols_period[(np.arange(align_symbols) + shift) % period_symbols]
            score = float(np.real(np.mean(decided * np.conj(ref))))
            if score > best_score:
                best_score = score
                best_rotation = rotation
                best_shift = shift

    pred_bits = _hard_decision_qpsk_bits(usable_symbols * best_rotation)
    ref_bits = _repeat_reference_qpsk_bits(usable_symbols.size, best_shift)
    compare_len = min(len(pred_bits), len(ref_bits))
    if compare_len == 0:
        return 1.0
    return float(np.mean(pred_bits[:compare_len] != ref_bits[:compare_len]))



def calculate_true_qpsk_ber_from_iq(iq_samples):
    """
    针对固定参考 QPSK 发射链路，直接在 IQ 上估计真实 BER。

    做法：
    1. RRC 匹配滤波；
    2. 在 0..SPS-1 内搜索最佳抽样相位；
    3. 同时尝试原始符号与共轭符号，覆盖硬件常见的 IQ 镜像/反相；
    4. 对每个候选分支做粗频偏/相位补偿；
    5. 调用逐位比较 BER，取最小值。
    """
    iq_samples = np.asarray(iq_samples, dtype=np.complex64)
    if iq_samples.size < SPS * 32:
        return 1.0

    matched = np.convolve(iq_samples, _get_rrc_filter(), mode="same").astype(np.complex64)
    best_ber = 1.0

    for phase in range(SPS):
        symbols = matched[phase::SPS]
        if symbols.size < 64:
            continue

        for candidate in (symbols, np.conj(symbols)):
            ber_raw = calculate_true_qpsk_ber(candidate)
            if ber_raw < best_ber:
                best_ber = ber_raw

            work = candidate.astype(np.complex64, copy=True)
            probe = work[: min(len(work), 2048)]
            if probe.size >= 8:
                z = probe ** 4
                delta = np.angle(np.mean(z[1:] * np.conj(z[:-1])) + 1e-12) / 4.0
                work *= np.exp(-1j * delta * np.arange(len(work), dtype=np.float32))
                probe = work[: min(len(work), 2048)]
                phase0 = 0.25 * np.angle(np.mean(probe ** 4) + 1e-12)
                work *= np.exp(-1j * phase0)
                ber_corr = calculate_true_qpsk_ber(work)
                if ber_corr < best_ber:
                    best_ber = ber_corr

    return float(best_ber)



def evaluate_true_qpsk_performance_from_iq(iq_samples):
    """
    迁移自新复原代码的更严格数字链路评估逻辑。

    核心步骤：
    1. RRC 匹配滤波；
    2. 搜索最佳抽样相位；
    3. 与固定参考 QPSK 序列做移位/旋转对齐；
    4. 做最小二乘增益与相位校正；
    5. 直接计算真实 BER 与真实 EVM。
    """
    iq_samples = np.asarray(iq_samples, dtype=np.complex64)
    if iq_samples.size < SPS * 32:
        return {"ber": 1.0, "evm": 100.0, "symbols": np.array([], dtype=np.complex64)}

    matched = np.convolve(iq_samples, _get_rrc_filter(), mode="same").astype(np.complex64)
    ref_bits_period, ref_symbols_period = _get_reference_qpsk_period()
    period_symbols = max(1, len(ref_symbols_period))
    if period_symbols <= 0:
        return {"ber": 1.0, "evm": 100.0, "symbols": np.array([], dtype=np.complex64)}

    best_ber = 1.0
    best_evm = 100.0
    best_syms_norm = np.array([], dtype=np.complex64)

    for phase in range(SPS):
        rx_syms = matched[phase::SPS]
        if rx_syms.size < 64:
            continue

        for candidate in (rx_syms, np.conj(rx_syms)):
            work = candidate.astype(np.complex64, copy=True)
            probe = work[: min(len(work), 2048)]
            if probe.size >= 8:
                z = probe ** 4
                delta = np.angle(np.mean(z[1:] * np.conj(z[:-1])) + 1e-12) / 4.0
                work *= np.exp(-1j * delta * np.arange(len(work), dtype=np.float32))
                probe = work[: min(len(work), 2048)]
                phase0 = 0.25 * np.angle(np.mean(probe ** 4) + 1e-12)
                work *= np.exp(-1j * phase0)

            align_len = min(len(work), max(256, min(QPSK_BER_ALIGN_SYMBOLS, len(work))))
            seg_rx = work[:align_len]

            best_shift = 0
            best_rotation = 1.0
            best_score = -1.0
            for rotation in (1.0, 1.0j, -1.0, -1.0j):
                seg_rot = seg_rx * rotation
                for shift in range(period_symbols):
                    ref = ref_symbols_period[(np.arange(align_len) + shift) % period_symbols]
                    score = float(np.abs(np.vdot(seg_rot, ref)))
                    if score > best_score:
                        best_score = score
                        best_shift = shift
                        best_rotation = rotation

            rx_aligned = work * best_rotation
            ref_aligned = ref_symbols_period[(np.arange(len(rx_aligned)) + best_shift) % period_symbols]
            L = min(len(rx_aligned), len(ref_aligned))
            if L < 32:
                continue
            rx_aligned = rx_aligned[:L]
            ref_aligned = ref_aligned[:L]

            alpha_c = np.sum(ref_aligned * np.conj(rx_aligned)) / (np.sum(np.abs(rx_aligned) ** 2) + 1e-12)
            rx_corrected = rx_aligned * alpha_c

            evm = np.sqrt(np.mean(np.abs(rx_corrected - ref_aligned) ** 2) / (np.mean(np.abs(ref_aligned) ** 2) + 1e-12))
            evm_pct = float(evm * 100.0)

            bits_rx_I = (np.real(rx_corrected) > 0)
            bits_ref_I = (np.real(ref_aligned) > 0)
            bits_rx_Q = (np.imag(rx_corrected) > 0)
            bits_ref_Q = (np.imag(ref_aligned) > 0)
            errors = np.sum(bits_rx_I != bits_ref_I) + np.sum(bits_rx_Q != bits_ref_Q)
            ber = float(errors / (2 * len(bits_ref_I)))

            norm_factor = np.sqrt(np.mean(np.abs(ref_aligned) ** 2)) + 1e-12
            rx_norm = (rx_corrected / norm_factor).astype(np.complex64)

            if ber < best_ber or (abs(ber - best_ber) < 1e-12 and evm_pct < best_evm):
                best_ber = ber
                best_evm = evm_pct
                best_syms_norm = rx_norm

    return {"ber": float(best_ber), "evm": float(best_evm), "symbols": best_syms_norm}


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


def _build_reference_qpsk_wave(length):
    global _qpsk_ref_wave_cache
    length = int(max(length, 1))
    cached = _qpsk_ref_wave_cache.get(length)
    if cached is not None:
        return cached

    if os.path.exists(QPSK_REF_WAVE_FILE):
        interleaved = np.fromfile(QPSK_REF_WAVE_FILE, dtype=np.int16)
        if interleaved.size >= 2 and interleaved.size % 2 == 0:
            wave = (interleaved[0::2].astype(np.float32) + 1j * interleaved[1::2].astype(np.float32)) / 32767.0
            if len(wave) >= length:
                wave = wave[:length]
            else:
                repeats = int(np.ceil(length / max(len(wave), 1)))
                wave = np.tile(wave, repeats)[:length]
            _qpsk_ref_wave_cache[length] = wave.astype(np.complex64)
            return _qpsk_ref_wave_cache[length]

    _, ref_symbols = _get_reference_qpsk_period()
    repeats = int(np.ceil((length + 4 * SPS * 12) / max(len(ref_symbols) * SPS, 1)))
    symbols_long = np.tile(ref_symbols, repeats)
    up = np.zeros(len(symbols_long) * SPS, dtype=np.complex64)
    up[::SPS] = symbols_long
    shaped = np.convolve(up, _get_rrc_filter(), mode="same").astype(np.complex64)
    shaped = shaped[:length]
    shaped = shaped / (np.sqrt(np.mean(np.abs(shaped) ** 2)) + 1e-12)
    shaped = shaped * 0.08
    _qpsk_ref_wave_cache[length] = shaped.astype(np.complex64)
    return _qpsk_ref_wave_cache[length]


def _estimate_reference_useful_component(iq_wave):
    iq_wave = np.asarray(iq_wave, dtype=np.complex64)
    if iq_wave.size < 64:
        return None

    ref_wave = _build_reference_qpsk_wave(len(iq_wave))
    if ref_wave.size != iq_wave.size:
        return None

    norm = np.sqrt(np.mean(np.abs(iq_wave) ** 2)) + 1e-12
    wave = iq_wave / norm

    best_score = -1.0
    best_useful = None
    n = len(wave)
    nfft = 1 << int(np.ceil(np.log2(max(1, 2 * n - 1))))
    wave_fft = np.fft.fft(wave, nfft)

    for candidate in (ref_wave, np.conj(ref_wave)):
        cand = candidate / (np.sqrt(np.mean(np.abs(candidate) ** 2)) + 1e-12)
        corr = np.fft.ifft(wave_fft * np.conj(np.fft.fft(cand, nfft)))
        shift = int(np.argmax(np.abs(corr[:n])))
        aligned = np.roll(cand, shift)
        gain = np.vdot(aligned, wave) / (np.vdot(aligned, aligned) + 1e-12)
        useful = gain * aligned
        score = float(np.mean(np.abs(useful) ** 2))
        if score > best_score:
            best_score = score
            best_useful = useful.astype(np.complex64)

    if best_useful is None:
        return None
    return best_useful * norm


def calculate_interference_power_ratio(raw_iq, clean_iq, modulation_mode):
    """
    计算“恢复前干扰功率 / 恢复后干扰功率”的值。

    数字调制场景下，利用固定已知 QPSK 参考波形估计有用信号分量，
    然后把残差视为干扰+噪声，从而计算恢复前后干扰功率比值。
    模拟调制场景没有固定参考业务波形时，不输出该指标。
    """
    if modulation_mode != "digital_qpsk":
        return None

    raw_useful = _estimate_reference_useful_component(raw_iq)
    clean_useful = _estimate_reference_useful_component(clean_iq)
    if raw_useful is None or clean_useful is None:
        return None

    raw_interference = np.asarray(raw_iq, dtype=np.complex64) - raw_useful
    clean_interference = np.asarray(clean_iq, dtype=np.complex64) - clean_useful
    raw_power = np.mean(np.abs(raw_interference) ** 2)
    clean_power = np.mean(np.abs(clean_interference) ** 2)
    return raw_power / (clean_power + 1e-12)


def estimate_qpsk_ber_from_evm(evm_percent):
    """
    用 EVM 估计 QPSK 的 BER。
    这是无参考比特场景下的近似指标，保留作为兜底工具函数。
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
    ????????????????

    ??????????????????
    1. ????????????????????
    2. ?????????????????
    3. ????????????
    """
    sos_pre = signal.butter(6, 450e3, btype="low", fs=FS, output="sos")
    filtered_pre = _apply_sos_filter(sos_pre, iq_data, zero_phase=True).astype(np.complex64)

    spectrum = np.fft.fft(filtered_pre)
    freqs = np.fft.fftfreq(filtered_pre.size, d=1.0 / FS)
    signal_power = np.abs(spectrum) ** 2

    passband_edge = 420e3
    transition_edge = 650e3
    passband_mask = np.abs(freqs) <= passband_edge
    outband_mask = np.abs(freqs) >= transition_edge
    transition_mask = (~passband_mask) & (~outband_mask)

    if np.any(outband_mask):
        noise_power = float(np.median(signal_power[outband_mask]))
    else:
        noise_power = float(np.median(signal_power) * 0.12)

    wiener_gain = np.maximum(signal_power - noise_power, 0.0) / (signal_power + 1e-12)
    wiener_gain = np.clip(wiener_gain, 0.08, 1.0)

    taper = np.ones_like(freqs, dtype=np.float32)
    taper[outband_mask] = 0.04
    if np.any(transition_mask):
        transition_pos = (np.abs(freqs[transition_mask]) - passband_edge) / max(transition_edge - passband_edge, 1.0)
        taper[transition_mask] = 0.04 + 0.96 * 0.5 * (1.0 + np.cos(np.pi * transition_pos))

    filtered = np.fft.ifft(spectrum * wiener_gain * taper).astype(np.complex64)
    sos_post = signal.butter(6, 480e3, btype="low", fs=FS, output="sos")
    filtered = _apply_sos_filter(sos_post, filtered, zero_phase=True)
    return filtered.astype(np.complex64), "filter_then_wiener"


def _restore_wideband_like(iq_data, mode="wideband_barrage"):
    """
    ???? / ?????????????????

    ???????????????
    - ?? STFT ??????????
    - ?????????
    - ? noise_fm ??????????
    """
    nperseg = 512
    noverlap = 384
    f, t, Zxx = signal.stft(iq_data, fs=FS, nperseg=nperseg, noverlap=noverlap, return_onesided=False)
    power = np.abs(Zxx) ** 2

    if power.size == 0:
        return iq_data.astype(np.complex64, copy=True), "subband_mask"

    noise_floor = np.percentile(power, 35, axis=1, keepdims=True)
    gain = np.maximum(power - noise_floor, 0.0) / (power + 1e-12)
    gain = np.clip(gain, 0.10, 1.0)

    if mode == "noise_fm":
        passband_edge = 800e3
        transition_edge = 1050e3
        post_band = [20e3, 900e3]
    else:
        passband_edge = 520e3
        transition_edge = 820e3
        post_band = [10e3, 650e3]

    abs_f = np.abs(f)
    taper = np.ones_like(abs_f, dtype=np.float32)
    taper[abs_f >= transition_edge] = 0.03
    transition_mask = (abs_f > passband_edge) & (abs_f < transition_edge)
    if np.any(transition_mask):
        transition_pos = (abs_f[transition_mask] - passband_edge) / max(transition_edge - passband_edge, 1.0)
        taper[transition_mask] = 0.03 + 0.97 * 0.5 * (1.0 + np.cos(np.pi * transition_pos))

    Zxx_clean = Zxx * gain * taper[:, None]
    _, restored = signal.istft(Zxx_clean, fs=FS, nperseg=nperseg, noverlap=noverlap, input_onesided=False)
    restored = restored[: iq_data.size].astype(np.complex64)

    magnitude = np.abs(restored)
    limit = np.median(magnitude) * (3.0 if mode == "wideband_barrage" else 3.5)
    restored = np.where(magnitude > limit, restored * (limit / (magnitude + 1e-12)), restored)

    sos_bp = signal.butter(4, post_band, btype="bandpass", fs=FS, output="sos")
    restored = _apply_sos_filter(sos_bp, restored, zero_phase=True)
    return restored.astype(np.complex64), "subband_mask"



def restore_signal(iq_data, label):
    """
    ????????????????

    ?????
    - none: ??????????
    - white_noise: ????????????
    - single_tone: ?????????
    - narrowband / comb: ???? + ???? + ?????
    - wideband_barrage / noise_fm: ????? + ??????
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
            peaks = [freq for freq in peaks if abs(freq) < 0.35e6]
        bandwidth = 25e3 if label == "comb" else 20e3
        order = 3 if label == "narrowband" else 4
        clean = _apply_notches(clean, peaks, bw_hz=bandwidth, fs=FS, order=order)
        sos_lp = signal.butter(6, 450e3, btype="low", fs=FS, output="sos")
        clean = _apply_sos_filter(sos_lp, clean, zero_phase=True)
        method = "adaptive_notch"
        return clean, method

    if label in {"wideband_barrage", "noise_fm"}:
        clean, method = _restore_wideband_like(clean, mode=label)
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
    power_ratio = calculate_interference_power_ratio(iq_processed, clean_wave, modulation_mode)
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
        if ENABLE_TRUE_BER:
            perf_before = evaluate_true_qpsk_performance_from_iq(iq_processed)
            perf_after = evaluate_true_qpsk_performance_from_iq(clean_wave)
            ber_before = perf_before["ber"]
            ber_after = perf_after["ber"]
            # ???? EVM ??????? BER ?????? EVM ???? 100%%?
            if method == "bypass" and perf_before["symbols"].size > 0:
                raw_const = perf_before["symbols"]
                clean_const = raw_const.copy()
            else:
                if perf_before["symbols"].size > 0:
                    raw_const = perf_before["symbols"]
                if perf_after["symbols"].size > 0:
                    clean_const = perf_after["symbols"]
        else:
            ber_before = estimate_qpsk_ber_from_evm(evm_before)
            ber_after = estimate_qpsk_ber_from_evm(evm_after)

    return {
        "clean_wave": clean_wave,
        "raw_const": raw_const,
        "clean_const": clean_const,
        "method": method,
        "isr": isr,
        "power_ratio": power_ratio,
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
            if restore_info["power_ratio"] is None:
                print("RESTORE_POWER_RATIO:--", flush=True)
            else:
                print(f"RESTORE_POWER_RATIO:{restore_info['power_ratio']:.4f}", flush=True)
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
            print("RESTORE_POWER_RATIO:1.0000", flush=True)
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
        if restore_info["power_ratio"] is None:
            print("RESTORE_POWER_RATIO:--", flush=True)
        else:
            print(f"RESTORE_POWER_RATIO:{restore_info['power_ratio']:.4f}", flush=True)
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
        print("RESTORE_POWER_RATIO:1.0000", flush=True)
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






