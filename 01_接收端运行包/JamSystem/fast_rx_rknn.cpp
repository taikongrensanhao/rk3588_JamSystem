#include <iio.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <condition_variable>
#include <string>
#include <mutex>
#include <thread>
#include <vector>

#if __has_include(<rknn_api.h>)
#include <rknn_api.h>
#define JAMSYSTEM_HAS_RKNN_C_API 1
#elif __has_include(<rknn_api/rknn_api.h>)
#include <rknn_api/rknn_api.h>
#define JAMSYSTEM_HAS_RKNN_C_API 1
#else
#define JAMSYSTEM_HAS_RKNN_C_API 0
#endif

namespace {

constexpr unsigned long long kRfFrequency = 200000000ULL;
constexpr long long kSamplingRate = 6400000;
constexpr long long kRfBandwidth = 10000000;
constexpr int kIqLength = 40960;
constexpr int kStftN = 1024;
constexpr int kHop = 512;
constexpr int kStftCols = 81;
constexpr int kNumClasses = 7;

const char *kLabels[kNumClasses] = {
    "none",
    "single_tone",
    "narrowband",
    "wideband_barrage",
    "comb",
    "white_noise",
    "noise_fm",
};

volatile sig_atomic_t g_run = 1;

void on_signal(int) {
    g_run = 0;
}

double env_double(const char *name, double default_value) {
    const char *value = std::getenv(name);
    if (!value || !*value) {
        return default_value;
    }
    char *end = nullptr;
    const double parsed = std::strtod(value, &end);
    return end != value ? parsed : default_value;
}

int env_int(const char *name, int default_value) {
    const char *value = std::getenv(name);
    if (!value || !*value) {
        return default_value;
    }
    char *end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    return end != value ? static_cast<int>(parsed) : default_value;
}

bool env_flag_enabled(const char *name, bool default_value) {
    const char *value = std::getenv(name);
    if (!value || !*value) {
        return default_value;
    }
    return std::strcmp(value, "1") == 0 ||
           std::strcmp(value, "true") == 0 ||
           std::strcmp(value, "TRUE") == 0 ||
           std::strcmp(value, "yes") == 0 ||
           std::strcmp(value, "on") == 0;
}

unsigned long long center_frequency_hz() {
    const char *value = std::getenv("JAMSYSTEM_CENTER_FREQ_HZ");
    if (!value || !*value) {
        value = std::getenv("JAMSYSTEM_RF_FREQUENCY_HZ");
    }
    if (!value || !*value) {
        return kRfFrequency;
    }
    char *end = nullptr;
    unsigned long long parsed = std::strtoull(value, &end, 10);
    return end != value && parsed > 0 ? parsed : kRfFrequency;
}

std::string model_path() {
    const char *env = std::getenv("JAMSYSTEM_RKNN_MODEL");
    if (env && *env) {
        return env;
    }
    const char *base = std::getenv("JAMSYSTEM_BASE_PATH");
    if (base && *base) {
        return std::string(base) + "/mobilenet_interference_rk3588.rknn";
    }
    return "/home/pi/Desktop/run/JamSystem/mobilenet_interference_rk3588.rknn";
}

void fft_radix2(std::vector<std::complex<float>> &a) {
    const int n = static_cast<int>(a.size());
    static std::vector<int> bitrev;
    if (static_cast<int>(bitrev.size()) != n) {
        bitrev.assign(n, 0);
        for (int i = 1; i < n; ++i) {
            bitrev[i] = (bitrev[i >> 1] >> 1) | ((i & 1) ? (n >> 1) : 0);
        }
    }
    for (int i = 0; i < n; ++i) {
        const int j = bitrev[i];
        if (i < j) {
            std::swap(a[i], a[j]);
        }
    }

    for (int len = 2; len <= n; len <<= 1) {
        const float angle = -2.0f * static_cast<float>(M_PI) / static_cast<float>(len);
        const std::complex<float> wlen(std::cos(angle), std::sin(angle));
        for (int i = 0; i < n; i += len) {
            std::complex<float> w(1.0f, 0.0f);
            for (int j = 0; j < len / 2; ++j) {
                const std::complex<float> u = a[i + j];
                const std::complex<float> v = a[i + j + len / 2] * w;
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
}

void make_hann(std::vector<float> &window) {
    window.resize(kStftN);
    for (int i = 0; i < kStftN; ++i) {
        window[i] = 0.5f - 0.5f * std::cos(2.0f * static_cast<float>(M_PI) * i / kStftN);
    }
}

void make_stft_input(const int16_t *iq_interleaved, std::vector<float> &input) {
    static std::vector<float> hann;
    static float hann_sum = 0.0f;
    static std::vector<std::complex<float>> iq(kIqLength);
    static std::vector<std::complex<float>> frame(kStftN);
    if (hann.empty()) {
        make_hann(hann);
        for (float v : hann) {
            hann_sum += v;
        }
    }

    std::complex<double> mean(0.0, 0.0);
    for (int i = 0; i < kIqLength; ++i) {
        const float re = static_cast<float>(iq_interleaved[i * 2]) / 32767.0f;
        const float im = static_cast<float>(iq_interleaved[i * 2 + 1]) / 32767.0f;
        iq[i] = std::complex<float>(re, im);
        mean += std::complex<double>(re, im);
    }
    mean /= static_cast<double>(kIqLength);

    float max_abs = 0.0f;
    for (int i = 0; i < kIqLength; ++i) {
        iq[i] -= std::complex<float>(static_cast<float>(mean.real()), static_cast<float>(mean.imag()));
        max_abs = std::max(max_abs, std::abs(iq[i]));
    }
    if (max_abs > 1e-6f) {
        const float inv = 1.0f / max_abs;
        for (auto &v : iq) {
            v *= inv;
        }
    }

    if (input.size() != static_cast<size_t>(kStftN * kStftCols)) {
        input.resize(kStftN * kStftCols);
    }
    int col_stride = env_int("JAMSYSTEM_FAST_STFT_STRIDE", 1);
    if (col_stride < 1) {
        col_stride = 1;
    }
    if (col_stride > 4) {
        col_stride = 4;
    }
    for (int col = 0; col < kStftCols; ++col) {
        if (col_stride > 1 && (col % col_stride) != 0) {
            const int src_col = col - 1;
            for (int row = 0; row < kStftN; ++row) {
                input[row * kStftCols + col] = input[row * kStftCols + src_col];
            }
            continue;
        }
        const int start = col * kHop - kStftN / 2;
        for (int n = 0; n < kStftN; ++n) {
            const int src = start + n;
            frame[n] = (src >= 0 && src < kIqLength) ? iq[src] * hann[n] : std::complex<float>(0.0f, 0.0f);
        }
        fft_radix2(frame);
        for (int row = 0; row < kStftN; ++row) {
            const int shifted = (row + kStftN / 2) & (kStftN - 1);
            const float mag = std::abs(frame[shifted]) / std::max(hann_sum, 1e-12f);
            input[row * kStftCols + col] = 10.0f * std::log10(mag + 1e-10f);
        }
    }
}

float softmax_conf(const float *logits, int count, int *best_index) {
    int best = 0;
    float max_logit = logits[0];
    for (int i = 1; i < count; ++i) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
            best = i;
        }
    }
    double sum = 0.0;
    for (int i = 0; i < count; ++i) {
        sum += std::exp(static_cast<double>(logits[i] - max_logit));
    }
    *best_index = best;
    return static_cast<float>(1.0 / sum);
}

double estimate_power_dbm(const int16_t *iq_interleaved) {
    double sum = 0.0;
    for (int i = 0; i < kIqLength; ++i) {
        const double re = static_cast<double>(iq_interleaved[i * 2]) / 32767.0;
        const double im = static_cast<double>(iq_interleaved[i * 2 + 1]) / 32767.0;
        sum += re * re + im * im;
    }
    const double mean_power = sum / std::max(1, kIqLength);
    return 10.0 * std::log10(mean_power + 1e-12) - 20.0;
}

struct ToneShape {
    double peak_to_median_db = 0.0;
    int peak_count_15db = 0;
};

ToneShape estimate_tone_shape(const int16_t *iq_interleaved) {
    static std::vector<float> hann;
    static std::vector<std::complex<float>> frame(kStftN);
    static std::vector<double> db(kStftN);
    if (hann.empty()) {
        make_hann(hann);
    }

    const int start = (kIqLength - kStftN) / 2;
    double mean_re = 0.0;
    double mean_im = 0.0;
    for (int i = 0; i < kIqLength; ++i) {
        mean_re += static_cast<double>(iq_interleaved[i * 2]);
        mean_im += static_cast<double>(iq_interleaved[i * 2 + 1]);
    }
    mean_re /= std::max(1, kIqLength);
    mean_im /= std::max(1, kIqLength);

    for (int i = 0; i < kStftN; ++i) {
        const int src = start + i;
        const float re = static_cast<float>(static_cast<double>(iq_interleaved[src * 2]) - mean_re) / 2048.0f;
        const float im = static_cast<float>(static_cast<double>(iq_interleaved[src * 2 + 1]) - mean_im) / 2048.0f;
        frame[i] = std::complex<float>(re, im) * hann[i];
    }
    fft_radix2(frame);

    for (int i = 0; i < kStftN; ++i) {
        const int shifted = (i + kStftN / 2) & (kStftN - 1);
        const float mag = std::norm(frame[shifted]);
        db[i] = 10.0 * std::log10(static_cast<double>(mag) + 1e-12);
    }

    std::vector<double> sorted = db;
    std::nth_element(sorted.begin(), sorted.begin() + sorted.size() / 2, sorted.end());
    const double median = sorted[sorted.size() / 2];
    const double max_db = *std::max_element(db.begin(), db.end());

    int peaks = 0;
    int last_peak = -1000000;
    const int min_distance_bins = 20;
    for (int i = 1; i + 1 < kStftN; ++i) {
        if (db[i] > median + 15.0 && db[i] >= db[i - 1] && db[i] >= db[i + 1] &&
            i - last_peak >= min_distance_bins) {
            ++peaks;
            last_peak = i;
        }
    }

    ToneShape shape;
    shape.peak_to_median_db = max_db - median;
    shape.peak_count_15db = peaks;
    return shape;
}

void refine_single_tone_vs_narrowband(const int16_t *iq_interleaved, int *label, float *conf) {
    if (!env_flag_enabled("JAMSYSTEM_FAST_REFINE_NARROWBAND", false)) {
        return;
    }
    constexpr int kSingleToneLabel = 1;
    constexpr int kNarrowbandLabel = 2;
    if (*label != kSingleToneLabel) {
        return;
    }

    const ToneShape shape = estimate_tone_shape(iq_interleaved);
    const bool multi_peak = shape.peak_count_15db >= env_int("JAMSYSTEM_FAST_NARROWBAND_MIN_PEAKS", 2);
    const bool soft_single_tone = *conf < static_cast<float>(env_double("JAMSYSTEM_FAST_SINGLE_TONE_CONF_KEEP", 0.88));
    const bool not_very_sharp = shape.peak_to_median_db < env_double("JAMSYSTEM_FAST_SINGLE_TONE_SHARP_DB", 24.0);

    if (multi_peak || (soft_single_tone && not_very_sharp)) {
        *label = kNarrowbandLabel;
        if (*conf < 0.72f) {
            *conf = 0.72f;
        }
    }
}

void maybe_dump_frame(const int16_t *iq_interleaved, int frame_index) {
    const char *path = std::getenv("JAMSYSTEM_FAST_DUMP_FILE");
    if (!path || !*path) {
        return;
    }
    FILE *fp = std::fopen(path, "wb");
    if (!fp) {
        std::printf("[FAST_RKNN] dump open failed: %s\n", path);
        return;
    }
    const size_t want = static_cast<size_t>(kIqLength) * 2;
    const size_t wrote = std::fwrite(iq_interleaved, sizeof(int16_t), want, fp);
    std::fclose(fp);
    if (wrote == want) {
        std::printf("[FAST_RKNN] dumped frame #%d: %s\n", frame_index, path);
    } else {
        std::printf("[FAST_RKNN] dump short write: %zu/%zu\n", wrote, want);
    }
}

#if JAMSYSTEM_HAS_RKNN_C_API
class RknnModel {
public:
    ~RknnModel() {
        if (ctx_) {
            rknn_destroy(ctx_);
        }
    }

    bool load(const std::string &path) {
        FILE *fp = std::fopen(path.c_str(), "rb");
        if (!fp) {
            std::printf("[FAST_RKNN] model open failed: %s\n", path.c_str());
            return false;
        }
        std::fseek(fp, 0, SEEK_END);
        const long size = std::ftell(fp);
        std::fseek(fp, 0, SEEK_SET);
        model_.resize(size);
        if (std::fread(model_.data(), 1, model_.size(), fp) != model_.size()) {
            std::fclose(fp);
            std::printf("[FAST_RKNN] model read failed\n");
            return false;
        }
        std::fclose(fp);

        int ret = rknn_init(&ctx_, model_.data(), model_.size(), 0, nullptr);
        if (ret != RKNN_SUCC) {
            std::printf("[FAST_RKNN] rknn_init failed: %d\n", ret);
            ctx_ = 0;
            return false;
        }
        std::printf("[FAST_RKNN] model loaded: %s\n", path.c_str());
        return true;
    }

    bool infer(std::vector<float> &input, int *label, float *conf) {
        rknn_input in;
        std::memset(&in, 0, sizeof(in));
        in.index = 0;
        in.buf = input.data();
        in.size = static_cast<uint32_t>(input.size() * sizeof(float));
        in.pass_through = 0;
        in.type = RKNN_TENSOR_FLOAT32;
        // RKNN runtime normalization only accepts NHWC source layout. The tensor
        // has one channel, so the contiguous data order is equivalent to
        // [1, 1024, 81, 1] for this model input.
        in.fmt = RKNN_TENSOR_NHWC;

        int ret = rknn_inputs_set(ctx_, 1, &in);
        if (ret != RKNN_SUCC) {
            std::printf("[FAST_RKNN] inputs_set failed: %d\n", ret);
            return false;
        }
        ret = rknn_run(ctx_, nullptr);
        if (ret != RKNN_SUCC) {
            std::printf("[FAST_RKNN] run failed: %d\n", ret);
            return false;
        }

        rknn_output out;
        std::memset(&out, 0, sizeof(out));
        out.want_float = 1;
        ret = rknn_outputs_get(ctx_, 1, &out, nullptr);
        if (ret != RKNN_SUCC) {
            std::printf("[FAST_RKNN] outputs_get failed: %d\n", ret);
            return false;
        }

        const float *logits = static_cast<const float *>(out.buf);
        int best = 0;
        const float c = softmax_conf(logits, kNumClasses, &best);
        rknn_outputs_release(ctx_, 1, &out);
        *label = best;
        *conf = c;
        return true;
    }

private:
    rknn_context ctx_ = 0;
    std::vector<unsigned char> model_;
};
#endif

struct IioRx {
    iio_context *ctx = nullptr;
    iio_device *phy = nullptr;
    iio_device *rx_dev = nullptr;
    iio_channel *rx_i = nullptr;
    iio_channel *rx_q = nullptr;
    iio_buffer *rx_buf = nullptr;

    ~IioRx() {
        if (rx_buf) {
            iio_buffer_destroy(rx_buf);
        }
        if (ctx) {
            iio_context_destroy(ctx);
        }
    }

    bool open(const char *uri) {
        ctx = iio_create_context_from_uri(uri);
        if (!ctx) {
            std::printf("[FAST_RKNN] connect failed: %s\n", uri);
            return false;
        }
        std::printf("[FAST_RKNN] connect ok: %s\n", iio_context_get_description(ctx));

        phy = iio_context_find_device(ctx, "ad9361-phy");
        rx_dev = iio_context_find_device(ctx, "cf-ad9361-lpc");
        if (!phy || !rx_dev) {
            std::printf("[FAST_RKNN] AD9361 rx device not found\n");
            return false;
        }

        iio_channel *p_rx1 = iio_device_find_channel(phy, "voltage0", false);
        iio_channel *rx_lo = iio_device_find_channel(phy, "altvoltage0", true);
        if (!p_rx1 || !rx_lo) {
            std::printf("[FAST_RKNN] AD9361 physical rx channel not found\n");
            return false;
        }

        iio_channel_attr_write_longlong(rx_lo, "frequency", center_frequency_hz());
        iio_channel_attr_write_longlong(p_rx1, "sampling_frequency", kSamplingRate);
        iio_channel_attr_write_longlong(p_rx1, "rf_bandwidth", kRfBandwidth);
        iio_channel_attr_write(p_rx1, "rf_port_select", "A_BALANCED");
        iio_channel_attr_write(p_rx1, "gain_control_mode", "manual");
        iio_channel_attr_write_double(p_rx1, "hardwaregain", env_double("JAMSYSTEM_RX_GAIN_DB", 45.0));
        iio_device_attr_write_longlong(rx_dev, "filter_fir_en", 0);

        long long actual_lo = 0;
        long long actual_fs = 0;
        iio_channel_attr_read_longlong(rx_lo, "frequency", &actual_lo);
        iio_channel_attr_read_longlong(p_rx1, "sampling_frequency", &actual_fs);
        std::printf("[FAST_RKNN] LO %.3f GHz, FS %.2f MHz\n", actual_lo / 1e9, actual_fs / 1e6);

        rx_i = iio_device_find_channel(rx_dev, "voltage0", false);
        rx_q = iio_device_find_channel(rx_dev, "voltage1", false);
        if (!rx_i || !rx_q) {
            std::printf("[FAST_RKNN] AD9361 stream channels not found\n");
            return false;
        }
        iio_channel_enable(rx_i);
        iio_channel_enable(rx_q);
        rx_buf = iio_device_create_buffer(rx_dev, kIqLength, false);
        if (!rx_buf) {
            std::printf("[FAST_RKNN] rx buffer create failed\n");
            return false;
        }
        return true;
    }
};

class RxFramePump {
public:
    explicit RxFramePump(IioRx *rx)
        : rx_(rx), latest_(kIqLength * 2), consumer_copy_(kIqLength * 2) {}

    void start() {
        worker_ = std::thread(&RxFramePump::loop, this);
        std::printf("[FAST_RKNN] RX producer thread started\n");
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        if (worker_.joinable()) {
            worker_.join();
        }
    }

    bool wait_next(const int16_t **raw, unsigned long long *seq) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] {
            return stop_ || !g_run || seq_ != consumed_seq_;
        });
        if (seq_ == consumed_seq_ || !ready_) {
            return false;
        }
        consumer_copy_ = latest_;
        consumed_seq_ = seq_;
        if (seq) {
            *seq = consumed_seq_;
        }
        *raw = consumer_copy_.data();
        return true;
    }

private:
    void loop() {
        while (g_run) {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (stop_) {
                    break;
                }
            }
            const ssize_t filled = iio_buffer_refill(rx_->rx_buf);
            if (filled < 0) {
                std::printf("[FAST_RKNN] rx refill failed: %zd\n", filled);
                continue;
            }
            const int16_t *raw = static_cast<const int16_t *>(iio_buffer_start(rx_->rx_buf));
            if (!raw) {
                continue;
            }
            {
                std::lock_guard<std::mutex> lock(mutex_);
                std::copy(raw, raw + kIqLength * 2, latest_.begin());
                ready_ = true;
                ++seq_;
            }
            cv_.notify_one();
        }
        cv_.notify_all();
    }

    IioRx *rx_;
    std::vector<int16_t> latest_;
    std::vector<int16_t> consumer_copy_;
    std::thread worker_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool ready_ = false;
    bool stop_ = false;
    unsigned long long seq_ = 0;
    unsigned long long consumed_seq_ = 0;
};

}  // namespace

int main(int argc, char **argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    signal(SIGINT, on_signal);

#if !JAMSYSTEM_HAS_RKNN_C_API
    std::printf("[FAST_RKNN] rknn_api.h not found. Install RKNN runtime headers to build this prototype.\n");
    return 2;
#else
    const char *uri = "ip:192.168.1.10";
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "ip:", 3) == 0) {
            uri = argv[i];
            break;
        }
    }
    IioRx rx;
    if (!rx.open(uri)) {
        return 1;
    }

    RknnModel model;
    if (!model.load(model_path())) {
        return 1;
    }

    RxFramePump pump(&rx);
    pump.start();

    std::vector<float> input;
    int count = 0;
    const int max_frames = env_int("JAMSYSTEM_FAST_MAX_FRAMES", 0);
    while (g_run) {
        const auto t_loop0 = std::chrono::steady_clock::now();
        const int16_t *raw = nullptr;
        unsigned long long frame_seq = 0;
        if (!pump.wait_next(&raw, &frame_seq)) {
            break;
        }

        const auto t_frame = std::chrono::steady_clock::now();
        const double power_dbm = estimate_power_dbm(raw);
        maybe_dump_frame(raw, count + 1);
        const auto t_power = std::chrono::steady_clock::now();
        if (env_flag_enabled("JAMSYSTEM_ENABLE_NO_SIGNAL_GATE", true)) {
            const double threshold_dbm = env_double("JAMSYSTEM_NO_SIGNAL_POWER_DBM", -80.0);
            if (power_dbm < threshold_dbm) {
                const auto t1 = std::chrono::steady_clock::now();
                const double wait_ms = std::chrono::duration<double, std::milli>(t_frame - t_loop0).count();
                const double power_ms = std::chrono::duration<double, std::milli>(t_power - t_frame).count();
                const double ms = std::chrono::duration<double, std::milli>(t1 - t_frame).count();
                const double total_ms = std::chrono::duration<double, std::milli>(t1 - t_loop0).count();
                std::printf("[py] RESULT_POWER_DBM:%.2f\n", power_dbm);
                std::printf("  [py] MODEL_BACKEND:rknn_c\n");
                std::printf("  [py] RECOGNITION_TIME_MS:%.3f\n", ms);
                std::printf("  [py] RESTORATION_TIME_MS:0.000\n");
                std::printf("  [py] STAGE_WAIT_FRAME_MS:%.3f\n", wait_ms);
                std::printf("  [py] STAGE_POWER_GATE_MS:%.3f\n", power_ms);
                std::printf("  [py] STAGE_STFT_MS:0.000\n");
                std::printf("  [py] STAGE_RKNN_MS:0.000\n");
                std::printf("  [py] STAGE_POST_MS:%.3f\n", std::max(0.0, ms - power_ms));
                std::printf("  [py] STAGE_TOTAL_MS:%.3f\n", total_ms);
                std::printf("  [py] RESULT_ID:no_signal\n");
                std::printf("  [py] RESULT_CONF:1.0000\n");
                std::printf("\033[1;36m[#%d] 预测=%-18s 置信=100.0%%  C++内存直推\033[0m\n",
                            ++count, "no_signal");
                if (max_frames > 0 && count >= max_frames) {
                    break;
                }
                continue;
            }
        }
        make_stft_input(raw, input);
        const auto t_stft = std::chrono::steady_clock::now();
        int label = 0;
        float conf = 0.0f;
        if (!model.infer(input, &label, &conf)) {
            continue;
        }
        const auto t_infer = std::chrono::steady_clock::now();
        refine_single_tone_vs_narrowband(raw, &label, &conf);
        const auto t1 = std::chrono::steady_clock::now();
        const double wait_ms = std::chrono::duration<double, std::milli>(t_frame - t_loop0).count();
        const double power_ms = std::chrono::duration<double, std::milli>(t_power - t_frame).count();
        const double stft_ms = std::chrono::duration<double, std::milli>(t_stft - t_power).count();
        const double rknn_ms = std::chrono::duration<double, std::milli>(t_infer - t_stft).count();
        const double post_ms = std::chrono::duration<double, std::milli>(t1 - t_infer).count();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t_frame).count();
        const double total_ms = std::chrono::duration<double, std::milli>(t1 - t_loop0).count();

        std::printf("[py] RESULT_POWER_DBM:%.2f\n", power_dbm);
        std::printf("  [py] MODEL_BACKEND:rknn_c\n");
        std::printf("  [py] RECOGNITION_TIME_MS:%.3f\n", ms);
        std::printf("  [py] RESTORATION_TIME_MS:0.000\n");
        std::printf("  [py] STAGE_WAIT_FRAME_MS:%.3f\n", wait_ms);
        std::printf("  [py] STAGE_POWER_GATE_MS:%.3f\n", power_ms);
        std::printf("  [py] STAGE_STFT_MS:%.3f\n", stft_ms);
        std::printf("  [py] STAGE_RKNN_MS:%.3f\n", rknn_ms);
        std::printf("  [py] STAGE_POST_MS:%.3f\n", post_ms);
        std::printf("  [py] STAGE_TOTAL_MS:%.3f\n", total_ms);
        std::printf("  [py] RESULT_ID:%s\n", kLabels[std::max(0, std::min(label, kNumClasses - 1))]);
        std::printf("  [py] RESULT_CONF:%.4f\n", conf);
        std::printf("\033[1;36m[#%d] 预测=%-18s 置信=%.1f%%  C++内存直推\033[0m\n",
                    ++count, kLabels[std::max(0, std::min(label, kNumClasses - 1))], conf * 100.0f);
        if (max_frames > 0 && count >= max_frames) {
            break;
        }
    }
    pump.stop();
    return 0;
#endif
}
