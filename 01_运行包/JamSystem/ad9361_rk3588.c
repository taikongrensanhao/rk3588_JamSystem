#include <iio.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <signal.h>
#include <unistd.h>
#include <errno.h>

/* ===================== 1. 配置 ===================== */
#define RF_FREQUENCY   2480000000ULL
#define SAMPLING_RATE  6400000
#define RF_BANDWIDTH   10000000        /* 10MHz，确保宽带信号可以通过 */
#define LENGTH         40960
#define WARMUP_FRAMES    20
#define CAPTURE_INTERVAL 10

#define DEFAULT_BASE_DIR "/home/pi/Desktop/JamSystem"

static char runtime_base_dir[512] = {0};
static char runtime_template_dir[512] = {0};
static char runtime_captured_file[512] = {0};
static char runtime_python_exe[512] = {0};
static char runtime_predict_script[512] = {0};

struct iio_context *ctx        = NULL;
struct iio_device  *ad9361_phy = NULL, *tx_dev = NULL, *rx_dev = NULL;
struct iio_buffer  *rx_buf     = NULL, *tx_buf = NULL;
struct iio_channel *tx_i = NULL, *tx_q = NULL,
                   *rx_i = NULL, *rx_q = NULL;
int16_t *jam_template = NULL;
volatile int run_flag = 1;

int  total_recognitions   = 0;
int  correct_recognitions = 0;
char expected_label[64];
char modulation_mode[64] = "digital_qpsk";
int  is_none_mode = 0;

void handle_sig(int sig) { run_flag = 0; }

int resolve_executable_dir(char *out, size_t out_size) {
    ssize_t len = readlink("/proc/self/exe", out, out_size - 1);
    if (len <= 0 || (size_t)len >= out_size) {
        return -1;
    }
    out[len] = '\0';

    char *last_slash = strrchr(out, '/');
    if (!last_slash) {
        return -1;
    }
    *last_slash = '\0';
    return 0;
}

void init_runtime_paths(void) {
    char exe_dir[512] = {0};
    const char *base_env = getenv("JAMSYSTEM_BASE_PATH");
    const char *python_env = getenv("JAMSYSTEM_PYTHON_EXE");
    const char *predict_env = getenv("JAMSYSTEM_PREDICT_SCRIPT");

    if (base_env && strlen(base_env) > 0) {
        snprintf(runtime_base_dir, sizeof(runtime_base_dir), "%s", base_env);
    } else if (resolve_executable_dir(exe_dir, sizeof(exe_dir)) == 0) {
        snprintf(runtime_base_dir, sizeof(runtime_base_dir), "%s", exe_dir);
    } else {
        snprintf(runtime_base_dir, sizeof(runtime_base_dir), "%s", DEFAULT_BASE_DIR);
    }

    snprintf(runtime_template_dir, sizeof(runtime_template_dir), "%s/templates", runtime_base_dir);
    snprintf(runtime_captured_file, sizeof(runtime_captured_file), "%s/output/captured.bin", runtime_base_dir);

    if (python_env && strlen(python_env) > 0) {
        snprintf(runtime_python_exe, sizeof(runtime_python_exe), "%s", python_env);
    } else {
        snprintf(runtime_python_exe, sizeof(runtime_python_exe), "%s/jam_env/bin/python3", runtime_base_dir);
    }

    if (predict_env && strlen(predict_env) > 0) {
        snprintf(runtime_predict_script, sizeof(runtime_predict_script), "%s", predict_env);
    } else {
        snprintf(runtime_predict_script, sizeof(runtime_predict_script), "%s/predict_single.py", runtime_base_dir);
    }
}

int check_connectivity(const char *uri) {
    printf("[INFO] checking %s ...\n", uri);
    errno = 0;
    struct iio_context *check_ctx = iio_create_context_from_uri(uri);
    if (!check_ctx) {
        printf("[ERR] connect failed errno=%d: %s\n", errno, strerror(errno));
        return -1;
    }

    printf("[INFO] connect ok: %s\n", iio_context_get_description(check_ctx));
    iio_context_destroy(check_ctx);
    return 0;
}

/* ===================== 2. 识别函数 ===================== */
void run_recognition_and_stat() {
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "JAMSYSTEM_MODULATION_MODE=%s %s %s --once %s 2>&1",
             modulation_mode, runtime_python_exe, runtime_predict_script, runtime_captured_file);
    FILE *fp = popen(cmd, "r");
    if (!fp) { printf("[ERR] popen failed\n"); return; }

    char  line[1024];
    char  result_id[64] = "";
    float confidence    = 0.0f;
    long  pwr           = 0;

    while (fgets(line, sizeof(line), fp) != NULL) {
        if (strstr(line, "DEBUG_POWER:"))
            sscanf(line, "DEBUG_POWER:%ld", &pwr);
        if (strstr(line, "RESULT_ID:"))
            sscanf(line, "RESULT_ID:%63s", result_id);
        if (strstr(line, "RESULT_CONF:"))
            sscanf(line, "RESULT_CONF:%f", &confidence);
        if (!strstr(line, "DEBUG_"))
            printf("  [py] %s", line);
    }
    pclose(fp);

    if (strlen(result_id) > 0) {
        total_recognitions++;
        int ok = (strcmp(result_id, expected_label) == 0);
        if (ok) correct_recognitions++;
        printf("\033[1;%sm[#%d] 预测=%-18s 置信=%.1f%%  "
               "功率=%ld  累计准确率=%.1f%%\033[0m\n",
               ok ? "32" : "31",
               total_recognitions, result_id,
               confidence * 100.0f, pwr,
               100.0 * correct_recognitions / total_recognitions);
    }
}

/* ===================== 3. 硬件初始化 ===================== */
int configure_hardware() {
    printf("[HW] 定位设备...\n");
    ad9361_phy = iio_context_find_device(ctx, "ad9361-phy");
    tx_dev     = iio_context_find_device(ctx, "cf-ad9361-dds-core-lpc");
    rx_dev     = iio_context_find_device(ctx, "cf-ad9361-lpc");
    if (!ad9361_phy || !tx_dev || !rx_dev) {
        printf("[ERR] 设备定位失败\n"); return -1;
    }

    struct iio_channel *p_tx1 =
        iio_device_find_channel(ad9361_phy, "voltage0", true);
    struct iio_channel *p_rx1 =
        iio_device_find_channel(ad9361_phy, "voltage0", false);
    struct iio_channel *rx_lo =
        iio_device_find_channel(ad9361_phy, "altvoltage0", true);
    struct iio_channel *tx_lo =
        iio_device_find_channel(ad9361_phy, "altvoltage1", true);
    if (!p_tx1 || !p_rx1 || !rx_lo || !tx_lo) {
        printf("[ERR] 找不到物理通道\n"); return -1;
    }

    /* 频率 */
    iio_channel_attr_write_longlong(rx_lo, "frequency", RF_FREQUENCY);
    iio_channel_attr_write_longlong(tx_lo, "frequency", RF_FREQUENCY);
    long long actual_lo = 0;
    iio_channel_attr_read_longlong(rx_lo, "frequency", &actual_lo);
    printf("[HW] 本振频率: %.3f GHz\n", actual_lo / 1e9);

    /* 采样率 */
    iio_channel_attr_write_longlong(p_tx1, "sampling_frequency", SAMPLING_RATE);
    iio_channel_attr_write_longlong(p_rx1, "sampling_frequency", SAMPLING_RATE);
    long long actual_fs = 0;
    iio_channel_attr_read_longlong(p_rx1, "sampling_frequency", &actual_fs);
    printf("[HW] 采样率: %.2f MHz\n", actual_fs / 1e6);

    /* 射频带宽：10MHz，确保宽带信号完整通过 */
    iio_channel_attr_write_longlong(p_tx1, "rf_bandwidth", RF_BANDWIDTH);
    iio_channel_attr_write_longlong(p_rx1, "rf_bandwidth", RF_BANDWIDTH);
    long long actual_bw = 0;
    iio_channel_attr_read_longlong(p_rx1, "rf_bandwidth", &actual_bw);
    printf("[HW] 射频带宽: %.2f MHz\n", actual_bw / 1e6);

    /* 端口 */
    iio_channel_attr_write(p_tx1, "rf_port_select", "A");
    iio_channel_attr_write(p_rx1, "rf_port_select", "A_BALANCED");

    /* 增益 */
    if (is_none_mode) {
        printf("[HW] none模式：RX AGC自动增益\n");
        iio_channel_attr_write(p_rx1, "gain_control_mode", "fast_attack");
    } else {
        printf("[HW] 干扰模式：TX hardwaregain=-3dB  RX manual=70dB\n");
        iio_channel_attr_write_double(p_tx1, "hardwaregain", -3.0);
        iio_channel_attr_write(p_rx1, "gain_control_mode", "manual");
        iio_channel_attr_write_double(p_rx1, "hardwaregain", 70.0);  /* 30 -> 70 */
    }

    /* 禁用FIR */
    iio_device_attr_write_longlong(tx_dev, "filter_fir_en", 0);
    iio_device_attr_write_longlong(rx_dev, "filter_fir_en", 0);

    /* 数据流通道 */
    printf("[HW] 绑定数据流通道...\n");
    tx_i = iio_device_find_channel(tx_dev, "voltage0", true);
    tx_q = iio_device_find_channel(tx_dev, "voltage1", true);
    rx_i = iio_device_find_channel(rx_dev, "voltage0", false);
    rx_q = iio_device_find_channel(rx_dev, "voltage1", false);
    if (!tx_i || !tx_q || !rx_i || !rx_q) {
        printf("[ERR] 数据流通道绑定失败\n"); return -1;
    }
    iio_channel_enable(tx_i); iio_channel_enable(tx_q);
    iio_channel_enable(rx_i); iio_channel_enable(rx_q);

    tx_buf = iio_device_create_buffer(tx_dev, LENGTH, false);
    rx_buf = iio_device_create_buffer(rx_dev, LENGTH, false);
    if (!tx_buf || !rx_buf) {
        printf("[ERR] 创建buffer失败\n"); return -1;
    }

    if (is_none_mode) {
        int16_t *tptr = (int16_t *)iio_buffer_start(tx_buf);
        if (tptr) memset(tptr, 0, LENGTH * 2 * sizeof(int16_t));
    }

    return 0;
}

/* ===================== 4. 主程序 ===================== */
int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    if (argc < 2) {
        printf("用法: %s <jam_type> [ip:x.x.x.x] [digital_qpsk|analog_fm]\n", argv[0]);
        printf("示例: %s single_tone ip:192.168.1.10 digital_qpsk\n", argv[0]);
        printf("类型: none single_tone narrowband wideband_barrage "
               "comb white_noise noise_fm\n");
        return -1;
    }
    if (strcmp(argv[1], "--check") == 0) {
        const char *uri = (argc > 2) ? argv[2] : "ip:192.168.1.10";
        return check_connectivity(uri);
    }

    init_runtime_paths();
    strncpy(expected_label, argv[1], sizeof(expected_label) - 1);
    if (argc > 3 && argv[3] && strlen(argv[3]) > 0) {
        strncpy(modulation_mode, argv[3], sizeof(modulation_mode) - 1);
    }
    is_none_mode = (strcmp(expected_label, "none") == 0);
    signal(SIGINT, handle_sig);

    printf("[INFO] runtime base dir: %s\n", runtime_base_dir);
    printf("[INFO] runtime python: %s\n", runtime_python_exe);
    printf("[INFO] runtime script: %s\n", runtime_predict_script);
    printf("[INFO] modulation mode: %s\n", modulation_mode);
    printf("[py] MODULATION_MODE:%s\n", modulation_mode);

    /* 加载模板 */
    char path[256];
    snprintf(path, sizeof(path), "%s/%s.bin", runtime_template_dir, argv[1]);
    FILE *fp = fopen(path, "rb");
    if (!fp) { printf("[ERR] 找不到模板: %s\n", path); return -1; }
    jam_template = malloc(LENGTH * 2 * sizeof(int16_t));
    if (!jam_template) { fclose(fp); return -1; }
    size_t nr = fread(jam_template, sizeof(int16_t), LENGTH * 2, fp);
    fclose(fp);
    printf("[INFO] 已加载模板: %s  (%zu 个int16)\n", path, nr);
    if (is_none_mode)
        printf("[INFO] none模式：只接收，不发射\n");

    /* 连接ANTSDR */
    const char *uri = (argc > 2) ? argv[2] : "ip:192.168.1.10";
    printf("[INFO] 连接 %s ...\n", uri);
    errno = 0;
    ctx = iio_create_context_from_uri(uri);
    if (!ctx) {
        printf("[ERR] 连接失败 errno=%d: %s\n", errno, strerror(errno));
        return -1;
    }
    printf("[INFO] 连接成功: %s\n", iio_context_get_description(ctx));

    if (configure_hardware() < 0) {
        iio_context_destroy(ctx); return -1;
    }

    /* 预热 */
    printf("[INFO] 硬件预热 %d 帧...\n", WARMUP_FRAMES);
    for (int i = 0; i < WARMUP_FRAMES; i++) {
        if (!is_none_mode) {
            int16_t *tptr = (int16_t *)iio_buffer_start(tx_buf);
            if (tptr) memcpy(tptr, jam_template, LENGTH * 2 * sizeof(int16_t));
            iio_buffer_push(tx_buf);
        }
        iio_buffer_refill(rx_buf);
    }
    printf("[INFO] 预热完成，开始识别循环\n");
    printf("[INFO] 期望类型: %s\n", expected_label);
    printf("--------------------------------------------\n");

    /* 主循环 */
    int frame_cnt = 0;
    while (run_flag) {
        if (!is_none_mode) {
            int16_t *tptr = (int16_t *)iio_buffer_start(tx_buf);
            if (tptr) memcpy(tptr, jam_template, LENGTH * 2 * sizeof(int16_t));
            ssize_t pushed = iio_buffer_push(tx_buf);
            if (pushed < 0) {
                printf("[WARN] TX push 失败: %zd\n", pushed);
                usleep(10000); continue;
            }
        }

        ssize_t filled = iio_buffer_refill(rx_buf);
        if (filled < 0) {
            printf("[WARN] RX refill 失败: %zd\n", filled);
            usleep(10000); continue;
        }

        frame_cnt++;

        if (frame_cnt % CAPTURE_INTERVAL == 0) {
            const int16_t *rptr =
                (const int16_t *)iio_buffer_start(rx_buf);
            if (!rptr) { printf("[WARN] RX buffer 为空\n"); continue; }

            FILE *f = fopen(runtime_captured_file, "wb");
            if (f) {
                fwrite(rptr, sizeof(int16_t), LENGTH * 2, f);
                fclose(f);
                run_recognition_and_stat();
            } else {
                printf("[ERR] 无法写入 %s\n", runtime_captured_file);
            }
        }
    }

    printf("\n--------------------------------------------\n");
    printf("[FINAL] 总识别次数: %d  正确: %d  准确率: %.1f%%\n",
           total_recognitions, correct_recognitions,
           total_recognitions > 0
               ? 100.0 * correct_recognitions / total_recognitions
               : 0.0);

    iio_buffer_destroy(tx_buf);
    iio_buffer_destroy(rx_buf);
    free(jam_template);
    iio_context_destroy(ctx);
    return 0;
}
