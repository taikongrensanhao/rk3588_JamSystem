#include <iio.h>
#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <math.h>
#include <signal.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/statvfs.h>
#include <sys/wait.h>
#include <pthread.h>

/* ===================== 1. 配置 ===================== */
#define RF_FREQUENCY   200000000ULL
#define SAMPLING_RATE  6400000
#define RF_BANDWIDTH   10000000        /* 10MHz，确保宽带信号可以通过 */
#define LENGTH         40960
#define SLIDE_STEP     (LENGTH / 2)
#define WARMUP_FRAMES    20
#define CAPTURE_INTERVAL 1
#define RECORD_SECONDS    5
#define RECORD_FRAME_INT16_COUNT (LENGTH * 2)
#define RECORD_PRE_FRAMES ((SAMPLING_RATE * RECORD_SECONDS + LENGTH - 1) / LENGTH)
#define RECORD_POST_FRAMES RECORD_PRE_FRAMES
#define RECORD_MIN_FREE_MB 300
#define RECORD_MAX_ACTIVE 16
#define RECORD_CLOSE_COMPLETED 1
#define RECORD_CLOSE_ABORTED   2
#define COLLECT_DEFAULT_FRAMES 200

#define DEFAULT_BASE_DIR "/home/pi/Desktop/JamSystem"
#define MAX_SELFTEST_SWITCH_ITEMS 8

static char runtime_base_dir[512] = {0};
static char runtime_template_dir[512] = {0};
static char runtime_captured_file[512] = {0};
static char runtime_python_exe[512] = {0};
static char runtime_predict_script[512] = {0};
static char runtime_useful_qpsk_file[512] = {0};
static char runtime_useful_bits_file[512] = {0};
static char runtime_useful_fm_file[512] = {0};
static char runtime_record_dir[512] = {0};

struct iio_context *ctx        = NULL;
struct iio_device  *ad9361_phy = NULL, *tx_dev = NULL, *rx_dev = NULL;
struct iio_buffer  *rx_buf     = NULL, *tx_buf = NULL;
struct iio_channel *tx_i = NULL, *tx_q = NULL,
                   *rx_i = NULL, *rx_q = NULL;
int16_t *jam_template = NULL;
static int16_t *selftest_switch_templates[MAX_SELFTEST_SWITCH_ITEMS] = {0};
static char selftest_switch_labels[MAX_SELFTEST_SWITCH_ITEMS][64];
static int selftest_switch_count = 0;
static int selftest_switch_index = 0;
static long long selftest_switch_interval_ms = 1000;
static long long selftest_switch_last_ms = 0;
int16_t *useful_qpsk_template = NULL;
int16_t *useful_fm_template = NULL;
volatile int run_flag = 1;

int  total_recognitions   = 0;
int  correct_recognitions = 0;
char expected_label[64];
char modulation_mode[64] = "digital_qpsk";
char run_mode[64] = "self_test";
int  is_none_mode = 0;
int  rx_only_mode = 0;
int  rx_unknown_mode = 0;
static int no_signal_valid_streak = 0;
static FILE *recognition_worker_in = NULL;
static FILE *recognition_worker_out = NULL;
static pid_t recognition_worker_pid = -1;
static int recognition_worker_failed = 0;

static int16_t *record_pre_cache = NULL;
static size_t record_pre_write_idx = 0;
static size_t record_pre_valid_frames = 0;
static char last_record_label[64] = "none";
static unsigned long record_event_id = 0;

typedef struct {
    FILE *fp;
    int active;
    size_t post_frames_left;
    char file_path[512];
    char label[64];
} record_task_t;

static record_task_t record_tasks[RECORD_MAX_ACTIVE];

typedef struct record_write_job {
    FILE *fp;
    int16_t *data;
    size_t int16_count;
    int close_after;
    char file_path[512];
    struct record_write_job *next;
} record_write_job_t;

static pthread_t record_writer_thread;
static pthread_mutex_t record_queue_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t record_queue_cond = PTHREAD_COND_INITIALIZER;
static record_write_job_t *record_queue_head = NULL;
static record_write_job_t *record_queue_tail = NULL;
static int record_writer_started = 0;
static int record_writer_stop = 0;

static unsigned long long get_center_frequency_hz(void) {
    const char *env = getenv("JAMSYSTEM_CENTER_FREQ_HZ");
    if (!env || strlen(env) == 0) {
        env = getenv("JAMSYSTEM_RF_FREQUENCY_HZ");
    }
    if (env && strlen(env) > 0) {
        char *endptr = NULL;
        unsigned long long value = strtoull(env, &endptr, 10);
        if (endptr && endptr != env && value > 0) {
            return value;
        }
    }
    return RF_FREQUENCY;
}

static double get_rx_full_scale_dbm(void) {
    const char *env = getenv("JAMSYSTEM_RX_FS_DBM");
    if (env && strlen(env) > 0) {
        char *endptr = NULL;
        double value = strtod(env, &endptr);
        if (endptr && endptr != env) {
            return value;
        }
    }
    return -20.0;
}

static double estimate_rx_power_dbm(const int16_t *iq_ptr, size_t complex_count) {
    if (!iq_ptr || complex_count == 0) {
        return -120.0;
    }

    const double full_scale_dbm = get_rx_full_scale_dbm();
    const double fs_ref = 32767.0;
    double accum = 0.0;
    for (size_t idx = 0; idx < complex_count; ++idx) {
        const double i_val = (double)iq_ptr[idx * 2 + 0];
        const double q_val = (double)iq_ptr[idx * 2 + 1];
        accum += (i_val * i_val + q_val * q_val) / (2.0 * fs_ref * fs_ref);
    }

    const double mean_norm_power = accum / (double)complex_count;
    return full_scale_dbm + 10.0 * log10(mean_norm_power + 1e-12);
}

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
    const char *captured_env = getenv("JAMSYSTEM_CAPTURED_FILE");

    if (base_env && strlen(base_env) > 0) {
        snprintf(runtime_base_dir, sizeof(runtime_base_dir), "%s", base_env);
    } else if (resolve_executable_dir(exe_dir, sizeof(exe_dir)) == 0) {
        snprintf(runtime_base_dir, sizeof(runtime_base_dir), "%s", exe_dir);
    } else {
        snprintf(runtime_base_dir, sizeof(runtime_base_dir), "%s", DEFAULT_BASE_DIR);
    }

    snprintf(runtime_template_dir, sizeof(runtime_template_dir), "%s/templates", runtime_base_dir);
    if (captured_env && strlen(captured_env) > 0) {
        snprintf(runtime_captured_file, sizeof(runtime_captured_file), "%s", captured_env);
    } else {
        snprintf(runtime_captured_file, sizeof(runtime_captured_file), "%s/output/captured.bin", runtime_base_dir);
    }
    snprintf(runtime_useful_qpsk_file, sizeof(runtime_useful_qpsk_file), "%s/templates/useful_qpsk.bin", runtime_base_dir);
    snprintf(runtime_useful_bits_file, sizeof(runtime_useful_bits_file), "%s/templates/useful_qpsk_bits.bin", runtime_base_dir);
    snprintf(runtime_useful_fm_file, sizeof(runtime_useful_fm_file), "%s/templates/useful_fm.bin", runtime_base_dir);

    const char *record_env = getenv("JAMSYSTEM_RECORD_DIR");
    if (record_env && strlen(record_env) > 0) {
        snprintf(runtime_record_dir, sizeof(runtime_record_dir), "%s", record_env);
    } else if (access("/mnt/usb", W_OK) == 0) {
        snprintf(runtime_record_dir, sizeof(runtime_record_dir), "%s", "/mnt/usb/JamRecords");
    } else if (access("/mnt/sdcard", W_OK) == 0) {
        snprintf(runtime_record_dir, sizeof(runtime_record_dir), "%s", "/mnt/sdcard/JamRecords");
    } else {
        snprintf(runtime_record_dir, sizeof(runtime_record_dir), "%s/output/records", runtime_base_dir);
    }

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

static double get_target_input_jsr_db(void) {
    const char *env = getenv("JAMSYSTEM_INPUT_JSR_DB");
    if (env && strlen(env) > 0) {
        char *endptr = NULL;
        double value = strtod(env, &endptr);
        if (endptr && endptr != env) {
            return value;
        }
    }
    return 30.0;
}

static double env_double_value(const char *name, double default_value) {
    const char *env = getenv(name);
    if (env && strlen(env) > 0) {
        char *endptr = NULL;
        double value = strtod(env, &endptr);
        if (endptr && endptr != env) {
            return value;
        }
    }
    return default_value;
}

static double get_tx_class_gain_db(void) {
    double gain_db = env_double_value("JAMSYSTEM_TX_CLASS_GAIN_DB", 0.0);

    if (strcmp(expected_label, "narrowband") == 0) {
        gain_db += env_double_value("JAMSYSTEM_TX_NARROWBAND_GAIN_DB", 10.0);
    } else if (strcmp(expected_label, "single_tone") == 0) {
        gain_db += env_double_value("JAMSYSTEM_TX_SINGLE_TONE_GAIN_DB", 0.0);
    } else if (strcmp(expected_label, "wideband_barrage") == 0) {
        gain_db += env_double_value("JAMSYSTEM_TX_WIDEBAND_GAIN_DB", 0.0);
    } else if (strcmp(expected_label, "comb") == 0) {
        gain_db += env_double_value("JAMSYSTEM_TX_COMB_GAIN_DB", 0.0);
    } else if (strcmp(expected_label, "white_noise") == 0) {
        gain_db += env_double_value("JAMSYSTEM_TX_WHITE_NOISE_GAIN_DB", 0.0);
    } else if (strcmp(expected_label, "noise_fm") == 0) {
        gain_db += env_double_value("JAMSYSTEM_TX_NOISE_FM_GAIN_DB", 0.0);
    }

    return gain_db;
}

static int is_digital_qpsk_mode(void) {
    return strcmp(modulation_mode, "digital_qpsk") == 0;
}

static int is_analog_fm_mode(void) {
    return strcmp(modulation_mode, "analog_fm") == 0;
}
static int is_rx_only_mode(void) {
    return rx_only_mode;
}

static int is_rx_unknown_mode(void) {
    return rx_unknown_mode;
}

static int should_push_tx(void) {
    return !is_rx_only_mode();
}

static double get_no_signal_power_threshold_dbm(void) {
    const char *env = getenv("JAMSYSTEM_NO_SIGNAL_POWER_DBM");
    if (env && strlen(env) > 0) {
        char *endptr = NULL;
        double value = strtod(env, &endptr);
        if (endptr && endptr != env) {
            return value;
        }
    }
    return -80.0;
}

static int no_signal_gate_enabled(void) {
    const char *env = getenv("JAMSYSTEM_ENABLE_NO_SIGNAL_GATE");
    if (env && strlen(env) > 0) {
        return strcmp(env, "0") != 0 && strcasecmp(env, "false") != 0 && strcasecmp(env, "off") != 0;
    }
    return is_rx_only_mode();
}

static int get_no_signal_confirm_frames(void) {
    const char *env = getenv("JAMSYSTEM_NO_SIGNAL_CONFIRM_FRAMES");
    if (env && strlen(env) > 0) {
        const int value = atoi(env);
        if (value >= 1) {
            return value;
        }
    }
    return 5;
}

static double compute_iq_power(const int16_t *iq_ptr, size_t complex_count) {
    if (!iq_ptr || complex_count == 0) {
        return 0.0;
    }
    double accum = 0.0;
    for (size_t idx = 0; idx < complex_count; ++idx) {
        const double i_val = (double)iq_ptr[idx * 2 + 0];
        const double q_val = (double)iq_ptr[idx * 2 + 1];
        accum += i_val * i_val + q_val * q_val;
    }
    return accum / (double)complex_count;
}

static int load_iq_template_file(const char *path, int16_t **buffer, size_t expected_count, const char *tag) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        printf("[ERR] 找不到%s: %s\n", tag, path);
        return -1;
    }

    int16_t *tmp = (int16_t *)calloc(expected_count, sizeof(int16_t));
    if (!tmp) {
        fclose(fp);
        printf("[ERR] 为%s分配内存失败\n", tag);
        return -1;
    }

    size_t nr = fread(tmp, sizeof(int16_t), expected_count, fp);
    fclose(fp);
    *buffer = tmp;
    printf("[INFO] 已加载%s: %s  (%zu 个int16)\n", tag, path, nr);
    return 0;
}


static long long monotonic_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000LL + (long long)(ts.tv_nsec / 1000000LL);
}

static int is_selftest_switch_mode(void) {
    return strcmp(run_mode, "selftest_switch") == 0 || strcmp(run_mode, "dynamic_selftest") == 0;
}

static int is_tx_jammer_only_mode(void) {
    return strcmp(run_mode, "tx_jammer_only") == 0 ||
           strcmp(run_mode, "jammer_only") == 0 ||
           strcmp(run_mode, "tx_pure_jammer") == 0;
}

static int is_tx_only_mode(void) {
    return strcmp(run_mode, "tx_only") == 0 ||
           strcmp(run_mode, "transmit_only") == 0 ||
           is_tx_jammer_only_mode();
}

static int get_selftest_switch_interval_ms(void) {
    const char *env = getenv("JAMSYSTEM_SELFTEST_SWITCH_MS");
    if (env && strlen(env) > 0) {
        const int value = atoi(env);
        if (value >= 20) {
            return value;
        }
    }
    return 1000;
}

static void clear_selftest_switch_templates(void) {
    if (selftest_switch_count > 0) {
        for (int idx = 0; idx < selftest_switch_count; ++idx) {
            free(selftest_switch_templates[idx]);
            selftest_switch_templates[idx] = NULL;
        }
        jam_template = NULL;
    } else {
        free(jam_template);
        jam_template = NULL;
    }
    selftest_switch_count = 0;
    selftest_switch_index = 0;
}

static int setup_selftest_switcher(void) {
    if (is_rx_only_mode()) {
        return 0;
    }
    if (!is_selftest_switch_mode()) {
        return 0;
    }
    const char *seq_env = getenv("JAMSYSTEM_SELFTEST_SWITCH_SEQ");

    const char *default_seq = "narrowband,wideband_barrage,comb,white_noise,noise_fm,single_tone";
    char seq_buf[512];
    snprintf(seq_buf, sizeof(seq_buf), "%s", (seq_env && strlen(seq_env) > 0) ? seq_env : default_seq);

    char *saveptr = NULL;
    char *token = strtok_r(seq_buf, ",; ", &saveptr);
    while (token && selftest_switch_count < MAX_SELFTEST_SWITCH_ITEMS) {
        char path[512];
        snprintf(path, sizeof(path), "%s/%s.bin", runtime_template_dir, token);
        int16_t *tmpl = NULL;
        if (load_iq_template_file(path, &tmpl, LENGTH * 2, "自测切换模板") == 0) {
            selftest_switch_templates[selftest_switch_count] = tmpl;
            snprintf(selftest_switch_labels[selftest_switch_count], sizeof(selftest_switch_labels[0]), "%s", token);
            ++selftest_switch_count;
        }
        token = strtok_r(NULL, ",; ", &saveptr);
    }

    if (selftest_switch_count <= 0) {
        printf("[WARN] 自测动态切换未加载到有效模板，保持固定干扰模式\n");
        return 0;
    }

    free(jam_template);
    selftest_switch_interval_ms = get_selftest_switch_interval_ms();
    selftest_switch_index = 0;
    selftest_switch_last_ms = monotonic_ms();
    jam_template = selftest_switch_templates[selftest_switch_index];
    snprintf(expected_label, sizeof(expected_label), "%s", selftest_switch_labels[selftest_switch_index]);
    is_none_mode = 0;
    printf("[INFO] 自测动态切换已启用: %d类, 周期=%lld ms, 当前=%s\n",
           selftest_switch_count, selftest_switch_interval_ms, expected_label);
    return 1;
}

static void update_selftest_switcher(void) {
    if (selftest_switch_count <= 1 || !should_push_tx()) {
        return;
    }
    const long long now = monotonic_ms();
    if (selftest_switch_last_ms <= 0) {
        selftest_switch_last_ms = now;
        return;
    }
    if (now - selftest_switch_last_ms < selftest_switch_interval_ms) {
        return;
    }
    while (now - selftest_switch_last_ms >= selftest_switch_interval_ms) {
        selftest_switch_last_ms += selftest_switch_interval_ms;
        selftest_switch_index = (selftest_switch_index + 1) % selftest_switch_count;
    }
    jam_template = selftest_switch_templates[selftest_switch_index];
    snprintf(expected_label, sizeof(expected_label), "%s", selftest_switch_labels[selftest_switch_index]);
    printf("[SWITCH] 自测干扰切换: %s\n", expected_label);
}
static void compose_tx_frame_from_useful(int16_t *dst, const int16_t *useful_template, size_t complex_count, int add_jammer) {
    const double target_jsr_db = get_target_input_jsr_db();
    const double class_gain_db = add_jammer ? get_tx_class_gain_db() : 0.0;
    const double class_gain = pow(10.0, class_gain_db / 20.0);
    const double useful_power = compute_iq_power(useful_template, complex_count) + 1e-12;
    const double jammer_power = compute_iq_power(jam_template, complex_count) + 1e-12;
    const double jammer_scale = add_jammer ? sqrt((useful_power * pow(10.0, target_jsr_db / 10.0)) / jammer_power) : 0.0;
    const double target_peak = 0.75 * 32767.0;
    double peak_abs = 1.0;
    static int tx_scale_reported = 0;

    for (size_t idx = 0; idx < complex_count; ++idx) {
        double i_val = (double)useful_template[idx * 2 + 0];
        double q_val = (double)useful_template[idx * 2 + 1];
        if (add_jammer) {
            i_val += jammer_scale * (double)jam_template[idx * 2 + 0];
            q_val += jammer_scale * (double)jam_template[idx * 2 + 1];
        }
        if (fabs(i_val) > peak_abs) peak_abs = fabs(i_val);
        if (fabs(q_val) > peak_abs) peak_abs = fabs(q_val);
    }

    const double tx_scale = peak_abs > target_peak ? target_peak / peak_abs : 1.0;
    if (!tx_scale_reported) {
        printf("[HW] TX合成: JSR=%.1fdB class_gain=%.1fdB jammer_scale=%.3f tx_scale=%.6f peak=%.1f\n",
               target_jsr_db, class_gain_db, jammer_scale, tx_scale, peak_abs);
        tx_scale_reported = 1;
    }

    for (size_t idx = 0; idx < complex_count; ++idx) {
        double i_val = (double)useful_template[idx * 2 + 0];
        double q_val = (double)useful_template[idx * 2 + 1];
        if (add_jammer) {
            i_val += jammer_scale * (double)jam_template[idx * 2 + 0];
            q_val += jammer_scale * (double)jam_template[idx * 2 + 1];
        }
        i_val *= tx_scale * class_gain;
        q_val *= tx_scale * class_gain;
        if (i_val > 32767.0) i_val = 32767.0;
        if (i_val < -32767.0) i_val = -32767.0;
        if (q_val > 32767.0) q_val = 32767.0;
        if (q_val < -32767.0) q_val = -32767.0;
        dst[idx * 2 + 0] = (int16_t)lrint(i_val);
        dst[idx * 2 + 1] = (int16_t)lrint(q_val);
    }
}

static int fill_tx_buffer_frame(void) {
    int16_t *tptr = (int16_t *)iio_buffer_start(tx_buf);
    if (!tptr) {
        return -1;
    }

    update_selftest_switcher();

    if (is_tx_jammer_only_mode()) {
        if (is_none_mode) {
            memset(tptr, 0, LENGTH * 2 * sizeof(int16_t));
        } else {
            memcpy(tptr, jam_template, LENGTH * 2 * sizeof(int16_t));
        }
        return 0;
    }

    if (is_digital_qpsk_mode()) {
        compose_tx_frame_from_useful(tptr, useful_qpsk_template, LENGTH, !is_none_mode);
        return 0;
    }

    if (is_analog_fm_mode()) {
        compose_tx_frame_from_useful(tptr, useful_fm_template, LENGTH, !is_none_mode);
        return 0;
    }

    if (is_none_mode) {
        memset(tptr, 0, LENGTH * 2 * sizeof(int16_t));
    } else {
        memcpy(tptr, jam_template, LENGTH * 2 * sizeof(int16_t));
    }
    return 0;
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

/* ===================== 2. 前后5秒数据缓存与外部存储 ===================== */
static int ensure_dir_recursive(const char *dir_path) {
    if (!dir_path || strlen(dir_path) == 0) {
        return -1;
    }

    char tmp[512];
    snprintf(tmp, sizeof(tmp), "%s", dir_path);
    size_t len = strlen(tmp);
    if (len == 0) {
        return -1;
    }
    if (tmp[len - 1] == '/') {
        tmp[len - 1] = '\0';
    }

    for (char *p = tmp + 1; *p; ++p) {
        if (*p == '/') {
            *p = '\0';
            if (mkdir(tmp, 0775) != 0 && errno != EEXIST) {
                return -1;
            }
            *p = '/';
        }
    }

    if (mkdir(tmp, 0775) != 0 && errno != EEXIST) {
        return -1;
    }
    return 0;
}

static int start_record_writer(void);

static int init_record_cache(void) {
    const size_t frame_bytes = RECORD_FRAME_INT16_COUNT * sizeof(int16_t);
    const size_t cache_bytes = RECORD_PRE_FRAMES * frame_bytes;
    record_pre_cache = (int16_t *)malloc(cache_bytes);
    if (!record_pre_cache) {
        printf("[WARN] 前5秒缓存分配失败，需要约 %.1f MB，文件输出功能关闭\n",
               cache_bytes / 1024.0 / 1024.0);
        return -1;
    }
    memset(record_pre_cache, 0, cache_bytes);

    if (ensure_dir_recursive(runtime_record_dir) != 0) {
        printf("[WARN] 无法创建记录目录: %s，文件输出功能关闭\n", runtime_record_dir);
        free(record_pre_cache);
        record_pre_cache = NULL;
        return -1;
    }

    printf("[INFO] 干扰前后数据记录目录: %s\n", runtime_record_dir);
    printf("[INFO] 前后缓存: 前%d秒 %zu帧 + 后%d秒 %zu帧，约 %.1f MB/次\n",
           RECORD_SECONDS, (size_t)RECORD_PRE_FRAMES,
           RECORD_SECONDS, (size_t)RECORD_POST_FRAMES,
           (RECORD_PRE_FRAMES + RECORD_POST_FRAMES) * frame_bytes / 1024.0 / 1024.0);
    if (start_record_writer() != 0) {
        free(record_pre_cache);
        record_pre_cache = NULL;
        return -1;
    }
    return 0;
}

static void cache_rx_frame_for_record(const int16_t *frame_ptr) {
    if (!record_pre_cache || !frame_ptr) {
        return;
    }
    memcpy(record_pre_cache + record_pre_write_idx * RECORD_FRAME_INT16_COUNT,
           frame_ptr,
           RECORD_FRAME_INT16_COUNT * sizeof(int16_t));
    record_pre_write_idx = (record_pre_write_idx + 1) % RECORD_PRE_FRAMES;
    if (record_pre_valid_frames < RECORD_PRE_FRAMES) {
        record_pre_valid_frames++;
    }
}

static int enqueue_record_write(FILE *fp, const int16_t *data, size_t int16_count,
                                int close_after, const char *file_path);

static void enqueue_pre_record_cache(FILE *fp, const char *file_path) {
    if (!fp || !record_pre_cache || record_pre_valid_frames == 0) {
        return;
    }

    const size_t start = (record_pre_valid_frames == RECORD_PRE_FRAMES) ? record_pre_write_idx : 0;
    for (size_t i = 0; i < record_pre_valid_frames; ++i) {
        const size_t idx = (start + i) % RECORD_PRE_FRAMES;
        enqueue_record_write(fp,
                             record_pre_cache + idx * RECORD_FRAME_INT16_COUNT,
                             RECORD_FRAME_INT16_COUNT,
                             0,
                             file_path);
    }
}

static int has_enough_record_space(void) {
    struct statvfs vfs;
    if (statvfs(runtime_record_dir, &vfs) != 0) {
        printf("[WARN] 无法检查记录目录剩余空间: %s\n", runtime_record_dir);
        return 0;
    }

    const unsigned long long free_bytes =
        (unsigned long long)vfs.f_bavail * (unsigned long long)vfs.f_frsize;
    const unsigned long long required_bytes =
        (unsigned long long)RECORD_MIN_FREE_MB * 1024ULL * 1024ULL;

    if (free_bytes < required_bytes) {
        printf("[WARN] U盘/记录目录剩余空间不足: %.1f MB < %d MB，本次不保存干扰前后数据\n",
               free_bytes / 1024.0 / 1024.0, RECORD_MIN_FREE_MB);
        return 0;
    }
    return 1;
}

static void *record_writer_loop(void *arg) {
    (void)arg;
    while (1) {
        pthread_mutex_lock(&record_queue_mutex);
        while (!record_queue_head && !record_writer_stop) {
            pthread_cond_wait(&record_queue_cond, &record_queue_mutex);
        }
        if (!record_queue_head && record_writer_stop) {
            pthread_mutex_unlock(&record_queue_mutex);
            break;
        }

        record_write_job_t *job = record_queue_head;
        record_queue_head = job->next;
        if (!record_queue_head) {
            record_queue_tail = NULL;
        }
        pthread_mutex_unlock(&record_queue_mutex);

        if (job->data && job->int16_count > 0 && job->fp) {
            fwrite(job->data, sizeof(int16_t), job->int16_count, job->fp);
        }
        free(job->data);

        if (job->close_after && job->fp) {
            fflush(job->fp);
            fclose(job->fp);
            if (job->close_after == RECORD_CLOSE_COMPLETED) {
                printf("[REC] 干扰变化前后各%d秒数据保存完成: %s\n", RECORD_SECONDS, job->file_path);
            } else {
                printf("[REC] 程序退出，已关闭未完成记录文件: %s\n", job->file_path);
            }
        }
        free(job);
    }
    return NULL;
}

static int start_record_writer(void) {
    if (record_writer_started) {
        return 0;
    }
    record_writer_stop = 0;
    if (pthread_create(&record_writer_thread, NULL, record_writer_loop, NULL) != 0) {
        printf("[WARN] 异步写盘线程启动失败，将无法记录bin文件\n");
        return -1;
    }
    record_writer_started = 1;
    printf("[INFO] 异步bin写入线程已启动\n");
    return 0;
}

static void stop_record_writer(void) {
    if (!record_writer_started) {
        return;
    }
    pthread_mutex_lock(&record_queue_mutex);
    record_writer_stop = 1;
    pthread_cond_signal(&record_queue_cond);
    pthread_mutex_unlock(&record_queue_mutex);
    pthread_join(record_writer_thread, NULL);
    record_writer_started = 0;
}

static int enqueue_record_write(FILE *fp, const int16_t *data, size_t int16_count,
                                int close_after, const char *file_path) {
    if (!record_writer_started || !fp) {
        return -1;
    }

    record_write_job_t *job = (record_write_job_t *)calloc(1, sizeof(record_write_job_t));
    if (!job) {
        printf("[WARN] 异步写入任务分配失败\n");
        return -1;
    }
    job->fp = fp;
    job->int16_count = int16_count;
    job->close_after = close_after;
    if (file_path) {
        snprintf(job->file_path, sizeof(job->file_path), "%s", file_path);
    }

    if (data && int16_count > 0) {
        const size_t bytes = int16_count * sizeof(int16_t);
        job->data = (int16_t *)malloc(bytes);
        if (!job->data) {
            printf("[WARN] 异步写入数据缓存分配失败，丢弃一帧记录数据\n");
            free(job);
            return -1;
        }
        memcpy(job->data, data, bytes);
    }

    pthread_mutex_lock(&record_queue_mutex);
    if (record_queue_tail) {
        record_queue_tail->next = job;
    } else {
        record_queue_head = job;
    }
    record_queue_tail = job;
    pthread_cond_signal(&record_queue_cond);
    pthread_mutex_unlock(&record_queue_mutex);
    return 0;
}

static record_task_t *alloc_record_task(void) {
    for (size_t i = 0; i < RECORD_MAX_ACTIVE; ++i) {
        if (!record_tasks[i].active && record_tasks[i].fp == NULL) {
            return &record_tasks[i];
        }
    }
    return NULL;
}

static void start_interference_recording(const char *result_id) {
    if (!record_pre_cache || is_none_mode) {
        return;
    }
    if (!result_id || strlen(result_id) == 0 || strcmp(result_id, "none") == 0) {
        return;
    }
    if (strcmp(result_id, last_record_label) == 0) {
        return;
    }

    if (ensure_dir_recursive(runtime_record_dir) != 0) {
        printf("[WARN] 记录目录不可用: %s\n", runtime_record_dir);
        return;
    }
    if (!has_enough_record_space()) {
        return;
    }

    record_task_t *task = alloc_record_task();
    if (!task) {
        printf("[WARN] 当前并行记录任务已满(%d)，本次干扰变化不新建文件\n", RECORD_MAX_ACTIVE);
        return;
    }

    time_t now = time(NULL);
    struct tm tm_now;
    localtime_r(&now, &tm_now);
    char stamp[64];
    strftime(stamp, sizeof(stamp), "%Y%m%d_%H%M%S", &tm_now);
    const unsigned long event_id = ++record_event_id;

    snprintf(task->file_path, sizeof(task->file_path),
             "%s/%s_evt%lu_%s_%s_%s_pre%d_post%d.bin",
             runtime_record_dir, stamp, event_id, modulation_mode, expected_label, result_id,
             RECORD_SECONDS, RECORD_SECONDS);

    task->fp = fopen(task->file_path, "wb");
    if (!task->fp) {
        printf("[WARN] 无法创建记录文件: %s\n", task->file_path);
        return;
    }

    enqueue_pre_record_cache(task->fp, task->file_path);
    task->active = 1;
    task->post_frames_left = RECORD_POST_FRAMES;
    snprintf(task->label, sizeof(task->label), "%s", result_id);
    snprintf(last_record_label, sizeof(last_record_label), "%s", result_id);
    printf("[REC] 触发干扰变化数据保存: %s\n", task->file_path);
    printf("[REC] 已写入触发前约%d秒缓存，继续保存触发后%d秒...\n",
           RECORD_SECONDS, RECORD_SECONDS);
}

static void append_post_record_frame(const int16_t *frame_ptr) {
    if (!frame_ptr) {
        return;
    }
    for (size_t i = 0; i < RECORD_MAX_ACTIVE; ++i) {
        record_task_t *task = &record_tasks[i];
        if (!task->active || !task->fp) {
            continue;
        }
        enqueue_record_write(task->fp, frame_ptr, RECORD_FRAME_INT16_COUNT, 0, task->file_path);
        if (task->post_frames_left > 0) {
            task->post_frames_left--;
        }
        if (task->post_frames_left == 0) {
            enqueue_record_write(task->fp, NULL, 0, RECORD_CLOSE_COMPLETED, task->file_path);
            task->fp = NULL;
            task->active = 0;
        }
    }
}

static void close_recording_if_needed(void) {
    for (size_t i = 0; i < RECORD_MAX_ACTIVE; ++i) {
        record_task_t *task = &record_tasks[i];
        if (task->fp) {
            enqueue_record_write(task->fp, NULL, 0, RECORD_CLOSE_ABORTED, task->file_path);
            task->fp = NULL;
        }
        task->active = 0;
        task->post_frames_left = 0;
    }
    stop_record_writer();
}

static int env_flag_enabled(const char *name, int default_value) {
    const char *value = getenv(name);
    if (!value || strlen(value) == 0) {
        return default_value;
    }
    return strcmp(value, "1") == 0
        || strcasecmp(value, "true") == 0
        || strcasecmp(value, "yes") == 0
        || strcasecmp(value, "on") == 0;
}

static int env_int_value(const char *name, int default_value) {
    const char *value = getenv(name);
    if (!value || strlen(value) == 0) {
        return default_value;
    }
    int parsed = atoi(value);
    return parsed > 0 ? parsed : default_value;
}

static void build_recognition_speed_env(char *speed_env, size_t speed_env_size) {
    const int fast_test = env_flag_enabled("JAMSYSTEM_FAST_TEST_MODE", 0);
    const int disable_plot = env_flag_enabled("JAMSYSTEM_DISABLE_PLOT_EXPORT", fast_test);
    const int enable_true_ber = env_flag_enabled("JAMSYSTEM_ENABLE_TRUE_BER", fast_test ? 0 : 1);
    const int disable_wideband_refine = env_flag_enabled("JAMSYSTEM_DISABLE_WIDEBAND_REFINE", 1);
    snprintf(speed_env, speed_env_size,
             "JAMSYSTEM_FAST_TEST_MODE=%d JAMSYSTEM_DISABLE_PLOT_EXPORT=%d "
             "JAMSYSTEM_FAST_RESTORE=1 JAMSYSTEM_FAST_SKIP_REFERENCE_METRICS=1 "
             "JAMSYSTEM_FAST_CLASSIFY_ONLY=%d JAMSYSTEM_RESULT_SLEEP_SEC=0 "
             "JAMSYSTEM_ENABLE_TRUE_BER=%d JAMSYSTEM_DISABLE_WIDEBAND_REFINE=%d",
             fast_test ? 1 : 0,
             disable_plot ? 1 : 0,
             fast_test ? 1 : 0,
             enable_true_ber ? 1 : 0,
             disable_wideband_refine ? 1 : 0);
}

static void stop_recognition_worker(void) {
    if (recognition_worker_in) {
        fprintf(recognition_worker_in, "__quit__\n");
        fflush(recognition_worker_in);
        fclose(recognition_worker_in);
        recognition_worker_in = NULL;
    }
    if (recognition_worker_out) {
        fclose(recognition_worker_out);
        recognition_worker_out = NULL;
    }
    if (recognition_worker_pid > 0) {
        waitpid(recognition_worker_pid, NULL, 0);
        recognition_worker_pid = -1;
    }
}

static int start_recognition_worker(const char *speed_env, const char *expected_env) {
    if (recognition_worker_in && recognition_worker_out && recognition_worker_pid > 0) {
        return 1;
    }
    if (recognition_worker_failed) {
        return 0;
    }

    int to_child[2] = {-1, -1};
    int from_child[2] = {-1, -1};
    if (pipe(to_child) != 0 || pipe(from_child) != 0) {
        printf("[WARN] 常驻Python识别管道创建失败，回退单次识别\n");
        recognition_worker_failed = 1;
        return 0;
    }

    char cmd[2048];
    if (is_digital_qpsk_mode()) {
        snprintf(cmd, sizeof(cmd),
                 "%s JAMSYSTEM_MODULATION_MODE=%s JAMSYSTEM_RUN_MODE=%s JAMSYSTEM_EXPECTED_LABEL=%s "
                 "JAMSYSTEM_QPSK_REF_BITS_FILE=%s %s -u %s --worker",
                 speed_env, modulation_mode, run_mode, expected_env,
                 runtime_useful_bits_file, runtime_python_exe, runtime_predict_script);
    } else if (is_analog_fm_mode()) {
        snprintf(cmd, sizeof(cmd),
                 "%s JAMSYSTEM_MODULATION_MODE=%s JAMSYSTEM_RUN_MODE=%s JAMSYSTEM_EXPECTED_LABEL=%s "
                 "JAMSYSTEM_FM_REF_WAVE_FILE=%s %s -u %s --worker",
                 speed_env, modulation_mode, run_mode, expected_env,
                 runtime_useful_fm_file, runtime_python_exe, runtime_predict_script);
    } else {
        snprintf(cmd, sizeof(cmd),
                 "%s JAMSYSTEM_MODULATION_MODE=%s JAMSYSTEM_RUN_MODE=%s JAMSYSTEM_EXPECTED_LABEL=%s "
                 "%s -u %s --worker",
                 speed_env, modulation_mode, run_mode, expected_env,
                 runtime_python_exe, runtime_predict_script);
    }

    pid_t pid = fork();
    if (pid < 0) {
        printf("[WARN] 常驻Python识别进程启动失败，回退单次识别\n");
        close(to_child[0]); close(to_child[1]);
        close(from_child[0]); close(from_child[1]);
        recognition_worker_failed = 1;
        return 0;
    }
    if (pid == 0) {
        dup2(to_child[0], STDIN_FILENO);
        dup2(from_child[1], STDOUT_FILENO);
        dup2(from_child[1], STDERR_FILENO);
        close(to_child[0]); close(to_child[1]);
        close(from_child[0]); close(from_child[1]);
        execl("/bin/sh", "sh", "-c", cmd, (char *)NULL);
        _exit(127);
    }

    close(to_child[0]);
    close(from_child[1]);
    recognition_worker_in = fdopen(to_child[1], "w");
    recognition_worker_out = fdopen(from_child[0], "r");
    recognition_worker_pid = pid;
    if (!recognition_worker_in || !recognition_worker_out) {
        printf("[WARN] 常驻Python识别管道打开失败，回退单次识别\n");
        stop_recognition_worker();
        recognition_worker_failed = 1;
        return 0;
    }
    setvbuf(recognition_worker_in, NULL, _IOLBF, 0);
    printf("[INFO] 常驻Python识别进程已启动，后续帧复用已加载模型\n");
    return 1;
}

static int run_recognition_worker(const char *speed_env, const char *expected_env,
                                  char *result_id, size_t result_id_size,
                                  float *confidence) {
    if (!start_recognition_worker(speed_env, expected_env)) {
        return 0;
    }
    if (result_id && result_id_size > 0) {
        result_id[0] = '\0';
    }
    if (confidence) {
        *confidence = 0.0f;
    }
    if (fprintf(recognition_worker_in, "%s\n", runtime_captured_file) < 0 || fflush(recognition_worker_in) != 0) {
        printf("[WARN] 常驻Python识别写入失败，回退单次识别\n");
        stop_recognition_worker();
        recognition_worker_failed = 1;
        return 0;
    }

    char line[1024];
    while (fgets(line, sizeof(line), recognition_worker_out) != NULL) {
        if (strstr(line, "RESULT_ID:") && result_id && result_id_size > 0) {
            sscanf(line, "RESULT_ID:%63s", result_id);
        }
        if (strstr(line, "RESULT_CONF:") && confidence) {
            sscanf(line, "RESULT_CONF:%f", confidence);
        }
        if (strstr(line, "WORKER_DONE")) {
            return result_id && strlen(result_id) > 0;
        }
        if (!strstr(line, "DEBUG_") && !strstr(line, "WORKER_BEGIN")) {
            printf("  [py] %s", line);
        }
    }

    printf("[WARN] 常驻Python识别进程无输出，回退单次识别\n");
    stop_recognition_worker();
    recognition_worker_failed = 1;
    return 0;
}

static int run_collect_only_loop(void) {
    if (ensure_dir_recursive(runtime_record_dir) != 0) {
        printf("[ERR] 采集目录不可用: %s\n", runtime_record_dir);
        return -1;
    }

    const int target_frames = env_int_value("JAMSYSTEM_COLLECT_FRAMES", COLLECT_DEFAULT_FRAMES);
    time_t now = time(NULL);
    struct tm tm_now;
    localtime_r(&now, &tm_now);
    char stamp[64];
    strftime(stamp, sizeof(stamp), "%Y%m%d_%H%M%S", &tm_now);

    char collect_path[1024];
    snprintf(collect_path, sizeof(collect_path),
             "%s/%s_collect_%s_%s_%dframes.bin",
             runtime_record_dir, stamp, modulation_mode, expected_label, target_frames);

    FILE *fp = fopen(collect_path, "wb");
    if (!fp) {
        printf("[ERR] 无法创建采集文件: %s\n", collect_path);
        return -1;
    }

    const double frame_mb = RECORD_FRAME_INT16_COUNT * sizeof(int16_t) / 1024.0 / 1024.0;
    printf("[COLLECT] 纯采集模式已启用，不运行识别/复原\n");
    printf("[COLLECT] 输出文件: %s\n", collect_path);
    printf("[COLLECT] 目标帧数: %d，每帧约 %.3f MB，总计约 %.1f MB\n",
           target_frames, frame_mb, target_frames * frame_mb);

    int saved_frames = 0;
    while (run_flag && saved_frames < target_frames) {
        if (should_push_tx() && (is_digital_qpsk_mode() || is_analog_fm_mode() || !is_none_mode)) {
            if (fill_tx_buffer_frame() == 0) {
                ssize_t pushed = iio_buffer_push(tx_buf);
                if (pushed < 0) {
                    printf("[WARN] TX push 失败: %zd\n", pushed);
                    usleep(10000);
                    continue;
                }
            }
        }

        ssize_t filled = iio_buffer_refill(rx_buf);
        if (filled < 0) {
            printf("[WARN] RX refill 失败: %zd\n", filled);
            usleep(10000);
            continue;
        }

        const int16_t *rptr = (const int16_t *)iio_buffer_start(rx_buf);
        if (!rptr) {
            printf("[WARN] RX buffer 为空\n");
            continue;
        }

        fwrite(rptr, sizeof(int16_t), RECORD_FRAME_INT16_COUNT, fp);
        saved_frames++;
        if (saved_frames == 1 || saved_frames % 20 == 0 || saved_frames == target_frames) {
            printf("[COLLECT] 已保存 %d/%d 帧\n", saved_frames, target_frames);
        }
    }

    fflush(fp);
    fclose(fp);
    printf("[COLLECT] 采集完成: %s  (%d帧，约 %.1f MB)\n",
           collect_path, saved_frames, saved_frames * frame_mb);
    return 0;
}

/* ===================== 2. 识别函数 ===================== */
int run_recognition_and_stat(double rx_power_dbm, char *detected_label, size_t detected_label_size) {
    char cmd[2048];
    char speed_env[384];
    const char *expected_env = is_rx_unknown_mode() ? "" : expected_label;
    const int fast_test = env_flag_enabled("JAMSYSTEM_FAST_TEST_MODE", 0);
    const int use_worker = env_flag_enabled("JAMSYSTEM_USE_PY_WORKER", fast_test);
    build_recognition_speed_env(speed_env, sizeof(speed_env));
    char  result_id[64] = "";
    float confidence    = 0.0f;

    printf("[py] RESULT_POWER_DBM:%.2f\n", rx_power_dbm);

    if (use_worker && run_recognition_worker(speed_env, expected_env,
                                             result_id, sizeof(result_id),
                                             &confidence)) {
        goto got_result;
    }

    if (is_digital_qpsk_mode()) {
        snprintf(cmd, sizeof(cmd),
                 "%s JAMSYSTEM_MODULATION_MODE=%s JAMSYSTEM_RUN_MODE=%s JAMSYSTEM_EXPECTED_LABEL=%s JAMSYSTEM_QPSK_REF_BITS_FILE=%s %s %s --once %s 2>&1",
                 speed_env, modulation_mode, run_mode, expected_env, runtime_useful_bits_file, runtime_python_exe, runtime_predict_script, runtime_captured_file);
    } else if (is_analog_fm_mode()) {
        snprintf(cmd, sizeof(cmd),
                 "%s JAMSYSTEM_MODULATION_MODE=%s JAMSYSTEM_RUN_MODE=%s JAMSYSTEM_EXPECTED_LABEL=%s JAMSYSTEM_FM_REF_WAVE_FILE=%s %s %s --once %s 2>&1",
                 speed_env, modulation_mode, run_mode, expected_env, runtime_useful_fm_file, runtime_python_exe, runtime_predict_script, runtime_captured_file);
    } else {
        snprintf(cmd, sizeof(cmd), "%s JAMSYSTEM_MODULATION_MODE=%s JAMSYSTEM_RUN_MODE=%s JAMSYSTEM_EXPECTED_LABEL=%s %s %s --once %s 2>&1",
                 speed_env, modulation_mode, run_mode, expected_env, runtime_python_exe, runtime_predict_script, runtime_captured_file);
    }
    FILE *fp = popen(cmd, "r");
    if (!fp) { printf("[ERR] popen failed\n"); return 0; }

    char  line[1024];

    while (fgets(line, sizeof(line), fp) != NULL) {
        if (strstr(line, "RESULT_ID:"))
            sscanf(line, "RESULT_ID:%63s", result_id);
        if (strstr(line, "RESULT_CONF:"))
            sscanf(line, "RESULT_CONF:%f", &confidence);
        if (!strstr(line, "DEBUG_"))
            printf("  [py] %s", line);
    }
    pclose(fp);

got_result:
    if (strlen(result_id) > 0) {
        if (detected_label && detected_label_size > 0) {
            snprintf(detected_label, detected_label_size, "%s", result_id);
        }
        total_recognitions++;
        if (is_rx_only_mode() || is_rx_unknown_mode()) {
            printf("\033[1;36m[#%d] 预测=%-18s 置信=%.1f%%  "
                   "功率=%.2f dBm  接收自动识别\033[0m\n",
                   total_recognitions, result_id,
                   confidence * 100.0f, rx_power_dbm);
        } else {
            int ok = (strcmp(result_id, expected_label) == 0);
            if (ok) correct_recognitions++;
            printf("\033[1;%sm[#%d] 预测=%-18s 置信=%.1f%%  "
                   "功率=%.2f dBm  累计准确率=%.1f%%\033[0m\n",
                   ok ? "32" : "31",
                   total_recognitions, result_id,
                   confidence * 100.0f, rx_power_dbm,
                   100.0 * correct_recognitions / total_recognitions);
        }
        return strcmp(result_id, "none") != 0;
    }
    return 0;
}

static void recognize_sliding_window(const int16_t *window_ptr) {
    if (!window_ptr) {
        return;
    }

    FILE *f = fopen(runtime_captured_file, "wb");
    if (!f) {
        printf("[ERR] 无法写入 %s\n", runtime_captured_file);
        return;
    }

    char detected_label[64] = "";
    const double rx_power_dbm = estimate_rx_power_dbm(window_ptr, LENGTH);
    fwrite(window_ptr, sizeof(int16_t), LENGTH * 2, f);
    fclose(f);

    const double no_signal_threshold_dbm = get_no_signal_power_threshold_dbm();
    if (no_signal_gate_enabled()) {
        const int confirm_frames = get_no_signal_confirm_frames();
        if (rx_power_dbm < no_signal_threshold_dbm) {
            no_signal_valid_streak = 0;
            printf("[py] RESULT_POWER_DBM:%.2f\n", rx_power_dbm);
            printf("[py] RESULT_ID:no_signal\n");
            printf("[py] RESULT_CONF:1.0000\n");
            printf("[INFO] 无有效信号: power=%.2f dBm threshold=%.2f dBm\n",
                   rx_power_dbm, no_signal_threshold_dbm);
            snprintf(last_record_label, sizeof(last_record_label), "%s", "none");
            return;
        }
        no_signal_valid_streak++;
        if (no_signal_valid_streak < confirm_frames) {
            printf("[py] RESULT_POWER_DBM:%.2f\n", rx_power_dbm);
            printf("[py] RESULT_ID:no_signal\n");
            printf("[py] RESULT_CONF:1.0000\n");
            printf("[INFO] 等待有效信号稳定: power=%.2f dBm threshold=%.2f dBm streak=%d/%d\n",
                   rx_power_dbm, no_signal_threshold_dbm, no_signal_valid_streak, confirm_frames);
            snprintf(last_record_label, sizeof(last_record_label), "%s", "none");
            return;
        }
    }

    const int has_interference =
        run_recognition_and_stat(rx_power_dbm, detected_label, sizeof(detected_label));
    if (strlen(detected_label) > 0 && strcmp(detected_label, "none") == 0) {
        snprintf(last_record_label, sizeof(last_record_label), "%s", "none");
    }
    if (has_interference) {
        start_interference_recording(detected_label);
    }
}

/* ===================== 3. 硬件初始化 ===================== */
int configure_hardware() {
    printf("[HW] 定位设备...\n");
    ad9361_phy = iio_context_find_device(ctx, "ad9361-phy");
    tx_dev     = iio_context_find_device(ctx, "cf-ad9361-dds-core-lpc");
    rx_dev     = iio_context_find_device(ctx, "cf-ad9361-lpc");
    if (!ad9361_phy || !rx_dev || (!is_rx_only_mode() && !tx_dev)) {
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
    if (!p_rx1 || !rx_lo || (!is_rx_only_mode() && (!p_tx1 || !tx_lo))) {
        printf("[ERR] 找不到物理通道\n"); return -1;
    }

    /* 频率 */
    const unsigned long long center_frequency_hz = get_center_frequency_hz();
    iio_channel_attr_write_longlong(rx_lo, "frequency", center_frequency_hz);
    if (!is_rx_only_mode()) {
        iio_channel_attr_write_longlong(tx_lo, "frequency", center_frequency_hz);
    }
    long long actual_lo = 0;
    iio_channel_attr_read_longlong(rx_lo, "frequency", &actual_lo);
    printf("[HW] 本振频率: %.3f GHz\n", actual_lo / 1e9);

    /* 采样率 */
    if (!is_rx_only_mode()) {
        iio_channel_attr_write_longlong(p_tx1, "sampling_frequency", SAMPLING_RATE);
    }
    iio_channel_attr_write_longlong(p_rx1, "sampling_frequency", SAMPLING_RATE);
    long long actual_fs = 0;
    iio_channel_attr_read_longlong(p_rx1, "sampling_frequency", &actual_fs);
    printf("[HW] 采样率: %.2f MHz\n", actual_fs / 1e6);

    /* 射频带宽：10MHz，确保宽带信号完整通过 */
    if (!is_rx_only_mode()) {
        iio_channel_attr_write_longlong(p_tx1, "rf_bandwidth", RF_BANDWIDTH);
    }
    iio_channel_attr_write_longlong(p_rx1, "rf_bandwidth", RF_BANDWIDTH);
    long long actual_bw = 0;
    iio_channel_attr_read_longlong(p_rx1, "rf_bandwidth", &actual_bw);
    printf("[HW] 射频带宽: %.2f MHz\n", actual_bw / 1e6);

    /* 端口 */
    if (!is_rx_only_mode()) {
        iio_channel_attr_write(p_tx1, "rf_port_select", "A");
    }
    iio_channel_attr_write(p_rx1, "rf_port_select", "A_BALANCED");

    /* 增益 */
    if (is_none_mode && !is_tx_only_mode()) {
        printf("[HW] none模式：RX AGC自动增益\n");
        iio_channel_attr_write(p_rx1, "gain_control_mode", "fast_attack");
    } else {
        const double rx_gain_db = env_double_value("JAMSYSTEM_RX_GAIN_DB", 60.0);
        if (is_rx_only_mode()) {
            printf("[HW] 双板接收模式：本机TX关闭  RX manual=%.1fdB\n", rx_gain_db);
            iio_channel_attr_write(p_rx1, "gain_control_mode", "manual");
            iio_channel_attr_write_double(p_rx1, "hardwaregain", rx_gain_db);
        } else {
            const double tx_gain_db = env_double_value("JAMSYSTEM_TX_GAIN_DB", -3.0);
            printf("[HW] 单板自测模式：TX hardwaregain=%.1fdB  RX manual=%.1fdB\n",
                   tx_gain_db, rx_gain_db);
            iio_channel_attr_write_double(p_tx1, "hardwaregain", tx_gain_db);
            iio_channel_attr_write(p_rx1, "gain_control_mode", "manual");
            iio_channel_attr_write_double(p_rx1, "hardwaregain", rx_gain_db);
        }
    }

    /* 禁用FIR */
    if (!is_rx_only_mode()) {
        iio_device_attr_write_longlong(tx_dev, "filter_fir_en", 0);
    }
    iio_device_attr_write_longlong(rx_dev, "filter_fir_en", 0);

    /* 数据流通道 */
    printf("[HW] 绑定数据流通道...\n");
    if (!is_rx_only_mode()) {
        tx_i = iio_device_find_channel(tx_dev, "voltage0", true);
        tx_q = iio_device_find_channel(tx_dev, "voltage1", true);
    }
    rx_i = iio_device_find_channel(rx_dev, "voltage0", false);
    rx_q = iio_device_find_channel(rx_dev, "voltage1", false);
    if (!rx_i || !rx_q || (!is_rx_only_mode() && (!tx_i || !tx_q))) {
        printf("[ERR] 数据流通道绑定失败\n"); return -1;
    }
    if (!is_rx_only_mode()) {
        iio_channel_enable(tx_i); iio_channel_enable(tx_q);
    }
    iio_channel_enable(rx_i); iio_channel_enable(rx_q);

    if (!is_rx_only_mode()) {
        tx_buf = iio_device_create_buffer(tx_dev, LENGTH, false);
    }
    rx_buf = iio_device_create_buffer(rx_dev, LENGTH, false);
    if (!rx_buf || (!is_rx_only_mode() && !tx_buf)) {
        printf("[ERR] 创建buffer失败\n"); return -1;
    }

    if (!is_rx_only_mode() && (is_digital_qpsk_mode() || is_analog_fm_mode() || is_none_mode)) {
        if (fill_tx_buffer_frame() < 0) {
            printf("[ERR] 初始化TX buffer失败\n"); return -1;
        }
    }

    return 0;
}
/* ===================== 4. 主程序 ===================== */
int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    if (argc < 2) {
        printf("用法: %s <jam_type|rx_unknown> [ip:x.x.x.x] [digital_qpsk|analog_fm] [self_test|rx_only|tx_only|tx_jammer_only|selftest_switch]\n", argv[0]);
        printf("示例: %s single_tone ip:192.168.1.10 digital_qpsk self_test\n", argv[0]);
        printf("纯干扰发射示例: %s wideband_barrage ip:192.168.1.10 digital_qpsk tx_jammer_only\n", argv[0]);
        printf("类型: none single_tone narrowband wideband_barrage "
               "comb white_noise noise_fm rx_unknown\n");
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
    if (argc > 4 && argv[4] && strlen(argv[4]) > 0) {
        strncpy(run_mode, argv[4], sizeof(run_mode) - 1);
    }
    rx_only_mode = (strcmp(run_mode, "rx_only") == 0 || strcmp(run_mode, "receive_only") == 0);
    rx_unknown_mode = (rx_only_mode && (strcmp(expected_label, "rx_unknown") == 0 || strcmp(expected_label, "unknown") == 0));
    is_none_mode = (strcmp(expected_label, "none") == 0);
    signal(SIGINT, handle_sig);
    const int fast_test = env_flag_enabled("JAMSYSTEM_FAST_TEST_MODE", 0);
    const int collect_only = env_flag_enabled("JAMSYSTEM_COLLECT_ONLY", 0);
    const int disable_record = env_flag_enabled(
        "JAMSYSTEM_DISABLE_RECORD",
        fast_test);

    printf("[INFO] runtime base dir: %s\n", runtime_base_dir);
    printf("[INFO] runtime python: %s\n", runtime_python_exe);
    printf("[INFO] runtime script: %s\n", runtime_predict_script);
    printf("[INFO] modulation mode: %s\n", modulation_mode);
    printf("[INFO] run mode: %s\n", rx_only_mode ? "rx_only" : (is_selftest_switch_mode() ? "selftest_switch" : (is_tx_jammer_only_mode() ? "tx_jammer_only" : (is_tx_only_mode() ? "tx_only" : "self_test"))));
    printf("[py] MODULATION_MODE:%s\n", modulation_mode);
    if (collect_only) {
        printf("[INFO] collect-only模式：只连续采集保存，不运行识别触发逻辑\n");
    } else if (disable_record) {
        printf("[INFO] 极速测试模式：关闭前后缓存和干扰变化记录\n");
    } else if (!is_none_mode) {
        init_record_cache();
    } else {
        printf("[INFO] none模式：不触发干扰前后数据保存\n");
    }

    /* 加载模板 */
    if (is_rx_unknown_mode()) {
        printf("[INFO] 双板接收自动识别：不加载干扰模板，不设置期望类型\n");
    } else {
        char path[256];
        snprintf(path, sizeof(path), "%s/%s.bin", runtime_template_dir, argv[1]);
        if (load_iq_template_file(path, &jam_template, LENGTH * 2, "干扰模板") < 0) {
            return -1;
        }
    }
    setup_selftest_switcher();

    if (is_tx_jammer_only_mode()) {
        printf("[INFO] 纯干扰发射模式：只发送 %s 模板，不叠加QPSK/FM参考信号\n", expected_label);
    } else if (is_digital_qpsk_mode()) {
        if (load_iq_template_file(runtime_useful_qpsk_file, &useful_qpsk_template, LENGTH * 2, "QPSK参考模板") < 0) {
            clear_selftest_switch_templates();
            return -1;
        }
        if (is_rx_only_mode()) {
            printf("[INFO] 数字调制链路：双板接收，本机QPSK发射关闭\n");
            printf("[INFO] 参考bits文件仅用于还原/误码指标: %s\n", runtime_useful_bits_file);
        } else {
            printf("[INFO] 数字调制链路：固定QPSK参考信号发射已启用\n");
            printf("[INFO] 参考bits文件: %s\n", runtime_useful_bits_file);
            if (is_none_mode) {
                printf("[INFO] none模式：发纯净QPSK参考信号，不加干扰\n");
            } else {
                printf("[INFO] 干扰模式：QPSK参考信号 + %s 干扰，目标JSR=%.1f dB\n", expected_label, get_target_input_jsr_db());
            }
        }
    } else if (is_analog_fm_mode()) {
        if (load_iq_template_file(runtime_useful_fm_file, &useful_fm_template, LENGTH * 2, "FM参考模板") < 0) {
            clear_selftest_switch_templates();
            return -1;
        }
        if (is_rx_only_mode()) {
            printf("[INFO] 模拟调制链路：双板接收，本机FM发射关闭\n");
            printf("[INFO] 参考FM文件仅用于还原指标: %s\n", runtime_useful_fm_file);
        } else {
            printf("[INFO] 模拟调制链路：固定FM参考信号发射已启用\n");
            printf("[INFO] 参考FM文件: %s\n", runtime_useful_fm_file);
            if (is_none_mode) {
                printf("[INFO] none模式：发纯净FM参考信号，不加干扰\n");
            } else {
                printf("[INFO] 干扰模式：FM参考信号 + %s 干扰，目标JSR=%.1f dB\n", expected_label, get_target_input_jsr_db());
            }
        }
    } else if (is_none_mode) {
        printf("[INFO] none模式：只接收，不发射\n");
    }

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
        if (should_push_tx() && (is_digital_qpsk_mode() || is_analog_fm_mode() || !is_none_mode)) {
            if (fill_tx_buffer_frame() == 0) {
                iio_buffer_push(tx_buf);
            }
        }
        iio_buffer_refill(rx_buf);
    }
    if (selftest_switch_count > 1) {
        selftest_switch_last_ms = monotonic_ms();
    }
    printf("[INFO] 预热完成，开始识别循环\n");
    if (is_rx_unknown_mode()) {
        printf("[INFO] 接收模式：自动识别，不使用期望类型\n");
    } else {
        printf("[INFO] 期望类型: %s\n", expected_label);
    }
    printf("--------------------------------------------\n");

    if (collect_only) {
        int collect_ret = run_collect_only_loop();
        if (tx_buf) iio_buffer_destroy(tx_buf);
        iio_buffer_destroy(rx_buf);
        free(record_pre_cache);
        clear_selftest_switch_templates();
        free(useful_qpsk_template);
        free(useful_fm_template);
        iio_context_destroy(ctx);
        return collect_ret;
    }

    if (is_tx_only_mode()) {
        printf("[INFO] 固定发射模式：仅执行TX发射，不运行Python识别\n");
        const int refill_rx_in_tx_only = env_flag_enabled("JAMSYSTEM_TX_ONLY_REFILL_RX", 0);
        unsigned long tx_frame_count = 0;
        time_t last_tx_log = 0;
        if (!refill_rx_in_tx_only) {
            printf("[INFO] 固定发射优化：TX-only模式默认不等待RX刷新，避免发射端阻塞\n");
        }
        while (run_flag) {
            if (should_push_tx()) {
                if (fill_tx_buffer_frame() == 0) {
                    ssize_t pushed = iio_buffer_push(tx_buf);
                    if (pushed < 0) {
                        printf("[WARN] TX push 失败: %zd\n", pushed);
                        usleep(10000);
                        continue;
                    }
                    tx_frame_count++;
                }
            }
            if (refill_rx_in_tx_only) {
                const ssize_t nbytes = iio_buffer_refill(rx_buf);
                if (nbytes < 0) {
                    printf("[WARN] RX refill 失败: %zd\n", nbytes);
                    usleep(10000);
                }
            } else {
                usleep(1000);
            }
            time_t now = time(NULL);
            if (now != last_tx_log) {
                last_tx_log = now;
                printf("[TX] 固定发射中: frame=%lu mode=%s jammer=%s\n",
                       tx_frame_count, modulation_mode, expected_label);
            }
        }

        printf("--------------------------------------------\n");
        printf("[FINAL] 固定发射结束\n");
        if (tx_buf) iio_buffer_destroy(tx_buf);
        iio_buffer_destroy(rx_buf);
        free(record_pre_cache);
        clear_selftest_switch_templates();
        free(useful_qpsk_template);
        free(useful_fm_template);
        iio_context_destroy(ctx);
        return 0;
    }

    if (is_selftest_switch_mode()) {
        printf("[INFO] 动态切换测试模式：仅执行发射切换和接收刷新，不运行Python识别\n");
        while (run_flag) {
            if (should_push_tx()) {
                if (fill_tx_buffer_frame() == 0) {
                    ssize_t pushed = iio_buffer_push(tx_buf);
                    if (pushed < 0) {
                        printf("[WARN] TX push 失败: %zd\n", pushed);
                        usleep(10000);
                        continue;
                    }
                }
            }
            ssize_t filled = iio_buffer_refill(rx_buf);
            if (filled < 0) {
                printf("[WARN] RX refill 失败: %zd\n", filled);
                usleep(10000);
                continue;
            }
        }

        printf("\n--------------------------------------------\n");
        printf("[FINAL] 动态切换测试结束\n");
        stop_recognition_worker();
        if (tx_buf) iio_buffer_destroy(tx_buf);
        iio_buffer_destroy(rx_buf);
        close_recording_if_needed();
        free(record_pre_cache);
        clear_selftest_switch_templates();
        free(useful_qpsk_template);
        free(useful_fm_template);
        iio_context_destroy(ctx);
        return 0;
    }

    /* 主循环 */
    int frame_cnt = 0;
    int prev_tail_valid = 0;
    int16_t *prev_tail = (int16_t *)malloc(SLIDE_STEP * 2 * sizeof(int16_t));
    int16_t *overlap_window = (int16_t *)malloc(LENGTH * 2 * sizeof(int16_t));
    if (!prev_tail || !overlap_window) {
        printf("[WARN] 滑动窗口缓存分配失败，退回整帧识别\n");
        free(prev_tail);
        free(overlap_window);
        prev_tail = NULL;
        overlap_window = NULL;
        prev_tail_valid = 0;
    } else {
        printf("[INFO] 滑动识别窗口: 长度=%d点，步长=%d点，重叠率=50%%\n", LENGTH, SLIDE_STEP);
    }

    while (run_flag) {
        if (should_push_tx() && (is_digital_qpsk_mode() || is_analog_fm_mode() || !is_none_mode)) {
            if (fill_tx_buffer_frame() == 0) {
                ssize_t pushed = iio_buffer_push(tx_buf);
                if (pushed < 0) {
                    printf("[WARN] TX push 失败: %zd\n", pushed);
                    usleep(10000);
                    continue;
                }
            }
        }

        ssize_t filled = iio_buffer_refill(rx_buf);
        if (filled < 0) {
            printf("[WARN] RX refill 失败: %zd\n", filled);
            usleep(10000); continue;
        }

        frame_cnt++;
        const int16_t *rptr = (const int16_t *)iio_buffer_start(rx_buf);
        if (!rptr) {
            printf("[WARN] RX buffer 为空\n");
            continue;
        }

        cache_rx_frame_for_record(rptr);
        append_post_record_frame(rptr);

        if (frame_cnt % CAPTURE_INTERVAL == 0) {
            if (!fast_test && overlap_window && prev_tail && prev_tail_valid) {
                memcpy(overlap_window, prev_tail, SLIDE_STEP * 2 * sizeof(int16_t));
                memcpy(overlap_window + SLIDE_STEP * 2,
                       rptr,
                       SLIDE_STEP * 2 * sizeof(int16_t));
                recognize_sliding_window(overlap_window);
            }
            recognize_sliding_window(rptr);
        }

        if (prev_tail) {
            memcpy(prev_tail,
                   rptr + SLIDE_STEP * 2,
                   SLIDE_STEP * 2 * sizeof(int16_t));
            prev_tail_valid = 1;
        }
    }

    printf("\n--------------------------------------------\n");
    if (is_rx_only_mode() || is_rx_unknown_mode()) {
        printf("[FINAL] 双板接收自动识别结束，总识别次数: %d\n", total_recognitions);
    } else {
        printf("[FINAL] 总识别次数: %d  正确: %d  准确率: %.1f%%\n",
               total_recognitions, correct_recognitions,
               total_recognitions > 0
                   ? 100.0 * correct_recognitions / total_recognitions
                   : 0.0);
    }

    stop_recognition_worker();
    if (tx_buf) iio_buffer_destroy(tx_buf);
    iio_buffer_destroy(rx_buf);
    close_recording_if_needed();
    free(prev_tail);
    free(overlap_window);
    free(record_pre_cache);
    clear_selftest_switch_templates();
    free(useful_qpsk_template);
    free(useful_fm_template);
    iio_context_destroy(ctx);
    return 0;
}
























