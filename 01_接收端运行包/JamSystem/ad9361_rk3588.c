#include <iio.h>
#include <stdio.h>
#include <string.h>
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
#include <pthread.h>

/* ===================== 1. 配置 ===================== */
#define RF_FREQUENCY   2480000000ULL
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

    if (base_env && strlen(base_env) > 0) {
        snprintf(runtime_base_dir, sizeof(runtime_base_dir), "%s", base_env);
    } else if (resolve_executable_dir(exe_dir, sizeof(exe_dir)) == 0) {
        snprintf(runtime_base_dir, sizeof(runtime_base_dir), "%s", exe_dir);
    } else {
        snprintf(runtime_base_dir, sizeof(runtime_base_dir), "%s", DEFAULT_BASE_DIR);
    }

    snprintf(runtime_template_dir, sizeof(runtime_template_dir), "%s/templates", runtime_base_dir);
    snprintf(runtime_captured_file, sizeof(runtime_captured_file), "%s/output/captured.bin", runtime_base_dir);
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

static int is_digital_qpsk_mode(void) {
    return strcmp(modulation_mode, "digital_qpsk") == 0;
}

static int is_analog_fm_mode(void) {
    return strcmp(modulation_mode, "analog_fm") == 0;
}
static int is_rx_only_mode(void) {
    return rx_only_mode;
}

static int should_push_tx(void) {
    return !is_rx_only_mode();
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

static void compose_tx_frame_from_useful(int16_t *dst, const int16_t *useful_template, size_t complex_count, int add_jammer) {
    const double target_jsr_db = get_target_input_jsr_db();
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
        printf("[HW] TX合成: JSR=%.1fdB jammer_scale=%.3f tx_scale=%.6f peak=%.1f\n",
               target_jsr_db, jammer_scale, tx_scale, peak_abs);
        tx_scale_reported = 1;
    }

    for (size_t idx = 0; idx < complex_count; ++idx) {
        double i_val = (double)useful_template[idx * 2 + 0];
        double q_val = (double)useful_template[idx * 2 + 1];
        if (add_jammer) {
            i_val += jammer_scale * (double)jam_template[idx * 2 + 0];
            q_val += jammer_scale * (double)jam_template[idx * 2 + 1];
        }
        i_val *= tx_scale;
        q_val *= tx_scale;
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
    char cmd[1024];
    if (is_digital_qpsk_mode()) {
        snprintf(cmd, sizeof(cmd),
                 "JAMSYSTEM_MODULATION_MODE=%s JAMSYSTEM_EXPECTED_LABEL=%s JAMSYSTEM_ENABLE_TRUE_BER=1 JAMSYSTEM_QPSK_REF_BITS_FILE=%s %s %s --once %s 2>&1",
                 modulation_mode, expected_label, runtime_useful_bits_file, runtime_python_exe, runtime_predict_script, runtime_captured_file);
    } else if (is_analog_fm_mode()) {
        snprintf(cmd, sizeof(cmd),
                 "JAMSYSTEM_MODULATION_MODE=%s JAMSYSTEM_EXPECTED_LABEL=%s JAMSYSTEM_FM_REF_WAVE_FILE=%s %s %s --once %s 2>&1",
                 modulation_mode, expected_label, runtime_useful_fm_file, runtime_python_exe, runtime_predict_script, runtime_captured_file);
    } else {
        snprintf(cmd, sizeof(cmd), "JAMSYSTEM_MODULATION_MODE=%s JAMSYSTEM_EXPECTED_LABEL=%s %s %s --once %s 2>&1",
                 modulation_mode, expected_label, runtime_python_exe, runtime_predict_script, runtime_captured_file);
    }
    FILE *fp = popen(cmd, "r");
    if (!fp) { printf("[ERR] popen failed\n"); return 0; }

    char  line[1024];
    char  result_id[64] = "";
    float confidence    = 0.0f;

    printf("[py] RESULT_POWER_DBM:%.2f\n", rx_power_dbm);

    while (fgets(line, sizeof(line), fp) != NULL) {
        if (strstr(line, "RESULT_ID:"))
            sscanf(line, "RESULT_ID:%63s", result_id);
        if (strstr(line, "RESULT_CONF:"))
            sscanf(line, "RESULT_CONF:%f", &confidence);
        if (!strstr(line, "DEBUG_"))
            printf("  [py] %s", line);
    }
    pclose(fp);

    if (strlen(result_id) > 0) {
        if (detected_label && detected_label_size > 0) {
            snprintf(detected_label, detected_label_size, "%s", result_id);
        }
        total_recognitions++;
        int ok = (strcmp(result_id, expected_label) == 0);
        if (ok) correct_recognitions++;
        printf("\033[1;%sm[#%d] 预测=%-18s 置信=%.1f%%  "
               "功率=%.2f dBm  累计准确率=%.1f%%\033[0m\n",
               ok ? "32" : "31",
               total_recognitions, result_id,
               confidence * 100.0f, rx_power_dbm,
               100.0 * correct_recognitions / total_recognitions);
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
    iio_channel_attr_write_longlong(rx_lo, "frequency", RF_FREQUENCY);
    if (!is_rx_only_mode()) {
        iio_channel_attr_write_longlong(tx_lo, "frequency", RF_FREQUENCY);
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
    if (is_none_mode) {
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

    if (!is_rx_only_mode() && (is_digital_qpsk_mode() || is_none_mode)) {
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
        printf("用法: %s <jam_type> [ip:x.x.x.x] [digital_qpsk|analog_fm] [self_test|rx_only]\n", argv[0]);
        printf("示例: %s single_tone ip:192.168.1.10 digital_qpsk self_test\n", argv[0]);
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
    if (argc > 4 && argv[4] && strlen(argv[4]) > 0) {
        strncpy(run_mode, argv[4], sizeof(run_mode) - 1);
    }
    rx_only_mode = (strcmp(run_mode, "rx_only") == 0 || strcmp(run_mode, "receive_only") == 0);
    is_none_mode = (strcmp(expected_label, "none") == 0);
    signal(SIGINT, handle_sig);
    const int collect_only = env_flag_enabled("JAMSYSTEM_COLLECT_ONLY", 0);

    printf("[INFO] runtime base dir: %s\n", runtime_base_dir);
    printf("[INFO] runtime python: %s\n", runtime_python_exe);
    printf("[INFO] runtime script: %s\n", runtime_predict_script);
    printf("[INFO] modulation mode: %s\n", modulation_mode);
    printf("[INFO] run mode: %s\n", rx_only_mode ? "rx_only" : "self_test");
    printf("[py] MODULATION_MODE:%s\n", modulation_mode);
    if (collect_only) {
        printf("[INFO] collect-only模式：只连续采集保存，不运行识别触发逻辑\n");
    } else if (!is_none_mode) {
        init_record_cache();
    } else {
        printf("[INFO] none模式：不触发干扰前后数据保存\n");
    }

    /* 加载模板 */
    char path[256];
    snprintf(path, sizeof(path), "%s/%s.bin", runtime_template_dir, argv[1]);
    if (load_iq_template_file(path, &jam_template, LENGTH * 2, "干扰模板") < 0) {
        return -1;
    }

    if (is_digital_qpsk_mode()) {
        if (load_iq_template_file(runtime_useful_qpsk_file, &useful_qpsk_template, LENGTH * 2, "QPSK参考模板") < 0) {
            free(jam_template);
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
            free(jam_template);
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
    printf("[INFO] 预热完成，开始识别循环\n");
    printf("[INFO] 期望类型: %s\n", expected_label);
    printf("--------------------------------------------\n");

    if (collect_only) {
        int collect_ret = run_collect_only_loop();
        if (tx_buf) iio_buffer_destroy(tx_buf);
        iio_buffer_destroy(rx_buf);
        free(record_pre_cache);
        free(jam_template);
        free(useful_qpsk_template);
        free(useful_fm_template);
        iio_context_destroy(ctx);
        return collect_ret;
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
            if (overlap_window && prev_tail && prev_tail_valid) {
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
    printf("[FINAL] 总识别次数: %d  正确: %d  准确率: %.1f%%\n",
           total_recognitions, correct_recognitions,
           total_recognitions > 0
               ? 100.0 * correct_recognitions / total_recognitions
               : 0.0);

    if (tx_buf) iio_buffer_destroy(tx_buf);
    iio_buffer_destroy(rx_buf);
    close_recording_if_needed();
    free(prev_tail);
    free(overlap_window);
    free(record_pre_cache);
    free(jam_template);
    free(useful_qpsk_template);
    free(useful_fm_template);
    iio_context_destroy(ctx);
    return 0;
}




















