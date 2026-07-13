#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QAbstractItemView>
#include <QComboBox>
#include <QDateTime>
#include <QDialog>
#include <QDir>
#include <QEvent>
#include <QFile>
#include <QFileInfo>
#include <QFormLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QLayout>
#include <QList>
#include <QListView>
#include <QListWidget>
#include <QProcessEnvironment>
#include <QPushButton>
#include <QRegularExpression>
#include <QSharedPointer>
#include <QSizePolicy>
#include <QStatusBar>
#include <QTextEdit>
#include <QTimer>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

constexpr int kPlotPointCount = 1024;
constexpr int kSpectrumPointCount = 512;
constexpr int kPlotColumnCount = 12;

constexpr const char *kDefaultBasePath = "/home/pi/Desktop/JamSystem";
constexpr const char *kDefaultDeviceIp = "192.168.1.10";
constexpr const char *kDefaultTrainServer = "http://192.168.137.2:8008";
constexpr const char *kDefaultUploadDir = "/mnt/usb/JamRecords";

QString stripAnsi(const QString &text)
{
    static const QRegularExpression ansiPattern(QStringLiteral("\\x1B\\[[0-9;]*[A-Za-z]"));
    QString cleaned = text;
    cleaned.remove(ansiPattern);
    return cleaned.trimmed();
}

QVector<double> buildIndexAxis(int count)
{
    QVector<double> axis;
    axis.reserve(count);
    for (int i = 0; i < count; ++i) {
        axis.append(i);
    }
    return axis;
}

} // namespace

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , m_backend(new QProcess(this))
    , m_onlineLearning(new QProcess(this))
    , m_manualStopRequested(false)
    , m_restoreMetricsSuppressed(false)
    , m_btnOpenTestPage(nullptr)
    , m_btnOpenPerformancePage(nullptr)
    , m_testDialog(nullptr)
    , m_performanceDialog(nullptr)
    , m_lineEditTrainServer(nullptr)
    , m_lineEditUploadDir(nullptr)
    , m_comboLearningMode(nullptr)
    , m_labelLearningStatus(nullptr)
    , m_textLearningLog(nullptr)
    , m_btnOnlineLearning(nullptr)
    , m_perfStatusValue(nullptr)
    , m_perfModeValue(nullptr)
    , m_perfExpectedValue(nullptr)
    , m_perfResultValue(nullptr)
    , m_perfConfValue(nullptr)
    , m_perfPowerValue(nullptr)
    , m_perfRecognitionTimeValue(nullptr)
    , m_perfRestorationTimeValue(nullptr)
    , m_perfTotalValue(nullptr)
    , m_perfCorrectValue(nullptr)
    , m_perfAccuracyValue(nullptr)
    , m_perfAvgRecognitionValue(nullptr)
    , m_perfBestRecognitionValue(nullptr)
    , m_perfAvgIntervalValue(nullptr)
    , m_perfUpdatedValue(nullptr)
    , m_perfTotalCount(0)
    , m_perfCorrectCount(0)
    , m_perfLastConfidence(-1.0)
    , m_perfLastPowerDbm(std::numeric_limits<double>::quiet_NaN())
    , m_perfLastRecognitionMs(std::numeric_limits<double>::quiet_NaN())
    , m_perfLastRestorationMs(std::numeric_limits<double>::quiet_NaN())
    , m_perfRecognitionSumMs(0.0)
    , m_perfBestRecognitionMs(std::numeric_limits<double>::quiet_NaN())
    , m_perfIntervalSumMs(0.0)
    , m_perfIntervalCount(0)
    , m_perfLastUpdateEpochMs(0)
{
    ui->setupUi(this);
    applyFullScreenLayout();
    applyUiTextOverrides();
    applyTouchScreenContrastStyle();
    applyTouchComboBoxStyle();
    setWindowTitle(QStringLiteral("RK3588无线电干扰识别与还原系统"));

    connect(m_backend, &QProcess::readyReadStandardOutput, this, &MainWindow::handleBackendOutput);
    connect(m_backend, &QProcess::errorOccurred, this, &MainWindow::handleBackendError);
    connect(m_backend,
            qOverload<int, QProcess::ExitStatus>(&QProcess::finished),
            this,
            &MainWindow::handleBackendFinished);
    connect(m_backend, &QProcess::readyReadStandardError, this, [this]() {
        const QString chunk = QString::fromLocal8Bit(m_backend->readAllStandardError()).trimmed();
        if (!chunk.isEmpty()) {
            m_lastStdErr = chunk;
            ui->statusbar->showMessage(QStringLiteral("后端日志: %1").arg(chunk), 5000);
        }
    });
    connect(m_onlineLearning,
            &QProcess::readyReadStandardOutput,
            this,
            &MainWindow::handleOnlineLearningOutput);
    connect(m_onlineLearning,
            &QProcess::readyReadStandardError,
            this,
            &MainWindow::handleOnlineLearningOutput);
    connect(m_onlineLearning,
            &QProcess::errorOccurred,
            this,
            &MainWindow::handleOnlineLearningError);
    connect(m_onlineLearning,
            qOverload<int, QProcess::ExitStatus>(&QProcess::finished),
            this,
            &MainWindow::handleOnlineLearningFinished);

    ui->lineEdit_ip->setText(QString::fromLatin1(kDefaultDeviceIp));
    ui->comboBox_2->setCurrentIndex(0);
    ui->comboBox_2->setEnabled(false);
    ui->comboBox_modulation->installEventFilter(this);
    connect(ui->btnConnect, &QPushButton::clicked, this, &MainWindow::on_btnConnect_clicked);
    connect(ui->comboBox_modulation,
            qOverload<int>(&QComboBox::currentIndexChanged),
            this,
            [this](int) {
                updateModulationMetricLabels();
                updatePerformancePage();
            });

    setupPlotStyles();
    setupTestPageEntry();
    resetPlots();
    resetMetrics();
    updatePlotPresentation();
    updateRunningState(false);
    refreshRuntimePaths();
    setStatusMessage(QStringLiteral("系统就绪，等待启动 AD9361 采集"), QStringLiteral("#3A7AFE"), 18);
    ui->statusbar->showMessage(QStringLiteral("就绪"));
}

MainWindow::~MainWindow()
{
    if (m_backend->state() != QProcess::NotRunning) {
        m_manualStopRequested = true;
        m_backend->terminate();
        if (!m_backend->waitForFinished(1500)) {
            m_backend->kill();
            m_backend->waitForFinished(1000);
        }
    }
    if (m_onlineLearning->state() != QProcess::NotRunning) {
        m_onlineLearning->terminate();
        if (!m_onlineLearning->waitForFinished(1500)) {
            m_onlineLearning->kill();
            m_onlineLearning->waitForFinished(1000);
        }
    }
    delete ui;
}

bool MainWindow::eventFilter(QObject *watched, QEvent *event)
{
    if (watched == ui->comboBox_modulation
        && (event->type() == QEvent::MouseButtonPress || event->type() == QEvent::TouchBegin)) {
        auto *combo = qobject_cast<QComboBox *>(watched);
        if (!combo || !combo->isEnabled()) {
            return true;
        }

        const QString title = QStringLiteral("选择调制方式");
        showTouchComboDialog(combo, title);
        event->accept();
        return true;
    }
    return QMainWindow::eventFilter(watched, event);
}

void MainWindow::applyUiTextOverrides()
{
    ui->groupBox->setTitle(QStringLiteral("系统控制与配置"));
    ui->label_ip->setText(QStringLiteral("设备 IP："));

    ui->groupBox_6->setTitle(QStringLiteral("运行模式"));
    ui->comboBox_2->clear();
    ui->comboBox_2->addItem(QStringLiteral("双板接收（关闭本机发射）"), QStringLiteral("rx_only"));
    ui->comboBox_2->setEnabled(false);

    ui->groupBox_modulation->setTitle(QStringLiteral("调制方式"));
    ui->comboBox_modulation->setItemText(0, QStringLiteral("数字调制（QPSK）"));
    ui->comboBox_modulation->setItemText(1, QStringLiteral("模拟调制（FM，240 kbaud）"));
    ui->comboBox_modulation->setItemData(0, QStringLiteral("digital_qpsk"));
    ui->comboBox_modulation->setItemData(1, QStringLiteral("analog_fm"));

    ui->groupBox_2->setTitle(QStringLiteral("外部发射端干扰"));
    ui->groupBox_2->setVisible(false);

    ui->btnStart->setText(QStringLiteral("启动采集与识别"));
    ui->btnStop->setText(QStringLiteral("停止"));
    ui->btnRefresh->setText(QStringLiteral("刷新状态"));

    ui->groupBox_7->setTitle(QStringLiteral("识别与还原结果"));
    ui->label_result->setText(QStringLiteral("系统就绪"));
    ui->progressBar_conf->setFormat(QStringLiteral("识别置信度 %p%"));
    ui->label_restore_status_title->setText(QStringLiteral("还原结论"));
    ui->label_restore_method_title->setText(QStringLiteral("还原方法"));
    ui->label_metric_isr_title->setText(QStringLiteral("ISR"));
    ui->label_metric_power_ratio_title->setText(QStringLiteral("干扰功率前后比"));
    ui->label_metric_power_dbm_title->setText(QStringLiteral("接收功率"));
    ui->label_metric_evm_before_title->setText(QStringLiteral("还原前 EVM"));
    ui->label_metric_evm_after_title->setText(QStringLiteral("还原后 EVM"));
    ui->label_modulation_title->setText(QStringLiteral("调制方式"));
    ui->label_recognition_time_title->setText(QStringLiteral("识别耗时"));
    ui->label_restoration_time_title->setText(QStringLiteral("复原耗时"));
    ui->label_record_status_title->setText(QStringLiteral("数据记录"));
    ui->label_record_status_value->setText(QStringLiteral("前后5秒，U盘/外部硬盘存储"));

    ui->groupBox_3->setTitle(QStringLiteral("IQ 波形前后对比"));
    ui->groupBox_5->setTitle(QStringLiteral("星座图前后对比"));
    ui->groupBox_4->setTitle(QStringLiteral("频谱前后对比"));

    updateModulationMetricLabels();
}

void MainWindow::applyTouchScreenContrastStyle()
{
    setStyleSheet(QStringLiteral(
        "QMainWindow, QWidget { background: #F2F2F2; color: #000000; }"
        "QGroupBox { background: #FFFFFF; color: #000000; border: 2px solid #000000; border-radius: 4px; margin-top: 14px; font-weight: 700; }"
        "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; background: #FFFFFF; color: #000000; }"
        "QLabel { background: transparent; color: #000000; }"
        "QLineEdit, QComboBox { background: #FFFFFF; color: #000000; border: 2px solid #000000; min-height: 34px; padding: 4px 8px; }"
        "QPushButton { background: #FFFFFF; color: #000000; border: 2px solid #000000; min-height: 34px; font-weight: 700; }"
        "QPushButton:pressed { background: #000000; color: #FFFFFF; }"
        "QPushButton:disabled, QComboBox:disabled, QLineEdit:disabled { background: #D8D8D8; color: #555555; border-color: #666666; }"
        "QProgressBar { background: #FFFFFF; color: #000000; border: 2px solid #000000; text-align: center; font-weight: 700; }"
        "QProgressBar::chunk { background: #2F6FDB; }"
        "QListWidget, QListView { background: #FFFFFF; color: #000000; border: 2px solid #000000; }"
        "QListWidget::item:selected, QListView::item:selected { background: #000000; color: #FFFFFF; }"));
}

void MainWindow::applyFullScreenLayout()
{
    auto *central = centralWidget();
    if (!central) {
        return;
    }

    QWidget *leftPanel = ui->groupBox ? ui->groupBox->parentWidget() : nullptr;
    QWidget *rightPanel = ui->groupBox_3 ? ui->groupBox_3->parentWidget() : nullptr;

    if (!leftPanel || !rightPanel || leftPanel == rightPanel) {
        return;
    }

    leftPanel->setParent(nullptr);
    rightPanel->setParent(nullptr);

    auto *mainLayout = new QHBoxLayout(central);
    mainLayout->setContentsMargins(4, 4, 4, 4);
    mainLayout->setSpacing(6);
    mainLayout->addWidget(leftPanel, 0);
    mainLayout->addWidget(rightPanel, 1);

    leftPanel->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
    rightPanel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    leftPanel->setMinimumWidth(245);
    leftPanel->setMaximumWidth(300);
    leftPanel->setMinimumHeight(0);
    rightPanel->setMinimumWidth(0);
    rightPanel->setMinimumHeight(0);

    const QList<QGroupBox *> groups = {
        ui->groupBox,
        ui->groupBox_3,
        ui->groupBox_4,
        ui->groupBox_5,
        ui->groupBox_6,
        ui->groupBox_7,
        ui->groupBox_modulation,
        ui->groupBox_2,
    };
    for (QGroupBox *group : groups) {
        if (!group) {
            continue;
        }
        group->setMinimumHeight(0);
        group->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
    }

    const QList<QCustomPlot *> plots = {
        ui->widget_iq,
        ui->widget_spec,
        ui->widget_const,
    };
    for (QCustomPlot *plot : plots) {
        if (!plot) {
            continue;
        }
        plot->setMinimumSize(0, 105);
        plot->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    }

    if (menuBar()) {
        menuBar()->hide();
    }
    if (statusBar()) {
        statusBar()->hide();
    }

    QTimer::singleShot(0, this, [this]() {
        if (centralWidget() && centralWidget()->layout()) {
            centralWidget()->layout()->invalidate();
            centralWidget()->layout()->activate();
        }
    });
}

void MainWindow::applyTouchComboBoxStyle()
{
    const QString comboStyle = QStringLiteral(
        "QComboBox { background: #FFFFFF; color: #000000; border: 2px solid #000000; min-height: 40px; padding: 4px 8px; }"
        "QComboBox QAbstractItemView { background: #FFFFFF; color: #000000; min-height: 220px; outline: 0; border: 2px solid #000000; }"
        "QComboBox QAbstractItemView::item { min-height: 42px; padding: 8px; }"
        "QComboBox QAbstractItemView::item:selected { background: #000000; color: #FFFFFF; }");

    const auto setupCombo = [comboStyle](QComboBox *combo) {
        auto *view = new QListView(combo);
        view->setMinimumHeight(220);
        view->setSpacing(2);
        view->setVerticalScrollMode(QAbstractItemView::ScrollPerPixel);
        view->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        combo->setView(view);
        combo->setMinimumHeight(40);
        combo->setMaxVisibleItems(10);
        combo->setStyleSheet(comboStyle);
        combo->setAttribute(Qt::WA_AcceptTouchEvents, true);
        combo->view()->setAttribute(Qt::WA_AcceptTouchEvents, true);
    };

    setupCombo(ui->comboBox);
    setupCombo(ui->comboBox_modulation);
    setupCombo(ui->comboBox_2);
}

void MainWindow::setupTestPageEntry()
{
    if (m_btnOpenTestPage) {
        return;
    }

    m_btnOpenPerformancePage = new QPushButton(QStringLiteral("性能展示"), ui->groupBox);
    m_btnOpenPerformancePage->setMinimumHeight(42);
    m_btnOpenPerformancePage->setStyleSheet(QStringLiteral(
        "QPushButton { background:#FFFFFF; color:#000000; border:2px solid #000000; font-size:16px; font-weight:700; }"
        "QPushButton:pressed { background:#000000; color:#FFFFFF; }"));

    m_btnOpenTestPage = new QPushButton(QStringLiteral("测试页面"), ui->groupBox);
    m_btnOpenTestPage->setMinimumHeight(42);
    m_btnOpenTestPage->setStyleSheet(QStringLiteral(
        "QPushButton { background:#FFFFFF; color:#000000; border:2px solid #000000; font-size:16px; font-weight:700; }"
        "QPushButton:pressed { background:#000000; color:#FFFFFF; }"));

    if (ui->groupBox && ui->groupBox->layout()) {
        ui->groupBox->layout()->addWidget(m_btnOpenPerformancePage);
        ui->groupBox->layout()->addWidget(m_btnOpenTestPage);
    }

    connect(m_btnOpenPerformancePage, &QPushButton::clicked, this, &MainWindow::openPerformancePage);
    connect(m_btnOpenTestPage, &QPushButton::clicked, this, &MainWindow::openTestPage);
}

QLabel *MainWindow::makePerformanceValueLabel(QWidget *parent, const QString &text) const
{
    auto *label = new QLabel(text.isEmpty() ? QStringLiteral("--") : text, parent);
    label->setAlignment(Qt::AlignCenter);
    label->setMinimumHeight(46);
    label->setWordWrap(true);
    label->setStyleSheet(QStringLiteral(
        "font-size:22px; font-weight:700; background:#FFFFFF; color:#000000; "
        "border:2px solid #000000; padding:6px;"));
    return label;
}

void MainWindow::buildPerformancePage()
{
    if (m_performanceDialog) {
        return;
    }

    m_performanceDialog = new QDialog(this);
    m_performanceDialog->setWindowTitle(QStringLiteral("接收端性能展示"));
    m_performanceDialog->setModal(false);
    m_performanceDialog->resize(980, 640);
    m_performanceDialog->setStyleSheet(QStringLiteral(
        "QDialog, QWidget { background:#F2F2F2; color:#000000; }"
        "QGroupBox { background:#FFFFFF; border:2px solid #000000; border-radius:4px; margin-top:14px; font-weight:700; }"
        "QGroupBox::title { subcontrol-origin: margin; left:8px; padding:0 4px; background:#FFFFFF; }"
        "QLabel { color:#000000; }"
        "QPushButton { background:#FFFFFF; color:#000000; border:2px solid #000000; min-height:42px; font-weight:700; }"
        "QPushButton:pressed { background:#000000; color:#FFFFFF; }"));

    auto *root = new QVBoxLayout(m_performanceDialog);
    root->setContentsMargins(16, 16, 16, 16);
    root->setSpacing(12);

    auto *title = new QLabel(QStringLiteral("接收端性能展示"), m_performanceDialog);
    title->setAlignment(Qt::AlignCenter);
    title->setStyleSheet(QStringLiteral(
        "font-size:26px; font-weight:700; background:#FFFFFF; border:2px solid #000000; padding:8px;"));
    root->addWidget(title);

    auto *note = new QLabel(QStringLiteral(
        "本端作为接收端运行：外部发送端负责发射或切换干扰样式，本端只负责实时接收、识别、复原和速度统计，不在接收端选择干扰样式。"),
        m_performanceDialog);
    note->setWordWrap(true);
    note->setStyleSheet(QStringLiteral(
        "font-size:16px; font-weight:700; background:#FFFFFF; border:2px solid #000000; padding:8px;"));
    root->addWidget(note);

    auto *statusBox = new QGroupBox(QStringLiteral("实时状态"), m_performanceDialog);
    auto *statusGrid = new QGridLayout(statusBox);
    statusGrid->setContentsMargins(12, 20, 12, 12);
    statusGrid->setHorizontalSpacing(10);
    statusGrid->setVerticalSpacing(10);

    const auto addMetric = [statusGrid, statusBox](int row, int col, const QString &name, QLabel *value) {
        auto *nameLabel = new QLabel(name, statusBox);
        nameLabel->setAlignment(Qt::AlignCenter);
        nameLabel->setMinimumHeight(32);
        nameLabel->setStyleSheet(QStringLiteral("font-size:16px; font-weight:700; background:#FFFFFF;"));
        statusGrid->addWidget(nameLabel, row * 2, col);
        statusGrid->addWidget(value, row * 2 + 1, col);
    };

    m_perfStatusValue = makePerformanceValueLabel(statusBox, QStringLiteral("等待启动"));
    m_perfModeValue = makePerformanceValueLabel(statusBox);
    m_perfExpectedValue = makePerformanceValueLabel(statusBox);
    m_perfResultValue = makePerformanceValueLabel(statusBox);
    m_perfConfValue = makePerformanceValueLabel(statusBox);
    m_perfPowerValue = makePerformanceValueLabel(statusBox);
    m_perfRecognitionTimeValue = makePerformanceValueLabel(statusBox);
    m_perfRestorationTimeValue = makePerformanceValueLabel(statusBox);
    m_perfTotalValue = makePerformanceValueLabel(statusBox);
    m_perfCorrectValue = makePerformanceValueLabel(statusBox);
    m_perfAccuracyValue = makePerformanceValueLabel(statusBox);
    m_perfAvgRecognitionValue = makePerformanceValueLabel(statusBox);
    m_perfBestRecognitionValue = makePerformanceValueLabel(statusBox);
    m_perfAvgIntervalValue = makePerformanceValueLabel(statusBox);
    m_perfUpdatedValue = makePerformanceValueLabel(statusBox);

    addMetric(0, 0, QStringLiteral("运行状态"), m_perfStatusValue);
    addMetric(0, 1, QStringLiteral("调制方式"), m_perfModeValue);
    addMetric(0, 2, QStringLiteral("外部干扰"), m_perfExpectedValue);
    addMetric(1, 0, QStringLiteral("识别结果"), m_perfResultValue);
    addMetric(1, 1, QStringLiteral("置信度"), m_perfConfValue);
    addMetric(1, 2, QStringLiteral("接收功率"), m_perfPowerValue);
    addMetric(2, 0, QStringLiteral("识别耗时"), m_perfRecognitionTimeValue);
    addMetric(2, 1, QStringLiteral("复原耗时"), m_perfRestorationTimeValue);
    addMetric(2, 2, QStringLiteral("累计次数"), m_perfTotalValue);
    addMetric(3, 0, QStringLiteral("统计方式"), m_perfCorrectValue);
    addMetric(3, 1, QStringLiteral("准确率"), m_perfAccuracyValue);
    addMetric(3, 2, QStringLiteral("平均识别耗时"), m_perfAvgRecognitionValue);
    addMetric(4, 0, QStringLiteral("最快识别耗时"), m_perfBestRecognitionValue);
    addMetric(4, 1, QStringLiteral("平均刷新周期"), m_perfAvgIntervalValue);
    addMetric(4, 2, QStringLiteral("最近更新"), m_perfUpdatedValue);

    root->addWidget(statusBox, 1);

    auto *buttonRow = new QHBoxLayout();
    auto *resetButton = new QPushButton(QStringLiteral("清零统计"), m_performanceDialog);
    auto *backButton = new QPushButton(QStringLiteral("返回主界面"), m_performanceDialog);
    buttonRow->addWidget(resetButton);
    buttonRow->addWidget(backButton);
    root->addLayout(buttonRow);

    connect(resetButton, &QPushButton::clicked, this, [this]() {
        resetPerformanceMetrics();
        updatePerformancePage();
    });
    connect(backButton, &QPushButton::clicked, this, [this]() {
        if (m_performanceDialog) {
            m_performanceDialog->hide();
        }
        showFullScreen();
        raise();
        activateWindow();
    });

    updatePerformancePage();
}

void MainWindow::buildTestPage()
{
    if (m_testDialog) {
        return;
    }

    m_testDialog = new QDialog(this);
    m_testDialog->setWindowTitle(QStringLiteral("系统测试页面"));
    m_testDialog->setModal(false);
    m_testDialog->resize(900, 560);
    m_testDialog->setStyleSheet(QStringLiteral(
        "QDialog, QWidget { background:#F2F2F2; color:#000000; }"
        "QGroupBox { background:#FFFFFF; border:2px solid #000000; border-radius:4px; margin-top:14px; font-weight:700; }"
        "QGroupBox::title { subcontrol-origin: margin; left:8px; padding:0 4px; background:#FFFFFF; }"
        "QLineEdit, QComboBox { background:#FFFFFF; color:#000000; border:2px solid #000000; min-height:34px; padding:4px 8px; }"
        "QPushButton { background:#FFFFFF; color:#000000; border:2px solid #000000; min-height:38px; font-weight:700; }"
        "QPushButton:pressed { background:#000000; color:#FFFFFF; }"
        "QPushButton:disabled { background:#D8D8D8; color:#555555; border-color:#666666; }"
        "QTextEdit { background:#FFFFFF; color:#000000; border:2px solid #000000; font-family:monospace; }"));

    auto *root = new QVBoxLayout(m_testDialog);
    root->setContentsMargins(16, 16, 16, 16);
    root->setSpacing(12);

    auto *title = new QLabel(QStringLiteral("系统测试页面"), m_testDialog);
    title->setAlignment(Qt::AlignCenter);
    title->setStyleSheet(QStringLiteral("font-size:24px; font-weight:700; background:#FFFFFF; border:2px solid #000000; padding:8px;"));
    root->addWidget(title);

    auto *learningBox = new QGroupBox(QStringLiteral("在线学习与自动部署"), m_testDialog);
    auto *learningLayout = new QVBoxLayout(learningBox);
    learningLayout->setContentsMargins(12, 18, 12, 12);
    learningLayout->setSpacing(10);

    auto *form = new QFormLayout();
    form->setLabelAlignment(Qt::AlignRight | Qt::AlignVCenter);
    form->setFormAlignment(Qt::AlignLeft | Qt::AlignTop);
    m_lineEditTrainServer = new QLineEdit(learningBox);
    QString trainServer = qEnvironmentVariable("JAMSYSTEM_TRAIN_SERVER");
    if (trainServer.isEmpty()) {
        trainServer = QString::fromLatin1(kDefaultTrainServer);
    }
    m_lineEditTrainServer->setText(trainServer);
    m_lineEditUploadDir = new QLineEdit(learningBox);
    QString uploadDirDefault = qEnvironmentVariable("JAMSYSTEM_UPLOAD_DIR");
    if (uploadDirDefault.isEmpty()) {
        uploadDirDefault = QString::fromLatin1(kDefaultUploadDir);
    }
    m_lineEditUploadDir->setText(uploadDirDefault);
    m_comboLearningMode = new QComboBox(learningBox);
    m_comboLearningMode->addItem(QStringLiteral("类增量学习（补充错分/新类样本）"), QStringLiteral("class_increment"));
    m_comboLearningMode->addItem(QStringLiteral("域增量学习（适配数字/模拟链路）"), QStringLiteral("domain_increment"));
    m_comboLearningMode->addItem(QStringLiteral("全量混合训练（原始链路+实采数据）"), QStringLiteral("mixed_train"));
    m_comboLearningMode->setMinimumHeight(38);
    form->addRow(QStringLiteral("电脑训练服务："), m_lineEditTrainServer);
    form->addRow(QStringLiteral("RK新数据目录："), m_lineEditUploadDir);
    form->addRow(QStringLiteral("学习模式："), m_comboLearningMode);
    learningLayout->addLayout(form);

    m_labelLearningStatus = new QLabel(QStringLiteral("状态：等待在线学习"), learningBox);
    m_labelLearningStatus->setStyleSheet(QStringLiteral("font-size:16px; font-weight:700; background:#FFFFFF; padding:6px;"));
    learningLayout->addWidget(m_labelLearningStatus);

    m_textLearningLog = new QTextEdit(learningBox);
    m_textLearningLog->setReadOnly(true);
    m_textLearningLog->setMinimumHeight(260);
    m_textLearningLog->setText(QStringLiteral(
        "点击“在线学习”后：\n"
        "1. RK通过另一个网口连接电脑训练服务；\n"
        "2. 上传RK本地新采集bin数据；\n"
        "3. 类增量：补充错分样本或新增干扰类别；\n"
        "4. 域增量：适配数字QPSK/模拟FM等链路差异；\n"
        "5. 全量混合训练：原始链路、历史实采和新采数据一起训练；\n"
        "6. 训练完成后转换并部署最新MobileNetV2/RKNN模型。\n"));
    learningLayout->addWidget(m_textLearningLog, 1);

    auto *buttonRow = new QHBoxLayout();
    m_btnOnlineLearning = new QPushButton(QStringLiteral("在线学习"), learningBox);
    auto *back = new QPushButton(QStringLiteral("返回主界面"), learningBox);
    m_btnOnlineLearning->setMinimumHeight(46);
    back->setMinimumHeight(46);
    buttonRow->addWidget(m_btnOnlineLearning);
    buttonRow->addWidget(back);
    learningLayout->addLayout(buttonRow);

    root->addWidget(learningBox, 1);

    connect(m_btnOnlineLearning, &QPushButton::clicked, this, &MainWindow::startOnlineLearning);
    connect(back, &QPushButton::clicked, this, [this]() {
        if (m_testDialog) {
            m_testDialog->hide();
        }
        showFullScreen();
        raise();
        activateWindow();
    });
}

void MainWindow::openTestPage()
{
    buildTestPage();
    if (!m_testDialog) {
        return;
    }
    showFullScreen();
    m_testDialog->showFullScreen();
    m_testDialog->raise();
    m_testDialog->activateWindow();
}

void MainWindow::openPerformancePage()
{
    buildPerformancePage();
    if (!m_performanceDialog) {
        return;
    }
    updatePerformancePage();
    showFullScreen();
    m_performanceDialog->showFullScreen();
    m_performanceDialog->raise();
    m_performanceDialog->activateWindow();
}

void MainWindow::resetPerformanceMetrics()
{
    m_perfTotalCount = 0;
    m_perfCorrectCount = 0;
    m_perfLastConfidence = -1.0;
    m_perfLastPowerDbm = std::numeric_limits<double>::quiet_NaN();
    m_perfLastRecognitionMs = std::numeric_limits<double>::quiet_NaN();
    m_perfLastRestorationMs = std::numeric_limits<double>::quiet_NaN();
    m_perfRecognitionSumMs = 0.0;
    m_perfBestRecognitionMs = std::numeric_limits<double>::quiet_NaN();
    m_perfIntervalSumMs = 0.0;
    m_perfIntervalCount = 0;
    m_perfLastUpdateEpochMs = 0;
    m_perfLastResultId.clear();
    m_perfLastUpdateText = QStringLiteral("--");
}

void MainWindow::updatePerformancePage()
{
    if (!m_performanceDialog) {
        return;
    }

    const bool running = (m_backend->state() != QProcess::NotRunning);
    const double avgRecognition = (m_perfTotalCount > 0 && m_perfRecognitionSumMs > 0.0)
        ? m_perfRecognitionSumMs / static_cast<double>(m_perfTotalCount)
        : std::numeric_limits<double>::quiet_NaN();
    const double avgInterval = m_perfIntervalCount > 0
        ? m_perfIntervalSumMs / static_cast<double>(m_perfIntervalCount)
        : std::numeric_limits<double>::quiet_NaN();

    const auto set = [](QLabel *label, const QString &text) {
        if (label) {
            label->setText(text);
        }
    };
    const auto msText = [](double value) {
        return std::isnan(value) ? QStringLiteral("--") : QStringLiteral("%1 ms").arg(value, 0, 'f', 3);
    };

    set(m_perfStatusValue, running ? QStringLiteral("接收识别中") : QStringLiteral("等待启动"));
    set(m_perfModeValue, QStringLiteral("%1 / %2").arg(mapModulationToCn(currentModulationArg()), mapRunModeToCn(currentRunModeArg())));
    set(m_perfExpectedValue, QStringLiteral("由外部发射端决定"));
    set(m_perfResultValue, m_perfLastResultId.isEmpty() ? QStringLiteral("--") : mapIdToCn(m_perfLastResultId));
    set(m_perfConfValue, m_perfLastConfidence < 0.0
            ? QStringLiteral("--")
            : QStringLiteral("%1 %").arg(m_perfLastConfidence * 100.0, 0, 'f', 1));
    set(m_perfPowerValue, std::isnan(m_perfLastPowerDbm)
            ? QStringLiteral("--")
            : QStringLiteral("%1 dBm").arg(m_perfLastPowerDbm, 0, 'f', 2));
    set(m_perfRecognitionTimeValue, msText(m_perfLastRecognitionMs));
    set(m_perfRestorationTimeValue, msText(m_perfLastRestorationMs));
    set(m_perfTotalValue, QString::number(m_perfTotalCount));
    set(m_perfCorrectValue, QStringLiteral("外部判定"));
    set(m_perfAccuracyValue, QStringLiteral("不统计"));
    set(m_perfAvgRecognitionValue, msText(avgRecognition));
    set(m_perfBestRecognitionValue, msText(m_perfBestRecognitionMs));
    set(m_perfAvgIntervalValue, msText(avgInterval));
    set(m_perfUpdatedValue, m_perfLastUpdateText.isEmpty() ? QStringLiteral("--") : m_perfLastUpdateText);
}

void MainWindow::notePerformanceResult(const QString &resultId)
{
    const qint64 now = QDateTime::currentMSecsSinceEpoch();
    if (m_perfLastUpdateEpochMs > 0) {
        m_perfIntervalSumMs += static_cast<double>(now - m_perfLastUpdateEpochMs);
        ++m_perfIntervalCount;
    }
    m_perfLastUpdateEpochMs = now;
    m_perfLastUpdateText = QDateTime::currentDateTime().toString(QStringLiteral("HH:mm:ss.zzz"));
    m_perfLastResultId = resultId;

    ++m_perfTotalCount;
    if (!std::isnan(m_perfLastRecognitionMs)) {
        m_perfRecognitionSumMs += m_perfLastRecognitionMs;
        if (std::isnan(m_perfBestRecognitionMs) || m_perfLastRecognitionMs < m_perfBestRecognitionMs) {
            m_perfBestRecognitionMs = m_perfLastRecognitionMs;
        }
    }

    updatePerformancePage();
}

QString MainWindow::onlineLearningPython() const
{
    const QString venvPython = m_basePath + QStringLiteral("/jam_env/bin/python3");
    if (QFileInfo::exists(venvPython)) {
        return venvPython;
    }
    return QStringLiteral("python3");
}

void MainWindow::startOnlineLearning()
{
    refreshRuntimePaths();
    if (m_onlineLearning->state() != QProcess::NotRunning) {
        if (m_labelLearningStatus) {
            m_labelLearningStatus->setText(QStringLiteral("状态：在线学习正在执行"));
        }
        return;
    }

    const QString script = m_basePath + QStringLiteral("/rk_online_learning_once.py");
    if (!QFileInfo::exists(script)) {
        if (m_labelLearningStatus) {
            m_labelLearningStatus->setText(QStringLiteral("状态：缺少脚本 rk_online_learning_once.py"));
        }
        if (m_textLearningLog) {
            m_textLearningLog->append(QStringLiteral("[ERR] 缺少脚本：%1").arg(QDir::toNativeSeparators(script)));
        }
        return;
    }

    QString server = m_lineEditTrainServer ? m_lineEditTrainServer->text().trimmed() : QString();
    if (server.isEmpty()) {
        server = QString::fromLatin1(kDefaultTrainServer);
    }
    QString uploadDir = m_lineEditUploadDir ? m_lineEditUploadDir->text().trimmed() : QString();
    if (uploadDir.isEmpty()) {
        uploadDir = QString::fromLatin1(kDefaultUploadDir);
    }
    const QString learningMode = m_comboLearningMode
        ? m_comboLearningMode->currentData().toString()
        : QStringLiteral("class_increment");

    m_onlineLearningBuffer.clear();
    if (m_textLearningLog) {
        m_textLearningLog->clear();
        m_textLearningLog->append(QStringLiteral("[GUI] 启动在线学习"));
        m_textLearningLog->append(QStringLiteral("[GUI] 训练服务：%1").arg(server));
        m_textLearningLog->append(QStringLiteral("[GUI] 新数据目录：%1").arg(uploadDir));
        m_textLearningLog->append(QStringLiteral("[GUI] 学习模式：%1").arg(learningMode));
    }
    if (m_labelLearningStatus) {
        m_labelLearningStatus->setText(QStringLiteral("状态：正在连接电脑并触发训练"));
    }
    if (m_btnOnlineLearning) {
        m_btnOnlineLearning->setEnabled(false);
    }

    const QStringList args = {
        script,
        QStringLiteral("--server"),
        server,
        QStringLiteral("--upload-dir"),
        uploadDir,
        QStringLiteral("--mode"),
        learningMode,
        QStringLiteral("--train-now"),
        QStringLiteral("--wait"),
    };

    m_onlineLearning->setWorkingDirectory(m_basePath);
    m_onlineLearning->start(onlineLearningPython(), args);
}

void MainWindow::handleOnlineLearningOutput()
{
    m_onlineLearningBuffer += QString::fromLocal8Bit(m_onlineLearning->readAllStandardOutput());
    m_onlineLearningBuffer += QString::fromLocal8Bit(m_onlineLearning->readAllStandardError());

    while (m_onlineLearningBuffer.contains('\n')) {
        const int pos = m_onlineLearningBuffer.indexOf('\n');
        const QString line = m_onlineLearningBuffer.left(pos).trimmed();
        m_onlineLearningBuffer.remove(0, pos + 1);
        if (line.isEmpty()) {
            continue;
        }
        if (m_textLearningLog) {
            m_textLearningLog->append(line);
        }
        if (m_labelLearningStatus && (line.startsWith(QStringLiteral("[ONLINE]")) ||
                                      line.startsWith(QStringLiteral("[SYNC]")) ||
                                      line.startsWith(QStringLiteral("[ERR]")))) {
            m_labelLearningStatus->setText(QStringLiteral("状态：%1").arg(line));
        }
    }
}

void MainWindow::handleOnlineLearningError(QProcess::ProcessError error)
{
    const QString text = processErrorToText(error);
    if (m_textLearningLog) {
        m_textLearningLog->append(QStringLiteral("[ERR] 在线学习进程错误：%1").arg(text));
    }
    if (m_labelLearningStatus) {
        m_labelLearningStatus->setText(QStringLiteral("状态：在线学习启动失败"));
    }
    if (m_btnOnlineLearning) {
        m_btnOnlineLearning->setEnabled(true);
    }
}

void MainWindow::handleOnlineLearningFinished(int exitCode, QProcess::ExitStatus exitStatus)
{
    handleOnlineLearningOutput();
    if (m_btnOnlineLearning) {
        m_btnOnlineLearning->setEnabled(true);
    }

    const bool ok = (exitStatus == QProcess::NormalExit && exitCode == 0);
    if (m_labelLearningStatus) {
        m_labelLearningStatus->setText(
            ok ? QStringLiteral("状态：在线学习完成，模型已自动部署到板端")
               : QStringLiteral("状态：在线学习失败，请查看日志"));
    }
    if (m_textLearningLog) {
        m_textLearningLog->append(
            ok ? QStringLiteral("[GUI] 在线学习完成")
               : QStringLiteral("[GUI] 在线学习失败，退出码 %1").arg(exitCode));
    }
}

void MainWindow::showTouchComboDialog(QComboBox *combo, const QString &title)
{
    if (!combo) {
        return;
    }

    QDialog dialog(this);
    dialog.setWindowTitle(title);
    dialog.setModal(true);
    dialog.resize(360, combo->count() > 3 ? 430 : 260);

    auto *layout = new QVBoxLayout(&dialog);
    layout->setSpacing(10);
    layout->setContentsMargins(14, 14, 14, 14);

    auto *label = new QLabel(title, &dialog);
    label->setAlignment(Qt::AlignCenter);
    label->setStyleSheet(QStringLiteral("font-size: 20px; font-weight: 700; color: #000000; background: #FFFFFF;"));
    layout->addWidget(label);

    auto *list = new QListWidget(&dialog);
    list->setStyleSheet(QStringLiteral(
        "QListWidget { font-size: 20px; background: #FFFFFF; color: #000000; border: 2px solid #000000; }"
        "QListWidget::item { min-height: 48px; padding: 8px; }"
        "QListWidget::item:selected { background: #000000; color: #FFFFFF; }"));
    list->setVerticalScrollMode(QAbstractItemView::ScrollPerPixel);
    list->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    list->setAttribute(Qt::WA_AcceptTouchEvents, true);

    for (int i = 0; i < combo->count(); ++i) {
        auto *item = new QListWidgetItem(combo->itemText(i));
        item->setData(Qt::UserRole, i);
        list->addItem(item);
        if (i == combo->currentIndex()) {
            list->setCurrentItem(item);
        }
    }
    layout->addWidget(list, 1);

    auto *cancel = new QPushButton(QStringLiteral("取消"), &dialog);
    cancel->setMinimumHeight(44);
    cancel->setStyleSheet(QStringLiteral("font-size: 18px; background: #FFFFFF; color: #000000; border: 2px solid #000000;"));
    layout->addWidget(cancel);

    connect(list, &QListWidget::itemClicked, &dialog, [&dialog, combo](QListWidgetItem *item) {
        if (!item) {
            return;
        }
        combo->setCurrentIndex(item->data(Qt::UserRole).toInt());
        dialog.accept();
    });
    connect(cancel, &QPushButton::clicked, &dialog, &QDialog::reject);

    dialog.exec();
}

void MainWindow::applyDarkPlotStyle(QCustomPlot *plot, const QString &xLabel, const QString &yLabel)
{
    plot->setBackground(QColor(250, 250, 250));
    plot->setNoAntialiasingOnDrag(true);
    plot->legend->setVisible(false);

    QPen axisPen(QColor(0, 0, 0));
    axisPen.setWidth(1);

    plot->xAxis->setBasePen(axisPen);
    plot->xAxis->setTickPen(axisPen);
    plot->xAxis->setSubTickPen(axisPen);
    plot->xAxis->setTickLabelColor(QColor(0, 0, 0));
    plot->xAxis->setLabelColor(QColor(0, 0, 0));
    plot->xAxis->setLabel(xLabel);
    plot->xAxis->grid()->setVisible(true);
    plot->xAxis->grid()->setPen(QPen(QColor(150, 150, 150), 1, Qt::DotLine));

    plot->yAxis->setBasePen(axisPen);
    plot->yAxis->setTickPen(axisPen);
    plot->yAxis->setSubTickPen(axisPen);
    plot->yAxis->setTickLabelColor(QColor(0, 0, 0));
    plot->yAxis->setLabelColor(QColor(0, 0, 0));
    plot->yAxis->setLabel(yLabel);
    plot->yAxis->grid()->setVisible(true);
    plot->yAxis->grid()->setPen(QPen(QColor(150, 150, 150), 1, Qt::DotLine));

    plot->axisRect()->setBackground(QColor(255, 255, 255));
}

void MainWindow::setupPlotStyles()
{
    applyDarkPlotStyle(ui->widget_iq, QStringLiteral("采样点"), QStringLiteral("幅度"));
    ui->widget_iq->addGraph();
    ui->widget_iq->graph(0)->setPen(QPen(QColor(180, 85, 35), 1.4, Qt::DashLine));
    ui->widget_iq->addGraph();
    ui->widget_iq->graph(1)->setPen(QPen(QColor(45, 95, 170), 1.4, Qt::DashLine));
    ui->widget_iq->addGraph();
    ui->widget_iq->graph(2)->setPen(QPen(QColor(210, 95, 20), 2.2));
    ui->widget_iq->addGraph();
    ui->widget_iq->graph(3)->setPen(QPen(QColor(20, 80, 170), 2.2));
    ui->widget_iq->xAxis->setRange(0, kPlotPointCount - 1);
    ui->widget_iq->yAxis->setRange(-1.2, 1.2);

    applyDarkPlotStyle(ui->widget_spec, QStringLiteral("频率 (kHz)"), QStringLiteral("功率 (dBm)"));
    ui->widget_spec->addGraph();
    ui->widget_spec->graph(0)->setPen(QPen(QColor(160, 70, 70), 1.5, Qt::DashLine));
    ui->widget_spec->addGraph();
    ui->widget_spec->graph(1)->setPen(QPen(QColor(0, 70, 160), 2.3));
    ui->widget_spec->xAxis->setRange(-3200, 3200);
    ui->widget_spec->yAxis->setRange(-120, 20);
    QSharedPointer<QCPAxisTickerFixed> ticker(new QCPAxisTickerFixed);
    ticker->setTickStep(1000.0);
    ui->widget_spec->xAxis->setTicker(ticker);

    applyDarkPlotStyle(ui->widget_const, QStringLiteral("I (同相)"), QStringLiteral("Q (正交)"));
    ui->widget_const->addGraph();
    ui->widget_const->graph(0)->setLineStyle(QCPGraph::lsNone);
    ui->widget_const->graph(0)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDisc, 3.5));
    ui->widget_const->graph(0)->setPen(QPen(QColor(170, 75, 40)));
    ui->widget_const->addGraph();
    ui->widget_const->graph(1)->setLineStyle(QCPGraph::lsNone);
    ui->widget_const->graph(1)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDisc, 4.0));
    ui->widget_const->graph(1)->setPen(QPen(QColor(0, 105, 70)));
    ui->widget_const->xAxis->setRange(-1.5, 1.5);
    ui->widget_const->yAxis->setRange(-1.5, 1.5);
    ui->widget_const->xAxis->setScaleRatio(ui->widget_const->yAxis, 1.0);
}

void MainWindow::resetPlots()
{
    const QVector<double> axis = buildIndexAxis(kPlotPointCount);
    QVector<double> zeros(kPlotPointCount, 0.0);
    QVector<double> specAxis;
    specAxis.reserve(kSpectrumPointCount);
    for (int i = 0; i < kSpectrumPointCount; ++i) {
        specAxis.append(-3200.0 + i * (6400.0 / (kSpectrumPointCount - 1)));
    }
    QVector<double> specBaseline(kSpectrumPointCount, -120.0);

    m_rawSpectrumSmooth = specBaseline;
    m_cleanSpectrumSmooth = specBaseline;

    for (int i = 0; i < ui->widget_iq->graphCount(); ++i) {
        ui->widget_iq->graph(i)->setData(axis, zeros);
    }
    ui->widget_iq->replot();

    ui->widget_spec->graph(0)->setData(specAxis, specBaseline);
    ui->widget_spec->graph(1)->setData(specAxis, specBaseline);
    ui->widget_spec->yAxis->setRange(-120, 20);
    ui->widget_spec->replot();

    for (int i = 0; i < ui->widget_const->graphCount(); ++i) {
        ui->widget_const->graph(i)->setData(QVector<double>(), QVector<double>());
    }
    ui->widget_const->replot();
}

void MainWindow::resetMetrics()
{
    m_restoreMetricsSuppressed = false;
    resetPerformanceMetrics();
    ui->progressBar_conf->setValue(0);
    ui->label_restore_status->setText(QStringLiteral("--"));
    ui->label_restore_method->setText(QStringLiteral("--"));
    ui->label_metric_isr->setText(QStringLiteral("--"));
    ui->label_metric_power_ratio->setText(QStringLiteral("--"));
    ui->label_metric_power_dbm->setText(QStringLiteral("--"));
    ui->label_metric_evm_before->setText(QStringLiteral("--"));
    ui->label_metric_evm_after->setText(QStringLiteral("--"));
    ui->label_modulation_value->setText(mapModulationToCn(currentModulationArg()));
    ui->label_extra_metric1_value->setText(QStringLiteral("--"));
    ui->label_extra_metric2_value->setText(QStringLiteral("--"));
    ui->label_recognition_time_value->setText(QStringLiteral("--"));
    ui->label_restoration_time_value->setText(QStringLiteral("--"));
    ui->label_record_status_value->setText(QStringLiteral("前后5秒，U盘/外部硬盘存储"));
    updateModulationMetricLabels();
    updatePerformancePage();
}

void MainWindow::updatePlotPresentation()
{
    if (m_restoreMetricsSuppressed) {
        ui->groupBox_3->setTitle(QStringLiteral("IQ 波形"));
        ui->groupBox_5->setTitle(QStringLiteral("星座图"));
        ui->groupBox_4->setTitle(QStringLiteral("频谱"));

        ui->widget_iq->graph(0)->setPen(QPen(QColor(210, 95, 20), 2.2));
        ui->widget_iq->graph(1)->setPen(QPen(QColor(20, 80, 170), 2.2));
        ui->widget_iq->graph(2)->setPen(QPen(QColor(170, 170, 170), 1.0, Qt::DotLine));
        ui->widget_iq->graph(3)->setPen(QPen(QColor(120, 120, 120), 1.0, Qt::DotLine));

        ui->widget_spec->graph(0)->setPen(QPen(QColor(0, 70, 160), 2.3));
        ui->widget_spec->graph(1)->setPen(QPen(QColor(180, 180, 180), 1.0, Qt::DotLine));

        ui->widget_const->graph(0)->setPen(QPen(QColor(0, 105, 70)));
        ui->widget_const->graph(1)->setPen(QPen(QColor(170, 170, 170)));
    } else {
        ui->groupBox_3->setTitle(QStringLiteral("IQ 波形前后对比"));
        ui->groupBox_5->setTitle(QStringLiteral("星座图前后对比"));
        ui->groupBox_4->setTitle(QStringLiteral("频谱前后对比"));

        ui->widget_iq->graph(0)->setPen(QPen(QColor(180, 85, 35), 1.4, Qt::DashLine));
        ui->widget_iq->graph(1)->setPen(QPen(QColor(45, 95, 170), 1.4, Qt::DashLine));
        ui->widget_iq->graph(2)->setPen(QPen(QColor(210, 95, 20), 2.2));
        ui->widget_iq->graph(3)->setPen(QPen(QColor(20, 80, 170), 2.2));

        ui->widget_spec->graph(0)->setPen(QPen(QColor(160, 70, 70), 1.5, Qt::DashLine));
        ui->widget_spec->graph(1)->setPen(QPen(QColor(0, 70, 160), 2.3));

        ui->widget_const->graph(0)->setPen(QPen(QColor(170, 75, 40)));
        ui->widget_const->graph(1)->setPen(QPen(QColor(0, 105, 70)));
    }
}

void MainWindow::updateRunningState(bool running)
{
    ui->btnStart->setEnabled(!running);
    ui->btnStop->setEnabled(running);
    ui->btnRefresh->setEnabled(true);
    ui->lineEdit_ip->setEnabled(!running);
    ui->comboBox_modulation->setEnabled(!running);
    ui->comboBox_2->setEnabled(false);
    updatePerformancePage();
}

bool MainWindow::refreshRuntimePaths()
{
    const QString envBase = qEnvironmentVariable("JAMSYSTEM_BASE_PATH");
    m_basePath = envBase.isEmpty() ? QString::fromLatin1(kDefaultBasePath) : QDir::fromNativeSeparators(envBase);

    const QString envExe = qEnvironmentVariable("JAMSYSTEM_AD9361_EXE");
    m_backendExe = envExe.isEmpty()
        ? (m_basePath + QStringLiteral("/ad9361_rk3588"))
        : QDir::fromNativeSeparators(envExe);

    m_backend->setWorkingDirectory(m_basePath);

    const bool ok = QFileInfo::exists(m_backendExe) &&
                    QFileInfo::exists(m_basePath + QStringLiteral("/output")) &&
                    QFileInfo::exists(m_basePath + QStringLiteral("/templates"));

    ui->statusbar->showMessage(
        QStringLiteral("JamSystem: %1 | ad9361: %2")
            .arg(QDir::toNativeSeparators(m_basePath))
            .arg(QDir::toNativeSeparators(m_backendExe)));
    return ok;
}

void MainWindow::setStatusMessage(const QString &text, const QString &color, int fontSize, bool bold)
{
    Q_UNUSED(color);
    const QString weight = bold ? QStringLiteral("700") : QStringLiteral("500");
    ui->label_result->setEnabled(true);
    ui->label_result->setStyleSheet(
        QStringLiteral("color:#000000; background:#FFFFFF; font-size:%1pt; font-weight:%2;")
            .arg(fontSize)
            .arg(weight));
    ui->label_result->setText(text);
    ui->statusbar->showMessage(text, 4000);
}

void MainWindow::on_btnStart_clicked()
{
    if (!refreshRuntimePaths()) {
        setStatusMessage(QStringLiteral("RK3588 路径检查失败，请确认 JamSystem 和 ad9361_rk3588 存在"),
                         QStringLiteral("#E05A5A"), 15, true);
        return;
    }

    const QString jamType = QStringLiteral("none");
    const QString modulationArg = currentModulationArg();
    const QString runModeArg = currentRunModeArg();
    const QString templateBin = m_basePath + QStringLiteral("/templates/") + jamType + QStringLiteral(".bin");
    if (!QFileInfo::exists(templateBin)) {
        setStatusMessage(QStringLiteral("模板文件不存在：%1").arg(QDir::toNativeSeparators(templateBin)),
                         QStringLiteral("#E05A5A"), 14, true);
        return;
    }

    QString ip = ui->lineEdit_ip->text().trimmed();
    if (ip.startsWith(QStringLiteral("ip:"), Qt::CaseInsensitive)) {
        ip = ip.mid(3).trimmed();
        ui->lineEdit_ip->setText(ip);
    }
    if (ip.isEmpty()) {
        ip = QString::fromLatin1(kDefaultDeviceIp);
        ui->lineEdit_ip->setText(ip);
    }
    const QString uri = ip.startsWith(QStringLiteral("ip:")) ? ip : QStringLiteral("ip:%1").arg(ip);

    m_outputBuffer.clear();
    m_lastStdErr.clear();
    m_manualStopRequested = false;

    if (m_backend->state() != QProcess::NotRunning) {
        m_manualStopRequested = true;
        m_backend->terminate();
        m_backend->waitForFinished(1000);
        m_manualStopRequested = false;
    }

    resetPlots();
    resetMetrics();
    updateRunningState(true);

    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    env.insert(QStringLiteral("JAMSYSTEM_MODULATION_MODE"), modulationArg);
    m_backend->setProcessEnvironment(env);

    const QStringList args = {
        jamType,
        uri,
        modulationArg,
        runModeArg,
    };
    m_backend->start(m_backendExe, args);
    setStatusMessage(QStringLiteral("正在启动 AD9361 %1...").arg(mapRunModeToCn(runModeArg)), QStringLiteral("#3A7AFE"), 18);
}

void MainWindow::on_btnStop_clicked()
{
    m_manualStopRequested = true;
    if (m_backend->state() != QProcess::NotRunning) {
        m_backend->terminate();
        if (!m_backend->waitForFinished(1500)) {
            m_backend->kill();
        }
    }
    updateRunningState(false);
    setStatusMessage(QStringLiteral("采集与识别已停止"), QStringLiteral("#6B7280"), 18);
    ui->statusbar->showMessage(QStringLiteral("监测已停止"));
}

void MainWindow::on_btnRefresh_clicked()
{
    m_manualStopRequested = true;
    if (m_backend->state() != QProcess::NotRunning) {
        m_backend->terminate();
        if (!m_backend->waitForFinished(1200)) {
            m_backend->kill();
            m_backend->waitForFinished(800);
        }
    }

    m_outputBuffer.clear();
    m_lastStdErr.clear();
    m_rawSpectrumSmooth.clear();
    m_cleanSpectrumSmooth.clear();
    m_restoreMetricsSuppressed = false;

    refreshRuntimePaths();
    resetPlots();
    resetMetrics();
    updatePlotPresentation();
    updateRunningState(false);
    setStatusMessage(QStringLiteral("状态已刷新，正在重新检测设备"), QStringLiteral("#3A7AFE"), 17);
    ui->statusbar->showMessage(QStringLiteral("状态已刷新"));

    checkDeviceConnection();
}

void MainWindow::handleBackendOutput()
{
    m_outputBuffer += QString::fromLocal8Bit(m_backend->readAllStandardOutput());

    while (m_outputBuffer.contains('\n')) {
        const int pos = m_outputBuffer.indexOf('\n');
        const QString line = m_outputBuffer.left(pos).trimmed();
        m_outputBuffer.remove(0, pos + 1);
        handleBackendLine(line);
    }
}

void MainWindow::handleBackendLine(const QString &rawLine)
{
    const QString line = stripAnsi(rawLine);
    if (line.isEmpty()) {
        return;
    }

    const int pyIndex = line.indexOf(QStringLiteral("[py]"));
    if (pyIndex >= 0) {
        const QString pyLine = line.mid(pyIndex + 4).trimmed();
        if (handlePythonProtocolLine(pyLine)) {
            return;
        }
    }

    if (handlePythonProtocolLine(line)) {
        return;
    }

    if (parseBackendSummaryLine(line)) {
        return;
    }

    if (line.startsWith(QStringLiteral("[REC]"))) {
        if (line.contains(QStringLiteral("触发干扰")) ||
            line.contains(QStringLiteral("继续保存"))) {
            ui->label_record_status_value->setText(QStringLiteral("正在保存至U盘/外部硬盘"));
        } else if (line.contains(QStringLiteral("保存完成")) ||
                   line.contains(QStringLiteral("已完成"))) {
            ui->label_record_status_value->setText(QStringLiteral("已保存至U盘/外部硬盘"));
        } else if (line.contains(QStringLiteral("空间不足")) ||
                   line.contains(QStringLiteral("跳过保存"))) {
            ui->label_record_status_value->setText(QStringLiteral("空间不足，未保存"));
        }
        ui->statusbar->showMessage(line, 5000);
        return;
    }

    if (line.startsWith(QStringLiteral("[INFO]")) || line.startsWith(QStringLiteral("[HW]"))) {
        ui->statusbar->showMessage(line, 3000);
        return;
    }

    if (line.startsWith(QStringLiteral("[WARN]"))) {
        ui->statusbar->showMessage(line, 5000);
        return;
    }

    if (line.startsWith(QStringLiteral("[ERR]"))) {
        setStatusMessage(line, QStringLiteral("#E05A5A"), 14, true);
        return;
    }

    if (line.startsWith(QStringLiteral("[FINAL]"))) {
        ui->statusbar->showMessage(line, 5000);
        return;
    }
}

bool MainWindow::handlePythonProtocolLine(const QString &line)
{
    if (line.isEmpty()) {
        return false;
    }

    if (line == QStringLiteral("PYTHON_STARTED")) {
        setStatusMessage(QStringLiteral("Python 已启动，正在加载模型..."), QStringLiteral("#3A7AFE"), 18);
        return true;
    }

    if (line == QStringLiteral("MODEL_LOADED")) {
        setStatusMessage(QStringLiteral("模型已加载，正在处理真实采集信号..."), QStringLiteral("#36B37E"), 18);
        return true;
    }

    if (line == QStringLiteral("PLOT_READY")) {
        updatePlots();
        return true;
    }

    if (line.startsWith(QStringLiteral("MODULATION_MODE:"))) {
        const QString mode = line.section(':', 1).trimmed();
        ui->label_modulation_value->setText(mapModulationToCn(mode));
        return true;
    }

    if (line.startsWith(QStringLiteral("RESULT_ID:"))) {
        const QString resultId = line.section(':', 1).trimmed();
        setStatusMessage(mapIdToCn(resultId), QStringLiteral("#FF6B35"), 24, true);
        notePerformanceResult(resultId);
        return true;
    }

    if (line.startsWith(QStringLiteral("RESULT_CONF:"))) {
        const double conf = line.section(':', 1).trimmed().toDouble();
        ui->progressBar_conf->setValue(qBound(0, static_cast<int>(conf * 100.0), 100));
        m_perfLastConfidence = conf;
        updatePerformancePage();
        return true;
    }

    if (line.startsWith(QStringLiteral("RESTORE_STATUS:"))) {
        const QString status = line.section(':', 1).trimmed();
        m_restoreMetricsSuppressed = (status == QStringLiteral("not_required"));
        updatePlotPresentation();
        ui->label_restore_status->setText(mapRestoreStatus(status));
        if (m_restoreMetricsSuppressed) {
            ui->label_restore_method->setText(QStringLiteral("无需还原"));
            ui->label_metric_isr->setText(QStringLiteral("--"));
            ui->label_metric_power_ratio->setText(QStringLiteral("--"));
            ui->label_metric_evm_before->setText(QStringLiteral("--"));
            ui->label_metric_evm_after->setText(QStringLiteral("--"));
            ui->label_extra_metric1_value->setText(QStringLiteral("--"));
            ui->label_extra_metric2_value->setText(QStringLiteral("--"));
            ui->label_restoration_time_value->setText(QStringLiteral("--"));
            m_perfLastRestorationMs = std::numeric_limits<double>::quiet_NaN();
            updatePerformancePage();
        }
        return true;
    }

    if (line.startsWith(QStringLiteral("RESTORE_METHOD:"))) {
        if (!m_restoreMetricsSuppressed) {
            ui->label_restore_method->setText(mapRestoreMethod(line.section(':', 1).trimmed()));
        }
        return true;
    }

    if (line.startsWith(QStringLiteral("RESTORE_ISR:"))) {
        if (!m_restoreMetricsSuppressed) {
            ui->label_metric_isr->setText(QStringLiteral("%1 dB").arg(line.section(':', 1).trimmed()));
        }
        return true;
    }

    if (line.startsWith(QStringLiteral("RESTORE_POWER_RATIO:"))) {
        if (!m_restoreMetricsSuppressed) {
            const QString value = line.section(':', 1).trimmed();
            if (value == QStringLiteral("--") || value.isEmpty()) {
                ui->label_metric_power_ratio->setText(QStringLiteral("--"));
            } else {
                ui->label_metric_power_ratio->setText(QStringLiteral("%1 倍").arg(value));
            }
        }
        return true;
    }

    if (line.startsWith(QStringLiteral("RESULT_POWER_DBM:"))) {
        const double powerDbm = line.section(':', 1).trimmed().toDouble();
        ui->label_metric_power_dbm->setText(QStringLiteral("%1 dBm").arg(line.section(':', 1).trimmed()));
        m_perfLastPowerDbm = powerDbm;
        updatePerformancePage();
        return true;
    }

    if (line.startsWith(QStringLiteral("RESTORE_EVM_BEFORE:"))) {
        if (!m_restoreMetricsSuppressed) {
            ui->label_metric_evm_before->setText(QStringLiteral("%1 %").arg(line.section(':', 1).trimmed()));
        }
        return true;
    }

    if (line.startsWith(QStringLiteral("RESTORE_EVM_AFTER:"))) {
        if (!m_restoreMetricsSuppressed) {
            ui->label_metric_evm_after->setText(QStringLiteral("%1 %").arg(line.section(':', 1).trimmed()));
        }
        return true;
    }

    if (line.startsWith(QStringLiteral("RESTORE_BER_BEFORE:"))) {
        if (!m_restoreMetricsSuppressed) {
            ui->label_extra_metric1_value->setText(line.section(':', 1).trimmed());
        }
        return true;
    }

    if (line.startsWith(QStringLiteral("RESTORE_BER_AFTER:"))) {
        if (!m_restoreMetricsSuppressed) {
            ui->label_extra_metric2_value->setText(line.section(':', 1).trimmed());
        }
        return true;
    }

    if (line.startsWith(QStringLiteral("RESTORE_CORR:"))) {
        if (!m_restoreMetricsSuppressed) {
            ui->label_extra_metric1_value->setText(line.section(':', 1).trimmed());
        }
        return true;
    }

    if (line.startsWith(QStringLiteral("RESTORE_NMSE:"))) {
        if (!m_restoreMetricsSuppressed) {
            ui->label_extra_metric2_value->setText(QStringLiteral("%1 dB").arg(line.section(':', 1).trimmed()));
        }
        return true;
    }

    if (line.startsWith(QStringLiteral("RECOGNITION_TIME_MS:"))) {
        const double recognitionMs = line.section(':', 1).trimmed().toDouble();
        ui->label_recognition_time_value->setText(QStringLiteral("%1 ms").arg(line.section(':', 1).trimmed()));
        m_perfLastRecognitionMs = recognitionMs;
        updatePerformancePage();
        return true;
    }

    if (line.startsWith(QStringLiteral("RESTORATION_TIME_MS:"))) {
        const double restorationMs = line.section(':', 1).trimmed().toDouble();
        if (!m_restoreMetricsSuppressed) {
            ui->label_restoration_time_value->setText(
                QStringLiteral("%1 ms").arg(line.section(':', 1).trimmed()));
            m_perfLastRestorationMs = restorationMs;
        } else {
            m_perfLastRestorationMs = std::numeric_limits<double>::quiet_NaN();
        }
        updatePerformancePage();
        return true;
    }

    if (line.startsWith(QStringLiteral("RESTORE_ERROR:"))) {
        ui->statusbar->showMessage(QStringLiteral("还原回退: %1").arg(line.section(':', 1).trimmed()), 5000);
        return true;
    }

    if (line.startsWith(QStringLiteral("MODEL_LOAD_ERROR:")) ||
        line.startsWith(QStringLiteral("FILE_READ_ERROR:"))) {
        setStatusMessage(line, QStringLiteral("#E05A5A"), 14, true);
        updateRunningState(false);
        return true;
    }

    if (line.startsWith(QStringLiteral(">>>"))) {
        ui->statusbar->showMessage(line, 3000);
        return true;
    }

    return false;
}

bool MainWindow::parseBackendSummaryLine(const QString &line)
{
    if (!line.startsWith(QStringLiteral("[#"))) {
        return false;
    }

    static const QRegularExpression summaryPattern(
        QStringLiteral(R"(\[#\d+\]\s+.*?=([A-Za-z0-9_]+)\s+.*?=([0-9.]+)%.*)"));
    const QRegularExpressionMatch match = summaryPattern.match(line);
    if (!match.hasMatch()) {
        return false;
    }

    const QString resultId = match.captured(1);
    const double confidence = match.captured(2).toDouble();
    setStatusMessage(mapIdToCn(resultId), QStringLiteral("#FF6B35"), 24, true);
    ui->progressBar_conf->setValue(qBound(0, static_cast<int>(confidence), 100));
    ui->statusbar->showMessage(line, 3000);
    return true;
}

void MainWindow::updatePlots()
{
    const QString binPath = m_basePath + QStringLiteral("/output/live_plot.bin");
    QFile file(binPath);
    if (!file.open(QIODevice::ReadOnly)) {
        return;
    }

    const QByteArray bytes = file.readAll();
    const int expectedBytes = kPlotPointCount * kPlotColumnCount * static_cast<int>(sizeof(float));
    if (bytes.size() < expectedBytes) {
        return;
    }

    const auto *data = reinterpret_cast<const float *>(bytes.constData());
    QVector<double> axis;
    QVector<double> rawI;
    QVector<double> rawQ;
    QVector<double> cleanI;
    QVector<double> cleanQ;
    QVector<double> rawFreq;
    QVector<double> rawPsd;
    QVector<double> cleanFreq;
    QVector<double> cleanPsd;
    QVector<double> rawConstI;
    QVector<double> rawConstQ;
    QVector<double> cleanConstI;
    QVector<double> cleanConstQ;

    axis.reserve(kPlotPointCount);
    rawI.reserve(kPlotPointCount);
    rawQ.reserve(kPlotPointCount);
    cleanI.reserve(kPlotPointCount);
    cleanQ.reserve(kPlotPointCount);
    rawFreq.reserve(kSpectrumPointCount);
    rawPsd.reserve(kSpectrumPointCount);
    cleanFreq.reserve(kSpectrumPointCount);
    cleanPsd.reserve(kSpectrumPointCount);
    rawConstI.reserve(kPlotPointCount);
    rawConstQ.reserve(kPlotPointCount);
    cleanConstI.reserve(kPlotPointCount);
    cleanConstQ.reserve(kPlotPointCount);

    for (int k = 0; k < kPlotPointCount; ++k) {
        const int offset = k * kPlotColumnCount;
        axis.append(k);
        rawI.append(static_cast<double>(data[offset + 0]));
        rawQ.append(static_cast<double>(data[offset + 1]));
        cleanI.append(static_cast<double>(data[offset + 4]));
        cleanQ.append(static_cast<double>(data[offset + 5]));
        rawConstI.append(static_cast<double>(data[offset + 8]));
        rawConstQ.append(static_cast<double>(data[offset + 9]));
        cleanConstI.append(static_cast<double>(data[offset + 10]));
        cleanConstQ.append(static_cast<double>(data[offset + 11]));

        if (k < kSpectrumPointCount) {
            rawFreq.append(static_cast<double>(data[offset + 2]) / 1000.0);
            rawPsd.append(static_cast<double>(data[offset + 3]));
            cleanFreq.append(static_cast<double>(data[offset + 6]) / 1000.0);
            cleanPsd.append(static_cast<double>(data[offset + 7]));
        }
    }

    const auto rawMinMax = std::minmax_element(rawI.cbegin(), rawI.cend());
    const auto rawQMinMax = std::minmax_element(rawQ.cbegin(), rawQ.cend());
    const auto cleanMinMax = std::minmax_element(cleanI.cbegin(), cleanI.cend());
    const auto cleanQMinMax = std::minmax_element(cleanQ.cbegin(), cleanQ.cend());
    const double wavePeak = std::max({
        std::abs(*rawMinMax.first),
        std::abs(*rawMinMax.second),
        std::abs(*rawQMinMax.first),
        std::abs(*rawQMinMax.second),
        std::abs(*cleanMinMax.first),
        std::abs(*cleanMinMax.second),
        std::abs(*cleanQMinMax.first),
        std::abs(*cleanQMinMax.second),
        0.3,
    });
    const double targetWaveRange = std::clamp(wavePeak * 1.25, 0.8, 2.2);
    const QCPRange currentWaveRange = ui->widget_iq->yAxis->range();
    const double blendedWave = currentWaveRange.upper * 0.75 + targetWaveRange * 0.25;
    ui->widget_iq->yAxis->setRange(-blendedWave, blendedWave);

    ui->widget_iq->graph(0)->setData(axis, rawI);
    ui->widget_iq->graph(1)->setData(axis, rawQ);
    if (m_restoreMetricsSuppressed) {
        ui->widget_iq->graph(2)->setData(QVector<double>(), QVector<double>());
        ui->widget_iq->graph(3)->setData(QVector<double>(), QVector<double>());
    } else {
        ui->widget_iq->graph(2)->setData(axis, cleanI);
        ui->widget_iq->graph(3)->setData(axis, cleanQ);
    }
    ui->widget_iq->replot(QCustomPlot::rpQueuedReplot);

    ui->widget_const->graph(0)->setData(rawConstI, rawConstQ);
    if (m_restoreMetricsSuppressed) {
        ui->widget_const->graph(1)->setData(QVector<double>(), QVector<double>());
    } else {
        ui->widget_const->graph(1)->setData(cleanConstI, cleanConstQ);
    }
    ui->widget_const->replot(QCustomPlot::rpQueuedReplot);

    if (m_rawSpectrumSmooth.size() != rawPsd.size()) {
        m_rawSpectrumSmooth = rawPsd;
        m_cleanSpectrumSmooth = cleanPsd;
    } else {
        for (int i = 0; i < rawPsd.size(); ++i) {
            m_rawSpectrumSmooth[i] = m_rawSpectrumSmooth[i] * 0.70 + rawPsd[i] * 0.30;
            m_cleanSpectrumSmooth[i] = m_cleanSpectrumSmooth[i] * 0.70 + cleanPsd[i] * 0.30;
        }
    }

    const auto rawSpecMinMax = std::minmax_element(m_rawSpectrumSmooth.cbegin(), m_rawSpectrumSmooth.cend());
    const auto cleanSpecMinMax = std::minmax_element(m_cleanSpectrumSmooth.cbegin(), m_cleanSpectrumSmooth.cend());
    const double targetLower = std::clamp(
        std::floor(std::min(*rawSpecMinMax.first, *cleanSpecMinMax.first) - 6.0), -140.0, -20.0);
    const double targetUpper = std::clamp(
        std::ceil(std::max(*rawSpecMinMax.second, *cleanSpecMinMax.second) + 6.0), -60.0, 30.0);
    const QCPRange currentSpecRange = ui->widget_spec->yAxis->range();
    const double blendedLower = currentSpecRange.lower * 0.82 + targetLower * 0.18;
    const double blendedUpper = currentSpecRange.upper * 0.82 + targetUpper * 0.18;
    ui->widget_spec->yAxis->setRange(blendedLower, blendedUpper);

    ui->widget_spec->graph(0)->setData(rawFreq, m_rawSpectrumSmooth);
    if (m_restoreMetricsSuppressed) {
        ui->widget_spec->graph(1)->setData(QVector<double>(), QVector<double>());
    } else {
        ui->widget_spec->graph(1)->setData(cleanFreq, m_cleanSpectrumSmooth);
    }
    ui->widget_spec->replot(QCustomPlot::rpQueuedReplot);
}

void MainWindow::handleBackendError(QProcess::ProcessError error)
{
    if (m_manualStopRequested && error == QProcess::Crashed) {
        return;
    }

    QString detail = processErrorToText(error);
    if (!m_lastStdErr.isEmpty()) {
        detail += QStringLiteral(" | %1").arg(m_lastStdErr);
    }
    setStatusMessage(detail, QStringLiteral("#E05A5A"), 15, true);
    updateRunningState(false);
}

void MainWindow::handleBackendFinished(int exitCode, QProcess::ExitStatus exitStatus)
{
    const bool crashed = exitStatus == QProcess::CrashExit;
    updateRunningState(false);

    if (m_manualStopRequested) {
        m_manualStopRequested = false;
        return;
    }

    if (crashed) {
        setStatusMessage(QStringLiteral("AD9361 后端异常退出"), QStringLiteral("#E05A5A"), 16, true);
        return;
    }

    if (exitCode != 0) {
        QString text = QStringLiteral("AD9361 后端已退出，退出码 %1").arg(exitCode);
        if (!m_lastStdErr.isEmpty()) {
            text += QStringLiteral(" | %1").arg(m_lastStdErr);
        }
        setStatusMessage(text, QStringLiteral("#E05A5A"), 15, true);
        return;
    }

    setStatusMessage(QStringLiteral("采集任务已结束"), QStringLiteral("#6B7280"), 16);
}

void MainWindow::on_lineEdit_ip_returnPressed()
{
    checkDeviceConnection();
}

void MainWindow::on_btnConnect_clicked()
{
    checkDeviceConnection();
}

void MainWindow::checkDeviceConnection()
{
    QString ip = ui->lineEdit_ip->text().trimmed();
    if (ip.startsWith(QStringLiteral("ip:"), Qt::CaseInsensitive)) {
        ip = ip.mid(3).trimmed();
        ui->lineEdit_ip->setText(ip);
    }
    if (ip.isEmpty()) {
        return;
    }

    if (!refreshRuntimePaths()) {
        setStatusMessage(QStringLiteral("RK3588 路径检查失败，请确认 JamSystem 和 ad9361_rk3588 存在"),
                         QStringLiteral("#E05A5A"), 15, true);
        return;
    }

    auto *checker = new QProcess(this);
    const QStringList args = {
        QStringLiteral("--check"),
        QStringLiteral("ip:%1").arg(ip),
    };

    connect(checker,
            qOverload<int, QProcess::ExitStatus>(&QProcess::finished),
            this,
            [this, checker, ip](int exitCode, QProcess::ExitStatus) {
                ui->label_status_light->setStyleSheet(
                    exitCode == 0
                        ? QStringLiteral("background-color:#148A3B; border-radius:10px; border:2px solid #000000;")
                        : QStringLiteral("background-color:#B00020; border-radius:10px; border:2px solid #000000;"));
                ui->statusbar->showMessage(
                    exitCode == 0
                        ? QStringLiteral("设备 %1 连通正常").arg(ip)
                        : QStringLiteral("设备 %1 连通失败，请检查 IIO 链路").arg(ip),
                    3000);
                checker->deleteLater();
            });

    checker->setWorkingDirectory(m_basePath);
    checker->start(m_backendExe, args);
}

QString MainWindow::processErrorToText(QProcess::ProcessError error) const
{
    switch (error) {
    case QProcess::FailedToStart:
        return QStringLiteral("后端启动失败，请检查 ad9361_rk3588 路径");
    case QProcess::Crashed:
        return QStringLiteral("后端进程异常退出");
    case QProcess::Timedout:
        return QStringLiteral("后端响应超时");
    case QProcess::WriteError:
        return QStringLiteral("向后端写入数据失败");
    case QProcess::ReadError:
        return QStringLiteral("读取后端输出失败");
    case QProcess::UnknownError:
    default:
        return QStringLiteral("后端发生未知错误");
    }
}

QString MainWindow::mapIdToCn(const QString &id) const
{
    if (id == QStringLiteral("none")) {
        return QStringLiteral("无干扰信号");
    }
    if (id == QStringLiteral("single_tone")) {
        return QStringLiteral("单频干扰");
    }
    if (id == QStringLiteral("narrowband")) {
        return QStringLiteral("窄带干扰");
    }
    if (id == QStringLiteral("wideband_barrage")) {
        return QStringLiteral("宽带阻塞干扰");
    }
    if (id == QStringLiteral("comb")) {
        return QStringLiteral("梳状谱干扰");
    }
    if (id == QStringLiteral("white_noise")) {
        return QStringLiteral("白噪声干扰");
    }
    if (id == QStringLiteral("noise_fm")) {
        return QStringLiteral("噪声调频干扰");
    }
    return QStringLiteral("分析中...");
}

QString MainWindow::mapRestoreMethod(const QString &method) const
{
    if (method == QStringLiteral("bypass")) {
        return QStringLiteral("无需还原");
    }
    if (method == QStringLiteral("wiener_lowpass")) {
        return QStringLiteral("维纳抑噪 + 低通");
    }
    if (method == QStringLiteral("notch_filter")) {
        return QStringLiteral("陷波 + 低通");
    }
    if (method == QStringLiteral("clip_bandpass")) {
        return QStringLiteral("限幅 + 带通");
    }
    if (method == QStringLiteral("fallback")) {
        return QStringLiteral("回退到原始信号");
    }
    return method;
}

QString MainWindow::mapRestoreStatus(const QString &status) const
{
    if (status == QStringLiteral("success")) {
        return QStringLiteral("还原成功");
    }
    if (status == QStringLiteral("partial")) {
        return QStringLiteral("部分恢复");
    }
    if (status == QStringLiteral("limited")) {
        return QStringLiteral("恢复有限");
    }
    if (status == QStringLiteral("not_required")) {
        return QStringLiteral("无需还原");
    }
    if (status == QStringLiteral("error")) {
        return QStringLiteral("还原失败");
    }
    return QStringLiteral("--");
}

QString MainWindow::getEngName(int index) const
{
    static const QStringList names = {
        QStringLiteral("none"),
        QStringLiteral("single_tone"),
        QStringLiteral("narrowband"),
        QStringLiteral("wideband_barrage"),
        QStringLiteral("comb"),
        QStringLiteral("white_noise"),
        QStringLiteral("noise_fm"),
    };
    if (index >= 0 && index < names.size()) {
        return names[index];
    }
    return QStringLiteral("none");
}

QString MainWindow::currentModulationArg() const
{
    return ui->comboBox_modulation->currentData().toString();
}

QString MainWindow::currentRunModeArg() const
{
    return QStringLiteral("rx_only");
}

QString MainWindow::mapRunModeToCn(const QString &mode) const
{
    if (mode == QStringLiteral("rx_only") || mode == QStringLiteral("receive_only")) {
        return QStringLiteral("双板接收");
    }
    return QStringLiteral("双板接收");
}

QString MainWindow::mapModulationToCn(const QString &mode) const
{
    if (mode == QStringLiteral("analog_fm")) {
        return QStringLiteral("模拟调制（FM，240 kbaud）");
    }
    return QStringLiteral("数字调制（QPSK）");
}

void MainWindow::updateModulationMetricLabels()
{
    const QString mode = currentModulationArg();
    const bool analogMode = (mode == QStringLiteral("analog_fm"));
    ui->label_modulation_value->setText(mapModulationToCn(mode));
    ui->groupBox_5->setVisible(!analogMode);

    QString metric1Text = analogMode
        ? ui->label_extra_metric1_title->property("analogText").toString()
        : ui->label_extra_metric1_title->property("digitalText").toString();
    QString metric2Text = analogMode
        ? ui->label_extra_metric2_title->property("analogText").toString()
        : ui->label_extra_metric2_title->property("digitalText").toString();

    if (metric1Text.isEmpty()) {
        metric1Text = analogMode ? QStringLiteral("皮尔逊相关系数") : QStringLiteral("恢复前 BER");
    }
    if (metric2Text.isEmpty()) {
        metric2Text = analogMode ? QStringLiteral("NMSE") : QStringLiteral("恢复后 BER");
    }

    ui->label_extra_metric1_title->setText(metric1Text);
    ui->label_extra_metric2_title->setText(metric2Text);
}






