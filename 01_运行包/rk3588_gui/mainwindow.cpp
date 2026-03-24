#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QRegularExpression>
#include <QSharedPointer>

#include <algorithm>

namespace {

constexpr int kPlotPointCount = 1024;
constexpr int kSpectrumPointCount = 512;
constexpr int kPlotColumnCount = 12;

constexpr const char *kDefaultBasePath = "/home/pi/Desktop/JamSystem";
constexpr const char *kDefaultDeviceIp = "192.168.1.10";

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
    , m_manualStopRequested(false)
    , m_restoreMetricsSuppressed(false)
{
    ui->setupUi(this);
    applyUiTextOverrides();
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

    ui->lineEdit_ip->setText(QString::fromLatin1(kDefaultDeviceIp));
    ui->comboBox_2->setCurrentIndex(0);
    ui->comboBox_2->setEnabled(false);

    setupPlotStyles();
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
    delete ui;
}

void MainWindow::applyUiTextOverrides()
{
    ui->groupBox->setTitle(QStringLiteral("系统控制与配置"));
    ui->label_ip->setText(QStringLiteral("设备 IP："));

    ui->groupBox_6->setTitle(QStringLiteral("采集链路"));
    ui->comboBox_2->setItemText(0, QStringLiteral("AD9361 实时采集"));
    ui->comboBox_2->setItemText(1, QStringLiteral("预留"));

    ui->groupBox_2->setTitle(QStringLiteral("干扰样式选择"));
    ui->comboBox->setItemText(0, QStringLiteral("无干扰"));
    ui->comboBox->setItemText(1, QStringLiteral("单频"));
    ui->comboBox->setItemText(2, QStringLiteral("窄带"));
    ui->comboBox->setItemText(3, QStringLiteral("宽带"));
    ui->comboBox->setItemText(4, QStringLiteral("梳状谱"));
    ui->comboBox->setItemText(5, QStringLiteral("白噪声"));
    ui->comboBox->setItemText(6, QStringLiteral("噪声调频"));

    ui->btnStart->setText(QStringLiteral("启动采集与识别"));
    ui->btnStop->setText(QStringLiteral("停止"));

    ui->groupBox_7->setTitle(QStringLiteral("识别与还原结果"));
    ui->label_result->setText(QStringLiteral("系统就绪"));
    ui->progressBar_conf->setFormat(QStringLiteral("识别置信度 %p%"));
    ui->label_restore_status_title->setText(QStringLiteral("还原结论"));
    ui->label_restore_method_title->setText(QStringLiteral("还原方法"));
    ui->label_metric_isr_title->setText(QStringLiteral("ISR"));
    ui->label_metric_evm_before_title->setText(QStringLiteral("还原前 EVM"));
    ui->label_metric_evm_after_title->setText(QStringLiteral("还原后 EVM"));

    ui->groupBox_3->setTitle(QStringLiteral("IQ 波形前后对比"));
    ui->groupBox_5->setTitle(QStringLiteral("星座图前后对比"));
    ui->groupBox_4->setTitle(QStringLiteral("频谱前后对比"));
}

void MainWindow::applyDarkPlotStyle(QCustomPlot *plot, const QString &xLabel, const QString &yLabel)
{
    plot->setBackground(QColor(8, 12, 18));
    plot->setNoAntialiasingOnDrag(true);
    plot->legend->setVisible(false);

    QPen axisPen(QColor(216, 223, 230));
    axisPen.setWidth(1);

    plot->xAxis->setBasePen(axisPen);
    plot->xAxis->setTickPen(axisPen);
    plot->xAxis->setSubTickPen(axisPen);
    plot->xAxis->setTickLabelColor(QColor(225, 232, 240));
    plot->xAxis->setLabelColor(QColor(225, 232, 240));
    plot->xAxis->setLabel(xLabel);
    plot->xAxis->grid()->setVisible(true);
    plot->xAxis->grid()->setPen(QPen(QColor(73, 89, 109), 1, Qt::DotLine));

    plot->yAxis->setBasePen(axisPen);
    plot->yAxis->setTickPen(axisPen);
    plot->yAxis->setSubTickPen(axisPen);
    plot->yAxis->setTickLabelColor(QColor(225, 232, 240));
    plot->yAxis->setLabelColor(QColor(225, 232, 240));
    plot->yAxis->setLabel(yLabel);
    plot->yAxis->grid()->setVisible(true);
    plot->yAxis->grid()->setPen(QPen(QColor(73, 89, 109), 1, Qt::DotLine));

    plot->axisRect()->setBackground(QColor(12, 18, 27));
}

void MainWindow::setupPlotStyles()
{
    applyDarkPlotStyle(ui->widget_iq, QStringLiteral("采样点"), QStringLiteral("幅度"));
    ui->widget_iq->addGraph();
    ui->widget_iq->graph(0)->setPen(QPen(QColor(255, 128, 64, 160), 1.2, Qt::DashLine));
    ui->widget_iq->addGraph();
    ui->widget_iq->graph(1)->setPen(QPen(QColor(74, 144, 226, 160), 1.2, Qt::DashLine));
    ui->widget_iq->addGraph();
    ui->widget_iq->graph(2)->setPen(QPen(QColor(255, 215, 0), 1.8));
    ui->widget_iq->addGraph();
    ui->widget_iq->graph(3)->setPen(QPen(QColor(0, 255, 255), 1.8));
    ui->widget_iq->xAxis->setRange(0, kPlotPointCount - 1);
    ui->widget_iq->yAxis->setRange(-1.2, 1.2);

    applyDarkPlotStyle(ui->widget_spec, QStringLiteral("频率 (kHz)"), QStringLiteral("功率 (dB)"));
    ui->widget_spec->addGraph();
    ui->widget_spec->graph(0)->setPen(QPen(QColor(238, 80, 155), 1.4, Qt::DashLine));
    ui->widget_spec->addGraph();
    ui->widget_spec->graph(1)->setPen(QPen(QColor(53, 210, 255), 2.0));
    ui->widget_spec->xAxis->setRange(-3200, 3200);
    ui->widget_spec->yAxis->setRange(-120, 20);
    QSharedPointer<QCPAxisTickerFixed> ticker(new QCPAxisTickerFixed);
    ticker->setTickStep(1000.0);
    ui->widget_spec->xAxis->setTicker(ticker);

    applyDarkPlotStyle(ui->widget_const, QStringLiteral("I (同相)"), QStringLiteral("Q (正交)"));
    ui->widget_const->addGraph();
    ui->widget_const->graph(0)->setLineStyle(QCPGraph::lsNone);
    ui->widget_const->graph(0)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDisc, 3.5));
    ui->widget_const->graph(0)->setPen(QPen(QColor(255, 140, 84, 150)));
    ui->widget_const->addGraph();
    ui->widget_const->graph(1)->setLineStyle(QCPGraph::lsNone);
    ui->widget_const->graph(1)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDisc, 4.0));
    ui->widget_const->graph(1)->setPen(QPen(QColor(94, 255, 143, 210)));
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
    ui->progressBar_conf->setValue(0);
    ui->label_restore_status->setText(QStringLiteral("--"));
    ui->label_restore_method->setText(QStringLiteral("--"));
    ui->label_metric_isr->setText(QStringLiteral("--"));
    ui->label_metric_evm_before->setText(QStringLiteral("--"));
    ui->label_metric_evm_after->setText(QStringLiteral("--"));
}

void MainWindow::updatePlotPresentation()
{
    if (m_restoreMetricsSuppressed) {
        ui->groupBox_3->setTitle(QStringLiteral("IQ 波形"));
        ui->groupBox_5->setTitle(QStringLiteral("星座图"));
        ui->groupBox_4->setTitle(QStringLiteral("频谱"));

        ui->widget_iq->graph(0)->setPen(QPen(QColor(255, 128, 64), 1.8));
        ui->widget_iq->graph(1)->setPen(QPen(QColor(74, 144, 226), 1.8));
        ui->widget_iq->graph(2)->setPen(QPen(QColor(255, 215, 0, 45), 1.0, Qt::DotLine));
        ui->widget_iq->graph(3)->setPen(QPen(QColor(0, 255, 255, 45), 1.0, Qt::DotLine));

        ui->widget_spec->graph(0)->setPen(QPen(QColor(238, 80, 155), 2.0));
        ui->widget_spec->graph(1)->setPen(QPen(QColor(53, 210, 255, 35), 1.0, Qt::DotLine));

        ui->widget_const->graph(0)->setPen(QPen(QColor(255, 140, 84, 210)));
        ui->widget_const->graph(1)->setPen(QPen(QColor(94, 255, 143, 45)));
    } else {
        ui->groupBox_3->setTitle(QStringLiteral("IQ 波形前后对比"));
        ui->groupBox_5->setTitle(QStringLiteral("星座图前后对比"));
        ui->groupBox_4->setTitle(QStringLiteral("频谱前后对比"));

        ui->widget_iq->graph(0)->setPen(QPen(QColor(255, 128, 64, 160), 1.2, Qt::DashLine));
        ui->widget_iq->graph(1)->setPen(QPen(QColor(74, 144, 226, 160), 1.2, Qt::DashLine));
        ui->widget_iq->graph(2)->setPen(QPen(QColor(255, 215, 0), 1.8));
        ui->widget_iq->graph(3)->setPen(QPen(QColor(0, 255, 255), 1.8));

        ui->widget_spec->graph(0)->setPen(QPen(QColor(238, 80, 155), 1.4, Qt::DashLine));
        ui->widget_spec->graph(1)->setPen(QPen(QColor(53, 210, 255), 2.0));

        ui->widget_const->graph(0)->setPen(QPen(QColor(255, 140, 84, 150)));
        ui->widget_const->graph(1)->setPen(QPen(QColor(94, 255, 143, 210)));
    }
}

void MainWindow::updateRunningState(bool running)
{
    ui->btnStart->setEnabled(!running);
    ui->btnStop->setEnabled(running);
    ui->comboBox->setEnabled(!running);
    ui->lineEdit_ip->setEnabled(!running);
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
    const QString weight = bold ? QStringLiteral("700") : QStringLiteral("500");
    ui->label_result->setEnabled(true);
    ui->label_result->setStyleSheet(
        QStringLiteral("color:%1; font-size:%2pt; font-weight:%3;")
            .arg(color)
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

    const QString jamType = getEngName(ui->comboBox->currentIndex());
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

    const QStringList args = {
        jamType,
        uri,
    };
    m_backend->start(m_backendExe, args);
    setStatusMessage(QStringLiteral("正在启动 AD9361 采集与识别..."), QStringLiteral("#3A7AFE"), 18);
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

    if (line.startsWith(QStringLiteral("RESULT_ID:"))) {
        setStatusMessage(mapIdToCn(line.section(':', 1).trimmed()), QStringLiteral("#FF6B35"), 24, true);
        return true;
    }

    if (line.startsWith(QStringLiteral("RESULT_CONF:"))) {
        const double conf = line.section(':', 1).trimmed().toDouble();
        ui->progressBar_conf->setValue(qBound(0, static_cast<int>(conf * 100.0), 100));
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
            ui->label_metric_evm_before->setText(QStringLiteral("--"));
            ui->label_metric_evm_after->setText(QStringLiteral("--"));
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
                        ? QStringLiteral("background-color:#22C55E; border-radius:10px; border:1px solid #0F172A;")
                        : QStringLiteral("background-color:#EF4444; border-radius:10px; border:1px solid #0F172A;"));
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
