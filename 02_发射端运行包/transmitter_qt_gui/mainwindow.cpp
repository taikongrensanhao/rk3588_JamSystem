#include "mainwindow.h"

#include <QComboBox>
#include <QCoreApplication>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QNetworkInterface>
#include <QProcessEnvironment>
#include <QPushButton>
#include <QStringList>
#include <QTimer>
#include <QSpinBox>
#include <QTextEdit>
#include <QVBoxLayout>

namespace {
QString shellQuote(const QString &text)
{
    QString escaped = text;
    escaped.replace('\'', QStringLiteral("'\"'\"'"));
    return QStringLiteral("'") + escaped + QStringLiteral("'");
}

QString backendSwitchSequence(const QString &visibleSequence)
{
    const QStringList parts = visibleSequence.split(',', Qt::SkipEmptyParts);
    QStringList backendParts;
    for (const QString &part : parts) {
        const QString item = part.trimmed();
        if (item.isEmpty()) {
            continue;
        }
        backendParts << item;
        if (item == QStringLiteral("wideband_barrage")) {
            backendParts << item << item << item << item;
        }
    }
    return backendParts.isEmpty() ? visibleSequence : backendParts.join(',');
}
}

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent),
      statusLabel(new QLabel("空闲")),
      ipEdit(new QLineEdit("192.168.1.10")),
      txModeBox(new QComboBox),
      centerFreqBox(new QComboBox),
      modulationBox(new QComboBox),
      jammerBox(new QComboBox),
      switchMsSpin(new QSpinBox),
      sequenceEdit(new QLineEdit("single_tone,narrowband,wideband_barrage,comb,white_noise,noise_fm")),
      checkButton(new QPushButton("检测 AD9361")),
      startButton(new QPushButton("开始发射")),
      stopButton(new QPushButton("停止发射")),
      refreshIpButton(new QPushButton("显示本机IP")),
      exitButton(new QPushButton("退出界面")),
      logEdit(new QTextEdit),
      process(new QProcess(this)),
      checkMode(false),
      transmitRunning(false),
      txLogOffset(0)
{
    setWindowTitle("干扰发射控制台");

    txModeBox->addItem("纯净参考信号（无干扰）", "tx_clean_qpsk");
    txModeBox->addItem("纯干扰识别测试", "tx_jammer_only");
    txModeBox->addItem("参考+干扰还原测试", "tx_only");
    txModeBox->addItem("循环切换（按序列轮流发）", "selftest_switch");

    centerFreqBox->addItem("100 MHz", "100000000");
    centerFreqBox->addItem("200 MHz", "200000000");
    centerFreqBox->addItem("300 MHz", "300000000");
    centerFreqBox->setCurrentIndex(1);

    modulationBox->addItem("数字调制 QPSK", "digital_qpsk");
    modulationBox->addItem("模拟调制 FM", "analog_fm");

    jammerBox->addItem("窄带干扰", "narrowband");
    jammerBox->addItem("宽带阻塞", "wideband_barrage");
    jammerBox->addItem("梳状干扰", "comb");
    jammerBox->addItem("白噪声", "white_noise");
    jammerBox->addItem("噪声调频", "noise_fm");
    jammerBox->addItem("单频干扰", "single_tone");

    switchMsSpin->setRange(10, 10000);
    switchMsSpin->setSingleStep(10);
    switchMsSpin->setValue(25);
    switchMsSpin->setSuffix(" ms");

    logEdit->setReadOnly(true);
    logEdit->setLineWrapMode(QTextEdit::NoWrap);

    statusLabel->setMinimumHeight(42);
    statusLabel->setStyleSheet("font-size:22px;font-weight:700;color:#0b5cad;");

    auto *form = new QGridLayout;
    form->addWidget(new QLabel("AD9361 地址"), 0, 0);
    form->addWidget(ipEdit, 0, 1);
    form->addWidget(new QLabel("发射方式"), 1, 0);
    form->addWidget(txModeBox, 1, 1);
    form->addWidget(new QLabel("中心频率"), 2, 0);
    form->addWidget(centerFreqBox, 2, 1);
    form->addWidget(new QLabel("调制方式"), 3, 0);
    form->addWidget(modulationBox, 3, 1);
    form->addWidget(new QLabel("干扰类型"), 4, 0);
    form->addWidget(jammerBox, 4, 1);
    form->addWidget(new QLabel("切换周期"), 5, 0);
    form->addWidget(switchMsSpin, 5, 1);
    form->addWidget(new QLabel("切换序列"), 6, 0);
    form->addWidget(sequenceEdit, 6, 1);

    auto *buttonLayout = new QVBoxLayout;
    buttonLayout->addWidget(checkButton);
    buttonLayout->addWidget(startButton);
    buttonLayout->addWidget(stopButton);
    buttonLayout->addWidget(refreshIpButton);
    buttonLayout->addWidget(exitButton);
    buttonLayout->addStretch(1);

    auto *leftLayout = new QVBoxLayout;
    leftLayout->addWidget(statusLabel);
    leftLayout->addLayout(form);
    leftLayout->addLayout(buttonLayout);
    auto *hint = new QLabel("参考+干扰用于还原测试；纯干扰用于优先保证识别准确率；循环切换用于展示变化跟踪速度。");
    hint->setWordWrap(true);
    leftLayout->addWidget(hint);

    auto *leftBox = new QGroupBox("发射控制");
    leftBox->setLayout(leftLayout);

    auto *logBoxLayout = new QVBoxLayout;
    logBoxLayout->addWidget(logEdit);
    auto *logBox = new QGroupBox("运行日志");
    logBox->setLayout(logBoxLayout);

    auto *mainLayout = new QHBoxLayout;
    mainLayout->addWidget(leftBox, 0);
    mainLayout->addWidget(logBox, 1);

    auto *central = new QWidget;
    central->setLayout(mainLayout);
    setCentralWidget(central);

    const QString buttonStyle =
        "QPushButton{font-size:22px;font-weight:700;min-height:58px;border-radius:8px;"
        "background:#1f6feb;color:white;padding:10px;}"
        "QPushButton:disabled{background:#9aa7b5;color:#eef2f6;}";
    checkButton->setStyleSheet(buttonStyle + "QPushButton{background:#57606a;}");
    startButton->setStyleSheet(buttonStyle);
    stopButton->setStyleSheet(buttonStyle + "QPushButton{background:#d1242f;}");
    refreshIpButton->setStyleSheet(buttonStyle + "QPushButton{background:#0a7f64;}");
    exitButton->setStyleSheet(buttonStyle + "QPushButton{background:#6e7781;}");
    stopButton->setEnabled(false);

    setStyleSheet("QLabel{font-size:18px;} QLineEdit,QComboBox,QSpinBox{font-size:20px;min-height:40px;}"
                  "QGroupBox{font-size:20px;font-weight:700;} QTextEdit{font-size:15px;background:#101820;color:#d6e2ff;}");

    connect(checkButton, &QPushButton::clicked, this, &MainWindow::checkAd9361);
    connect(startButton, &QPushButton::clicked, this, &MainWindow::startTransmit);
    connect(stopButton, &QPushButton::clicked, this, &MainWindow::stopTransmit);
    connect(refreshIpButton, &QPushButton::clicked, this, &MainWindow::refreshLocalIp);
    connect(exitButton, &QPushButton::clicked, this, &MainWindow::close);
    connect(txModeBox, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int) {
        const QString selectedMode = txModeBox->currentData().toString();
        const bool cleanMode = selectedMode == QStringLiteral("tx_clean_qpsk");
        const bool dynamicMode = selectedMode == QStringLiteral("selftest_switch");
        switchMsSpin->setEnabled(dynamicMode);
        sequenceEdit->setEnabled(dynamicMode);
        jammerBox->setEnabled(!cleanMode);
        modulationBox->setEnabled(true);
        const bool pureMode = selectedMode == QStringLiteral("tx_jammer_only");
        startButton->setText(cleanMode ? "开始纯净无干扰" :
                             (dynamicMode ? "开始循环切换" :
                             (pureMode ? "开始纯干扰识别测试" : "开始参考+干扰还原测试")));
    });
    connect(centerFreqBox, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &MainWindow::restartTransmitIfRunning);
    auto *txLogTimer = new QTimer(this);
    txLogTimer->setInterval(500);
    connect(txLogTimer, &QTimer::timeout, this, &MainWindow::pollTxLog);
    txLogTimer->start();
    connect(process, &QProcess::readyReadStandardOutput, this, &MainWindow::handleReadyRead);
    connect(process, &QProcess::readyReadStandardError, this, &MainWindow::handleReadyRead);
    connect(process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            this, &MainWindow::handleFinished);

    appendLog("发射控制台已启动");
    appendLog("JamSystem目录: " + jamSystemDir());
    killStaleTransmitters();
    appendLog("已清理残留发射进程");
    switchMsSpin->setEnabled(false);
    sequenceEdit->setEnabled(false);
    jammerBox->setEnabled(false);
    modulationBox->setEnabled(true);
    startButton->setText("开始纯净无干扰");
    QTimer::singleShot(300, this, &MainWindow::refreshLocalIp);
}

MainWindow::~MainWindow()
{
    stopProcessBlocking(2000);
    killStaleTransmitters();
}

QString MainWindow::jamSystemDir() const
{
    const QString envPath = qEnvironmentVariable("JAMSYSTEM_BASE_PATH");
    if (!envPath.isEmpty()) {
        return envPath;
    }

    QDir appDir(QCoreApplication::applicationDirPath());
    if (QFileInfo(appDir.filePath("../JamSystem/ad9361_rk3588")).exists()) {
        appDir.cd("../JamSystem");
        return appDir.absolutePath();
    }
    if (QFileInfo("/home/pi/Desktop/run/JamSystem/ad9361_rk3588").exists()) {
        return "/home/pi/Desktop/run/JamSystem";
    }
    return appDir.absolutePath();
}

QString MainWindow::backendPath() const
{
    return QDir(jamSystemDir()).filePath("ad9361_rk3588");
}

QString MainWindow::ipArg() const
{
    QString ip = ipEdit->text().trimmed();
    if (ip.isEmpty()) {
        ip = "192.168.1.10";
    }
    if (!ip.startsWith("ip:")) {
        ip = "ip:" + ip;
    }
    return ip;
}

void MainWindow::appendLog(const QString &text)
{
    const QString ts = QDateTime::currentDateTime().toString("HH:mm:ss");
    logEdit->append("[" + ts + "] " + text.trimmed());
}

void MainWindow::setRunning(bool running)
{
    transmitRunning = running;
    statusLabel->setText(running ? "发射运行中" : "空闲");
    checkButton->setEnabled(!running && process->state() == QProcess::NotRunning);
    startButton->setEnabled(!running);
    stopButton->setEnabled(running);
    exitButton->setEnabled(!running);
}

bool MainWindow::stopProcessBlocking(int timeoutMs)
{
    if (process->state() == QProcess::NotRunning) {
        return true;
    }
    process->terminate();
    if (!process->waitForFinished(timeoutMs)) {
        process->kill();
        return process->waitForFinished(timeoutMs);
    }
    return true;
}

void MainWindow::killStaleTransmitters()
{
    QProcess killer;
    killer.setProgram(QStringLiteral("pkill"));
    killer.setArguments(QStringList()
                        << QStringLiteral("-9")
                        << QStringLiteral("-f")
                        << QStringLiteral("ad9361_rk3588"));
    killer.start();
    killer.waitForFinished(1000);
}

void MainWindow::runCheckProcess(const QStringList &args)
{
    if (!QFileInfo::exists(backendPath())) {
        QMessageBox::warning(this, "缺少程序", "找不到 ad9361_rk3588，请先编译发射程序。");
        return;
    }

    process->setWorkingDirectory(jamSystemDir());
    process->setProgram(backendPath());
    process->setArguments(args);
    process->setProcessEnvironment(QProcessEnvironment::systemEnvironment());
    appendLog("执行: " + backendPath() + " " + args.join(" "));
    process->start();
}

void MainWindow::checkAd9361()
{
    if (process->state() != QProcess::NotRunning) {
        QMessageBox::information(this, "正在运行", "请先停止当前任务。");
        return;
    }
    checkMode = true;
    statusLabel->setText("检测中");
    checkButton->setEnabled(false);
    startButton->setEnabled(false);
    runCheckProcess(QStringList() << "--check" << ipArg());
}

void MainWindow::startTransmit()
{
    if (transmitRunning) {
        QMessageBox::information(this, "正在运行", "发射任务已经在运行。");
        return;
    }
    if (process->state() != QProcess::NotRunning) {
        QMessageBox::information(this, "正在检测", "请等待检测任务结束。");
        return;
    }

    checkMode = false;
    killStaleTransmitters();

    const QString selectedMode = txModeBox->currentData().toString();
    const bool cleanMode = selectedMode == QStringLiteral("tx_clean_qpsk");
    const QString jammer = cleanMode ? QStringLiteral("none") : jammerBox->currentData().toString();
    const QString modulation = modulationBox->currentData().toString();
    const QString runMode = cleanMode ? QStringLiteral("tx_only") : selectedMode;
    QStringList envParts;
    envParts << QStringLiteral("JAMSYSTEM_CENTER_FREQ_HZ=%1")
                    .arg(centerFreqBox->currentData().toString());
    if (runMode == QStringLiteral("selftest_switch")) {
        const QString visibleSeq = sequenceEdit->text().trimmed();
        envParts << QStringLiteral("JAMSYSTEM_SELFTEST_SWITCH_MS=%1").arg(switchMsSpin->value());
        envParts << QStringLiteral("JAMSYSTEM_SELFTEST_SWITCH_SEQ=%1").arg(shellQuote(backendSwitchSequence(visibleSeq)));
    }
    const QStringList args = QStringList() << jammer << ipArg() << modulation << runMode;

    if (!QFileInfo::exists(backendPath())) {
        QMessageBox::warning(this, "缺少程序", "找不到 ad9361_rk3588，请先编译发射程序。");
        return;
    }

    const QString logPath = QStringLiteral("/tmp/jamsystem_tx.log");
    QFile::remove(logPath);
    txLogOffset = 0;
    const QString envPrefix = envParts.isEmpty() ? QString() : (envParts.join(' ') + QStringLiteral(" "));
    const QString command = QStringLiteral("cd %1 && %2nohup %3 %4 > %5 2>&1 &")
                               .arg(shellQuote(jamSystemDir()),
                                    envPrefix,
                                    shellQuote(backendPath()),
                                    args.join(' '),
                                    shellQuote(logPath));

    appendLog("执行: " + backendPath() + " " + args.join(" "));
    if (selectedMode == QStringLiteral("selftest_switch")) {
        appendLog("完整命令: 已按当前循环设置启动");
    } else {
        appendLog("完整命令: " + command);
    }
    appendLog("日志: " + logPath);
    appendLog("发射模式: " + txModeBox->currentText());
    appendLog("中心频率: " + centerFreqBox->currentText());
    if (cleanMode) {
        appendLog("纯净无干扰：按当前调制方式发射有用信号，不叠加干扰");
    } else if (runMode == QStringLiteral("selftest_switch")) {
        appendLog("切换周期: " + QString::number(switchMsSpin->value()) + " ms");
        appendLog("切换序列: " + sequenceEdit->text().trimmed());
    } else {
        appendLog("固定干扰: " + jammerBox->currentText());
    }
    if (!QProcess::startDetached(QStringLiteral("/bin/sh"), QStringList() << QStringLiteral("-c") << command)) {
        appendLog("启动失败: 无法启动后台发射命令");
        setRunning(false);
        return;
    }
    setRunning(true);
    appendLog("后台发射已启动，切换干扰时会自动清理旧进程");
}

void MainWindow::pollTxLog()
{
    if (!transmitRunning) {
        return;
    }

    QFile file(QStringLiteral("/tmp/jamsystem_tx.log"));
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        return;
    }

    if (file.size() < txLogOffset) {
        txLogOffset = 0;
    }
    if (!file.seek(txLogOffset)) {
        return;
    }

    const QByteArray data = file.readAll();
    txLogOffset = file.pos();
    if (data.isEmpty()) {
        return;
    }

    const QStringList lines = QString::fromLocal8Bit(data).split('\n', Qt::SkipEmptyParts);
    for (const QString &line : lines) {
        appendLog(line);
    }
}

void MainWindow::restartTransmitIfRunning()
{
    if (checkMode || !transmitRunning) {
        return;
    }

    appendLog("参数已变化，自动重启发射进程...");
    killStaleTransmitters();
    setRunning(false);
    QTimer::singleShot(100, this, &MainWindow::startTransmit);
}

void MainWindow::stopTransmit()
{
    if (!transmitRunning) {
        return;
    }
    appendLog("正在停止发射...");
    killStaleTransmitters();
    setRunning(false);
}

void MainWindow::handleReadyRead()
{
    const QString out = QString::fromLocal8Bit(process->readAllStandardOutput())
                        + QString::fromLocal8Bit(process->readAllStandardError());
    const QStringList lines = out.split('\n', Qt::SkipEmptyParts);
    for (const QString &line : lines) {
        appendLog(line);
    }
}

void MainWindow::handleFinished(int exitCode, QProcess::ExitStatus exitStatus)
{
    Q_UNUSED(exitStatus);
    appendLog("进程结束，退出码: " + QString::number(exitCode));
    if (checkMode) {
        transmitRunning = false;
        statusLabel->setText(exitCode == 0 ? "检测成功" : "检测失败");
        checkButton->setEnabled(true);
        startButton->setEnabled(true);
    }
    checkMode = false;
}

void MainWindow::refreshLocalIp()
{
    QStringList rows;
    const QList<QNetworkInterface> interfaces = QNetworkInterface::allInterfaces();
    for (const QNetworkInterface &iface : interfaces) {
        if (!(iface.flags() & QNetworkInterface::IsUp) || !(iface.flags() & QNetworkInterface::IsRunning)) {
            continue;
        }
        if (iface.flags() & QNetworkInterface::IsLoopBack) {
            continue;
        }
        QStringList ips;
        const QList<QNetworkAddressEntry> entries = iface.addressEntries();
        for (const QNetworkAddressEntry &entry : entries) {
            if (entry.ip().protocol() == QAbstractSocket::IPv4Protocol) {
                ips << entry.ip().toString();
            }
        }
        if (!ips.isEmpty()) {
            rows << iface.humanReadableName() + ": " + ips.join(", ");
        }
    }
    if (rows.isEmpty()) {
        rows << "未检测到已连接的IPv4网口";
    }
    appendLog("本机IP: " + rows.join(" | "));
    QMessageBox::information(this, "本机IP", rows.join("\n"));
}
