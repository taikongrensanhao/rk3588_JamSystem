#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QProcess>

class QLabel;
class QLineEdit;
class QComboBox;
class QSpinBox;
class QPushButton;
class QTextEdit;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow() override;

private slots:
    void checkAd9361();
    void startTransmit();
    void stopTransmit();
    void handleReadyRead();
    void handleFinished(int exitCode, QProcess::ExitStatus exitStatus);
    void refreshLocalIp();
    void restartTransmitIfRunning();
    void pollTxLog();

private:
    QString jamSystemDir() const;
    QString backendPath() const;
    QString ipArg() const;
    void appendLog(const QString &text);
    void setRunning(bool running);
    void runCheckProcess(const QStringList &args);
    bool stopProcessBlocking(int timeoutMs);
    void killStaleTransmitters();

    QLabel *statusLabel;
    QLineEdit *ipEdit;
    QComboBox *txModeBox;
    QComboBox *centerFreqBox;
    QComboBox *modulationBox;
    QComboBox *jammerBox;
    QSpinBox *switchMsSpin;
    QLineEdit *sequenceEdit;
    QPushButton *checkButton;
    QPushButton *startButton;
    QPushButton *stopButton;
    QPushButton *refreshIpButton;
    QPushButton *exitButton;
    QTextEdit *logEdit;
    QProcess *process;
    bool checkMode;
    bool transmitRunning;
    qint64 txLogOffset;
};

#endif
