#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QProcess>
#include <QVector>

#include "qcustomplot.h"

class QComboBox;
class QDialog;
class QLabel;
class QLineEdit;
class QPushButton;
class QTextEdit;

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow() override;

protected:
    bool eventFilter(QObject *watched, QEvent *event) override;

private slots:
    void on_btnStart_clicked();
    void on_btnStop_clicked();
    void on_btnRefresh_clicked();
    void on_lineEdit_ip_returnPressed();
    void on_btnConnect_clicked();
    void handleBackendOutput();
    void handleBackendError(QProcess::ProcessError error);
    void handleBackendFinished(int exitCode, QProcess::ExitStatus exitStatus);
    void openTestPage();
    void startOnlineLearning();
    void handleOnlineLearningOutput();
    void handleOnlineLearningError(QProcess::ProcessError error);
    void handleOnlineLearningFinished(int exitCode, QProcess::ExitStatus exitStatus);
    void openPerformancePage();

private:
    Ui::MainWindow *ui;
    QProcess *m_backend;
    QProcess *m_onlineLearning;
    QString m_outputBuffer;
    QString m_onlineLearningBuffer;
    QString m_lastStdErr;
    QString m_basePath;
    QString m_backendExe;
    QPushButton *m_btnOpenTestPage;
    QPushButton *m_btnOpenPerformancePage;
    QDialog *m_testDialog;
    QDialog *m_performanceDialog;
    QLineEdit *m_lineEditTrainServer;
    QLineEdit *m_lineEditUploadDir;
    QComboBox *m_comboLearningMode;
    QLabel *m_labelLearningStatus;
    QTextEdit *m_textLearningLog;
    QPushButton *m_btnOnlineLearning;
    QLabel *m_perfStatusValue;
    QLabel *m_perfModeValue;
    QLabel *m_perfExpectedValue;
    QLabel *m_perfResultValue;
    QLabel *m_perfConfValue;
    QLabel *m_perfPowerValue;
    QLabel *m_perfRecognitionTimeValue;
    QLabel *m_perfRestorationTimeValue;
    QLabel *m_perfTotalValue;
    QLabel *m_perfCorrectValue;
    QLabel *m_perfAccuracyValue;
    QLabel *m_perfAvgRecognitionValue;
    QLabel *m_perfBestRecognitionValue;
    QLabel *m_perfAvgIntervalValue;
    QLabel *m_perfUpdatedValue;
    bool m_manualStopRequested;
    bool m_restoreMetricsSuppressed;
    int m_perfTotalCount;
    int m_perfCorrectCount;
    double m_perfLastConfidence;
    double m_perfLastPowerDbm;
    double m_perfLastRecognitionMs;
    double m_perfLastRestorationMs;
    double m_perfRecognitionSumMs;
    double m_perfBestRecognitionMs;
    double m_perfIntervalSumMs;
    int m_perfIntervalCount;
    qint64 m_perfLastUpdateEpochMs;
    QString m_perfLastResultId;
    QString m_perfLastUpdateText;

    QVector<double> m_rawSpectrumSmooth;
    QVector<double> m_cleanSpectrumSmooth;

    void applyUiTextOverrides();
    void applyFullScreenLayout();
    void applyTouchScreenContrastStyle();
    void applyTouchComboBoxStyle();
    void setupTestPageEntry();
    void buildTestPage();
    void buildPerformancePage();
    QLabel *makePerformanceValueLabel(QWidget *parent, const QString &text = QString()) const;
    void resetPerformanceMetrics();
    void updatePerformancePage();
    void notePerformanceResult(const QString &resultId);
    QString onlineLearningPython() const;
    void showTouchComboDialog(QComboBox *combo, const QString &title);
    void setupPlotStyles();
    void applyDarkPlotStyle(QCustomPlot *plot, const QString &xLabel, const QString &yLabel);
    void updatePlots();
    void resetPlots();
    void resetMetrics();
    void updatePlotPresentation();
    void updateRunningState(bool running);
    bool refreshRuntimePaths();
    void checkDeviceConnection();
    void setStatusMessage(const QString &text, const QString &color, int fontSize, bool bold = false);
    void handleBackendLine(const QString &line);
    bool handlePythonProtocolLine(const QString &line);
    bool parseBackendSummaryLine(const QString &line);
    QString processErrorToText(QProcess::ProcessError error) const;
    QString mapIdToCn(const QString &id) const;
    QString mapRestoreMethod(const QString &method) const;
    QString mapRestoreStatus(const QString &status) const;
    QString getEngName(int index) const;
    QString currentModulationArg() const;
    QString currentRunModeArg() const;
    QString mapRunModeToCn(const QString &mode) const;
    QString mapModulationToCn(const QString &mode) const;
    void updateModulationMetricLabels();
};

#endif // MAINWINDOW_H

