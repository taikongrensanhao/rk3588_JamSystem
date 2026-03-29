#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QProcess>
#include <QVector>

#include "qcustomplot.h"

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

private slots:
    void on_btnStart_clicked();
    void on_btnStop_clicked();
    void on_lineEdit_ip_returnPressed();
    void handleBackendOutput();
    void handleBackendError(QProcess::ProcessError error);
    void handleBackendFinished(int exitCode, QProcess::ExitStatus exitStatus);

private:
    Ui::MainWindow *ui;
    QProcess *m_backend;
    QString m_outputBuffer;
    QString m_lastStdErr;
    QString m_basePath;
    QString m_backendExe;
    bool m_manualStopRequested;
    bool m_restoreMetricsSuppressed;

    QVector<double> m_rawSpectrumSmooth;
    QVector<double> m_cleanSpectrumSmooth;

    void applyUiTextOverrides();
    void setupPlotStyles();
    void applyDarkPlotStyle(QCustomPlot *plot, const QString &xLabel, const QString &yLabel);
    void updatePlots();
    void resetPlots();
    void resetMetrics();
    void updatePlotPresentation();
    void updateRunningState(bool running);
    bool refreshRuntimePaths();
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
    QString mapModulationToCn(const QString &mode) const;
    void updateModulationMetricLabels();
};

#endif // MAINWINDOW_H
