#include "mainwindow.h"

#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    QApplication::setApplicationName(QStringLiteral("rk3588_gui"));
    QApplication::setApplicationDisplayName(QStringLiteral("RK3588无线电干扰识别与还原系统"));

    MainWindow window;
    window.show();
    return app.exec();
}
