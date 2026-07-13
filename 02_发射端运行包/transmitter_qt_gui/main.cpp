#include "mainwindow.h"

#include <QApplication>
#include <QTimer>

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    MainWindow window;
    window.showFullScreen();
    QTimer::singleShot(1000, &window, [&window]() {
        window.showFullScreen();
        window.raise();
        window.activateWindow();
    });
    return app.exec();
}
