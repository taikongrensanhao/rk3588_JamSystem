RK3588 专用 Qt 工程

工程说明
- 这是给 RK3588 板端使用的 Qt 前端。
- Qt 不直接启动 `predict_single.py`，而是直接拉起 `/home/pi/Desktop/JamSystem/ad9361_rk3588`。
- `ad9361_rk3588` 会采集真实信号、写入 `output/captured.bin`，再调用 `predict_single.py`，并持续刷新 `output/live_plot.bin`。
- Qt 负责显示识别结果、还原结果，以及 IQ、星座图、频谱图。

板端默认路径
- JamSystem 根目录：`/home/pi/Desktop/JamSystem`
- 后端可执行文件：`/home/pi/Desktop/JamSystem/ad9361_rk3588`
- 绘图数据文件：`/home/pi/Desktop/JamSystem/output/live_plot.bin`

可选环境变量
- `JAMSYSTEM_BASE_PATH`
- `JAMSYSTEM_AD9361_EXE`

RK3588 编译命令
```bash
cd /home/pi/Desktop/rk3588_gui
qmake rk3588_gui.pro
make -j4
```

运行命令
```bash
./rk3588_gui
```

如果板子上是 Qt6，也可以用
```bash
qmake6 rk3588_gui.pro
make -j4
```
