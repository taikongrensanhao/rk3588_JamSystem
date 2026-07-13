#!/bin/bash
sleep 20
export DISPLAY=:0
export XAUTHORITY=/home/pi/.Xauthority
cd /home/pi/Desktop/run/receiver_run
pkill -f rk3588_gui || true
pkill -f ad9361_rk3588 || true
./start_gui.sh >> /home/pi/Desktop/run/receiver_run/autostart.log 2>&1
