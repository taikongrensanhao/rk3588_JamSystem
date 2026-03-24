#!/bin/bash
set -e

pkill -f rk3588_gui || true
pkill -f ad9361_rk3588 || true
