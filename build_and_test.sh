#!/bin/bash
# Script di compilazione e benchmark per NexaQuant v2.0
cd "$(dirname "$0")"

echo "[BUILD] Compiling NexaQuant v2.0 Engine..."
g++ -O3 -mavx2 -mfma main.cpp -o nexa_bench -lpthread -ldl

if [ $? -eq 0 ]; then
    echo "[BUILD] Compilation successful!"
    echo "[RUN] Executing NexaQuant v2.0 VRAM Multiplexer Benchmarks..."
    ./nexa_bench
else
    echo "[BUILD ERROR] Compilation failed!"
fi
