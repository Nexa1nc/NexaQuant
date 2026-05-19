#!/bin/bash
# Script di compilazione, unit testing e benchmark per NexaQuant v2.0
cd "$(dirname "$0")"

echo "[BUILD] Compiling automated test suite (tests.cpp)..."
g++ -O3 -mavx2 -mfma tests.cpp -o nexa_test -lpthread -ldl

if [ $? -ne 0 ]; then
    echo "[BUILD ERROR] Automated test suite compilation failed!"
    exit 1
fi

echo "[TEST] Running NexaQuant Automated Test Suite..."
./nexa_test

if [ $? -ne 0 ]; then
    echo "[TEST ERROR] Automated tests failed! Inference math or cache logic is broken."
    exit 1
fi

echo "[BUILD] Compiling NexaQuant v2.0 Engine..."
g++ -O3 -mavx2 -mfma main.cpp -o nexa_bench -lpthread -ldl

if [ $? -eq 0 ]; then
    echo "[BUILD] Compilation successful!"
    echo "[RUN] Executing NexaQuant v2.0 VRAM Multiplexer Benchmarks..."
    ./nexa_bench
else
    echo "[BUILD ERROR] Engine compilation failed!"
    exit 1
fi
