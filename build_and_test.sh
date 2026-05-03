#!/bin/bash
# Script di compilazione e benchmark per NexaQuant
# Traduciamo i percorsi Windows per WSL se necessario
cd "$(dirname "$0")"
g++ -O3 -mavx2 -mfma main.cpp -o nexa_bench -lpthread
if [ $? -eq 0 ]; then
    ./nexa_bench
else
    echo "Errore di compilazione!"
fi
