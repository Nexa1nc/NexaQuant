cmake_minimum_required(VERSION 3.14)
project(Nexa_Quant VERSION 0.1.0 LANGUAGES C CXX)

# Richiedi C++17 come minimo
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# --- OTTIMIZZAZIONI CPU (AVX / AVX2 / FMA) ---
# Questi flag indicano al compilatore di usare le istruzioni vettoriali della CPU.
# Su Windows (MSVC) usiamo /arch:AVX2. Su GCC/Clang usiamo -mavx2 -mfma.
if(MSVC)
    add_compile_options(/O2 /arch:AVX2 /fp:fast)
    # Ignora warning specifici di C su MSVC
    add_compile_options(/wd4244 /wd4267 /wd4334)
else()
    add_compile_options(-O3 -mavx -mavx2 -mfma -march=native)
endif()

# --- INTEGRAZIONE LIBRERIA GGML ---
# Usiamo FetchContent per scaricare in automatico il motore matematico (GGML)
# dal repository ufficiale di llama.cpp senza doverlo scaricare manualmente.
include(FetchContent)

message(STATUS "Scaricamento di GGML in corso...")
FetchContent_Declare(
    llama_cpp
    GIT_REPOSITORY https://github.com/ggerganov/llama.cpp.git
    GIT_TAG        master
)

# GGML fa parte del progetto llama.cpp, ma noi vogliamo compilare SOLO ggml,
# senza tutte le altre dipendenze inutili per mantenere il binario minuscolo.
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "Disable tests" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "Disable examples" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "Disable server" FORCE)
# Disabilitiamo CUDA/GPU, vogliamo girare solo su CPU datate!
set(LLAMA_CUBLAS OFF CACHE BOOL "Disable CUDA" FORCE)
set(LLAMA_METAL OFF CACHE BOOL "Disable Metal" FORCE)

FetchContent_MakeAvailable(llama_cpp)

# --- DEFINIZIONE DELL'ESEGUIBILE ---
# Aggiungi qui tutti i file .cpp del tuo progetto
add_executable(nexa_quant main.cpp)

# Collega le librerie
# 'ggml' è il core matematico.
# 'llama' contiene i parser GGUF di base se ci servono in futuro.
target_link_libraries(nexa_quant PRIVATE ggml)

if(NOT MSVC)
    # Su Linux/Mac serve linkare pthreads per il multithreading
    target_link_libraries(nexa_quant PRIVATE pthread)
endif()

message(STATUS "Configurazione di Nexa_Quant completata con successo. Ottimizzato per CPU Edge/Datate.")
