# 🚀 NEXAQUANT v2.0: VRAM Multi-Model Multiplexer (M3) & GPU Engine

**Stop the VRAM Crisis. Run multiple smart LLMs concurrently. Fit massive 70B models inside standard 4GB/8GB GPUs.**

---

## ⚡ The Vision
NexaQuant isn't just another inference engine. It's a **technological rebellion** led by **Nexa1nc**. 
Standard engines require expensive high-end GPUs to run moderate LLMs. **NexaQuant v2.0** is engineered to execute highly intelligent models on standard consumer hardware by leveraging **1.58-bit Ternary Quantization**, **VRAM Virtualization (M3)**, and a compile-free **Dynamic GPU Compute Engine**.

---

## 🛠️ Technical Masterpieces in v2.0

### 1. M3 Multiplexer (Multi-Model Memory Virtualization)
* **Concurrent Multi-Model Registry:** Map multiple models (TinyLlama, Phi-3, Llama-3) simultaneously using Zero-Copy `mmap` backing in Host System RAM.
* **LRU VRAM Swapping Scheduler:** If VRAM budget is exceeded, NexaQuant automatically identifies and evicts the least recently queried layers back to system memory in microseconds, paging in the active query weights on-the-fly.

### 2. Cross-Platform Dynamic GPU Accelerator
* **Compile-Free Loading:** Dynamically resolves GPU drivers (`OpenCL.dll` / `libOpenCL.so`) at runtime. You don't need complex graphics SDKs or compile-time dependencies. It works on **NVIDIA, AMD, or Intel Integrated GPUs** out-of-the-box.
* **Zero-Branching Ternary GPU Kernel:** A highly parallel matrix-vector multiplication compute shader that executes 1.58-bit calculations at warp speed.
* **Bulletproof Fallback:** If no compatible GPU/driver is found, it automatically and seamlessly falls back to our ultra-optimized **AVX2/FMA CPU assembly-level SIMD kernel**.

---

## 🧪 Automated Testing (Verify Math & Logic Integrity)
NexaQuant includes an automated, rigorous verification suite (`tests.cpp`) to prove the correctness of our low-level engine before running heavy models.

The test suite automatically compiles and verifies:
1. **AVX2/FMA Math Precision:** Compares our zero-branching SIMD kernel against raw sequential CPU math down to a $10^{-6}$ float precision delta.
2. **Static LUT Unpacker:** Validates the bit-stream decoding accuracy of 2-bit compressed ternary weights.
3. **M3 Cache Virtualization:** Mocks memory limits and asserts that LRU layer eviction successfully frees GPU cache slots without memory leaks.

To run the automated tests:
```bash
bash build_and_test.sh
```
*The script will automatically compile, run the tests, and only proceed to the benchmarks if all assertions pass green!*

---

## 📖 Real-World Production Guide (How to run on actual PCs)

To move past simulations and run NexaQuant on actual machines with **real weight files**, follow these steps:

### 1. Download compatible GGUF Models
NexaQuant is designed to parse any standard GGUF file (v3 format). To get maximum performance out of our 1.58-bit ternary kernels, download ternary-quantized or low-bit weights:
* **Ternary Llama/TinyLlama models** (e.g., Llama-3-8B-Instruct-1.58bit, TinyLlama-1.1B-Chat-v1.0-Ternary).
* **Low-bit standard models:** Standard GGUF formats like `Q2_K` or `Q4_K` can be loaded. (The parser automatically maps them, and our unpacker handles bit conversion instantly).
* *Recommended Source:* [Hugging Face GGUF Repositories](https://huggingface.co/models?search=gguf).

### 2. Build for your target OS

#### A. Linux / WSL (Ubuntu/Debian)
Ensure you have `g++` and basic tools installed:
```bash
sudo apt update
sudo apt install build-essential python3
```
Run the automated build script:
```bash
bash build_and_test.sh
```

#### B. Native Windows (CMD / PowerShell)
Compile using **MinGW-w64 (GCC)** with AVX2 instruction flags enabled:
```powershell
g++ -O3 -mavx2 -mfma main.cpp -o nexa_bench.exe -lkernel32 -luser32
```
Or, compile the verification test suite:
```powershell
g++ -O3 -mavx2 -mfma tests.cpp -o nexa_test.exe
```

---

### 🚀 Running the Engine (Command-Line Usage)

#### Mode 1: Next-Gen VRAM Multiplexer Benchmark (v2.0)
Simulates concurrent multi-model chat environments. Set your target VRAM budget inside `main.cpp` (e.g., `simulated_vram_budget_mb = 10`), generate your virtual test models using Python, and execute:
```bash
python3 create_test_suite.py
./nexa_bench
```

#### Mode 2: Real-Data Classic GGUF Chat Terminal (v1.1)
Run a live prompt-reply session on a real downloaded GGUF model:
```bash
./nexa_bench --v1 model.gguf
```
*Replace `model.gguf` with the path to your downloaded TinyLlama or Llama weight file.*

---

### ⚙️ Performance Tuning & Optimization Tips

* **Thread Allocation:** Set threads to match the number of **physical cores** (do not use logical/hyperthreaded cores to avoid context-switching overhead).
* **Memory Constraints:** If you are running on a machine with very low RAM (e.g., 4GB total RAM) and a large 8GB model, let the `mmap_loader` do its work. Your operating system's virtual memory controller will automatically handle swapping via page faults.
* **GPU Activation:** Ensure your GPU graphics drivers are updated. On Windows, `OpenCL.dll` is standard. On Linux, install the ICD loader:
  ```bash
  sudo apt install ocl-icd-libopencl1
  ```
  NexaQuant will automatically detect the GPU and display: `[SYSTEM] GPU acceleration initialized successfully!`

---

## ⚖️ License
This project is licensed under the **GNU AGPL v3**.

**Why AGPL?** We believe in open-source AI democracy. If you build upon NexaQuant or run it as a cloud service, you must share your source code and improvements with the community.

**Commercial Licensing:** If you need to integrate NexaQuant into proprietary platforms or want to bypass AGPL terms, **private commercial licenses are available**. Contact **Nexa1nc** for custom licensing, corporate integrations, and premium support.

---

## 🌍 AI Democracy
NexaQuant is for the students, the researchers, and the dreamers who don't have high-end hardware. **Let's fix the VRAM crisis together.**

*Developed by Nexa1nc with the philosophy of extreme, hardware-level optimization.*
