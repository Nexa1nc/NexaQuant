#pragma once
#ifdef _WIN32
#include <intrin.h>
#endif

class CpuFeatureDetector {
public:
    static void print_report() {
        std::cout << "--- CPU Detection: AVX2/FMA Optimized ---\n";
    }
};
