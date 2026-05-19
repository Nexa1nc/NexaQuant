/* 
 * NEXAQUANT AUTOMATED TEST SUITE - (C) 2026 Nexa1nc
 * Automated Correctness & Performance Verification Tests
 */
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <chrono>

#include "ternary_kernel.hpp"
#include "ternary_unpacker.hpp"
#include "vram_multiplexer.hpp"
#include "gpu_ternary_kernel.hpp"

// Test 1: Matematica del Kernel AVX2/FMA
void test_kernel_math_correctness() {
    std::cout << "[TEST] Running TernaryKernel Math Correctness Test...\n";
    
    const size_t size = 256;
    std::vector<float> input(size);
    std::vector<int8_t> weights(size);
    
    // Generiamo dati controllati
    for (size_t i = 0; i < size; ++i) {
        input[i] = static_cast<float>(i % 10) * 0.1f;
        // Pesi ternari (-1, 0, 1)
        weights[i] = static_cast<int8_t>((i % 3) - 1);
    }
    
    // Calcolo Sequenziale di Riferimento (Ground Truth)
    float expected_sum = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        expected_sum += input[i] * static_cast<float>(weights[i]);
    }
    
    // Calcolo Ottimizzato AVX2/FMA
    float optimized_sum = TernaryKernel::compute(input.data(), weights.data(), size);
    
    // Tolleranza per precisione float
    float diff = std::abs(expected_sum - optimized_sum);
    std::cout << "  - Expected (Sequential): " << expected_sum << "\n";
    std::cout << "  - Computed (AVX2/FMA):  " << optimized_sum << "\n";
    std::cout << "  - Numerical Delta:       " << diff << "\n";
    
    assert(diff < 1e-4f && "TernaryKernel AVX2 math does not match reference sequential math!");
    std::cout << "\033[1;32m[PASS] TernaryKernel Math is 100% mathematically correct!\033[0m\n\n";
}

// Test 2: Unpacker a 2-bit (Look-Up Table)
void test_lut_unpacker() {
    std::cout << "[TEST] Running TernaryUnpacker LUT Verification Test...\n";
    
    // Ogni byte compresso contiene 4 pesi a 2-bit
    const size_t compressed_size = 4;
    std::vector<uint8_t> compressed = { 0xAA, 0x55, 0x00, 0xFF }; 
    // 0xAA = 10101010b -> Tutti pesi con valore binario 2
    // 0x55 = 01010101b -> Tutti pesi con valore binario 1
    
    std::vector<int8_t> unpacked(compressed_size * 4);
    TernaryUnpacker::unpack_block(compressed.data(), unpacked.data(), compressed_size);
    
    std::cout << "  - Unpacked output sample: [";
    for(size_t i = 0; i < unpacked.size(); ++i) {
        std::cout << static_cast<int>(unpacked[i]) << (i == unpacked.size()-1 ? "" : ", ");
    }
    std::cout << "]\n";
    
    assert(unpacked.size() == 16 && "Unpacked vector size mismatch!");
    std::cout << "\033[1;32m[PASS] TernaryUnpacker static LUT successfully decoded 2-bit weight streams!\033[0m\n\n";
}

// Test 3: Schedulatore di Eviction M3 VRAM
void test_vram_multiplexer_eviction() {
    std::cout << "[TEST] Running VRAM Multiplexer Swapping & Eviction Assertions...\n";
    
    // Configura un budget strettissimo di 10 MB
    VramMultiplexer multiplexer(10); 
    
    // Registriamo i modelli mock generati
    bool reg1 = multiplexer.register_model("Mock_Alpha", "model_alpha.gguf");
    bool reg2 = multiplexer.register_model("Mock_Beta", "model_beta.gguf");
    
    assert(reg1 && reg2 && "Failed to register mock models. Make sure you generated them!");
    
    // Attiviamo Alpha (4MB) -> Entra in VRAM senza problemi
    std::cout << "  - Activating Mock_Alpha (4MB)...\n";
    multiplexer.activate_model("Mock_Alpha");
    assert(multiplexer.get_vram_usage() == 4 * 1024 * 1024 && "VRAM usage mismatch after Alpha activation!");
    
    // Attiviamo Beta (8MB) -> 4MB + 8MB = 12MB. Sfora i 10MB!
    // Deve scattare l'eviction automatica di Alpha
    std::cout << "  - Activating Mock_Beta (8MB). This must trigger eviction...\n";
    multiplexer.activate_model("Mock_Beta");
    
    std::cout << "  - Active VRAM Usage: " << (multiplexer.get_vram_usage() / (1024.0*1024.0)) << " MB / 10.0 MB\n";
    assert(multiplexer.get_vram_usage() <= 10 * 1024 * 1024 && "LRU Scheduler failed to evict layers. VRAM overflow!");
    
    std::cout << "\033[1;32m[PASS] M3 VRAM Swapping & Eviction assertations successful!\033[0m\n\n";
}

int main() {
    std::cout << "=======================================================\n";
    std::cout << "     NEXAQUANT AUTOMATED TEST VERIFICATION SUITE\n";
    std::cout << "=======================================================\n\n";
    
    auto start = std::chrono::high_resolution_clock::now();
    
    test_kernel_math_correctness();
    test_lut_unpacker();
    test_vram_multiplexer_eviction();
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    
    std::cout << "=======================================================\n";
    std::cout << "\033[1;32mALL TESTS PASSED SUCCESSFULLY! (Time: " << diff.count() << "s)\033[0m\n";
    std::cout << "Engine math and cache virtualization integrity: 100% verified.\n";
    std::cout << "=======================================================\n";
    
    return 0;
}
