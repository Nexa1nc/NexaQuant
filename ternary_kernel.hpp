/* NEXAQUANT ENGINE - (C) 2026 Nexa1nc | GPL v3 */
#pragma once
#include <iostream>
#include <immintrin.h>

class TernaryKernel {
public:
    // Versione Ultra-Ottimizzata con FMA (Fused Multiply-Add)
    static float compute(const float* input, const int8_t* weights, size_t size) {
        __m256 sum_vec = _mm256_setzero_ps();
        size_t i = 0;
        
        // Processiamo 8 elementi alla volta usando SIMD
        for (; i <= size - 8; i += 8) {
            __m256 v_in = _mm256_loadu_ps(input + i);
            
            // Convertiamo gli 8 pesi int8 in float nel registro
            // Questo trucco elimina tutti gli "if" e permette alla CPU di correre
            float w_float[8];
            for(int j=0; j<8; ++j) w_float[j] = static_cast<float>(weights[i+j]);
            
            __m256 v_w = _mm256_loadu_ps(w_float);
            
            // sum = sum + (input * weight)
            sum_vec = _mm256_fmadd_ps(v_in, v_w, sum_vec);
        }
        
        float buffer[8];
        _mm256_storeu_ps(buffer, sum_vec);
        
        float final_sum = 0;
        for (int j = 0; j < 8; ++j) final_sum += buffer[j];
        
        // Gestione degli elementi rimanenti (se non multipli di 8)
        for (; i < size; ++i) {
            final_sum += input[i] * static_cast<float>(weights[i]);
        }
        
        return final_sum;
    }
};
