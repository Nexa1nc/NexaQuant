#pragma once
#include <iostream>
#include <immintrin.h>

class TernaryKernel {
public:
    static float compute(const float* input, const int8_t* weights, size_t size) {
        __m256 sum_vec = _mm256_setzero_ps();
        size_t i = 0;
        for (; i <= size - 8; i += 8) {
            __m256 v_in = _mm256_loadu_ps(input + i);
            for (int j = 0; j < 8; ++j) {
                int8_t w = weights[i + j];
                if (w == 1) sum_vec = _mm256_add_ps(sum_vec, v_in);
                else if (w == -1) sum_vec = _mm256_sub_ps(sum_vec, v_in);
            }
        }
        float buffer[8];
        _mm256_storeu_ps(buffer, sum_vec);
        float final_sum = 0;
        for (int j = 0; j < 8; ++j) final_sum += buffer[j];
        for (; i < size; ++i) {
            if (weights[i] == 1) final_sum += input[i];
            else if (weights[i] == -1) final_sum -= input[i];
        }
        return final_sum;
    }
};
