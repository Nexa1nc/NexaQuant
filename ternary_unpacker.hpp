/* NEXAQUANT ENGINE - (C) 2026 Nexa1nc | GPL v3 */
#pragma once
#include <cstdint>

class TernaryUnpacker {
public:
    // Tabella di decodifica statica: 00->0, 01->1, 10->-1, 11->0
    static constexpr int8_t DECODE_LUT[4] = {0, 1, -1, 0};

    static void unpack_block(const uint8_t* packed_data, int8_t* out_weights, size_t packed_size) {
        for (size_t i = 0; i < packed_size; ++i) {
            uint8_t b = packed_data[i];
            
            // Decodifica istantanea tramite LUT senza branch
            out_weights[i * 4 + 0] = DECODE_LUT[(b >> 0) & 0x03];
            out_weights[i * 4 + 1] = DECODE_LUT[(b >> 2) & 0x03];
            out_weights[i * 4 + 2] = DECODE_LUT[(b >> 4) & 0x03];
            out_weights[i * 4 + 3] = DECODE_LUT[(b >> 6) & 0x03];
        }
    }
};
