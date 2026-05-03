#pragma once
#include <cstdint>

class TernaryUnpacker {
public:
    static void unpack_block(const uint8_t* packed_data, int8_t* out_weights, size_t packed_size) {
        for (size_t i = 0; i < packed_size; ++i) {
            uint8_t b = packed_data[i];
            out_weights[i * 4 + 0] = decode_2bit((b >> 0) & 0x03);
            out_weights[i * 4 + 1] = decode_2bit((b >> 2) & 0x03);
            out_weights[i * 4 + 2] = decode_2bit((b >> 4) & 0x03);
            out_weights[i * 4 + 3] = decode_2bit((b >> 6) & 0x03);
        }
    }
private:
    static inline int8_t decode_2bit(uint8_t two_bits) {
        if (two_bits == 0) return 0;
        if (two_bits == 1) return 1;
        if (two_bits == 2) return -1;
        return 0;
    }
};
