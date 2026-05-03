/* NEXAQUANT ENGINE - (C) 2026 NexaQuant Author | GPL v3 */
#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include <cstring>
#include <stdexcept>

// Costante Magic Number per GGUF (0x46554747 in Little Endian -> "GGUF")
constexpr uint32_t GGUF_MAGIC = 0x46554747;

// Tipi di dati dei metadati in GGUF
enum GGUF_METADATA_VALUE_TYPE {
    GGUF_TYPE_UINT8 = 0,
    GGUF_TYPE_INT8 = 1,
    GGUF_TYPE_UINT16 = 2,
    GGUF_TYPE_INT16 = 3,
    GGUF_TYPE_UINT32 = 4,
    GGUF_TYPE_INT32 = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL = 7,
    GGUF_TYPE_STRING = 8,
    GGUF_TYPE_ARRAY = 9,
    GGUF_TYPE_UINT64 = 10,
    GGUF_TYPE_INT64 = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

struct GgufTensorInfo {
    std::string name;
    uint32_t n_dims;
    uint64_t ne[4];
    uint32_t type;
    uint64_t offset;
};

class GgufMinimalParser {
private:
    const uint8_t* raw_data;
    size_t file_size;
    size_t offset;
    std::vector<GgufTensorInfo> tensors;

    template<typename T>
    T read_data() {
        if (offset + sizeof(T) > file_size) return T();
        T value;
        std::memcpy(&value, raw_data + offset, sizeof(T));
        offset += sizeof(T);
        return value;
    }

    std::string read_string() {
        uint64_t len = read_data<uint64_t>();
        if (offset + len > file_size) return "";
        std::string str(reinterpret_cast<const char*>(raw_data + offset), len);
        offset += len;
        return str;
    }

public:
    GgufMinimalParser(const void* mmap_data, size_t size) 
        : raw_data(static_cast<const uint8_t*>(mmap_data)), file_size(size), offset(0) {}

    bool parse_all() {
        if (file_size < 4) return false;
        uint32_t magic = read_data<uint32_t>();
        if (magic != GGUF_MAGIC) return false;
        uint32_t version = read_data<uint32_t>();
        uint64_t tensor_count = read_data<uint64_t>();
        uint64_t kv_count = read_data<uint64_t>();

        // Saltiamo i KV pairs per velocità in questo test
        // In un'implementazione reale leggeremmo i metadati qui

        // Parsing dei Tensori
        for (uint64_t i = 0; i < tensor_count; ++i) {
            GgufTensorInfo t;
            t.name = read_string();
            t.n_dims = read_data<uint32_t>();
            for (uint32_t j = 0; j < t.n_dims; ++j) t.ne[j] = read_data<uint64_t>();
            t.type = read_data<uint32_t>();
            t.offset = read_data<uint64_t>();
            tensors.push_back(t);
        }
        return true;
    }

    const std::vector<GgufTensorInfo>& get_tensors() const { return tensors; }
};
