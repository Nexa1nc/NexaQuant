/* NEXAQUANT ENGINE - (C) 2026 Nexa1nc | GPL v3 */
#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include <cstring>

constexpr uint32_t GGUF_MAGIC = 0x46554747;

enum GGUF_METADATA_VALUE_TYPE {
    GGUF_TYPE_UINT8 = 0, GGUF_TYPE_INT8 = 1, GGUF_TYPE_UINT16 = 2, GGUF_TYPE_INT16 = 3,
    GGUF_TYPE_UINT32 = 4, GGUF_TYPE_INT32 = 5, GGUF_TYPE_FLOAT32 = 6, GGUF_TYPE_BOOL = 7,
    GGUF_TYPE_STRING = 8, GGUF_TYPE_ARRAY = 9, GGUF_TYPE_UINT64 = 10, GGUF_TYPE_INT64 = 11,
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
        if (len > 1024 * 1024 || offset + len > file_size) {
            // Se la stringa è palesemente assurda (>1MB), c'è un disallineamento
            offset = file_size; 
            return "";
        }
        std::string str(reinterpret_cast<const char*>(raw_data + offset), (size_t)len);
        offset += (size_t)len;
        return str;
    }

    void skip_value(uint32_t type) {
        if (offset > file_size) return;
        switch (type) {
            case GGUF_TYPE_UINT8: case GGUF_TYPE_INT8: case GGUF_TYPE_BOOL: offset += 1; break;
            case GGUF_TYPE_UINT16: case GGUF_TYPE_INT16: offset += 2; break;
            case GGUF_TYPE_UINT32: case GGUF_TYPE_INT32: case GGUF_TYPE_FLOAT32: offset += 4; break;
            case GGUF_TYPE_UINT64: case GGUF_TYPE_INT64: case GGUF_TYPE_FLOAT64: offset += 8; break;
            case GGUF_TYPE_STRING: {
                uint64_t len = read_data<uint64_t>();
                if (offset + len > file_size) { offset = file_size; return; }
                offset += len;
                break;
            }
            case GGUF_TYPE_ARRAY: {
                uint32_t item_type = read_data<uint32_t>();
                uint64_t count = read_data<uint64_t>();
                if (count > 100000) count = 100000; // Supporto per tokenizer grandi
                for (uint64_t i = 0; i < count; ++i) {
                    if (offset >= file_size) break;
                    skip_value(item_type);
                }
                break;
            }
            default: break;
        }
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

        std::cout << "[DEBUG] GGUF v" << version << " | Tensors: " << tensor_count << " | KV Pairs: " << kv_count << "\n";

        // Saltiamo correttamente i KV pairs
        for (uint64_t i = 0; i < kv_count; ++i) {
            std::string key = read_string();
            uint32_t type = read_data<uint32_t>();
            std::cout << "  [KV " << i << "] Key: " << key << " | Type: " << type << " | Offset: " << offset << "\n";
            if (offset > file_size) {
                std::cerr << "[ERROR] Offset exceeded file size during KV parsing at index " << i << "\n";
                return false;
            }
            skip_value(type);
        }

        // Parsing dei Tensori
        for (uint64_t i = 0; i < tensor_count; ++i) {
            GgufTensorInfo t;
            t.name = read_string();
            t.n_dims = read_data<uint32_t>();
            if (t.n_dims > 4) return false; // Protezione
            for (uint32_t j = 0; j < t.n_dims; ++j) t.ne[j] = read_data<uint64_t>();
            t.type = read_data<uint32_t>();
            t.offset = read_data<uint64_t>();
            tensors.push_back(t);
        }
        return true;
    }

    const std::vector<GgufTensorInfo>& get_tensors() const { return tensors; }
};
