#pragma once
#include <iostream>
#include <vector>
#include <unordered_map>
#include <sys/mman.h>

struct LayerData {
    size_t offset;
    size_t size;
    void* active_pointer;
};

class DynamicLayerManager {
private:
    size_t ram_budget_bytes;
    size_t current_usage;
    std::vector<LayerData> layers;
    
public:
    DynamicLayerManager(size_t budget_mb = 512) 
        : ram_budget_bytes(budget_mb * 1024 * 1024), current_usage(0) {}

    void add_layer(size_t offset, size_t size) {
        layers.push_back({offset, size, nullptr});
    }

    void* activate_layer(size_t layer_idx, void* mmap_base_ptr) {
        LayerData& layer = layers[layer_idx];
        if (current_usage + layer.size > ram_budget_bytes) {
            clear_cache();
        }
        layer.active_pointer = (void*)((uint8_t*)mmap_base_ptr + layer.offset);
#ifndef _WIN32
        madvise(layer.active_pointer, layer.size, MADV_WILLNEED);
#endif
        current_usage += layer.size;
        return layer.active_pointer;
    }

    void clear_cache() {
        for (auto& layer : layers) {
            if (layer.active_pointer) {
#ifndef _WIN32
                madvise(layer.active_pointer, layer.size, MADV_DONTNEED);
#endif
                layer.active_pointer = nullptr;
            }
        }
        current_usage = 0;
    }
};
