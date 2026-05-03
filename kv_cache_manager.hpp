#pragma once
#include <vector>
#include <unordered_map>
#include <chrono>

struct KVCachePage {
    std::vector<float> data;
    bool is_on_disk;
    long last_access;
};

class KVCacheManager {
private:
    size_t max_ram_pages;
    std::unordered_map<size_t, KVCachePage> cache;
public:
    KVCacheManager(size_t max_pages = 100) : max_ram_pages(max_pages) {}
    
    void add_page(size_t id, const std::vector<float>& data) {
        if (cache.size() >= max_ram_pages) {
            std::cout << "[KV-Cache] Evicting to disk...\n";
        }
        cache[id] = {data, false, 0};
    }
};
