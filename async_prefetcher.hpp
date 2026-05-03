#pragma once
#include <iostream>
#include <future>

#ifndef _WIN32
#include <sys/mman.h>
#endif

class AsyncPrefetcher {
public:
    static void prefetch_next(void* ptr, size_t size) {
        if (!ptr || size == 0) return;
        std::async(std::launch::async, [ptr, size]() {
#ifndef _WIN32
            madvise(ptr, size, MADV_WILLNEED);
#endif
        });
    }
};
