#pragma once

#include <iostream>
#include <vector>
#include <thread>
#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <future>
#include <stdexcept>

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#endif

class CpuOptimizer {
private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;

    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
    size_t num_physical_cores;

    // Funzione helper per ottenere il numero reale di core fisici (ignorando l'Hyper-Threading)
    // Su CPU vecchie, l'Hyper-threading spesso peggiora le performance matematiche pesanti (es. GEMM)
    // perché i due thread logici competono per la stessa cache L1/L2 e unità vettoriali (AVX).
    size_t get_physical_cores() {
        // Un'implementazione robusta richiederebbe parsing complesso.
        // Per semplicità qui assumiamo che se abbiamo N core logici, abbiamo N/2 fisici
        // (comune per Intel/AMD con SMT). Nelle librerie reali si usa CPUID o sysfs.
        size_t logical = std::thread::hardware_concurrency();
        return logical > 1 ? logical / 2 : 1; 
    }

    // Funzione per vincolare un thread a un core specifico (Thread Affinity)
    void pin_thread_to_core(std::thread& t, size_t core_id) {
#ifdef _WIN32
        HANDLE thread_handle = t.native_handle();
        // Crea una maschera a bit (es. core 0 -> 0001, core 1 -> 0010, core 2 -> 0100)
        DWORD_PTR affinity_mask = (1ULL << core_id); 
        DWORD_PTR result = SetThreadAffinityMask(thread_handle, affinity_mask);
        if (result == 0) {
            std::cerr << "Avviso: Impossibile impostare l'affinity per il core " << core_id << "\n";
        }
#else
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(core_id, &cpuset);
        pthread_t native_thread = t.native_handle();
        int rc = pthread_setaffinity_np(native_thread, sizeof(cpu_set_t), &cpuset);
        if (rc != 0) {
            std::cerr << "Avviso: Impossibile impostare l'affinity per il core " << core_id << "\n";
        }
#endif
    }

public:
    CpuOptimizer(size_t threads = 0) : stop(false) {
        num_physical_cores = get_physical_cores();
        
        // Se l'utente non specifica i thread, usa i core fisici per massima efficienza
        size_t num_threads = (threads == 0) ? num_physical_cores : threads;
        
        std::cout << "Inizializzazione ThreadPool con " << num_threads << " thread (ottimizzato per LLM)..." << std::endl;

        for (size_t i = 0; i < num_threads; ++i) {
            workers.emplace_back([this] {
                for (;;) {
                    std::function<void()> task;

                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock, [this] { return this->stop || !this->tasks.empty(); });
                        if (this->stop && this->tasks.empty())
                            return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }

                    task(); // Esegue il task
                }
            });

            // Applica il "Pinning" dei thread ai core fisici.
            // Mappiamo i thread: 0->Core 0, 1->Core 2, 2->Core 4...
            // Questo perché spesso i sistemi operativi assegnano core logici adiacenti (0 e 1) allo stesso core fisico.
            size_t target_core = (i % num_physical_cores) * 2; // Salta di 2 per evitare l'HT
            pin_thread_to_core(workers.back(), target_core);
        }
    }

    // Aggiungi un task alla coda (es. un blocco di moltiplicazione di matrici)
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<typename std::invoke_result<F, Args...>::type> {
        using return_type = typename std::invoke_result<F, Args...>::type;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );

        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (stop)
                throw std::runtime_error("Impossibile accodare: ThreadPool fermato.");
            tasks.emplace([task]() { (*task)(); });
        }
        condition.notify_one();
        return res;
    }

    ~CpuOptimizer() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread& worker : workers) {
            worker.join();
        }
    }
};

/* ESEMPIO DI UTILIZZO
int main() {
    // Inizializza il pool che configurerà automaticamente i thread per i core fisici
    CpuOptimizer pool;

    // Simuliamo 4 task pesanti (es. moltiplicazioni riga-colonna usando AVX2)
    std::vector<std::future<int>> results;
    for(int i = 0; i < 4; ++i) {
        results.push_back(pool.enqueue([i] {
            // Qui inseriremo chiamate a GGML o funzioni SIMD custom
            std::cout << "Esecuzione Task Tensoriale " << i << " sul thread corrente.\n";
            return i * i;
        }));
    }

    for(auto && result: results)
        std::cout << "Risultato: " << result.get() << "\n";

    return 0;
}
*/
