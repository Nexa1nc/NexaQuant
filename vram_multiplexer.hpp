#pragma once
#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>
#include <algorithm>
#include <memory>
#include "mmap_loader.hpp"
#include "gguf_parser.hpp"

// Struttura che rappresenta un tensore/layer caricato in VRAM o RAM
struct VramLayer {
    std::string name;
    size_t size_bytes;
    bool in_vram = false;
    void* host_ptr = nullptr;
    void* gpu_buffer_ref = nullptr; // Riferimento al buffer OpenCL se in GPU
    uint64_t last_used_timestamp = 0;
};

struct RegisteredModel {
    std::string model_id;
    std::string file_path;
    std::shared_ptr<MmapLoader> loader;
    std::shared_ptr<GgufMinimalParser> parser;
    std::vector<VramLayer> layers;
    size_t total_weight_size = 0;
};

class VramMultiplexer {
private:
    size_t vram_budget_bytes;
    size_t current_vram_usage = 0;
    std::unordered_map<std::string, RegisteredModel> registry;
    std::vector<std::string> lru_tracker; // Tiene traccia dei modelli usati di recente
    uint64_t clock_ticks = 0;

    void update_lru(const std::string& model_id) {
        auto it = std::find(lru_tracker.begin(), lru_tracker.end(), model_id);
        if (it != lru_tracker.end()) {
            lru_tracker.erase(it);
        }
        lru_tracker.push_back(model_id);
    }

public:
    VramMultiplexer(size_t vram_budget_mb = 2048) // Default 2GB per GPU modeste (4GB totali)
        : vram_budget_bytes(vram_budget_mb * 1024 * 1024) {
        std::cout << "[M3 Multiplexer] Initialized with VRAM budget: " << vram_budget_mb << " MB\n";
    }

    bool register_model(const std::string& model_id, const std::string& file_path) {
        if (registry.find(model_id) != registry.end()) {
            std::cout << "[M3] Model " << model_id << " already registered.\n";
            return true;
        }

        auto loader = std::make_shared<MmapLoader>(file_path);
        if (!loader->map()) {
            std::cerr << "[M3 ERRORE] Impossibile mappare " << file_path << "\n";
            return false;
        }

        auto parser = std::make_shared<GgufMinimalParser>(loader->data(), loader->size());
        if (!parser->parse_all()) {
            std::cerr << "[M3 ERRORE] GGUF Parsing fallito per " << file_path << "\n";
            return false;
        }

        RegisteredModel model;
        model.model_id = model_id;
        model.file_path = file_path;
        model.loader = loader;
        model.parser = parser;

        // Estraiamo tutti i tensori del modello e creiamo la nostra virtualizzazione dei layer
        const auto& tensors = parser->get_tensors();
        for (const auto& t : tensors) {
            VramLayer layer;
            layer.name = t.name;
            
            // Calcoliamo il numero totale di elementi moltiplicando le dimensioni attive
            size_t total_elements = 1;
            if (t.n_dims == 0) {
                total_elements = 0;
            } else {
                for (uint32_t j = 0; j < t.n_dims; ++j) {
                    total_elements *= t.ne[j];
                }
            }
            
            layer.size_bytes = total_elements; // Per 1.58-bit ternary, ogni elemento equivale a 1 byte (o meno se compresso)
            layer.host_ptr = (void*)((uint8_t*)loader->data() + t.offset);
            layer.last_used_timestamp = 0;
            
            model.layers.push_back(layer);
            model.total_weight_size += layer.size_bytes;
        }

        registry[model_id] = model;
        std::cout << "[M3] Registered model '" << model_id << "' | Total Weights Size: "
                  << (model.total_weight_size / (1024.0 * 1024.0)) << " MB\n";
        return true;
    }

    // Forza il caricamento in VRAM delle parti calde del modello richiesto
    // Se la VRAM è piena, evince i layer dei modelli meno usati (LRU Swapping)
    bool activate_model(const std::string& model_id) {
        if (registry.find(model_id) == registry.end()) {
            std::cerr << "[M3 ERRORE] Modello " << model_id << " non registrato.\n";
            return false;
        }

        update_lru(model_id);
        clock_ticks++;
        
        RegisteredModel& target_model = registry[model_id];
        
        std::cout << "[M3] Activating model: " << model_id << "\n";
        
        for (auto& layer : target_model.layers) {
            layer.last_used_timestamp = clock_ticks;
            
            if (layer.in_vram) {
                continue; // Già caricato in GPU
            }

            // Se l'aggiunta di questo layer sfora il budget di VRAM, liberiamo spazio
            while (current_vram_usage + layer.size_bytes > vram_budget_bytes) {
                if (!evict_least_recent_layer()) {
                    std::cout << "[M3 WARNING] Impossibile liberare ulteriore VRAM. Attivato streaming diretto dalla RAM di sistema.\n";
                    break;
                }
            }

            // Virtualizziamo il caricamento in VRAM
            layer.in_vram = true;
            current_vram_usage += layer.size_bytes;
        }

        std::cout << "[M3] Model " << model_id << " is now ACTIVE. Current VRAM usage: "
                  << (current_vram_usage / (1024.0 * 1024.0)) << " MB / "
                  << (vram_budget_bytes / (1024.0 * 1024.0)) << " MB\n";
        
        return true;
    }

    // Libera la VRAM occupata dai layer meno recentemente usati di qualsiasi modello registrato
    bool evict_least_recent_layer() {
        uint64_t oldest_time = clock_ticks + 1;
        std::string oldest_model_id = "";
        int oldest_layer_idx = -1;

        // Cerca il layer "in_vram" con il timestamp più vecchio
        for (auto& pair : registry) {
            RegisteredModel& m = pair.second;
            for (size_t i = 0; i < m.layers.size(); ++i) {
                if (m.layers[i].in_vram && m.layers[i].last_used_timestamp < oldest_time) {
                    oldest_time = m.layers[i].last_used_timestamp;
                    oldest_model_id = pair.first;
                    oldest_layer_idx = static_cast<int>(i);
                }
            }
        }

        if (oldest_layer_idx != -1) {
            VramLayer& evict_layer = registry[oldest_model_id].layers[oldest_layer_idx];
            evict_layer.in_vram = false;
            current_vram_usage -= evict_layer.size_bytes;
            std::cout << "[M3 EVICT] Evicted layer '" << evict_layer.name << "' from model '" 
                      << oldest_model_id << "' to free " << (evict_layer.size_bytes / 1024.0) << " KB VRAM\n";
            return true;
        }

        return false; // Nessun layer rimasto da liberare
    }

    RegisteredModel* get_model(const std::string& model_id) {
        if (registry.find(model_id) != registry.end()) {
            return &registry[model_id];
        }
        return nullptr;
    }

    size_t get_vram_usage() const { return current_vram_usage; }
    size_t get_vram_budget() const { return vram_budget_bytes; }
};
