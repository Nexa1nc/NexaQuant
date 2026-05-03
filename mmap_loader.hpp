#include <iostream>
#include <string>
#include <stdexcept>
#include <cstdint>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

// Classe per gestire il Memory Mapping di file GGUF
class MmapLoader {
/* NEXAQUANT ENGINE - (C) 2026 NexaQuant Author | GPL v3 */
#pragma once
private:
    std::string file_path;
    size_t file_size;
    void* mapped_data;

#ifdef _WIN32
    HANDLE hFile;
    HANDLE hMapping;
#else
    int fd;
#endif

public:
    MmapLoader(const std::string& path) : file_path(path), file_size(0), mapped_data(nullptr) {
#ifdef _WIN32
        hFile = INVALID_HANDLE_VALUE;
        hMapping = NULL;
#else
        fd = -1;
#endif
    }

    ~MmapLoader() {
        unmap();
    }

    // Disabilita copia per evitare unmapping accidentali
    MmapLoader(const MmapLoader&) = delete;
    MmapLoader& operator=(const MmapLoader&) = delete;

    bool map() {
#ifdef _WIN32
        // Implementazione Windows
        hFile = CreateFileA(file_path.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (hFile == INVALID_HANDLE_VALUE) {
            std::cerr << "Errore: Impossibile aprire il file " << file_path << std::endl;
            return false;
        }

        LARGE_INTEGER size;
        if (!GetFileSizeEx(hFile, &size)) {
            std::cerr << "Errore: Impossibile ottenere la dimensione del file." << std::endl;
            CloseHandle(hFile);
            return false;
        }
        file_size = size.QuadPart;

        hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
        if (hMapping == NULL) {
            std::cerr << "Errore: Impossibile creare il mapping del file." << std::endl;
            CloseHandle(hFile);
            return false;
        }

        mapped_data = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
        if (mapped_data == NULL) {
            std::cerr << "Errore: Impossibile mappare la vista del file." << std::endl;
            CloseHandle(hMapping);
            CloseHandle(hFile);
            return false;
        }

#else
        // Implementazione POSIX (Linux/macOS)
        fd = open(file_path.c_str(), O_RDONLY);
        if (fd == -1) {
            std::cerr << "Errore: Impossibile aprire il file " << file_path << std::endl;
            return false;
        }

        struct stat sb;
        if (fstat(fd, &sb) == -1) {
            std::cerr << "Errore: Impossibile ottenere le informazioni sul file." << std::endl;
            close(fd);
            return false;
        }
        file_size = sb.st_size;

        // MAP_SHARED è essenziale per permettere all'OS di gestire i page faults e liberare RAM
        mapped_data = mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0);
        if (mapped_data == MAP_FAILED) {
            std::cerr << "Errore: Impossibile mappare il file in memoria." << std::endl;
            close(fd);
            return false;
        }

        // Hint al Kernel (Opzionale ma utile)
        // Diciamo all'OS che accederemo al file in modo sequenziale all'inizio (es. per leggere l'header)
        // e poi in modo casuale durante l'inferenza
#if defined(MADV_RANDOM)
        madvise(mapped_data, file_size, MADV_RANDOM);
#endif

#endif
        std::cout << "File mappato con successo. Dimensione: " << file_size / (1024 * 1024) << " MB." << std::endl;
        return true;
    }

    void unmap() {
        if (mapped_data != nullptr) {
#ifdef _WIN32
            UnmapViewOfFile(mapped_data);
            CloseHandle(hMapping);
            CloseHandle(hFile);
#else
            munmap(mapped_data, file_size);
            close(fd);
#endif
            mapped_data = nullptr;
            file_size = 0;
            std::cout << "Mapping rilasciato." << std::endl;
        }
    }

    // Ottieni il puntatore raw ai dati
    const void* data() const { return mapped_data; }
    
    // Ottieni la dimensione del file
    size_t size() const { return file_size; }
};

