#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "ternary_kernel.hpp"

// Tipi per i puntatori a funzione di OpenCL per il caricamento dinamico
typedef int cl_int;
typedef unsigned int cl_uint;
typedef unsigned long cl_ulong;
typedef void* cl_platform_id;
typedef void* cl_device_id;
typedef void* cl_context;
typedef void* cl_command_queue;
typedef void* cl_mem;
typedef void* cl_program;
typedef void* cl_kernel;

#define CL_SUCCESS 0
#define CL_DEVICE_TYPE_GPU (1 << 2)
#define CL_MEM_READ_ONLY (1 << 2)
#define CL_MEM_WRITE_ONLY (1 << 0)
#define CL_MEM_COPY_HOST_PTR (1 << 5)

class GpuTernaryKernel {
private:
    void* opencl_lib = nullptr;
    bool gpu_available = false;
    std::string active_gpu_name = "None";

    // Puntatori alle funzioni OpenCL
    cl_int (*clGetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*) = nullptr;
    cl_int (*clGetDeviceIDs)(cl_platform_id, cl_ulong, cl_uint, cl_device_id*, cl_uint*) = nullptr;
    cl_int (*clGetDeviceInfo)(cl_device_id, cl_uint, size_t, void*, size_t*) = nullptr;
    cl_context (*clCreateContext)(const void*, cl_uint, const cl_device_id*, void (*)(const char*, const void*, size_t, void*), void*, cl_int*) = nullptr;
    cl_command_queue (*clCreateCommandQueue)(cl_context, cl_device_id, cl_ulong, cl_int*) = nullptr;
    cl_program (*clCreateProgramWithSource)(cl_context, cl_uint, const char**, const size_t*, cl_int*) = nullptr;
    cl_int (*clBuildProgram)(cl_program, cl_uint, const cl_device_id*, const char*, void (*)(cl_program, void*), void*) = nullptr;
    cl_kernel (*clCreateKernel)(cl_program, const char*, cl_int*) = nullptr;
    cl_mem (*clCreateBuffer)(cl_context, cl_ulong, size_t, void*, cl_int*) = nullptr;
    cl_int (*clEnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_uint, size_t, size_t, const void*, cl_uint, const void*, void*) = nullptr;
    cl_int (*clSetKernelArg)(cl_kernel, cl_uint, size_t, const void*) = nullptr;
    cl_int (*clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, const void*, void*) = nullptr;
    cl_int (*clEnqueueReadBuffer)(cl_command_queue, cl_mem, cl_uint, size_t, size_t, void*, cl_uint, const void*, void*) = nullptr;
    cl_int (*clFinish)(cl_command_queue) = nullptr;
    cl_int (*clReleaseMemObject)(cl_mem) = nullptr;
    cl_int (*clReleaseKernel)(cl_kernel) = nullptr;
    cl_int (*clReleaseProgram)(cl_program) = nullptr;
    cl_int (*clReleaseCommandQueue)(cl_command_queue) = nullptr;
    cl_int (*clReleaseContext)(cl_context) = nullptr;

    // Risorse GPU attive
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel matmul_kernel = nullptr;

    bool resolve_symbols() {
#ifdef _WIN32
        auto get_sym = [&](const char* name) { return GetProcAddress((HMODULE)opencl_lib, name); };
#else
        auto get_sym = [&](const char* name) { return dlsym(opencl_lib, name); };
#endif
        clGetPlatformIDs = (cl_int(*)(cl_uint, cl_platform_id*, cl_uint*))get_sym("clGetPlatformIDs");
        clGetDeviceIDs = (cl_int(*)(cl_platform_id, cl_ulong, cl_uint, cl_device_id*, cl_uint*))get_sym("clGetDeviceIDs");
        clGetDeviceInfo = (cl_int(*)(cl_device_id, cl_uint, size_t, void*, size_t*))get_sym("clGetDeviceInfo");
        clCreateContext = (cl_context(*)(const void*, cl_uint, const cl_device_id*, void (*)(const char*, const void*, size_t, void*), void*, cl_int*))get_sym("clCreateContext");
        clCreateCommandQueue = (cl_command_queue(*)(cl_context, cl_device_id, cl_ulong, cl_int*))get_sym("clCreateCommandQueue");
        clCreateProgramWithSource = (cl_program(*)(cl_context, cl_uint, const char**, const size_t*, cl_int*))get_sym("clCreateProgramWithSource");
        clBuildProgram = (cl_int(*)(cl_program, cl_uint, const cl_device_id*, const char*, void (*)(cl_program, void*), void*))get_sym("clBuildProgram");
        clCreateKernel = (cl_kernel(*)(cl_program, const char*, cl_int*))get_sym("clCreateKernel");
        clCreateBuffer = (cl_mem(*)(cl_context, cl_ulong, size_t, void*, cl_int*))get_sym("clCreateBuffer");
        clEnqueueWriteBuffer = (cl_int(*)(cl_command_queue, cl_mem, cl_uint, size_t, size_t, const void*, cl_uint, const void*, void*))get_sym("clEnqueueWriteBuffer");
        clSetKernelArg = (cl_int(*)(cl_kernel, cl_uint, size_t, const void*))get_sym("clSetKernelArg");
        clEnqueueNDRangeKernel = (cl_int(*)(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, const void*, void*))get_sym("clEnqueueNDRangeKernel");
        clEnqueueReadBuffer = (cl_int(*)(cl_command_queue, cl_mem, cl_uint, size_t, size_t, void*, cl_uint, const void*, void*))get_sym("clEnqueueReadBuffer");
        clFinish = (cl_int(*)(cl_command_queue))get_sym("clFinish");
        clReleaseMemObject = (cl_int(*)(cl_mem))get_sym("clReleaseMemObject");
        clReleaseKernel = (cl_int(*)(cl_kernel))get_sym("clReleaseKernel");
        clReleaseProgram = (cl_int(*)(cl_program))get_sym("clReleaseProgram");
        clReleaseCommandQueue = (cl_int(*)(cl_command_queue))get_sym("clReleaseCommandQueue");
        clReleaseContext = (cl_int(*)(cl_context))get_sym("clReleaseContext");

        return clGetPlatformIDs && clGetDeviceIDs && clCreateContext && clCreateCommandQueue &&
               clCreateProgramWithSource && clBuildProgram && clCreateKernel && clCreateBuffer &&
               clSetKernelArg && clEnqueueNDRangeKernel && clEnqueueReadBuffer && clFinish;
    }

public:
    GpuTernaryKernel() {
        // Carichiamo la libreria OpenCL dinamicamente
#ifdef _WIN32
        opencl_lib = LoadLibraryA("OpenCL.dll");
#else
        opencl_lib = dlopen("libOpenCL.so", RTLD_NOW);
        if (!opencl_lib) {
            opencl_lib = dlopen("libOpenCL.so.1", RTLD_NOW);
        }
#endif

        if (!opencl_lib) {
            std::cout << "[SYSTEM] OpenCL driver library not found. Seamless fallback to AVX2/FMA CPU active.\n";
            return;
        }

        if (!resolve_symbols()) {
            std::cout << "[SYSTEM] OpenCL symbols could not be fully resolved. Falling back to AVX2 CPU.\n";
            return;
        }

        // Inizializziamo il primo dispositivo GPU trovato
        cl_platform_id platform = nullptr;
        cl_uint num_platforms = 0;
        if (clGetPlatformIDs(1, &platform, &num_platforms) != CL_SUCCESS || num_platforms == 0) {
            return;
        }

        cl_uint num_devices = 0;
        if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &num_devices) != CL_SUCCESS || num_devices == 0) {
            return;
        }

        char device_name[128];
        size_t size_ret;
        // 0x102B = CL_DEVICE_NAME
        clGetDeviceInfo(device, 0x102B, sizeof(device_name), device_name, &size_ret);
        active_gpu_name = std::string(device_name);

        cl_int err;
        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        if (err != CL_SUCCESS) return;

        queue = clCreateCommandQueue(context, device, 0, &err);
        if (err != CL_SUCCESS) return;

        // Kernel GPU per la moltiplicazione matriciale ternaria ottimizzata
        // Pesi compressi caricati come int8/char. Il kernel esegue l'inferenza 1.58-bit in parallelo
        const char* kernel_source = 
        "__kernel void ternary_gemv(\n"
        "    __global const float* input,\n"
        "    __global const char* weights,\n"
        "    __global float* output,\n"
        "    const unsigned int size)\n"
        "{\n"
        "    int gid = get_global_id(0); // Rappresenta la riga (output neuron)\n"
        "    float sum = 0.0f;\n"
        "    int row_offset = gid * size;\n"
        "    for (unsigned int i = 0; i < size; i++) {\n"
        "        sum += input[i] * (float)weights[row_offset + i];\n"
        "    }\n"
        "    output[gid] = sum;\n"
        "}\n";

        program = clCreateProgramWithSource(context, 1, &kernel_source, nullptr, &err);
        if (err != CL_SUCCESS) return;

        err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS) return;

        matmul_kernel = clCreateKernel(program, "ternary_gemv", &err);
        if (err == CL_SUCCESS) {
            gpu_available = true;
            std::cout << "[SYSTEM] GPU acceleration initialized successfully!\n";
            std::cout << "[GPU] Active Device: " << active_gpu_name << "\n";
        }
    }

    ~GpuTernaryKernel() {
        if (matmul_kernel) clReleaseKernel(matmul_kernel);
        if (program) clReleaseProgram(program);
        if (queue) clReleaseCommandQueue(queue);
        if (context) clReleaseContext(context);
#ifdef _WIN32
        if (opencl_lib) FreeLibrary((HMODULE)opencl_lib);
#else
        if (opencl_lib) dlclose(opencl_lib);
#endif
    }

    bool is_gpu_accelerated() const { return gpu_available; }
    std::string get_gpu_name() const { return active_gpu_name; }

    // Esegue il calcolo di un intero layer: Moltiplicazione Matrice-Vettore
    // input: vettore di input (size)
    // weights: matrice di pesi ternari piatti (rows * cols)
    // output: vettore dei risultati (rows)
    void matmul(const float* input, const int8_t* weights, float* output, size_t rows, size_t cols) {
        if (!gpu_available) {
            // Se non c'è GPU, eseguiamo in parallelo CPU AVX2 su tutti i thread
#pragma omp parallel for
            for (size_t r = 0; r < rows; ++r) {
                output[r] = TernaryKernel::compute(input, weights + (r * cols), cols);
            }
            return;
        }

        cl_int err;
        // Creiamo i buffer di VRAM
        cl_mem input_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * cols, (void*)input, &err);
        cl_mem weights_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int8_t) * rows * cols, (void*)weights, &err);
        cl_mem output_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * rows, nullptr, &err);

        unsigned int cols_uint = static_cast<unsigned int>(cols);
        clSetKernelArg(matmul_kernel, 0, sizeof(cl_mem), &input_buf);
        clSetKernelArg(matmul_kernel, 1, sizeof(cl_mem), &weights_buf);
        clSetKernelArg(matmul_kernel, 2, sizeof(cl_mem), &output_buf);
        clSetKernelArg(matmul_kernel, 3, sizeof(unsigned int), &cols_uint);

        size_t global_work_size = rows;
        err = clEnqueueNDRangeKernel(queue, matmul_kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);
        
        clEnqueueReadBuffer(queue, output_buf, 1, 0, sizeof(float) * rows, output, 0, nullptr, nullptr);
        clFinish(queue);

        clReleaseMemObject(input_buf);
        clReleaseMemObject(weights_buf);
        clReleaseMemObject(output_buf);
    }
};
