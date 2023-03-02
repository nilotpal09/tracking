#ifndef MEMORYSCHEDULER_H
#define MEMORYSCHEDULER_H

#include <cub/util_allocator.cuh>

#include "../include/utility.cuh"



class MemoryScheduler {
    public:
        static cub::CachingDeviceAllocator _allocator;

    public:
        template <typename T>
        inline static void allocate(T** d_ptr, size_t size, cudaStream_t stream = 0) {
            CUDA_WARN(_allocator.DeviceAllocate((void**)d_ptr, size * sizeof(T), stream));
        }

        template <typename T>
        inline static void free(T** ptr) {
            CUDA_WARN(_allocator.DeviceFree(*ptr));
        }
};

#endif