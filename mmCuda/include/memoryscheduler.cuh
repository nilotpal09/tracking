#ifndef MEMORYSCHEDULER_H
#define MEMORYSCHEDULER_H

#include <cub/util_allocator.cuh>
#include <iostream>

#include "../include/utility.cuh"

/*
  * Memory Scheduler

  * This class is used to allocate and free device memory.

  * Warning: This reassigns allocated memory so when allocating memory
  *          for a variable, the values may be preset to the values of
  *          the previous variable.
*/
class MemoryScheduler {
public:
  static unsigned _total_memory;
  static cub::CachingDeviceAllocator _allocator;

public:
  template <typename T>
  inline static void allocate(T **d_ptr, size_t size, cudaStream_t stream = 0) {
    CUDA_WARN(
        _allocator.DeviceAllocate((void **)d_ptr, size * sizeof(T), stream));
    _total_memory += size * sizeof(T);
  }

  template <typename T> inline static void free(T **ptr) {
    CUDA_WARN(_allocator.DeviceFree(*ptr));
  }

  template <typename T>
  inline static void memset(T **ptr, size_t size, T value,
                            cudaStream_t stream = 0) {
    CUDA_WARN(cudaMemset(*ptr, value, size * sizeof(T)));
  }

  inline static void total_memory() {
    std::cout << "Total device memory allocated: " << _total_memory / 1e9
              << " GB" << std::endl;
  }
};

#endif