#ifndef UTILITY_H
#define UTILITY_H

// cub
#include <cstdio>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>

#define CUDA_WARN(XXX)                                                         \
  do {                                                                         \
    cudaError_t err = XXX;                                                     \
    if (err != cudaSuccess) {                                                  \
      std::cout << "CUDA Warning: " << cudaGetErrorString(err) << " in file "  \
                << __FILE__ << ", at line " << __LINE__ << std::endl;          \
      CubDebugExit(err);                                                       \
    }                                                                          \
  } while (0)

#define CUDA_CHECK()                                                           \
  do {                                                                         \
    cudaError_t err = cudaGetLastError();                                      \
    if (err != cudaSuccess) {                                                  \
      std::cout << "CUDA Error: " << cudaGetErrorString(err) << " in file "    \
                << __FILE__ << ", at line " << __LINE__ << std::endl;          \
      CubDebugExit(err);                                                       \
    }                                                                          \
  } while (0)

#define DEBUG(XXX)
#ifndef REL do {XXX; } while (0)
#endif

template <typename T>
inline void PrefixSum(T **d_in, T **d_out, unsigned size) {
  // Copy hit offsets to device
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;

  // Set inital offsets to 0
  // CUDA_WARN(cudaMemset(d_out, 0, size * sizeof(T)));

  // Determine temporary device storage requirements
  CUDA_WARN(cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                          *d_in, &((*d_out)[1]), size));
  cudaDeviceSynchronize();

  // Allocate temporary storage
  CUDA_WARN(cudaMalloc(&d_temp_storage, temp_storage_bytes));

  // Run inclusive prefix sum
  CUDA_WARN(cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                          *d_in, &((*d_out)[1]), size));
  cudaDeviceSynchronize();

  // Free temporary storage
  // MemoryScheduler::free(d_in);
  CUDA_WARN(cudaFree(d_temp_storage));
}

template <typename K, typename V>
inline void Sort(K **d_keys_in, V **d_values_in, K **d_keys_out,
                 V **d_values_out, unsigned size) {
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;

  CUDA_WARN(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                            *d_keys_in, *d_keys_out,
                                            *d_values_in, *d_values_out, size));
  cudaDeviceSynchronize();

  // Allocate temporary storage
  CUDA_WARN(cudaMalloc(&d_temp_storage, temp_storage_bytes));

  CUDA_WARN(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                            *d_keys_in, *d_keys_out,
                                            *d_values_in, *d_values_out, size));
  cudaDeviceSynchronize();
}

template <typename F, typename V>
inline void ReduceFlagged(F **d_flags_in, V **d_values_in, V **d_values_out,
                          unsigned size) {
  unsigned *d_num_selected_out;
  cudaMalloc(&d_num_selected_out, sizeof(unsigned));

  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  CUDA_WARN(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes,
                                       *d_values_in, *d_flags_in, *d_values_out,
                                       d_num_selected_out, size));

  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);

  // Run selection
  CUDA_WARN(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes,
                                       *d_values_in, *d_flags_in, *d_values_out,
                                       d_num_selected_out, size));

  cudaFree(d_num_selected_out);
  cudaFree(d_temp_storage);
}

// Module coordinates
__device__ __host__ float Phi(const float &x, const float &y);

__device__ __host__ float R(const float &x, const float &y);

__device__ __host__ float Eta(const float &x, const float &y, const float &z);

#endif