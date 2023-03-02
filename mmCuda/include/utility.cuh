#ifndef UTILITY_H
#define UTILITY_H

// cub
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cstdio>

#define CUDA_WARN(XXX) \
    do { cudaError_t err = XXX; if (err != cudaSuccess) { \
        std::cout << "CUDA Warning: " << cudaGetErrorString(err) << " in file " << \
        __FILE__ << ", at line " << __LINE__ << std::endl; CubDebugExit(err);}} while (0) 

#define CUDA_CHECK() \
    do { cudaError_t err = cudaGetLastError(); if (err != cudaSuccess) { \
        std::cout << "CUDA Error: " << cudaGetErrorString(err) << " in file " << \
        __FILE__ << ", at line " << __LINE__ << std::endl; CubDebugExit(err);}} while (0)

inline void PrefixSum(unsigned* arr, unsigned size, unsigned* prefixSum) {
    prefixSum[0] = 0;
    // Adding present element with previous element
    for (int i = 0; i < size; i++) {
        prefixSum[i + 1] = prefixSum[i] + arr[i];
    }
}

template <typename T>
T* DevicePrefixSum(T* d_in, unsigned size) {
    T* prefixSum;
    cudaMalloc((T**)&prefixSum, (size + 1)*sizeof(T));
    DeviceScan(d_in, &prefixSum[1], size);
    return prefixSum;
}

template <typename T>
T* DeviceScan(T* d_in, T* d_out, unsigned int len) {
    // Determine temporary device storage requirements
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, len);
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run inclusive prefix sum
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, len);
    // Free temporary storage
    cudaFree(d_temp_storage);
    return d_out;
}

// Need to template this
template <typename T>
void DeviceSort(unsigned** d_keys_in,
              T** d_values_in,
              unsigned** d_keys_out,
              T** d_values_out,
              unsigned len) {
    // Determine temporary device storage requirements
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                    *d_keys_in, *d_keys_out, *d_values_in, *d_values_out, len);

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Run sorting operation
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                    *d_keys_in, *d_keys_out, *d_values_in, *d_values_out, len);

    // Free memory
    cudaFree(d_keys_in);
    cudaFree(d_values_in);
    cudaFree(d_keys_out);
    cudaFree(d_temp_storage);
}

#endif