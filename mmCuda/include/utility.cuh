#ifndef UTILITY_H
#define UTILITY_H

// cub
#include <cub/device/device_radix_sort.cuh>

inline void PrefixSum(unsigned* arr, unsigned size, unsigned* prefixSum) {
    prefixSum[0] = arr[0];
    // Adding present element with previous element
    for (int i = 1; i < size; i++) {
        prefixSum[i] = prefixSum[i - 1] + arr[i];
    }
}

// Need to template this
template <typename T>
std::pair<unsigned*,T*> DeviceSort(unsigned* h_keys_in,
                                   T* h_values_in,
                                   unsigned int len) {
    // Output
    unsigned h_keys_out_array[len];
    unsigned* h_keys_out = h_keys_out_array;
    T h_values_out_array[len];
    T* h_values_out = h_values_out_array;
    // Create device pointers
    unsigned* d_keys_in;
    unsigned* d_keys_out;
    T* d_values_in;
    T* d_values_out;
    // Assign memory on device
    cudaMalloc((unsigned**)&d_keys_in, len*sizeof(unsigned));
    cudaMalloc((unsigned**)&d_keys_out, len*sizeof(unsigned));
    cudaMalloc((T**)&d_values_in, len*sizeof(T));
    cudaMalloc((T**)&d_values_out, len*sizeof(T));
    // Transfer to device
    cudaMemcpy(d_keys_in, h_keys_in, len*sizeof(unsigned), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_keys_out, h_keys_out, len*sizeof(unsigned), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values_in, h_values_in, len*sizeof(T), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_values_out, h_values_out, len*sizeof(T), cudaMemcpyHostToDevice);

    // Determine temporary device storage requirements
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                    d_keys_in, d_keys_out, d_values_in, d_values_out, len);

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Run sorting operation
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                    d_keys_in, d_keys_out, d_values_in, d_values_out, len);

    // Copy output back to host
    cudaMemcpy(h_keys_out, d_keys_out, len*sizeof(unsigned), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_values_out, d_values_out, len*sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(d_keys_in);
    cudaFree(d_keys_out);
    cudaFree(d_values_in);
    cudaFree(d_values_out);

    return std::pair<unsigned*, T*>(h_keys_out, h_values_out);
}


#endif