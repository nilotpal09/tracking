#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define MAX_ERR 1


__global__ void kernel(int *x, int *y, int *out, int n){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int total_threds = blockDim.x * gridDim.x;

    for (int i=tid; i<n; i+=total_threds){
        out[i] = x[i] + y[i];         
    }
}


int main(int argc, char *argv[]){


    int N, grid_size, block_size;
    if (argc != 4){
        std::cout << "Error! Requires three args - N, grid_size, block_size" << std::endl;
        return 0;
    } else {
        N = atoi(argv[1]);
        grid_size = atoi(argv[2]);
        block_size = atoi(argv[3]);
        std::cout << "N: " << N << ", grid_size: " << grid_size << ", block_size: " << block_size << std::endl;
    }

    std::vector<int> x_h;
    std::vector<int> y_h;
    std::vector<int> out_h;

    for (int i=0; i<N; i++){
        x_h.push_back(rand() % 101);
        y_h.push_back(rand() % 101);
        out_h.push_back(13);
    }

    thrust::device_vector<int> x = x_h;
    thrust::device_vector<int> y = y_h;
    thrust::device_vector<int> out = out_h;


    int* d_x   = thrust::raw_pointer_cast(x.data());
    int* d_y   = thrust::raw_pointer_cast(y.data());
    int* d_out = thrust::raw_pointer_cast(out.data());

    kernel<<<grid_size, block_size>>>(d_x, d_y, d_out, N);

    thrust::host_vector<int> h_x   = x;
    thrust::host_vector<int> h_y   = y;
    thrust::host_vector<int> h_out = out;

    // Verification
    for(int i=0; i<N; i++){
        assert(fabs(h_out[i] - h_x[i] - h_y[i]) < MAX_ERR);
    }
    
    return 0;

}
