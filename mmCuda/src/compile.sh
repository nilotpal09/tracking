nvcc -std=c++11 -o ~/projects/tracking/mmCuda/bin/cuda_kernels.o -c ~/projects/tracking/mmCuda/src/cuda_kernels.cu
nvcc -std=c++11 -o ~/projects/tracking/mmCuda/bin/main.o -c ~/projects/tracking/mmCuda/src/main.cu
nvcc -o ~/projects/tracking/mmCuda/bin/main ~/projects/tracking/mmCuda/bin/cuda_kernels.o ~/projects/tracking/mmCuda/bin/main.o
