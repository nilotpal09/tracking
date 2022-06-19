# Intro

Demonstration of speedup from CUDA paralellizing graph creation for GNN tracking.

# Setup

Compile on CUDA node:
```
nvcc -std=c++11 main.cu -o main_cu
```

# Run
Syntax: 
```
./main_cu <N_TRIPLETS> <N_MODULES> <grid_size> <block_size>
```
Use `nvprof` to profile performance.
Examples:
- Serial: `nvprof ./main_cu 100000 10000 1 1`
- Parellel: `nvprof ./main_cu 100000 10000 1000 100`
