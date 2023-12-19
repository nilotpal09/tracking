#ifndef _CUDA_KERNELS_
#define _CUDA_KERNELS_

void launch_kernel_edge_count(
    unsigned int *flatten_doublets, unsigned int *input_hits_chunk_idx,
    unsigned int *edge_count, // could use int here, but same vector will be
                              // used to store cumsum
    unsigned int n, int grid_size, int block_size);

void launch_kernel_make_edges(unsigned int *flatten_triplets,
                              unsigned int *flatten_t2d_links,
                              unsigned int *edge_count, float *input_hits,
                              unsigned int *input_hits_chunk_idx,
                              unsigned int *srcs, unsigned int *dsts,
                              unsigned int n, int grid_size, int block_size);

void launch_kernel_triplet_iter_count(unsigned int *flatten_triplets,
                                      unsigned int *input_hits_chunk_idx,
                                      unsigned int *iter_count, unsigned int n,
                                      int grid_size, int block_size);

void launch_kernel_make_edges(
    unsigned int *flatten_triplets, unsigned int *flatten_t2d_links,
    unsigned int *edge_count, float *input_hits,
    unsigned int *input_hits_chunk_idx, unsigned int *srcs, unsigned int *dsts,
    unsigned int m_triplet_idx, unsigned long int offset, int grid_size,
    int block_size, unsigned int shared_mem_size, cudaStream_t stream);

#endif