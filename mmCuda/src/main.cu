#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <vector>
#include <map>
#include <set>
#include <tuple>
#include <random>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "../include/cuda_kernels.h"

#define N_MAX_HITS 10



int main(int argc, char *argv[]){

    // argc, argv
    //----------------------

    unsigned long int N_TRIPLETS, N_MODULES;
    int grid_size, block_size;

    if (argc != 5){
        std::cout << "Error! Requires 4 args - N_TRIPLETS, N_MODULES, grid_size, block_size" << std::endl;
        return 0;
    } else {
        N_TRIPLETS = atoi(argv[1]);
        N_MODULES  = atoi(argv[2]);
        grid_size  = atoi(argv[3]);
        block_size = atoi(argv[4]);

        std::cout << "Inputs -\n" << "N_TRIPLETS: " << N_TRIPLETS << ", N_MODULES: " << N_MODULES 
            << ", grid_size: " << grid_size << ", block_size: " << block_size << std::endl << std::endl;
    }



    // Seeding
    //----------------------

    // random seeding (module triple indices)
    std::random_device rd1;
    std::mt19937 rng1(rd1());
    std::uniform_int_distribution<unsigned long int> uni1(0,N_MODULES-1);

    // random seeding (num hits)
    std::random_device rd2;
    std::mt19937 rng2(rd2());
    std::uniform_int_distribution<int> uni2(0,N_MAX_HITS);

    // random seeding (hit values)
    std::random_device rd3;
    std::mt19937 rng3(rd3());
    std::uniform_real_distribution<float> uni3(-5,5);



    // Preprocessed data
    //----------------------

    std::set<std::tuple<unsigned long int, unsigned long int, unsigned long int>> triplets;
    std::vector<unsigned long int> flatten_triplets;

    std::map<std::tuple<unsigned long int, unsigned long int>, unsigned long int> doublets;
    std::vector<unsigned long int> flatten_doublets;

    std::vector<unsigned long int> flatten_t2d_links;
    unsigned long int t2d_link_count = 0;

    for (unsigned long int i=0; i<N_TRIPLETS; i++){

        unsigned long int a = uni1(rng1);
        unsigned long int b = uni1(rng1);
        unsigned long int c = uni1(rng1);

        auto tmp_triplet  = std::make_tuple(a, b, c);
        auto tmp_doublet1 = std::make_tuple(a, b);
        auto tmp_doublet2 = std::make_tuple(b, c);

        // unique triplet
        if (triplets.find(tmp_triplet) == triplets.end()){
            triplets.insert(tmp_triplet);
            flatten_triplets.push_back(a);
            flatten_triplets.push_back(b);
            flatten_triplets.push_back(c);

            // the doublet not exist already
            if (doublets.find(tmp_doublet1) == doublets.end()){
                doublets.insert({tmp_doublet1, t2d_link_count});
                flatten_doublets.push_back(a);
                flatten_doublets.push_back(b);
                flatten_t2d_links.push_back(t2d_link_count);
                t2d_link_count++;
            } else {
                flatten_t2d_links.push_back(doublets[tmp_doublet1]);
            }

            // the doublet not exist already
            if (doublets.find(tmp_doublet2) == doublets.end()){
                doublets.insert({tmp_doublet2, t2d_link_count});
                flatten_doublets.push_back(b);
                flatten_doublets.push_back(c);
                flatten_t2d_links.push_back(t2d_link_count);
                t2d_link_count++;
            } else {
                flatten_t2d_links.push_back(doublets[tmp_doublet2]);
            }
        }
    }

    unsigned long int n_triplets = flatten_triplets.size() / 3;
    unsigned long int n_doublets = flatten_doublets.size() / 2;
    std::cout << "n_triplets: " << n_triplets << ", n_doublets: " << n_doublets << std::endl;



    // Event data (input hits)
    //----------------------

    std::vector<std::vector<float> > input_hits;
    for (unsigned long int i=0; i<N_MODULES; i++){

        int num_hits = uni2(rng2);
        std::vector<float> hit_vector;

        for (int j=0; j<3*num_hits; j++){
            float random_float = uni3(rng3);
            hit_vector.push_back(random_float);
        }

        input_hits.push_back(hit_vector);
    }



    // flattening input_hits
    //----------------------

    std::vector<float> input_hits_flatten;
    std::vector<unsigned long int> input_hits_chunk_idx;
    unsigned long int hit_count = 0;

    for (unsigned long int i=0; i<input_hits.size(); i++){

        input_hits_chunk_idx.push_back(hit_count);
        hit_count += input_hits.at(i).size() / 3;

        for (unsigned int j=0; j<input_hits.at(i).size(); j++){
            input_hits_flatten.push_back(input_hits.at(i).at(j));
        }
    }

    input_hits_chunk_idx.push_back(hit_count);



    // stdout
    //----------------------

    printf("flatten sizes -\n");
    printf("triplets: %d, input_hits: %d, chunk_idx: %d\n\n", 
        flatten_triplets.size(), input_hits_flatten.size(), input_hits_chunk_idx.size());



    // edge count (respecting only the doublets)
    //----------------------
    
    // the last guy will be random/0 and will be ignored during exclusive_scan
    std::vector<unsigned long int> edge_count(n_doublets + 1);

    thrust::device_vector<unsigned long int> td_flatten_doublets     = flatten_doublets;
    thrust::device_vector<unsigned long int> td_flatten_triplets     = flatten_triplets;
    thrust::device_vector<unsigned long int> td_flatten_t2d_links    = flatten_t2d_links;
    thrust::device_vector<float> td_input_hits                       = input_hits_flatten;
    thrust::device_vector<unsigned long int> td_input_hits_chunk_idx = input_hits_chunk_idx;
    thrust::device_vector<unsigned long int> td_edge_count           = edge_count;

    unsigned long int* d_flatten_doublets     = thrust::raw_pointer_cast(td_flatten_doublets.data());
    unsigned long int* d_flatten_triplets     = thrust::raw_pointer_cast(td_flatten_triplets.data());
    unsigned long int* d_flatten_t2d_links    = thrust::raw_pointer_cast(td_flatten_t2d_links.data());
    float* d_input_hits                       = thrust::raw_pointer_cast(td_input_hits.data());
    unsigned long int* d_input_hits_chunk_idx = thrust::raw_pointer_cast(td_input_hits_chunk_idx.data());
    unsigned long int* d_edge_count           = thrust::raw_pointer_cast(td_edge_count.data());

    launch_kernel_edge_count(
        d_flatten_doublets, d_input_hits_chunk_idx, d_edge_count, n_doublets,
        grid_size, block_size);

    // d_edge_count is a cumusum
    thrust::exclusive_scan(thrust::device, d_edge_count, d_edge_count+n_doublets+1, d_edge_count, 0);
    unsigned long int n_edges = td_edge_count[n_doublets];
    std::cout << "num edges (respecting the doublet-map only): " << n_edges << std::endl;



    // filling the edges (src & dst)
    //----------------------

    thrust::device_vector<unsigned long int> td_src(n_edges);
    thrust::device_vector<unsigned long int> td_dst(n_edges);

    unsigned long int* d_src = thrust::raw_pointer_cast(td_src.data());
    unsigned long int* d_dst = thrust::raw_pointer_cast(td_dst.data());

    launch_kernel_make_edges(
        d_flatten_triplets, d_flatten_t2d_links, d_edge_count, d_input_hits, d_input_hits_chunk_idx, d_src, d_dst, n_triplets,
        grid_size, block_size);

    printf("PASSED\n");
}


