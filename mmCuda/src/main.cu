#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <vector>

// thrust
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "../include/utility.h"
#include "../include/cuda_kernels.h"



int main(int argc, char *argv[]){

    // argc, argv
    //----------------------

    unsigned int grid_size, block_size;

    if (argc != 3){
        std::cout << "Error! Requires 2 args - grid_size, block_size" << std::endl;
        return 0;
    } else {
        grid_size  = atoi(argv[1]);
        block_size = atoi(argv[2]);

        std::cout << "Inputs -\n" 
            << "grid_size: " << grid_size << ", block_size: " << block_size << std::endl << std::endl;
    }




    // parse the module map
    //-------------------------------

    std::string mm_path{"/storage/agrp/nilotpal/tracking/transformed_data/module_map/df_MMTriplet_3hits_ptCut1GeV_woutSec_woutOC_90kevents_woutElectron.csv"};
    ModuleMap mm = ModuleMap(mm_path);
    mm.print_summary();
    mm.cuda();

    unsigned int n_triplets = mm.num_triplets;
    unsigned int n_doublets = mm.num_doublets;




    // parse an event
    //-------------------------------

    std::string event_path{"/storage/agrp/nilotpal/tracking/transformed_data/events/event000000001-truth.csv"};
    EventData event = EventData(event_path, 18960);
    event.print_summary();




    // edge count (respecting only the doublets)
    //----------------------
    
    // the last guy will be random/0 and will be ignored during exclusive_scan
    std::vector<unsigned int> edge_count(n_doublets + 1);

    thrust::device_vector<float> td_input_hits                  = event.input_hits_flatten;
    thrust::device_vector<unsigned int> td_input_hits_chunk_idx = event.input_hits_chunk_idx;
    thrust::device_vector<unsigned int> td_edge_count      = edge_count;

    float* d_input_hits                  = thrust::raw_pointer_cast(td_input_hits.data());
    unsigned int* d_input_hits_chunk_idx = thrust::raw_pointer_cast(td_input_hits_chunk_idx.data());
    unsigned int* d_edge_count      = thrust::raw_pointer_cast(td_edge_count.data());

    launch_kernel_edge_count(
        mm.d_flatten_doublets, d_input_hits_chunk_idx, d_edge_count, n_doublets,
        grid_size, block_size);

    // d_edge_count is a cumusum [0, ..., total_doublet_hit]
    thrust::exclusive_scan(thrust::device, d_edge_count, d_edge_count+n_doublets+1, d_edge_count, 0);
    unsigned int n_edges = td_edge_count[n_doublets];
    std::cout << "num edges (respecting the doublet-map only): " << n_edges << std::endl;




    // counting iters in triplets
    //----------------------

    std::vector<unsigned int> iter_count(n_triplets);
    thrust::device_vector<unsigned int> td_iter_count = iter_count;
    unsigned int* d_iter_count = thrust::raw_pointer_cast(td_iter_count.data());

    launch_kernel_triplet_iter_count(
        mm.d_flatten_triplets, d_input_hits_chunk_idx, d_iter_count, n_triplets,
        grid_size, block_size);




    // filling the edges (src & dst)
    //----------------------

    thrust::device_vector<unsigned int> td_src(n_edges);
    thrust::device_vector<unsigned int> td_dst(n_edges);

    unsigned int* d_src = thrust::raw_pointer_cast(td_src.data());
    unsigned int* d_dst = thrust::raw_pointer_cast(td_dst.data());


    // without stream (inefficient)
    launch_kernel_make_edges(
        mm.d_flatten_triplets, mm.d_flatten_t2d_links, d_edge_count, d_input_hits, d_input_hits_chunk_idx, d_src, d_dst, n_triplets,
        grid_size, block_size);



    // // with stream
    // unsigned int n_stream_max = 100000;

    // cudaStream_t stream[n_stream_max];
    // unsigned int shared_mem_size = 0;
    // unsigned int max_grid_size  = 1000;
    // unsigned int max_block_size = 1000;

    // unsigned long int offset = 0; // needed for defining the thread_id (ONLY)
    // int stream_i = 0;


    // for (int i=0; i<n_stream_max; i++){

    //     if (td_iter_count[i] == 0){
    //         continue;
    //     }

    //     cudaStreamCreate(&stream[i]);

    //     grid_size  = std::min(max_grid_size, (unsigned int)td_iter_count[i]);
    //     if (td_iter_count[i] % grid_size){
    //         block_size = std::min((unsigned int)td_iter_count[i]/grid_size, max_block_size);
    //     } else {
    //         block_size = std::min((unsigned int)td_iter_count[i]/grid_size + 1, max_block_size);            
    //     }

    //     launch_kernel_make_edges(
    //         mm.d_flatten_triplets, mm.d_flatten_t2d_links, d_edge_count, d_input_hits, d_input_hits_chunk_idx, d_src, d_dst, i, offset,
    //         grid_size, block_size, shared_mem_size, stream[stream_i]);
        
    //     offset += (unsigned long int)td_iter_count[i];
    //     stream_i++;

    //     // break;
    // }

    // std::cout << "# streams:" << stream_i << std::endl;

    // for (int i=0; i<stream_i; i++){
    //     cudaStreamDestroy(stream[i]);
    // }

    printf("PASSED\n");
}









    // // write them to a file
    // std::ofstream myfile ("/srv01/agrp/nilotpal/projects/tracking/mmCuda/run/triplet_iter_count_event1.txt");
    // if (myfile.is_open()) {
    //     for (unsigned int i=0; i<n_triplets; i++){
    //         myfile << td_iter_count[i] << std::endl;
    //     }        
    //     myfile.close();
    // }


