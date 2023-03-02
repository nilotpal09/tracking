#include "../include/triplet_finder.cuh"
#include "../include/datatypes.cuh"

#include <cuda_profiler_api.h>

#define N_MOD_PAIRS 509461

void checkCudaError(unsigned line) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(err) << " at line " << line << std::endl;
    } else {
        std::cout << "No CUDA error at line " << line << std::endl;
    }
}

int main(int argc, char *argv[]) {

    std::string mm_path{"/srv01/agrp/shieldse/storage/ML/trackingData/transformed_data/module_map/df_MMTriplet_3hits_ptCut1GeV_woutSec_woutOC_90kevents_woutElectron.csv"};
    std::string mm_pairs_path{"/srv01/agrp/shieldse/storage/ML/trackingData/transformed_data/module_map/df_MMTriplet_3hits_ptCut1GeV_woutSec_woutOC_90kevents_woutElectron_pairs.csv"};
    ModuleMap mm = ModuleMap(mm_path, mm_pairs_path);
    mm.print_summary();
    
    CUDA_CHECK();

    std::string event_path{"/storage/agrp/nilotpal/tracking/transformed_data/events/event000000001-truth.csv"};
    EventData event = EventData(event_path, mm, 18960);
    event.print_summary();

    CUDA_CHECK();
    
    dim3 grid_dim(65535);
    dim3 block_dim(32,32);
    // Call doublet finding kernal
    doublet_finding<<<grid_dim, block_dim>>>(
        ModuleMap::num_modules(),
        ModuleMap::num_doublets(),
        *mm.d_modules(),
        *mm.d_pairs(),
        *event.d_hits(),
        *event.d_hit_offsets(),
        *event.d_hit_pairs(),
        *event.d_hit_pairs_offsets()
    );
    
    cudaDeviceSynchronize();

    checkCudaError(__LINE__);
    cudaProfilerStop();
    
    /**
    // Transfer doublets atomics back onto host
    cudaMemcpy(h_atomics, d_atomics, d_size, cudaMemcpyDeviceToHost);
    
    // Running cumulative sum of number of pairs that pass
    unsigned h_atomics_cumsum_arr[n_mod_pairs];
    unsigned* h_atomics_cumsum = h_atomics_cumsum_arr;
    PrefixSum(h_atomics, n_mod_pairs, h_atomics_cumsum);
    
    // Transer summed atomics back onto device
    cudaMemcpy(d_atomics, h_atomics_cumsum, d_size, cudaMemcpyHostToDevice);

    bool* d_doublet_acceptance;
    bool* d_hit_acceptance;
    cudaMalloc((bool**)&d_doublet_acceptance, n_mod_pairs * sizeof(bool) * n_max_doublets);
    cudaMalloc((bool**)&d_hit_acceptance, n_hits * sizeof(bool));


    unsigned n_mod_triplets = mm.num_triplets();
    
    // Call triplet finding kernal
    triplet_finding<<<256, block_dim>>>(
        n_mod_triplets,
        d_mod_triplets,
        d_mod_pairs,
        d_mod_pairs_offsets,
        d_hits,
        d_hits_offsets,
        d_hits_a_idx,
        d_hits_b_idx,
        d_atomics,
        d_doublet_acceptance,
        d_hit_acceptance,
        n_max_doublets
    );
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();  // add
    if (err != cudaSuccess) std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl; // add
    cudaProfilerStop();
    return 0;
    */
    // Free up device memory
    // cudaFree(d_mod_pairs);
    // cudaFree(d_mod_pairs_offsets);

    cudaDeviceSynchronize();
}