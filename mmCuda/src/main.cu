#include "../include/triplet_finder.cuh"
#include "../include/datatypes.cuh"


int main(int argc, char *argv[]) {

    std::string mm_path{"/storage/agrp/nilotpal/tracking/transformed_data/module_map/df_MMTriplet_3hits_ptCut1GeV_woutSec_woutOC_90kevents_woutElectron.csv"};
    ModuleMap mm = ModuleMap(mm_path);
    mm.print_summary();
    // mm.cuda();

    // unsigned int n_triplets = mm.num_triplets;
    unsigned int n_doublets = mm.len_doublets();


    std::string event_path{"/storage/agrp/nilotpal/tracking/transformed_data/events/event000000001-truth.csv"};
    EventData event = EventData(event_path, 18960);
    event.print_summary();
    
    const unsigned n_max_doublets = 1000;

    // Transfer module pairs onto the device
    // Only needs to be done once
    unsigned* h_mod_pairs = mm.doublets(); // Get from module map
    const unsigned n_mod_pairs = mm.len_doublets(); // Fix this to the correct number from module map
    unsigned* d_mod_pairs;
    const unsigned d_size = n_mod_pairs*sizeof(unsigned);
    cudaMalloc((unsigned**)&d_mod_pairs, d_size); // Assign the memory on the device
    cudaMemcpy(d_mod_pairs, h_mod_pairs, d_size, cudaMemcpyHostToDevice); // Copy to device
    
    // Transfer module pairs offsets onto device
    // Only needs to be done once
    unsigned* h_mod_pairs_offsets = mm.offsets(); // Get from module map
    const unsigned n_mod = 18359; // Counted this from file
    unsigned* d_mod_pairs_offsets;
    const unsigned d_size_offsets = n_mod*sizeof(unsigned);
    cudaMalloc((unsigned**)&d_mod_pairs_offsets, d_size_offsets); // Assign the memory on the device
    cudaMemcpy(d_mod_pairs_offsets, h_mod_pairs_offsets, d_size_offsets, cudaMemcpyHostToDevice);
    
    // If needed can perform a loop over events with streams
    // Need to figure our most efficient way to do asynchronous memory transfers
    // const unsigned n_streams = 1; // Maximum number that can run concurrently
    // Create the streams
    // cudaStream_t streams[n_stream];
    // for (auto i : n_streams) {
    //     cudaStreamCreate(&streams[i]);
    // }

    // Transfer atomics onto device
    // Only needs to be done once
    unsigned h_atomics_arr[n_mod_pairs]; // Initialise to zero
    unsigned* h_atomics = h_atomics_arr;
    unsigned* d_atomics;
    cudaMalloc((unsigned**)&d_atomics, d_size);
    cudaMemcpy(d_atomics, h_atomics, d_size, cudaMemcpyHostToDevice);

    // Transfer triplets onto device
    unsigned* h_mod_triplets = mm.triplets();
    unsigned* d_mod_triplets;
    unsigned len_mod_triplets = mm.len_triplets();
    unsigned d_triplets_size = len_mod_triplets*sizeof(unsigned);
    cudaMalloc((unsigned**)&d_mod_triplets, d_triplets_size);
    cudaMemcpy(d_mod_triplets, h_mod_triplets, d_triplets_size, cudaMemcpyHostToDevice);

    // Assign device memory for containers of hit indices, this stays on the device
    // and no memory transfers are needed.
    unsigned* d_hits_a_idx;
    unsigned* d_hits_b_idx;
    cudaMalloc((unsigned**)&d_hits_a_idx, d_size * n_max_doublets);
    cudaMalloc((unsigned**)&d_hits_b_idx, d_size * n_max_doublets);

    // Transfer hits onto device
    // Needs to be done for every event
    // If loop over events can be clever with assigning max memory needed
    Hit* h_hits = event.hits;
    Hit* d_hits;
    const unsigned n_hits = event.len(); // Get correct number
    const unsigned d_size_hits = n_hits * sizeof(Hit); // Fix for correct hit type
    cudaMalloc((Hit**)&d_hits, d_size_hits); // Fix for correct hit type
    cudaMemcpy(d_hits, h_hits, d_size_hits, cudaMemcpyHostToDevice);

    // Transfer hit offsets onto device
    unsigned* h_hits_offsets = event.offsets();
    unsigned* d_hits_offsets;
    const unsigned d_size_hits_offsets = n_mod * sizeof(unsigned);
    cudaMalloc((unsigned**)&d_hits_offsets, d_size_hits_offsets);
    cudaMemcpy(d_hits_offsets, h_hits_offsets, d_size_hits_offsets, cudaMemcpyHostToDevice);

    dim3 grid_dim(65535,65535,1);
    dim3 block_dim(512,512,1);
    // Call doublet finding kernal
    doublet_finding<<<grid_dim, block_dim>>>(
        n_mod,
        d_mod_pairs,
        d_mod_pairs_offsets,
        d_hits,
        d_hits_offsets,
        d_hits_a_idx,
        d_hits_b_idx,
        d_atomics,
        n_max_doublets
    );
    /*
    // Transfer doublets atomics back onto host
    cudaMemcpy(h_atomics, d_atomics, d_size, cudaMemcpyDeviceToHost);

    // Running cumulative sum of number of pairs that pass
    unsigned* h_atomics_cumsum = PrefixSum(h_atomics, d_size);
    // Transer summed atomics back onto device
    cudaMemcpy(d_atomics, h_atomics_cumsum, d_size, cudaMemcpyHostToDevice);

    bool* d_doublet_acceptance;
    bool* d_hit_acceptance;
    cudaMalloc((bool**)&d_doublet_acceptance, n_mod_pairs * sizeof(bool) * n_max_doublets);
    cudaMalloc((bool**)&d_hit_acceptance, n_hits * sizeof(bool));


    unsigned n_mod_triplets = mm.num_triplets();

    // Call triplet finding kernal
    triplet_finding<<<grid_dim, block_dim>>>(
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
    */
    // Free up device memory
    cudaFree(d_mod_pairs);
    cudaFree(d_mod_pairs_offsets);

}