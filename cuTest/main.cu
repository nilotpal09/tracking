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

#define N_MAX_HITS 10



// kernel function (works on module triplets)
//----------------------

__global__ void kernel_make_edges(
	unsigned long int *flatten_triplets,
	unsigned long int *flatten_t2d_links,
	unsigned long int *edge_count,
	float *input_hits, 
	unsigned long int *input_hits_chunk_idx,
	unsigned long int *srcs,
	unsigned long int *dsts,
	unsigned long int n) {

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int total_threds = blockDim.x * gridDim.x;

    for (int i=tid; i<n; i+=total_threds){

        unsigned long int m1_idx = flatten_triplets[i*3];
        unsigned long int m2_idx = flatten_triplets[i*3 + 1];
        unsigned long int m3_idx = flatten_triplets[i*3 + 2];

        unsigned long int
    	m1_start_idx = input_hits_chunk_idx[m1_idx], 
		m1_end_idx   = input_hits_chunk_idx[m1_idx + 1],

    	m2_start_idx = input_hits_chunk_idx[m2_idx], 
		m2_end_idx   = input_hits_chunk_idx[m2_idx + 1],

    	m3_start_idx = input_hits_chunk_idx[m3_idx], 
		m3_end_idx   = input_hits_chunk_idx[m3_idx + 1];

		int n_hits_m1 = m1_end_idx - m1_start_idx;
		int n_hits_m2 = m2_end_idx - m2_start_idx;
		// int n_hits_m3 = m3_end_idx - m3_start_idx;

		// -1 coz there is a a zero in the beginning
		unsigned long int offset_d1 = edge_count[flatten_t2d_links[i*2] - 1];
		unsigned long int offset_d2 = edge_count[flatten_t2d_links[i*2+1] - 1];

		// loop over m2 hits
		for (unsigned long int j=m2_start_idx; j<m2_end_idx; j++){

			float x2 = input_hits[j*3];
			float y2 = input_hits[j*3+1];
			float z2 = input_hits[j*3+2];

			// no dynamic allocation allowed (!! need a better solution probably !!)
			unsigned long int pair_12[200], pair_23[200]; // holds the node (hit) indices
			int pair_12_count = 0, pair_23_count = 0;

			// loop over m1 hits
			for (unsigned long int k=m1_start_idx; k<m1_end_idx; k++){

				float x1 = input_hits[k*3];
				float y1 = input_hits[k*3+1];
				float z1 = input_hits[k*3+2];

				bool pass = (x1*x1 + y1*y1 + z1*z1) > (x2*x2 + y2*y2 + z2*z2);

				if (pass){
					pair_12[pair_12_count*2]     = k;
					pair_12[pair_12_count*2 + 1] = j;
					pair_12_count++;
				}
			}

			// loop over m3 hits
			for (unsigned long int k=m3_start_idx; k<m3_end_idx; k++){

				float x3 = input_hits[k*3];
				float y3 = input_hits[k*3+1];
				float z3 = input_hits[k*3+2];

				bool pass = (x3*x3 + y3*y3 + z3*z3) > (x2*x2 + y2*y2 + z2*z2);

				if (pass){
					pair_23[pair_23_count*2]     = j;
					pair_23[pair_23_count*2 + 1] = k;
					pair_23_count++;
				}
			}

			// loop over the pairs
			for (unsigned long int k=0; k<pair_12_count; k++){
				for (unsigned long int l=0; l<pair_23_count; l++){
					
					// some more selections here
					// need to store the hits in these 3 modules temporarily
					// or some more indexing...

					// edges in pair_12 (flipped, annoying)
					unsigned long int edge_ind_12 = (n_hits_m1 * pair_12[k*2] + pair_12[k*2+1]) + offset_d1;
					srcs[edge_ind_12] =  pair_12[k*2];
					dsts[edge_ind_12] =  pair_12[k*2+1];

					// edges in pair_23
					unsigned long int edge_ind_23 = (n_hits_m2 * pair_23[l*2] + pair_23[l*2+1]) + offset_d2;
					srcs[edge_ind_23] =  pair_23[l*2];
					dsts[edge_ind_23] =  pair_23[l*2+1];
				}				
			}
		}
    }
}



__global__ void kernel_edge_count(
	unsigned long int *flatten_doublets, 
	unsigned long int *input_hits_chunk_idx,
	unsigned long int *edge_count, // could use int here, but same vector will be used to store cumsum
	unsigned long int n){

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int total_threds = blockDim.x * gridDim.x;

    for (int i=tid; i<n; i+=total_threds){

        unsigned long int m1_idx = flatten_doublets[i*2];
        unsigned long int m2_idx = flatten_doublets[i*2 + 1];

        unsigned long int
    	m1_start_idx = input_hits_chunk_idx[m1_idx], 
		m1_end_idx   = input_hits_chunk_idx[m1_idx + 1],

    	m2_start_idx = input_hits_chunk_idx[m2_idx], 
		m2_end_idx   = input_hits_chunk_idx[m2_idx + 1];

		edge_count[i] = (m1_end_idx - m1_start_idx) * (m2_end_idx - m2_start_idx);
	}
}




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
	    hit_count++;

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

    kernel_edge_count<<<grid_size, block_size>>>(
    	d_flatten_doublets, d_input_hits_chunk_idx, d_edge_count, n_doublets);

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

    kernel_make_edges<<<grid_size, block_size>>>(
    	d_flatten_triplets, d_flatten_t2d_links, d_edge_count, d_input_hits, d_input_hits_chunk_idx, d_src, d_dst, n_triplets);

    printf("PASSED\n");
}


