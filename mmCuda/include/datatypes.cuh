#ifndef __DATATYPE_CUH__
#define __DATATYPE_CUH__

#include "../include/utility.cuh"
#include "../include/memoryscheduler.cuh"

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <map>
#include <set>
#include <tuple>
#include <algorithm>

// print numbers with formatting
#include <locale>
#include <iomanip>

__global__ inline void  DeviceCalculateHitPairs(
    unsigned n_mod_pairs,
    unsigned* d_mod_pair_inds,
    unsigned* d_mod_pairs,
    unsigned* d_hit_offsets,
    unsigned* d_hit_pairs_offsets
) {
    for (int i = blockIdx.x; i < n_mod_pairs; i += gridDim.x) {
        unsigned mod1 = d_mod_pair_inds[i];
        unsigned mod2 = d_mod_pairs[i];

        unsigned n_hits_mod1 = d_hit_offsets[mod1 + 1] - d_hit_offsets[mod1];
        unsigned n_hits_mod2 = d_hit_offsets[mod2 + 1] - d_hit_offsets[mod2];

        d_hit_pairs_offsets[i] = n_hits_mod1 * n_hits_mod2;
    }
}

__global__ inline void getMax(
    unsigned* max,
    unsigned n_mod_pairs,
    unsigned* d_hit_pairs_offsets
) {
    *max = d_hit_pairs_offsets[n_mod_pairs - 1];
    printf("Max: %d \n", *max);
}

__global__ inline void printHitIndices(
    unsigned* d_hit_offsets,
    unsigned n_hits
) {
    for (int i = blockIdx.x; i < n_hits; i += gridDim.x) {
        //unsigned diff = d_hit_offsets[i + 1] - d_hit_offsets[i];
        //printf("Hit %u: %u %u \n", i, diff, d_hit_offsets[i]);
        printf("Hit %u: %u \n", i, d_hit_offsets[i]);
    }
    // printf("Last hit: %u \n", d_hit_offsets[n_hits] - d_hit_offsets[n_hits - 1]);
    // printf("Total hits: %u \n", d_hit_offsets[n_hits]);
}



class ModuleMap{

    private:
        // Counted from file
        static constexpr unsigned n_mod = 18360; 
        static constexpr unsigned n_mod_pairs = 509461;
        static constexpr unsigned n_mod_triplets = 1242265;

        /*
        * Device pointers
        */
        struct DevicePtrs{
            unsigned* modules;
            unsigned* pairs;
        };

        DevicePtrs d_ptrs;

        void allocate_device_memory(){
            MemoryScheduler::allocate(&d_ptrs.modules, n_mod_pairs);
            MemoryScheduler::allocate(&d_ptrs.pairs, n_mod_pairs);
            std::cout << "Allocated device memory for module map" << std::endl;
            unsigned long long int total_mem = 0;
            total_mem += n_mod_pairs * sizeof(unsigned);
            total_mem += n_mod_pairs * sizeof(unsigned);
            std::cout << "Total memory allocated: " << total_mem / 1e9 << " GB" << std::endl;
        }

    
        std::string mm_path;
        std::vector<unsigned> h_mod_ind;
        std::vector<unsigned> h_mod_pairs;
        std::vector<unsigned> h_mod_pairs_offsets;
        std::vector<unsigned> h_mod_triplets;

    public:
        unsigned** d_modules() { return &d_ptrs.modules; }

        unsigned** d_pairs() { return &d_ptrs.pairs; }
    
    public:
        static constexpr unsigned num_modules() { return n_mod; }
        static constexpr unsigned num_doublets() { return n_mod_pairs; }
        static constexpr unsigned num_triplets() { return n_mod_triplets; }

        // constructor
        ModuleMap(std::string mm_path_arg, std::string mm_pairs_path){

            h_mod_ind.reserve(n_mod_pairs); // Reserve a bit of memory
            h_mod_pairs.reserve(n_mod_pairs);

            //h_mod_triplets.reserve(n_mod_triplets * 3);


            // Reserve a bit of memory
            h_mod_pairs_offsets.resize(n_mod + 1);
            h_mod_ind.reserve(n_mod_pairs);
            h_mod_pairs.reserve(n_mod_pairs);
            h_mod_triplets.reserve(n_mod_triplets * 3);

            mm_path = mm_path_arg;
            std::ifstream mm_file (mm_path);
            std::string mm_line;
            std::string delim = " ";

            std::vector<unsigned int> m_ids;
            m_ids.resize(3);

            /*********************************
            / Module triplets
            *********************************/
            if (mm_file.is_open()){
                while (getline(mm_file, mm_line)){

                    auto start = 0U;
                    auto end   = mm_line.find(delim);

                    for (int i=0; i<3; i++){
                        int m_id = std::stoi(mm_line.substr(start, end - start));
                        m_ids[i] = m_id;

                        start = end + delim.length();
                        end   = mm_line.find(delim, start);
                    }
                    // Add triplets
                    // Is a check to ensure they are all unique needed?
                    h_mod_triplets.insert(h_mod_triplets.end(),
                                          static_cast<unsigned*>(m_ids.data()),
                                          static_cast<unsigned*>(m_ids.data()) + 3);
        
                }
                mm_file.close();
            }
            h_mod_triplets.resize(h_mod_triplets.size());


            std::ifstream mm_pairs_file (mm_pairs_path);
            m_ids.resize(2);

            /*********************************
            / Module doublets
            *********************************/
            if (mm_pairs_file.is_open()){
                while (getline(mm_pairs_file, mm_line)){

                    auto start = 0U;
                    auto end   = mm_line.find(delim);

                    for (int i=0; i<2; i++){
                        int m_id = std::stoi(mm_line.substr(start, end - start));
                        m_ids[i] = m_id;

                        start = end + delim.length();
                        end   = mm_line.find(delim, start);
                    }
                    
                    h_mod_ind.push_back(m_ids[0]);
                    h_mod_pairs.push_back(m_ids[1]);
                }
                mm_pairs_file.close();
            }

            
            allocate_device_memory();
            CUDA_WARN(cudaMemcpy(d_ptrs.modules,
                                 static_cast<unsigned*>(h_mod_ind.data()),
                                 h_mod_ind.size() * sizeof(unsigned),
                                 cudaMemcpyHostToDevice));
            CUDA_WARN(cudaMemcpy(d_ptrs.pairs,
                                 static_cast<unsigned*>(h_mod_pairs.data()),
                                 h_mod_pairs.size() * sizeof(unsigned),
                                 cudaMemcpyHostToDevice));

        } // constructor ends


        void print_summary(){
            std::cout << std::string(50, '-') << std::endl;
            std::cout << "\tModule map summary" << std::endl;
            std::cout << std::string(50, '-') << std::endl;

            std::cout << "Module map taken from -" << std::endl;
            std::cout << mm_path << std::endl << std::endl;

            std::cout.imbue(std::locale(""));
            std::cout << std::fixed << std::showpoint << std::setprecision(3);

            std::cout << "# triplets: " << num_triplets() << std::endl;
            std::cout << "# doublets: " << num_doublets() << std::endl;

            std::cout << std::string(50, '-') << std::endl;
        }

};

struct Module {
    // Store information about the module
};


struct Hit {
    float x;
    float y;
    float z;
};



class EventData{

    private:
        struct DevicePtrs{
            Hit* hits;
            unsigned* hit_offsets;
            bool* hit_pairs;
            unsigned* hit_pairs_offsets;
        };

        DevicePtrs d_ptrs;

        void allocate_device_memory_hits(unsigned n_hits){
            MemoryScheduler::allocate(&d_ptrs.hits, n_hits);
            MemoryScheduler::allocate(&d_ptrs.hit_offsets, ModuleMap::num_modules() + 1);
        }

        std::string event_path;

        std::vector<unsigned> h_hit_inds;
        std::vector<Hit> h_hits;
        std::vector<unsigned> h_hit_offsets;
        std::vector<unsigned> h_hit_pairs_offsets;
        unsigned n_hits;
        unsigned n_hit_pairs;

    public:
        // Hit* hits;

        Hit** d_hits() { return &d_ptrs.hits; }

        unsigned** d_hit_offsets() { return &d_ptrs.hit_offsets; }

        bool** d_hit_pairs() { return &d_ptrs.hit_pairs; }

        unsigned** d_hit_pairs_offsets() { return &d_ptrs.hit_pairs_offsets; }

    
    public:
        // constructor
        EventData(std::string event_path_arg, ModuleMap& mm, unsigned int n_modules){
            n_hits = 0;
            event_path = event_path_arg;

            // read the file and make the flatten triplets
            std::ifstream event_file (event_path);

            std::string hit_line;
            std::string delim = ",";

            if (event_file.is_open()){

                // header (we ignore it for now)
                getline(event_file, hit_line);

                std::vector<std::string> col_entries;
                col_entries.reserve(15);
                h_hit_offsets.resize(ModuleMap::num_modules()); // Resize with number of modules
                while (getline(event_file, hit_line)){

                    //boost::split(col_entries, hit_line, boost::is_any_of(", "), boost::token_compress_on);
                    
                    auto start = 0U;
                    auto end   = hit_line.find(delim);

                    while (end != std::string::npos){
                        col_entries.push_back(hit_line.substr(start, end - start));

                        start = end + delim.length();
                        end   = hit_line.find(delim, start);
                    }
                    col_entries.push_back(hit_line.substr(start, end - start));

                    Hit hit;
                    // Need to optimize this into one memory copy
                    hit.x = std::stof(col_entries.at(1));
                    hit.y = std::stof(col_entries.at(2));
                    hit.z = std::stof(col_entries.at(3));
                    h_hits.push_back(hit);
                    unsigned mod = std::stoi(col_entries.back());
                    if (mod >= ModuleMap::num_modules())
                        std::cout << "Module ID is greater than number of modules" << std::endl;
                    h_hit_inds.push_back(mod);
                    h_hit_offsets[mod] += 1;
                    col_entries.clear();
                    n_hits += 1;
                } // end of while loop
            }
            event_file.close();

            allocate_device_memory_hits(n_hits);

            /*
                Sort Hits by module
            */
            unsigned* d_hit_inds;
            MemoryScheduler::allocate(&d_hit_inds, n_hits);
            CUDA_WARN(cudaMemcpy(d_hit_inds,
                                 static_cast<unsigned*>(h_hit_inds.data()),
                                 n_hits * sizeof(unsigned),
                                 cudaMemcpyHostToDevice));
            
            
            Hit* d_hits;
            MemoryScheduler::allocate(&d_hits, n_hits);
            CUDA_WARN(cudaMemcpy(d_hits,
                                 static_cast<Hit*>(h_hits.data()),
                                 n_hits * sizeof(Hit),
                                 cudaMemcpyHostToDevice));

            
            unsigned* d_hit_ind_out;
            MemoryScheduler::allocate(&d_hit_ind_out, n_hits);
            
            void     *d_temp_storage = NULL;
            size_t   temp_storage_bytes = 0;
            CUDA_WARN(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                                      d_hit_inds, d_hit_ind_out, d_hits, d_ptrs.hits, n_hits));
            cudaDeviceSynchronize();
            
            // Allocate temporary storage
            CUDA_WARN(cudaMalloc(&d_temp_storage, temp_storage_bytes));
            
            CUDA_WARN(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                                      d_hit_inds, d_hit_ind_out, d_hits, d_ptrs.hits, n_hits));
            cudaDeviceSynchronize();


            MemoryScheduler::free(&d_hit_inds);
            MemoryScheduler::free(&d_hits);
            MemoryScheduler::free(&d_hit_ind_out);
            CUDA_WARN(cudaFree(d_temp_storage));

            /*
                Calculate Hit offsets
            */
            
            // Copy hit offsets to device
            d_temp_storage = NULL;
            temp_storage_bytes = 0;
            unsigned* d_hit_offsets;
            MemoryScheduler::allocate(&d_hit_offsets, ModuleMap::num_modules());
            CUDA_WARN(cudaMemcpy(d_hit_offsets,
                                 static_cast<unsigned*>(h_hit_offsets.data()),
                                 ModuleMap::num_modules() * sizeof(unsigned),
                                 cudaMemcpyHostToDevice));

            cudaDeviceSynchronize();

            // Maybe a check to ensure all offsets are 0
            CUDA_WARN(cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                                    d_hit_offsets, &d_ptrs.hit_offsets[1], ModuleMap::num_modules()));
            cudaDeviceSynchronize();
            
            // Allocate temporary storage
            CUDA_WARN(cudaMalloc(&d_temp_storage, temp_storage_bytes));

            // Run inclusive prefix sum
            CUDA_WARN(cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                                    d_hit_offsets, &d_ptrs.hit_offsets[1], ModuleMap::num_modules()));
            cudaDeviceSynchronize();

            // Free temporary storage
            MemoryScheduler::free(&d_hit_offsets);
            CUDA_WARN(cudaFree(d_temp_storage));
            
            /*
                Calculate Hit Pairs
            */
            unsigned* d_hit_pairs_offsets;
            MemoryScheduler::allocate(&d_hit_pairs_offsets, ModuleMap::num_doublets());
            
            CUDA_CHECK();
            
            // Strangely this is quicker with 65535 blocks rather than num_doublets
            // By like a factor of 5
            DeviceCalculateHitPairs<<<65535, 1>>>(
                ModuleMap::num_doublets(),
                *mm.d_modules(),
                *mm.d_pairs(),
                d_ptrs.hit_offsets,
                d_hit_pairs_offsets
            );

            CUDA_CHECK();

            cudaDeviceSynchronize();


            n_hit_pairs = 0;
            unsigned* d_n_hit_pairs;
            MemoryScheduler::allocate(&d_n_hit_pairs, 1);
            CUDA_WARN(cudaMemcpy(d_n_hit_pairs,
                                 &n_hit_pairs,
                                 sizeof(unsigned),
                                 cudaMemcpyHostToDevice));
            
            CUDA_CHECK();
            
            MemoryScheduler::allocate(&d_ptrs.hit_pairs_offsets, ModuleMap::num_doublets() + 1);

            d_temp_storage = NULL;
            temp_storage_bytes = 0;
            
            CUDA_WARN(cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                                    d_hit_pairs_offsets, &d_ptrs.hit_pairs_offsets[1], ModuleMap::num_doublets()));
            cudaDeviceSynchronize();

            // Allocate temporary storage
            CUDA_WARN(cudaMalloc(&d_temp_storage, temp_storage_bytes));

            CUDA_WARN(cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                                    d_hit_pairs_offsets, &d_ptrs.hit_pairs_offsets[1], ModuleMap::num_doublets()));
            cudaDeviceSynchronize();

            MemoryScheduler::free(&d_hit_pairs_offsets);
            CUDA_WARN(cudaFree(d_temp_storage));

            CUDA_CHECK();

            cudaDeviceSynchronize();

            getMax<<<1, 1>>>(d_n_hit_pairs, ModuleMap::num_doublets(), &d_ptrs.hit_pairs_offsets[1]);

            CUDA_WARN(cudaMemcpy(&n_hit_pairs,
                                 d_n_hit_pairs,
                                 sizeof(unsigned),
                                 cudaMemcpyDeviceToHost));

            MemoryScheduler::free(&d_n_hit_pairs);

            MemoryScheduler::allocate(&d_ptrs.hit_pairs, n_hit_pairs);
            
        } // end of constructor


        void print_summary(){
            std::cout << std::string(50, '-') << std::endl;
            std::cout << "\tEvent summary" << std::endl;
            std::cout << std::string(50, '-') << std::endl;

            std::cout << "Event taken from -" << std::endl;
            std::cout << event_path << std::endl << std::endl;

            std::cout.imbue(std::locale(""));
            std::cout << std::fixed << std::showpoint << std::setprecision(3);

            std::cout << "# hits: " << n_hits << std::endl;
            std::cout << "# hit pairs: " << n_hit_pairs << std::endl;

            std::cout << std::string(50, '-') << std::endl;
        }

};



#endif