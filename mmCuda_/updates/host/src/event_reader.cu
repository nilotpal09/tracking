#include "host/include/event_reader.cuh"


__global__ void flatten_hits(
  unsigned n_hits,
  Hit* d_hits,
  float *d_x, float *d_y, float *d_z, float *d_eta, float *d_phi, float *d_r
) {
  unsigned tid = threadIdx.x + blockIdx.x * blockDim.x;

  for (unsigned i = tid; i < n_hits; i += blockDim.x * gridDim.x) {
    d_x[i] = d_hits[i].x;
    d_y[i] = d_hits[i].y;
    d_z[i] = d_hits[i].z;
    d_eta[i] = d_hits[i].eta;
    d_phi[i] = d_hits[i].phi;
    d_r[i] = d_hits[i].r;
  }
}

__global__ void calculate_hit_pairs(
  unsigned n_mod_pairs,
  ModuleDoublet* module_doublets,
  unsigned *hit_offsets,
  unsigned *n_hit_pairs,
  unsigned *n_hit_a
) {
  unsigned tid = threadIdx.x + blockIdx.x * blockDim.x;

  for (unsigned mod_idx = tid; mod_idx < n_mod_pairs; mod_idx += blockDim.x * gridDim.x) {
    unsigned mod_i = module_doublets[mod_idx].module_i;
    unsigned mod_j = module_doublets[mod_idx].module_j;

    unsigned n_hits_i = hit_offsets[mod_i + 1] - hit_offsets[mod_i];
    unsigned n_hits_j = hit_offsets[mod_j + 1] - hit_offsets[mod_j];
    n_hit_pairs[mod_idx] = n_hits_i * n_hits_j;
    n_hit_a[mod_idx] = n_hits_i;
  }
}

__global__ void init_hit_pairs(
  unsigned n_mod_pairs,
  ModuleDoublet *d_mod_doublets,
  unsigned *d_hit_offsets,
  unsigned *hits_pairs_acc_offsets,
  unsigned *d_hits_a, unsigned *d_hits_b
) {
  for (unsigned mod_idx = blockIdx.x; mod_idx < n_mod_pairs;
       mod_idx += gridDim.x) {
    unsigned mod1 = d_mod_doublets[mod_idx].module_i;
    unsigned mod2 = d_mod_doublets[mod_idx].module_j;

    unsigned n_hits_mod1 = d_hit_offsets[mod1 + 1] - d_hit_offsets[mod1];
    unsigned n_hits_mod2 = d_hit_offsets[mod2 + 1] - d_hit_offsets[mod2];

    unsigned hits_pairs_acc_offset = hits_pairs_acc_offsets[mod_idx];
    unsigned *hits_a = &d_hits_a[hits_pairs_acc_offset];
    unsigned *hits_b = &d_hits_b[hits_pairs_acc_offset];

    for (unsigned i = threadIdx.y; i < n_hits_mod1; i += blockDim.y) {
      unsigned *hits_a_row = &hits_a[i * n_hits_mod2];
      unsigned *hits_b_row = &hits_b[i * n_hits_mod2];
      for (unsigned j = threadIdx.x; j < n_hits_mod2; j += blockDim.x) {
        hits_a_row[j] = i;
        hits_b_row[j] = j;
      }
    }
  }
}

__global__ void calculate_hit_triplets(
  unsigned n_mod_triplets,
  ModuleTriplet *d_mod_triplets,
  unsigned *d_hit_offsets,
  unsigned long long *d_hit_triplets_offsets) {

  unsigned tid = threadIdx.x + blockIdx.x * blockDim.x;

  for (unsigned mod_idx = tid; mod_idx < n_mod_triplets; mod_idx += blockDim.x * gridDim.x) {
    unsigned mod_i = d_mod_triplets[mod_idx].module_i;
    unsigned mod_j = d_mod_triplets[mod_idx].module_j;
    unsigned mod_k = d_mod_triplets[mod_idx].module_k;

    unsigned n_hits_i = d_hit_offsets[mod_i + 1] - d_hit_offsets[mod_i];
    unsigned n_hits_j = d_hit_offsets[mod_j + 1] - d_hit_offsets[mod_j];
    unsigned n_hits_k = d_hit_offsets[mod_k + 1] - d_hit_offsets[mod_k];

    d_hit_triplets_offsets[mod_idx] = n_hits_i * n_hits_j * n_hits_k;
  }
}



EventReader::EventReader(std::string event_path_arg) {
  n_hits = 0;
  event_path = event_path_arg;
}

Event EventReader::read(ModuleMap &mm, unsigned n_modules) {
  std::ifstream event_file(event_path);

  std::string hit_line;
  std::string delim = ",";

  if (event_file.is_open()) {
    // header (we ignore it for now)
    getline(event_file, hit_line);

    std::vector<std::string> col_entries;
    col_entries.reserve(27);
    h_hit_offsets.resize(
        ModuleMap::num_modules()); // Resize with number of modules
    while (getline(event_file, hit_line)) {

      auto start = 0U;
      auto end = hit_line.find(delim);

      while (end != std::string::npos) {
        col_entries.push_back(hit_line.substr(start, end - start));

        start = end + delim.length();
        end = hit_line.find(delim, start);
      }
      col_entries.push_back(hit_line.substr(start, end - start));
      Hit hit;
      hit.x = std::stof(col_entries.at(2));
      hit.y = std::stof(col_entries.at(3));
      hit.z = std::stof(col_entries.at(4));
      hit.eta = Eta(hit.x, hit.y, hit.z);
      hit.phi = Phi(hit.x, hit.y);
      hit.r = R(hit.x, hit.y);
      h_hits.push_back(hit);
      unsigned mod = std::stoi(col_entries.back());
      if (mod >= ModuleMap::num_modules()) {
        std::cout << "Module ID is greater than number of modules" << std::endl;
      }
      h_hit_inds.push_back(mod);
      h_hit_offsets[mod] += 1;
      col_entries.clear();
      n_hits += 1;
    } // end of while loop
  } else {
    std::cout << "Unable to open file" << event_path << std::endl;
  }
  event_file.close();

  Event event;
  allocate_device_memory_hits(event, n_hits);


  // Sort Hits by module
  //--------------------------------------------------------------------------------  
  unsigned *d_hit_inds;
  MemoryScheduler::allocate(&d_hit_inds, n_hits);
  CUDA_WARN(cudaMemcpy(d_hit_inds, static_cast<unsigned *>(h_hit_inds.data()),
                       n_hits * sizeof(unsigned), cudaMemcpyHostToDevice));

  Hit *d_hits;
  MemoryScheduler::allocate(&d_hits, n_hits);
  CUDA_WARN(cudaMemcpy(d_hits, static_cast<Hit *>(h_hits.data()),
                       n_hits * sizeof(Hit), cudaMemcpyHostToDevice));

  Hit *d_hits_out;
  MemoryScheduler::allocate(&d_hits_out, n_hits);
  unsigned *d_hit_ind_out;
  MemoryScheduler::allocate(&d_hit_ind_out, n_hits);

  Sort(&d_hit_inds, &d_hits, &d_hit_ind_out, &d_hits_out, n_hits);

  flatten_hits<<<65535, 256>>>(n_hits, d_hits_out,
                               event.x, event.y, event.z,
                               event.eta, event.phi, event.r);

  MemoryScheduler::free(&d_hit_inds);
  MemoryScheduler::free(&d_hits);
  MemoryScheduler::free(&d_hit_ind_out);
  MemoryScheduler::free(&d_hits_out);

  
  // Calculate Hit offsets
  //--------------------------------------------------------------------------------
  unsigned *d_hit_offsets;
  MemoryScheduler::allocate(&d_hit_offsets, ModuleMap::num_modules());

  CUDA_WARN(cudaMemcpy(
      d_hit_offsets, static_cast<unsigned *>(h_hit_offsets.data()),
      ModuleMap::num_modules() * sizeof(unsigned), cudaMemcpyHostToDevice));

  MemoryScheduler::memset(&event.hit_offsets, ModuleMap::num_modules() + 1,
                          (unsigned)0);

  PrefixSum(&d_hit_offsets, &event.hit_offsets, ModuleMap::num_modules());

  MemoryScheduler::free(&d_hit_offsets);

  
  // Calculate Hit Pairs
  //--------------------------------------------------------------------------------
  unsigned *d_n_hit_pairs; // Number of hit pairs per doublet
  MemoryScheduler::allocate(&d_n_hit_pairs, ModuleMap::num_doublets());
  unsigned *d_n_hit_a; // Number of hits in first module of doublet
  MemoryScheduler::allocate(&d_n_hit_a, ModuleMap::num_doublets());

  CUDA_CHECK();

  calculate_hit_pairs<<<65535, 256>>>(ModuleMap::num_doublets(),
                                      *mm.doublets(), event.hit_offsets,
                                      d_n_hit_pairs, d_n_hit_a);

  CUDA_CHECK();
  cudaDeviceSynchronize();

  
  // Calculate Hit Pairs Offsets Prefix Sum
  //--------------------------------------------------------------------------------
  MemoryScheduler::allocate(&event.hit_pairs_offsets,
                            ModuleMap::num_doublets() + 1);

  MemoryScheduler::memset(&event.hit_pairs_offsets,
                          ModuleMap::num_doublets() + 1, (unsigned)0);

  PrefixSum(&d_n_hit_pairs, &event.hit_pairs_offsets,
            ModuleMap::num_doublets());

  MemoryScheduler::free(&d_n_hit_pairs);

  CUDA_CHECK();
  cudaDeviceSynchronize();

  // Calculate Hit Sum Offsets Prefix Sum
  //--------------------------------------------------------------------------------
  MemoryScheduler::allocate(&event.hit_sum_offsets,
                            ModuleMap::num_doublets() + 1);

  MemoryScheduler::memset(&event.hit_sum_offsets,
                          ModuleMap::num_doublets() + 1, (unsigned)0);

  PrefixSum(&d_n_hit_a, &event.hit_sum_offsets,
            ModuleMap::num_doublets());

  MemoryScheduler::free(&d_n_hit_a);

  CUDA_CHECK();
  cudaDeviceSynchronize();

  n_hit_pairs = getMax(&event.hit_pairs_offsets, ModuleMap::num_doublets());

  cudaDeviceSynchronize();
  

  MemoryScheduler::allocate(&event.hit_pairs, n_hit_pairs);
  MemoryScheduler::allocate(&event.hits_a, n_hit_pairs);
  MemoryScheduler::allocate(&event.hits_b, n_hit_pairs);
  // TODO: When selection is added erduce the memory required for this.
  MemoryScheduler::allocate(&event.hits_a_reduced, n_hit_pairs);
  MemoryScheduler::allocate(&event.hits_b_reduced, n_hit_pairs);

  unsigned total = getMax(&event.hit_sum_offsets, ModuleMap::num_doublets());

  cudaDeviceSynchronize();

  MemoryScheduler::allocate(&event.hit_sum, total);
  
  // MemoryScheduler::free(&d_total);

  dim3 grid_dim(65535);
  dim3 block_dim(32, 32);
  init_hit_pairs<<<grid_dim, block_dim>>>(
      ModuleMap::num_doublets(), *mm.doublets(), event.hit_offsets,
      event.hit_pairs_offsets, event.hits_a, event.hits_b);

  
  MemoryScheduler::allocate(&event.hit_module_sum, ModuleMap::num_doublets());
  MemoryScheduler::allocate(&event.hit_module_offsets,
                            ModuleMap::num_doublets() + 1);


  unsigned long long *d_hit_triplets_offsets;
  MemoryScheduler::allocate(&d_hit_triplets_offsets, ModuleMap::num_triplets());
  CUDA_CHECK();
  MemoryScheduler::memset(&d_hit_triplets_offsets, ModuleMap::num_triplets(),
                          (unsigned long long)0);
  
  calculate_hit_triplets<<<65535, 256>>>(ModuleMap::num_triplets(),
                                           *mm.triplets(), event.hit_offsets,
                                           d_hit_triplets_offsets);
  cudaDeviceSynchronize();
  CUDA_CHECK();
  
  unsigned long long *d_hit_triplets_sum;
  MemoryScheduler::allocate(&d_hit_triplets_sum, ModuleMap::num_triplets() + 1);
  MemoryScheduler::memset(&d_hit_triplets_sum, ModuleMap::num_triplets() + 1,
                          (unsigned long long)0);
  
  PrefixSum(&d_hit_triplets_offsets, &d_hit_triplets_sum,
            ModuleMap::num_triplets());

  cudaDeviceSynchronize();
  

  // Get the total number of hit triplets
  //n_hit_triplets = getMax(&d_hit_triplets_sum, ModuleMap::num_triplets() - 1);
  
  cudaDeviceSynchronize();

  MemoryScheduler::free(&d_hit_triplets_offsets);
  MemoryScheduler::free(&d_hit_triplets_sum);
  
  return event;
}

void EventReader::print_summary() {
  std::cout << std::string(50, '-') << std::endl;
  std::cout << "\tEvent summary" << std::endl;
  std::cout << std::string(50, '-') << std::endl;

  std::cout << "Event taken from -" << std::endl;
  std::cout << event_path << std::endl << std::endl;

  std::cout.imbue(std::locale(""));
  std::cout << std::fixed << std::showpoint << std::setprecision(3);

  std::cout << "# hits: " << n_hits << std::endl;
  std::cout << "# hit pairs: " << n_hit_pairs << std::endl;
  std::cout << "# hit triplets: " << n_hit_triplets << std::endl;

  std::cout << std::string(50, '-') << std::endl;
}

void EventReader::allocate_device_memory_hits(Event& event, unsigned n_hits) {
  MemoryScheduler::allocate(&event.x, n_hits);
  MemoryScheduler::allocate(&event.y, n_hits);
  MemoryScheduler::allocate(&event.z, n_hits);
  MemoryScheduler::allocate(&event.r, n_hits);
  MemoryScheduler::allocate(&event.phi, n_hits);
  MemoryScheduler::allocate(&event.eta, n_hits);
  MemoryScheduler::allocate(&event.hit_offsets, ModuleMap::num_modules() + 1);
  MemoryScheduler::allocate(&event.hit_module_sum, ModuleMap::num_doublets());
  MemoryScheduler::allocate(&event.hit_module_offsets,
                            ModuleMap::num_doublets() + 1);
}