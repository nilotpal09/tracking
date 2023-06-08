#include "../include/datatypes.cuh"

__device__ unsigned long long counter5 = 0;

__global__ void print_counter5() { printf("Counter5: %ld \n", counter5); }

__host__ float h_Phi(const float &x, const float &y) {
  return atan2(y, x);
}

__host__ float h_R(const float &x, const float &y) {
  return sqrtf(x * x + y * y);
}

__host__ float h_Eta(const float &x, const float &y, const float &z) {
  float r = h_R(x, y);
  float r3 = sqrtf(r * r + z * z);
  float theta = acos(z / r3);

  return -log(tan(theta * 0.5));
}

__global__ void DeviceCalculateHitPairs(unsigned n_mod_pairs,
                                        ModuleDoublet *d_mod_doublets,
                                        unsigned *d_hit_offsets,
                                        unsigned *d_n_hit_pairs,
                                        unsigned *d_n_hits_a) {
  for (int i = blockIdx.x; i < n_mod_pairs; i += gridDim.x) {
    unsigned mod1 = d_mod_doublets[i].module_a;
    unsigned mod2 = d_mod_doublets[i].module_b;

    unsigned n_hits_mod1 = d_hit_offsets[mod1 + 1] - d_hit_offsets[mod1];
    unsigned n_hits_mod2 = d_hit_offsets[mod2 + 1] - d_hit_offsets[mod2];

    d_n_hits_a[i] = n_hits_mod1 + 1;
    d_n_hit_pairs[i] = n_hits_mod1 * n_hits_mod2;
  }
}

__global__ void DeviceCalculateHitTriplets(unsigned n_mod_triplets,
                                           ModuleTriplet *d_mod_triplets,
                                           unsigned *d_hit_offsets,
                                           unsigned long long *d_hit_triplets_offsets) {
  for (unsigned i = blockIdx.x; i < n_mod_triplets; i += gridDim.x) {
    unsigned mod1 = d_mod_triplets[i].module_a;
    unsigned mod2 = d_mod_triplets[i].module_b;
    unsigned mod3 = d_mod_triplets[i].module_c;

    unsigned n_hits_mod1 = d_hit_offsets[mod1 + 1] - d_hit_offsets[mod1];
    unsigned n_hits_mod2 = d_hit_offsets[mod2 + 1] - d_hit_offsets[mod2];
    unsigned n_hits_mod3 = d_hit_offsets[mod3 + 1] - d_hit_offsets[mod3];

    d_hit_triplets_offsets[i] = n_hits_mod1 * n_hits_mod2 * n_hits_mod3;
    atomicAdd(&counter5, n_hits_mod1 * n_hits_mod2 * n_hits_mod3);
  }
}

__global__ void DeviceInitHitPairs(unsigned n_mod_pairs,
                                   ModuleDoublet *d_mod_doublets,
                                   unsigned *d_hit_offsets,
                                   unsigned *hits_pairs_acc_offsets,
                                   unsigned *d_hits_a, unsigned *d_hits_b) {
  for (unsigned mod_idx = blockIdx.x; mod_idx < n_mod_pairs;
       mod_idx += gridDim.x) {
    unsigned mod1 = d_mod_doublets[mod_idx].module_a;
    unsigned mod2 = d_mod_doublets[mod_idx].module_b;

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

ModuleMap::ModuleMap(std::string mm_path_arg, std::string mm_pairs_path) {

  h_mod_doublets.reserve(ModuleMap::num_doublets());
  h_mod_triplets.reserve(ModuleMap::num_triplets());

  mm_path = mm_path_arg;

  std::string mm_line;
  std::string delim = " ";

  std::vector<unsigned> m_ids;
  m_ids.resize(2);
  std::vector<float> m_cuts;
  m_cuts.resize(10);
  std::vector<unsigned> m_pairs_sum;
  std::vector<unsigned> m_idx_b;
  m_pairs_sum.resize(num_modules());
  m_idx_b.reserve(num_doublets());
  /*********************************
  / Module doublets
  *********************************/
  std::ifstream mm_pairs_file(mm_pairs_path);
  if (mm_pairs_file.is_open()) {
    while (getline(mm_pairs_file, mm_line)) {

      auto start = 0U;
      auto end = mm_line.find(delim);

      for (int i = 0; i < 10; i++) {
        if (i < 2) {
          int m_id = std::stoi(mm_line.substr(start, end - start));
          m_ids[i] = m_id;
        } else if (i < 10) {
          float m_cut = std::stof(mm_line.substr(start, end - start));
          m_cuts[i] = m_cut;
        }
        start = end + delim.length();
        end = mm_line.find(delim, start);
      }

      ModuleDoublet m_doublet{m_ids[0],  m_ids[1],  m_cuts[3], m_cuts[2],
                              m_cuts[5], m_cuts[4], m_cuts[7], m_cuts[6],
                              m_cuts[9], m_cuts[8]};
      h_mod_doublets.push_back(m_doublet);
      m_pairs_sum[m_ids[0]] += 1;
      m_idx_b.push_back(m_ids[1]);
    }
    mm_pairs_file.close();
  } else {
    std::cout << "Unable to open file" << mm_pairs_path << std::endl;
  }
  std::cout << "Formed module doublets" << std::endl;

  // Prefix sum m_pairs_sum
  for (int i = 1; i < num_modules(); i++) {
    m_pairs_sum[i] += m_pairs_sum[i - 1];
  }
  // Add zero to the beginning
  m_pairs_sum.insert(m_pairs_sum.begin(), 0);

  m_ids.resize(5);
  m_cuts.resize(26);
  /*********************************
  / Module triplets
  *********************************/
  std::ifstream mm_file(mm_path);
  if (mm_file.is_open()) {
    while (getline(mm_file, mm_line)) {

      auto start = 0U;
      auto end = mm_line.find(delim);

      for (int i = 0; i < 26; i++) {
        if (i < 3) {
          int m_id = std::stoi(mm_line.substr(start, end - start));
          m_ids[i] = m_id;
        } else if (i == 24) {
          int m_id = std::stoi(mm_line.substr(
              start, end - start)); // Horrible hacky to get doublet pairs
          m_ids[3] = m_id;
        } else if (i == 25) {
          int m_id = std::stoi(mm_line.substr(start, end - start));
          m_ids[4] = m_id;
        } else {
          float m_cut = std::stof(mm_line.substr(start, end - start));
          m_cuts[i] = m_cut;
        }
        start = end + delim.length();
        end = mm_line.find(delim, start);
      }
      unsigned pair_i = findPairIndex(m_ids[0], m_ids[1], m_pairs_sum, m_idx_b);
      unsigned pair_j = findPairIndex(m_ids[1], m_ids[2], m_pairs_sum, m_idx_b);

      //std::cout << "i: " << m_ids[0] << " j: " << m_ids[1] << " k: " << m_ids[2] << " pair_i: " << pair_i << " pair_j: " << pair_j << std::endl;

      ModuleTriplet trip{
          m_ids[0],   m_ids[1],   m_ids[2],   pair_i,   pair_j,
          m_cuts[4],  m_cuts[5],  m_cuts[6],  m_cuts[7],  m_cuts[8],
          m_cuts[9],  m_cuts[10], m_cuts[11], m_cuts[12], m_cuts[13],
          m_cuts[14], m_cuts[15], m_cuts[16], m_cuts[17], m_cuts[18],
          m_cuts[19], m_cuts[20], m_cuts[21], m_cuts[22], m_cuts[23]};
      h_mod_triplets.push_back(trip);
    }
    mm_file.close();
  } else {
    std::cout << "Unable to open file" << mm_path << std::endl;
  }
  std::cout << "Formed module triplets" << std::endl;

  allocate_device_memory();

  CUDA_WARN(cudaMemcpy(d_ptrs.doublets, h_mod_doublets.data(),
                       n_mod_pairs * sizeof(ModuleDoublet),
                       cudaMemcpyHostToDevice));
  CUDA_WARN(cudaMemcpy(d_ptrs.triplets, h_mod_triplets.data(),
                       n_mod_triplets * sizeof(ModuleTriplet),
                       cudaMemcpyHostToDevice));
}

void ModuleMap::print_summary() {
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

void ModuleMap::allocate_device_memory() {
  MemoryScheduler::allocate(&d_ptrs.doublets, n_mod_pairs);
  MemoryScheduler::allocate(&d_ptrs.triplets, n_mod_triplets);

  std::cout << "Allocated device memory for module map" << std::endl;
  unsigned long long int total_mem = 0;
  total_mem += n_mod_pairs * sizeof(ModuleDoublet);
  total_mem += n_mod_triplets * sizeof(ModuleTriplet);
  std::cout << "Total memory allocated: " << total_mem / 1e9 << " GB"
            << std::endl;
}

EventData::EventData(std::string event_path_arg, ModuleMap &mm,
                     unsigned int n_modules) {
  n_hits = 0;
  event_path = event_path_arg;

  // read the file and make the flatten triplets
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
    float x;
    float y;
    while (getline(event_file, hit_line)) {

      // boost::split(col_entries, hit_line, boost::is_any_of(", "),
      // boost::token_compress_on);

      auto start = 0U;
      auto end = hit_line.find(delim);

      while (end != std::string::npos) {
        col_entries.push_back(hit_line.substr(start, end - start));

        start = end + delim.length();
        end = hit_line.find(delim, start);
      }
      col_entries.push_back(hit_line.substr(start, end - start));
      Hit hit;
      // Need to optimize this into one memory copy
      hit.x = std::stof(col_entries.at(2));
      hit.y = std::stof(col_entries.at(3));
      hit.z = std::stof(col_entries.at(4));
      hit.eta = h_Eta(hit.x, hit.y, hit.z);
      hit.phi = h_Phi(hit.x, hit.y);
      hit.r = h_R(hit.x, hit.y);
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
  unsigned *d_hit_inds;
  MemoryScheduler::allocate(&d_hit_inds, n_hits);
  CUDA_WARN(cudaMemcpy(d_hit_inds, static_cast<unsigned *>(h_hit_inds.data()),
                       n_hits * sizeof(unsigned), cudaMemcpyHostToDevice));

  Hit *d_hits;
  MemoryScheduler::allocate(&d_hits, n_hits);
  CUDA_WARN(cudaMemcpy(d_hits, static_cast<Hit *>(h_hits.data()),
                       n_hits * sizeof(Hit), cudaMemcpyHostToDevice));

  unsigned *d_hit_ind_out;
  MemoryScheduler::allocate(&d_hit_ind_out, n_hits);

  Sort(&d_hit_inds, &d_hits, &d_hit_ind_out, &d_ptrs.hits, n_hits);

  DEBUG(test_hit_sort(&d_hit_ind_out, n_hits));

  MemoryScheduler::free(&d_hit_inds);
  MemoryScheduler::free(&d_hits);
  MemoryScheduler::free(&d_hit_ind_out);

  /*
      Calculate Hit offsets
  */
  unsigned *d_hit_offsets;
  MemoryScheduler::allocate(&d_hit_offsets, ModuleMap::num_modules());
  CUDA_WARN(cudaMemcpy(
      d_hit_offsets, static_cast<unsigned *>(h_hit_offsets.data()),
      ModuleMap::num_modules() * sizeof(unsigned), cudaMemcpyHostToDevice));

  MemoryScheduler::memset(&d_ptrs.hit_offsets, ModuleMap::num_modules() + 1,
                          (unsigned)0);

  PrefixSum(&d_hit_offsets, &d_ptrs.hit_offsets, ModuleMap::num_modules());

  DEBUG(test_hit_offsets(h_hit_offsets, &d_ptrs.hit_offsets, ModuleMap::num_modules()));

  MemoryScheduler::free(&d_hit_offsets);

  /*
      Calculate Hit Pairs
  */
  unsigned *d_n_hit_pairs; // Number of hit pairs per doublet
  MemoryScheduler::allocate(&d_n_hit_pairs, ModuleMap::num_doublets());
  unsigned *d_n_hit_a; // Number of hits in first module of doublet
  MemoryScheduler::allocate(&d_n_hit_a, ModuleMap::num_doublets());

  CUDA_CHECK();

  // Strangely this is quicker with 65535 blocks rather than num_doublets
  // By like a factor of 5
  DeviceCalculateHitPairs<<<65535, 1>>>(ModuleMap::num_doublets(),
                                        *mm.d_doublets(), d_ptrs.hit_offsets,
                                        d_n_hit_pairs, d_n_hit_a);

  CUDA_CHECK();

  cudaDeviceSynchronize();

  /*
    Calculate Hit Pairs Offsets Prefix Sum
  */
  MemoryScheduler::allocate(&d_ptrs.hit_pairs_offsets,
                            ModuleMap::num_doublets() + 1);

  MemoryScheduler::memset(&d_ptrs.hit_pairs_offsets,
                          ModuleMap::num_doublets() + 1, (unsigned)0);

  PrefixSum(&d_n_hit_pairs, &d_ptrs.hit_pairs_offsets,
            ModuleMap::num_doublets());

  DEBUG(test_hit_pairs_offsets(&d_n_hit_pairs, &d_ptrs.hit_pairs_offsets, ModuleMap::num_doublets()));

  MemoryScheduler::free(&d_n_hit_pairs);

  CUDA_CHECK();
  cudaDeviceSynchronize();

  /*
      Calculate Hit Sum Offsets Prefix Sum
  */
  MemoryScheduler::allocate(&d_ptrs.hit_sum_offsets,
                            ModuleMap::num_doublets() + 1);

  MemoryScheduler::memset(&d_ptrs.hit_sum_offsets,
                          ModuleMap::num_doublets() + 1, (unsigned)0);

  PrefixSum(&d_n_hit_a, &d_ptrs.hit_sum_offsets,
            ModuleMap::num_doublets());

  DEBUG(test_hit_sum_offsets(&d_n_hit_a, &d_ptrs.hit_sum_offsets, ModuleMap::num_doublets()));

  MemoryScheduler::free(&d_n_hit_a);

  CUDA_CHECK();
  cudaDeviceSynchronize();

  n_hit_pairs = getMax(&d_ptrs.hit_pairs_offsets, ModuleMap::num_doublets());

  cudaDeviceSynchronize();


  MemoryScheduler::allocate(&d_ptrs.hit_pairs, n_hit_pairs);
  MemoryScheduler::allocate(&d_ptrs.hits_a, n_hit_pairs);
  MemoryScheduler::allocate(&d_ptrs.hits_b, n_hit_pairs);
  // TODO: When selection is added erduce the memory required for this.
  MemoryScheduler::allocate(&d_ptrs.hits_a_reduced, n_hit_pairs);
  MemoryScheduler::allocate(&d_ptrs.hits_b_reduced, n_hit_pairs);

  unsigned total = getMax(&d_ptrs.hit_sum_offsets, ModuleMap::num_doublets());

  cudaDeviceSynchronize();

  MemoryScheduler::allocate(&d_ptrs.hit_sum, total);

  // MemoryScheduler::free(&d_total);

  dim3 grid_dim(65535);
  dim3 block_dim(32, 32);
  DeviceInitHitPairs<<<grid_dim, block_dim>>>(
      ModuleMap::num_doublets(), *mm.d_doublets(), d_ptrs.hit_offsets,
      d_ptrs.hit_pairs_offsets, d_ptrs.hits_a, d_ptrs.hits_b);

  MemoryScheduler::allocate(&d_ptrs.hit_module_sum, ModuleMap::num_doublets());

  unsigned long long *d_hit_triplets_offsets;
  MemoryScheduler::allocate(&d_hit_triplets_offsets, ModuleMap::num_triplets());
  MemoryScheduler::memset(&d_hit_triplets_offsets, ModuleMap::num_triplets(),
                          (unsigned long long)0);

  DeviceCalculateHitTriplets<<<65535, 1>>>(ModuleMap::num_triplets(),
                                           *mm.d_triplets(), d_ptrs.hit_offsets,
                                           d_hit_triplets_offsets);

  cudaDeviceSynchronize();
  unsigned long long *d_hit_triplets_sum;
  MemoryScheduler::allocate(&d_hit_triplets_sum, ModuleMap::num_triplets() + 1);
  MemoryScheduler::memset(&d_hit_triplets_sum, ModuleMap::num_triplets() + 1,
                          (unsigned long long)0);

  PrefixSum(&d_hit_triplets_offsets, &d_hit_triplets_sum,
            ModuleMap::num_triplets());

  cudaDeviceSynchronize();

  print_counter5<<<1,1>>>();

  // Get the total number of hit triplets
  n_hit_triplets = getMax(&d_hit_triplets_sum, ModuleMap::num_triplets());
  
  print_counter5<<<1,1>>>();
  cudaDeviceSynchronize();
}

void EventData::print_summary() {
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

void EventData::allocate_device_memory_hits(unsigned n_hits) {
  MemoryScheduler::allocate(&d_ptrs.hits, n_hits);
  MemoryScheduler::allocate(&d_ptrs.hit_offsets, ModuleMap::num_modules() + 1);
  MemoryScheduler::allocate(&d_ptrs.hit_module_sum, ModuleMap::num_doublets());
  MemoryScheduler::allocate(&d_ptrs.hit_module_offsets,
                            ModuleMap::num_doublets() + 1);
}

bool EventData::test_hit_sort(unsigned** d_hit_ind, unsigned n_hits) {
  unsigned *h_hit_ind = new unsigned[n_hits];

  CUDA_WARN(cudaMemcpy(h_hit_ind, *d_hit_ind, n_hits * sizeof(unsigned),
                       cudaMemcpyDeviceToHost));

  for (unsigned i = 0; i < n_hits - 1; i++) {
    if (h_hit_ind[i] > h_hit_ind[i + 1]) {
      std::cout << "\033[31m" << "Hit sort failed at " << i << "\033[0m" << std::endl;
      return false;
    }
  }

  delete[] h_hit_ind;
  std::cout << "\033[32m" << "Hit sort passed" << "\033[0m" << std::endl;
  return true;
}

bool EventData::test_hit_offsets(std::vector<unsigned>& h_hit_offsets, unsigned** d_hit_offsets, unsigned n_modules) {
  unsigned *h_hit_offsets_host = new unsigned[n_modules + 1];

  CUDA_WARN(cudaMemcpy(h_hit_offsets_host, *d_hit_offsets, (n_modules + 1) * sizeof(unsigned),
                       cudaMemcpyDeviceToHost));

  for (unsigned i = 0; i < n_modules; i++) {
    if (h_hit_offsets_host[i + 1] - h_hit_offsets_host[i] != h_hit_offsets[i]) {
      std::cout << "\033[31m" << "Hit offsets failed at " << i << "\033[0m" << std::endl;
      return false;
    }
  }

  delete[] h_hit_offsets_host;
  std::cout << "\033[32m" << "Hit offsets passed" << "\033[0m" << std::endl;
  return true;
}

bool EventData::test_hit_pairs_offsets(unsigned** d_n_hit_pairs, unsigned** d_hit_pairs_offsets, unsigned num_doublets) {
  unsigned *h_n_hit_pairs = new unsigned[num_doublets];
  unsigned *h_hit_pairs_offsets = new unsigned[num_doublets + 1];

  CUDA_WARN(cudaMemcpy(h_n_hit_pairs, *d_n_hit_pairs, num_doublets * sizeof(unsigned),
                       cudaMemcpyDeviceToHost));
  CUDA_WARN(cudaMemcpy(h_hit_pairs_offsets, *d_hit_pairs_offsets, (num_doublets + 1) * sizeof(unsigned),
                       cudaMemcpyDeviceToHost));

  for (unsigned i = 0; i < num_doublets; i++) {
    if (h_hit_pairs_offsets[i + 1] - h_hit_pairs_offsets[i] != h_n_hit_pairs[i]) {
      std::cout << "\033[31m" << "Hit pairs offsets failed at " << i << "\033[0m" << std::endl;
      return false;
    }
  }

  delete[] h_n_hit_pairs;
  delete[] h_hit_pairs_offsets;
  std::cout << "\033[32m" << "Hit pairs offsets passed" << "\033[0m" << std::endl;
  return true;
}

bool EventData::test_hit_sum_offsets(unsigned** d_n_hit_a, unsigned** d_hit_sum_offsets, unsigned num_doublets) {
  unsigned *h_n_hit_a = new unsigned[num_doublets];
  unsigned *h_hit_sum_offsets = new unsigned[num_doublets + 1];

  CUDA_WARN(cudaMemcpy(h_n_hit_a, *d_n_hit_a, num_doublets * sizeof(unsigned),
                       cudaMemcpyDeviceToHost));
  CUDA_WARN(cudaMemcpy(h_hit_sum_offsets, *d_hit_sum_offsets, (num_doublets + 1) * sizeof(unsigned),
                       cudaMemcpyDeviceToHost));

  for (unsigned i = 0; i < num_doublets; i++) {
    if (h_hit_sum_offsets[i + 1] - h_hit_sum_offsets[i] != h_n_hit_a[i]) {
      std::cout << "\033[31m" << "Hit sum offsets failed at " << i << "\033[0m" << std::endl;
      return false;
    }
  }

  delete[] h_n_hit_a;
  delete[] h_hit_sum_offsets;
  std::cout << "\033[32m" << "Hit sum offsets passed" << "\033[0m" << std::endl;
  return true;
}