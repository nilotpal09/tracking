#include "host/include/module_map_reader.cuh"

ModuleMapReader::ModuleMapReader(std::string mm_path_arg, std::string mm_pairs_path_arg) {

  h_mod_doublets.reserve(ModuleMap::num_doublets());
  h_mod_triplets.reserve(ModuleMap::num_triplets());

  mm_path = mm_path_arg;
  mm_pairs_path = mm_pairs_path_arg;
}

ModuleMap ModuleMapReader::read() {
  std::string mm_line;
  std::string delim = " ";

  std::vector<unsigned> m_ids;
  m_ids.resize(2);
  std::vector<float> m_cuts;
  m_cuts.resize(10);
  std::vector<unsigned> m_pairs_sum;
  std::vector<unsigned> m_idx_b;
  m_pairs_sum.resize(ModuleMap::num_modules());
  m_idx_b.reserve(ModuleMap::num_doublets());

  // Module doublets
  //--------------------------------------------------
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
  for (int i = 1; i < ModuleMap::num_modules(); i++) {
    m_pairs_sum[i] += m_pairs_sum[i - 1];
  }
  // Add zero to the beginning
  m_pairs_sum.insert(m_pairs_sum.begin(), 0);

  m_ids.resize(5);
  m_cuts.resize(26);


  // Module triplets
  //--------------------------------------------------
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
      unsigned pair_ij = findPairIndex(m_ids[0], m_ids[1], m_pairs_sum, m_idx_b);
      unsigned pair_jk = findPairIndex(m_ids[1], m_ids[2], m_pairs_sum, m_idx_b);

      ModuleTriplet trip;
      trip.module_i = m_ids[0];
      trip.module_j = m_ids[1];
      trip.module_k = m_ids[2];
      trip.doublet_ij_idx = pair_ij;
      trip.doublet_jk_idx = pair_jk;
      //ModuleTriplet trip{
      //    m_ids[0],   m_ids[1],   m_ids[2],   pair_i,   pair_j,
      //    m_cuts[4],  m_cuts[5],  m_cuts[6],  m_cuts[7],  m_cuts[8],
      //    m_cuts[9],  m_cuts[10], m_cuts[11], m_cuts[12], m_cuts[13],
      //    m_cuts[14], m_cuts[15], m_cuts[16], m_cuts[17], m_cuts[18],
      //    m_cuts[19], m_cuts[20], m_cuts[21], m_cuts[22], m_cuts[23]};
      h_mod_triplets.push_back(trip);
    }
    mm_file.close();
  } else {
    std::cout << "Unable to open file" << mm_path << std::endl;
  }
  std::cout << "Formed module triplets" << std::endl;

  // Allocate device memory
  //--------------------------------------------------
  ModuleMap mm;

  MemoryScheduler::allocate(mm.doublets(), ModuleMap::num_doublets());
  MemoryScheduler::allocate(mm.triplets(), ModuleMap::num_triplets());


  CUDA_WARN(cudaMemcpy(*mm.doublets(), h_mod_doublets.data(),
                       ModuleMap::num_doublets() * sizeof(ModuleDoublet),
                       cudaMemcpyHostToDevice));
  CUDA_WARN(cudaMemcpy(*mm.triplets(), h_mod_triplets.data(),
                       ModuleMap::num_triplets() * sizeof(ModuleTriplet),
                       cudaMemcpyHostToDevice));
  
  return mm;
}

void ModuleMapReader::print_summary() {
  std::cout << std::string(50, '-') << std::endl;
  std::cout << "\tModule map summary" << std::endl;
  std::cout << std::string(50, '-') << std::endl;

  std::cout << "Module map taken from -" << std::endl;
  std::cout << mm_path << std::endl << std::endl;

  std::cout.imbue(std::locale(""));
  std::cout << std::fixed << std::showpoint << std::setprecision(3);

  std::cout << "# triplets: " << ModuleMap::num_triplets() << std::endl;
  std::cout << "# doublets: " << ModuleMap::num_doublets() << std::endl;

  std::cout << std::string(50, '-') << std::endl;
}

unsigned ModuleMapReader::findPairIndex(unsigned a, unsigned b, std::vector<unsigned>& offsets, std::vector<unsigned>& indices) {
  unsigned offset = offsets[a];
  unsigned end = offsets[a + 1] - offset;

  unsigned idx;
  bool found = false;
  for (unsigned i = 0; i < end; i++) {
    if (indices[offset + i] == b) {
      idx = i;
      found = true;
    }
  }
  if (!found) {
    std::cout << "Pair not found" << std::endl;
  }
  //std::cout << "Pair " << a << " " << b << " found at " << offset + idx << std::endl;
  return offset + idx;
}