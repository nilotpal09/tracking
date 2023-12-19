#include "../include/datatypes.cuh"
#include "../include/triplet_finder.cuh"

#include <cuda_profiler_api.h>

void graph_builder_baseline(ModuleMap &mm, EventData &event) {
  //print_all<<<1,1>>>(ModuleMap::num_triplets(), *mm.d_triplets(), *event.d_hits(),
  //    *event.d_hit_offsets(), *event.d_hit_pairs(),
  //    *event.d_hit_pairs_offsets());
  
  MemoryScheduler::memset(event.d_hit_pairs(),
                          event.num_hit_pairs(), (unsigned)0);
  dim3 grid_dim(65535);
  dim3 block_dim(8, 8, 8);
  triplet_finding_baseline<<<grid_dim, block_dim>>>(
      ModuleMap::num_triplets(), *mm.d_triplets(), *event.d_hits(),
      *event.d_hit_offsets(), *event.d_hit_pairs(),
      *event.d_hit_pairs_offsets());

  cudaDeviceSynchronize();
  CUDA_CHECK();

  unsigned *d_hit_totals;
  MemoryScheduler::allocate(&d_hit_totals, event.num_hit_pairs() + 1);
  PrefixSum(event.d_hit_pairs(), &d_hit_totals, event.num_hit_pairs());
  unsigned max = getMax(&d_hit_totals, event.num_hit_pairs());
  print_counter4<<<1,1>>>();
  
}

void graph_builder(ModuleMap &mm, EventData &event) {
  cudaSetDevice(0);
  cudaDeviceSynchronize();
  // Doublet finder
  // --------------
  dim3 grid_dim(65535);
  dim3 block_dim(512, 1);
  // Call doublet finding kernal
  doublet_finding<<<grid_dim, block_dim>>>(
      ModuleMap::num_doublets(), *mm.d_doublets(), *event.d_hits(),
      *event.d_hit_offsets(), *event.d_hit_pairs(),
      *event.d_hit_pairs_offsets(), *event.d_hit_sum(),
      *event.d_hit_sum_offsets(), *event.d_hit_module_sum());
  cudaDeviceSynchronize();
  CUDA_CHECK();

  unsigned *d_hit_totals;
  MemoryScheduler::allocate(&d_hit_totals, event.num_hit_pairs() + 1);
  PrefixSum(event.d_hit_pairs(), &d_hit_totals, event.num_hit_pairs());
  unsigned max = getMax(&d_hit_totals, event.num_hit_pairs());

  print_counter<<<1,1>>>();
  print_counter2<<<1,1>>>();
  

  cudaDeviceSynchronize();
  CUDA_CHECK();

  // Reduce hit pairs
  // ----------------
  ReduceFlagged(event.d_hit_pairs(), event.d_hits_a(), event.d_hits_a_reduced(),
                event.num_hit_pairs());

  ReduceFlagged(event.d_hit_pairs(), event.d_hits_b(), event.d_hits_b_reduced(),
                event.num_hit_pairs());

  cudaDeviceSynchronize();
  CUDA_CHECK();

  // Prefix sum number of pairs per module doublet
  // ---------------------------------------------
  MemoryScheduler::memset(event.d_hit_module_offsets(),
                          ModuleMap::num_doublets() + 1, (unsigned)0);
  PrefixSum(event.d_hit_module_sum(), event.d_hit_module_offsets(),
            ModuleMap::num_doublets());

  unsigned total =
      getMax(event.d_hit_module_offsets(), ModuleMap::num_doublets());

  MemoryScheduler::free(event.d_hit_module_sum());

  cudaDeviceSynchronize();
  CUDA_CHECK();
  
  dim3 block_dim2(32, 32);
  count_reduced_triplets<<<grid_dim,block_dim2>>>(
      ModuleMap::num_triplets(), *mm.d_triplets(), *mm.d_doublets(),
      *event.d_hits(),
      *event.d_hit_offsets(),
      *event.d_hits_a_reduced(), *event.d_hits_b_reduced(),
      *event.d_hit_module_offsets(),
      *event.d_hit_sum(), *event.d_hit_sum_offsets());
  cudaDeviceSynchronize();
  CUDA_CHECK();
  print_counter3<<<1,1>>>();
  

  MemoryScheduler::memset(event.d_hit_pairs(),
                          event.num_hit_pairs(), (unsigned)0);
  // Triplet finder
  // --------------
  triplet_finding<<<grid_dim, block_dim>>>(
      ModuleMap::num_triplets(), *mm.d_triplets(), *event.d_hits(),
      *event.d_hit_offsets(), *event.d_hit_pairs(),
      *event.d_hit_pairs_offsets(),
      *event.d_hits_a_reduced(), *event.d_hits_b_reduced(),
      *event.d_hit_module_offsets(),
      *event.d_hit_sum(), *event.d_hit_sum_offsets());
  print_counter4<<<1,1>>>();
  cudaDeviceSynchronize();
  CUDA_CHECK();

  //unsigned *d_hit_pair_totals;
  //MemoryScheduler::allocate(&d_hit_pair_totals, event.num_hit_pairs() + 1);
  //PrefixSum(event.d_hit_pairs(), &d_hit_pair_totals, event.num_hit_pairs());
  //unsigned new_max = getMax(&d_hit_pair_totals, event.num_hit_pairs());
  
}

int main(int argc, char *argv[]) {

  std::string mm_path{"/srv01/agrp/shieldse/storage/data/trackingData/transformed_data/module_map/SortedMMtripletMerged_3hits_ptCut1GeV_woutSec_events_0kto90k_woutElectron_woutOverlapSP.csv"};
  std::string mm_pairs_path{"/srv01/agrp/shieldse/storage/data/trackingData/transformed_data/module_map/SortedMMtripletMerged_3hits_ptCut1GeV_woutSec_events_0kto90k_woutElectron_woutOverlapSP_loose_pairs.csv"};
  ModuleMap mm = ModuleMap(mm_path, mm_pairs_path);
  mm.print_summary();

  CUDA_CHECK();

  std::string event_path{"/srv01/agrp/shieldse/storage/data/trackingData/"
                         "transformed_data/event000000001-truth.csv"};
  EventData event = EventData(event_path, mm, 18960);
  event.print_summary();
  cudaDeviceSynchronize();
  CUDA_CHECK();

  // test_PrefixSum();

  // test_Sort();

  // test_ReduceFlagged();

  graph_builder(mm, event);

  // MemoryScheduler::memset(event.d_hit_pairs(), event.num_hit_pairs(),
  // (unsigned)0);

  //graph_builder_baseline(mm, event);

  cudaDeviceSynchronize();
}