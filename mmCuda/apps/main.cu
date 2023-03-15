#include "../include/datatypes.cuh"
#include "../include/triplet_finder.cuh"

#include <cuda_profiler_api.h>

void graph_builder_baseline(
    ModuleMap& mm,
    EventData& event
) {
    dim3 grid_dim(65535);
    dim3 block_dim(16, 8, 8);
    triplet_finding_baseline<<<grid_dim, block_dim>>>(
        ModuleMap::num_triplets(), *mm.d_triplets(), *event.d_hits(),
        *event.d_hit_offsets(), *event.d_hit_pairs(),
      *event.d_hit_pairs_offsets());
    cudaDeviceSynchronize();
    CUDA_CHECK();
}


void graph_builder(
  ModuleMap& mm,
  EventData& event
) {
  /*
    Doublet finder
  */
  dim3 grid_dim(65535);
  dim3 block_dim(32, 32);
  // Call doublet finding kernal
  doublet_finding<<<grid_dim, block_dim>>>(
      ModuleMap::num_doublets(), *mm.d_doublets(), *event.d_hits(),
      *event.d_hit_offsets(), *event.d_hit_pairs(),
      *event.d_hit_pairs_offsets(), *event.d_hit_sum(),
      *event.d_hit_sum_offsets(), *event.d_hit_module_sum());

  cudaDeviceSynchronize();

  CUDA_CHECK();

  /*
    Reduce hit pairs
  */
  ReduceFlagged(
    event.d_hit_pairs(),
    event.d_hits_a(),
    event.d_hits_a_reduced(),
    event.num_hit_pairs()
  );

  ReduceFlagged(
    event.d_hit_pairs(),
    event.d_hits_b(),
    event.d_hits_b_reduced(),
    event.num_hit_pairs()
  );

  cudaDeviceSynchronize();
  CUDA_CHECK();

  /*
    Prefix sum number of pairs per module doublet
  */
  MemoryScheduler::memset(event.d_hit_module_offsets(),
                          ModuleMap::num_doublets() + 1, (unsigned)0);
  PrefixSum(
    event.d_hit_module_sum(),
    event.d_hit_module_offsets(),
    ModuleMap::num_doublets()
  );

  unsigned* total;
  cudaMalloc(&total, 1 * sizeof(unsigned));
  getMax<<<1,1>>>(total, ModuleMap::num_doublets(),
                   *event.d_hit_module_offsets());

  MemoryScheduler::free(event.d_hit_module_sum());

  cudaDeviceSynchronize();
  CUDA_CHECK();

  /*
    Triplet finder
  */
  triplet_finding<<<grid_dim,32>>>(
      ModuleMap::num_triplets(), *mm.d_triplets(), *event.d_hits(),
      *event.d_hit_offsets(), *event.d_hit_pairs(), *event.d_hit_sum(),
      *event.d_hit_sum_offsets(), *event.d_hits_a_reduced(),
      *event.d_hits_b_reduced(), *event.d_hit_module_offsets());

  cudaDeviceSynchronize();
  CUDA_CHECK();
}



int main(int argc, char *argv[]) {

  std::string mm_path{
      "/srv01/agrp/shieldse/storage/ml/trackingData/transformed_data/"
      "module_map/"
      "df_MMTriplet_3hits_ptCut1GeV_woutSec_woutOC_90kevents_woutElectron.csv"};
  std::string mm_pairs_path{"/srv01/agrp/shieldse/storage/ml/trackingData/"
                            "transformed_data/module_map/"
                            "df_MMTriplet_3hits_ptCut1GeV_woutSec_woutOC_"
                            "90kevents_woutElectron_pairs.csv"};
  ModuleMap mm = ModuleMap(mm_path, mm_pairs_path);
  mm.print_summary();

  CUDA_CHECK();

  std::string event_path{"/storage/agrp/nilotpal/tracking/transformed_data/"
                         "events/event000000001-truth.csv"};
  EventData event = EventData(event_path, mm, 18960);
  event.print_summary();

  CUDA_CHECK();

  graph_builder(mm, event);

  //graph_builder_baseline(mm, event);

  cudaProfilerStop();
}