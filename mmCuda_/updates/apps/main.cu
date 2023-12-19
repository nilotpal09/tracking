#include "host/include/module_map_reader.cuh"
#include "host/include/event_reader.cuh"
#include "host/include/memory_scheduler.cuh"
#include "common/include/utils.cuh"
#include "device/include/doublet_finder.cuh"
#include "device/include/triplet_finder.cuh"


int main(int argc, char *argv[]) {

  std::string mm_path{"/srv01/agrp/shieldse/storage/data/trackingData/transformed_data/module_map/SortedMMtripletMerged_3hits_ptCut1GeV_woutSec_events_0kto90k_woutElectron_woutOverlapSP.csv"};
  std::string mm_pairs_path{"/srv01/agrp/shieldse/storage/data/trackingData/transformed_data/module_map/SortedMMtripletMerged_3hits_ptCut1GeV_woutSec_events_0kto90k_woutElectron_woutOverlapSP_loose_pairs.csv"};
  
  ModuleMapReader reader = ModuleMapReader(mm_path, mm_pairs_path);
  ModuleMap mm = reader.read();
  reader.print_summary();

  CUDA_CHECK();

  std::string event_path{"/srv01/agrp/shieldse/storage/data/trackingData/"
                         "transformed_data/event000000001-truth.csv"};
  EventReader event_reader = EventReader(event_path);
  Event event = event_reader.read(mm, ModuleMap::num_modules());
  event_reader.print_summary();
  cudaDeviceSynchronize();
  CUDA_CHECK();

  doublet_finder(event, mm);
  cudaDeviceSynchronize();
  print_counter(0);
  print_counter(1);
  print_counter(2);
  cudaDeviceSynchronize();
  CUDA_CHECK();
}