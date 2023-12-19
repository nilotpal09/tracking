#ifndef __EVENT_READER__
#define __EVENT_READER__

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>

#include "common/include/event.cuh"
#include "common/include/module_map.cuh"
#include "common/include/coordinate_converters.cuh"
#include "common/include/utils.cuh"
#include "host/include/memory_scheduler.cuh"


__global__ void flatten_hits(
  unsigned n_hits,
  Hit* d_hits,
  float *d_x, float *d_y, float *d_z, float *d_eta, float *d_phi, float *d_r
);

__global__ void calculate_hit_pairs(
  unsigned n_mod_pairs,
  ModuleDoublet* module_doublets,
  unsigned *hit_offsets,
  unsigned *n_hit_pairs,
  unsigned *n_hit_a
);

__global__ void init_hit_pairs(
  unsigned n_mod_pairs,
  ModuleDoublet *d_mod_doublets,
  unsigned *d_hit_offsets,
  unsigned *hits_pairs_acc_offsets,
  unsigned *d_hits_a, unsigned *d_hits_b
);


class EventReader {
private:
  std::string event_path;

  std::vector<unsigned> h_hit_inds;
  std::vector<Hit> h_hits;
  std::vector<unsigned> h_hit_offsets;
  std::vector<unsigned> h_hit_pairs_offsets;

  unsigned n_hits;
  unsigned n_hit_pairs;
  unsigned long long n_hit_triplets;

  
  void allocate_device_memory_hits(Event& event, unsigned n_hits);

public:
  // constructor
  EventReader(std::string event_path_arg);

  Event read(ModuleMap &mm, unsigned n_modules);

  void print_summary();

  unsigned num_hit_pairs() { return n_hit_pairs; }
};

#endif