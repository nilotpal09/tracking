#ifndef __DATATYPE_CUH__
#define __DATATYPE_CUH__

#include "../include/memoryscheduler.cuh"
#include "../include/utility.cuh"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <stdio.h>
#include <stdlib.h>
#include <tuple>
#include <vector>

// print numbers with formatting
#include <iomanip>
#include <locale>

struct ModuleTriplet {
  unsigned module_a;
  unsigned module_b;
  unsigned module_c;

  unsigned pair_i;
  unsigned pair_j;
};

struct ModuleDoublet {
  unsigned module_a;
  unsigned module_b;
};

__global__ void DeviceCalculateHitPairs(unsigned n_mod_pairs,
                                        ModuleDoublet *d_mod_doublets,
                                        unsigned *d_hit_offsets,
                                        unsigned *d_hit_pairs_offsets,
                                        unsigned *d_hit_sum_offsets);
__global__ void DeviceInitHitPairs(unsigned n_mod_pairs,
                                   ModuleDoublet *d_mod_doublets,
                                   unsigned *d_hit_offsets,
                                   unsigned *hits_pairs_acc_offsets,
                                   unsigned *d_hits_a, unsigned *d_hits_b);

__global__ void getMax(unsigned *max, unsigned n_mod_pairs,
                       unsigned *d_hit_pairs_offsets);

class ModuleMap {

private:
  // Counted from file
  static constexpr unsigned n_mod = 18360;
  static constexpr unsigned n_mod_pairs = 509461;
  static constexpr unsigned n_mod_triplets = 1242265;

  /*
   * Device pointers
   */
  struct DevicePtrs {
    ModuleDoublet *doublets;
    ModuleTriplet *triplets;
  };

  DevicePtrs d_ptrs;

  std::string mm_path;
  std::vector<ModuleDoublet> h_mod_doublets;
  std::vector<ModuleTriplet> h_mod_triplets;

  void allocate_device_memory();

public:
  static constexpr unsigned num_modules() { return n_mod; }
  static constexpr unsigned num_doublets() { return n_mod_pairs; }
  static constexpr unsigned num_triplets() { return n_mod_triplets; }

  // constructor
  ModuleMap(std::string mm_path_arg, std::string mm_pairs_path);

  void print_summary();

  /*
    Device pointers
  */
  ModuleDoublet **d_doublets() { return &d_ptrs.doublets; }

  ModuleTriplet **d_triplets() { return &d_ptrs.triplets; }
};

struct Hit {
  float x;
  float y;
  float z;
};

class EventData {

private:
  struct DevicePtrs {
    Hit *hits;
    unsigned *hit_offsets;
    unsigned *hit_pairs;
    unsigned *hit_pairs_offsets;
    unsigned *hits_a;
    unsigned *hits_b;
    unsigned *hits_a_reduced;
    unsigned *hits_b_reduced;
    unsigned *hit_sum;
    unsigned *hit_sum_offsets;
    unsigned *hit_module_sum;
    unsigned *hit_module_offsets;
  };

  DevicePtrs d_ptrs;

  std::string event_path;

  std::vector<unsigned> h_hit_inds;
  std::vector<Hit> h_hits;
  std::vector<unsigned> h_hit_offsets;
  std::vector<unsigned> h_hit_pairs_offsets;
  unsigned n_hits;
  unsigned n_hit_pairs;

  void allocate_device_memory_hits(unsigned n_hits);

public:
  // constructor
  EventData(std::string event_path_arg, ModuleMap &mm, unsigned int n_modules);

  void print_summary();

  unsigned num_hit_pairs() { return n_hit_pairs; }

  /*
    Device pointers
  */
  Hit **d_hits() { return &d_ptrs.hits; }

  unsigned **d_hit_offsets() { return &d_ptrs.hit_offsets; }

  unsigned **d_hit_pairs() { return &d_ptrs.hit_pairs; }

  unsigned **d_hit_pairs_offsets() { return &d_ptrs.hit_pairs_offsets; }

  unsigned **d_hits_a() { return &d_ptrs.hits_a; }

  unsigned **d_hits_b() { return &d_ptrs.hits_b; }

  unsigned **d_hits_a_reduced() { return &d_ptrs.hits_a_reduced; }

  unsigned **d_hits_b_reduced() { return &d_ptrs.hits_b_reduced; }

  unsigned **d_hit_sum() { return &d_ptrs.hit_sum; }

  unsigned **d_hit_sum_offsets() { return &d_ptrs.hit_sum_offsets; }

  unsigned **d_hit_module_sum() { return &d_ptrs.hit_module_sum; }

  unsigned **d_hit_module_offsets() { return &d_ptrs.hit_module_offsets; }
};

#endif