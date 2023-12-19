#ifndef __MODULE_MAP_READER__
#define __MODULE_MAP_READER__

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>

#include "common/include/utils.cuh"
#include "common/include/module_map.cuh"
#include "host/include/memory_scheduler.cuh"

class ModuleMapReader {
private:
  std::string mm_path;
  std::string mm_pairs_path;
  
  std::vector<ModuleDoublet> h_mod_doublets;
  std::vector<ModuleTriplet> h_mod_triplets;

  unsigned findPairIndex(unsigned a, unsigned b, std::vector<unsigned>& offsets, std::vector<unsigned>& indices);

public:
  // constructor
  ModuleMapReader(std::string mm_path_arg, std::string mm_pairs_path);

  ModuleMap read();

  void print_summary();

};

#endif