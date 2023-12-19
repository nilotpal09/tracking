#ifndef __MODULE_MAP__
#define __MODULE_MAP__

struct ModuleDoublet {
    unsigned module_i;
    unsigned module_j;

    float z0_min;
    float z0_max;
    float dphi_min;
    float dphi_max;
    float phi_slope_min;
    float phi_slope_max;
    float deta_min;
    float deta_max;
};

struct ModuleTriplet {
  unsigned module_i;
  unsigned module_j;
  unsigned module_k;

  unsigned doublet_ij_idx;
  unsigned doublet_jk_idx;

  ModuleDoublet doublet_ij;
  ModuleDoublet doublet_jk;

  float diff_dzdr_max;
  float diff_dzdr_min;
  float diff_dydx_max;
  float diff_dydx_min;
};

struct ModuleMap {
  static constexpr unsigned n_mod = 18360;
  static constexpr unsigned n_mod_pairs = 296502;
  static constexpr unsigned n_mod_triplets = 1242265;

  static constexpr unsigned num_modules() { return n_mod; }
  static constexpr unsigned num_doublets() { return n_mod_pairs; }
  static constexpr unsigned num_triplets() { return n_mod_triplets; }
  
  ModuleDoublet *_doublets;
  ModuleTriplet *_triplets;

  ModuleDoublet **doublets() { return &_doublets; }
  ModuleTriplet **triplets() { return &_triplets; }
};

#endif