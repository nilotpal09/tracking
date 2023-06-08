#ifndef __EVENT__
#define __EVENT__

struct Hit {
  float x;
  float y;
  float z;
  float r;
  float eta;
  float phi;
};

struct Event {
    float *x;
    float *y;
    float *z;
    float *r;
    float *eta;
    float *phi;

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

#endif