#ifndef __HIT_CONTAINERS__
#define __HIT_CONTAINERS__

#include <cub/cub.cuh>
#include "common/include/event.cuh"

struct HitContainer {
  // Structure of arrays for hits
  //--------------------------------
  float *x;
  float *y;
  float *z;
  float *r;
  float *phi;
  float *eta;
};

#endif