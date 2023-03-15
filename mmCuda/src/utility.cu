#include "../include/utility.cuh"

// Module coordinates
__device__ __host__ float Phi(const float &x, const float &y) {
  return atan2(y, x);
}

__device__ __host__ float R(const float &x, const float &y) {
  return sqrtf(x * x + y * y);
}
__device__ __host__ float Eta(const float &x, const float &y, const float &z) {
  float r = R(x, y);
  float r3 = sqrtf(r * r + z * z);
  float theta = acos(z / r3);

  return -log(tan(theta * 0.5));
}