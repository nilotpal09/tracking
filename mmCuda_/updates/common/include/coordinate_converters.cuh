#ifndef __COORDINARE_CONVERTERS__
#define __COORDINARE_CONVERTERS__

#include <math_constants.h>

__device__ __host__ inline float Phi(const float &x, const float &y) {
    return atan2(y, x);
}

__device__ __host__ inline float R(const float &x, const float &y) {
    return sqrtf(x * x + y * y);
}

__device__ __host__ inline float Eta(const float &x, const float &y, const float &z) {
    float r = R(x, y);
    float r3 = sqrtf(r * r + z * z);
    float theta = acos(z / r3);

    return -log(tan(theta * 0.5));
}

#endif