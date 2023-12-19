#ifndef __DEVICE_DEFINITIONS__
#define __DEVICE_DEFINITIONS__

#include <cub/cub.cuh>

struct HitContainer {
  /*
  Structure of arrays for hits
  */
  float *x;
  float *y;
  float *z;
  float *r;
  float *phi;
  float *eta;

  unsigned n_hits;
};

struct Hit {
  float x;
  float y;
  float z;
  float r;
  float phi;
  float eta;
};

/*
  I think this needs to be split into two classes
*/
template <unsigned SIZE, typename ALGORITHM> struct HitSubContainer {
  using WarpLoadT = WarpLoad<unsigned, 1, ALGORITHM, SIZE>;
  __shared__ typename WarpLoadT::TempStorage temp_storage[SIZE];

  const HitContainer &_hc;
  const unsigned _wid;
  WarpLoadT _loader;

  HitSubContainer() = delete;
  HitSubContainer(const HitContainer &hc, const unsigned tid)
      : _hc(hc), _wid(tid / SIZE), _loader(temp_storage[_wid]) {}

  inline __device__ void load(const unsigned offset,
                              const unsigned max = SIZE) {
    x() = _loader.Load(_hc.x + offset, _wid, max);
    y() = _loader.Load(_hc.y + offset, _wid, max);
    z() = _loader.Load(_hc.z + offset, _wid, max);
    r() = _loader.Load(_hc.r + offset, _wid, max);
    phi() = _loader.Load(_hc.phi + offset, _wid, max);
    eta() = _loader.Load(_hc.eta + offset, _wid, max);
  }

  // This is private memory in the register
  Hit _hit;
  Hit &hit() { return _hit; }

  inline __device__ float &x() { return _hit.x; }
  inline __device__ float &y() { return _hit.y; }
  inline __device__ float &z() { return _hit.z; }
  inline __device__ float &r() { return _hit.r; }
  inline __device__ float &phi() { return _hit.phi; }
  inline __device__ float &eta() { return _hit.eta; }
};

#endif