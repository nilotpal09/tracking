#include "device/include/selection.cuh"

/*
__device__ bool doublet_selection(
    Hit& hit_bottom,
    Hit& hit_top,
    ModuleDoublet& mod_doub
) {
    return true;
}

__device__ bool triplet_selection(
    Hit& hit_bottom,
    Hit& hit_top,
    ModuleTriplet& mod_trip
) {
    return true;
}

__device__ float Dphi(const float &x1, const float &y1, const float& x2, const float& y2) {
  float phi1 = Phi(x1,y1);
  float phi2 = Phi(x2,y2);
  float dphi = phi2 - phi1;

    if (dphi>CUDART_PI_F) dphi-=2*CUDART_PI_F;
    if (dphi<-CUDART_PI_F) dphi+=2*CUDART_PI_F;

    return dphi;
}

__device__ float Dr(const float& x1, const float& y1, const float& x2, const float& y2) {
  float r1 = R(x1,y1);
  float r2 = R(x2,y2);
  return r2 - r1;
}

__device__ float PhiSlope(const float& x1, const float& y1, const float& x2, const float& y2) {
  float dphi = Dphi(x1,y1,x2,y2);
  float dr   = Dr(x1,y1,x2,y2);

  float phi_slope = -999;

  if( fabs(dr)>0 ){
      phi_slope = dphi / dr; 
  } else {
    if (dphi > 0) {
      // set to max value of float
      phi_slope = 3.402823466e+38F; 
    } else if (dphi < 0) {
      // set to min value of float
      phi_slope = -3.402823466e+38F; 
    } else {
      // set to 0
      phi_slope = 0; 
    }
  }
  return phi_slope;
}

__device__ float Dz(const float &z1, const float &z2) { return z2 - z1; }

__device__ float Z0(const float& x1, const float& y1, const float& z1,
          const float& x2, const float& y2, const float& z2) {
  float dz = Dz(z1,z2);
    float dr = Dr(x1,y1,x2,y2);
    float r1 = R(x1,y1);

    float z0 = -999;

    if( fabs(dr)>0 ){
        z0 = z1 - r1 * dz / dr;
    } else {
      z0 = dz >= 0 ? 3.402823466e+38F : -3.402823466e+38F;
    }
    return z0;
}

__device__ float Deta(const float& x1, const float& y1, const float& z1,
            const float& x2, const float& y2, const float& z2) {
  float eta1 = Eta(x1,y1,z1);
  float eta2 = Eta(x2,y2,z2);
  // TODO: understand why we do 1-2, while for the other (dphi, dz) we do 2-1
  return eta1 - eta2;
}

__device__ float DiffDzDr(const float& x1, const float& x2, const float& x3,
            const float& y1, const float& y2, const float& y3,
            const float& z1, const float& z2, const float& z3) {
  float dz_12 = Dz(z1,z2);
    float dz_23 = Dz(z2,z3);
    float dr_12 = Dr(x1, y1, x2, y2);
    float dr_23 = Dr(x2, y2, x3, y3);

    float diff_dzdr = 999;

    if (dr_12!=0 && dr_23!=0) {
        diff_dzdr = (dz_12 / dr_12) - (dz_23 / dr_23);
    } else if (dr_12 == 0 && dr_23 == 0) {
      diff_dzdr = 0;
    } else if (dr_12 == 0) {
      diff_dzdr = ((-dz_23 / dr_23) >= 0) ? 3.402823466e+38F : -3.402823466e+38F;
    } else {
      diff_dzdr = ((dz_12 / dr_12) >= 0) ? 3.402823466e+38F : -3.402823466e+38F;
    }
    return diff_dzdr;
}

__device__ float DiffDzDr(const float& dz_12, const float& dz_23,
            const float& dr_12, const float& dr_23) {
  float diff_dzdr = 999;

    if (dr_12!=0 && dr_23!=0) {
        diff_dzdr = (dz_12 / dr_12) - (dz_23 / dr_23);
    } else if (dr_12 == 0 && dr_23 == 0) {
      diff_dzdr = 0;
    } else if (dr_12 == 0) {
      diff_dzdr = ((-dz_23 / dr_23) >= 0) ? 3.402823466e+38F : -3.402823466e+38F;
    } else {
      diff_dzdr = ((dz_12 / dr_12) >= 0) ? 3.402823466e+38F : -3.402823466e+38F;
    }
    return diff_dzdr;
}

__device__ float DiffDyDx(const float& x1, const float& x2, const float& x3,
            const float& y1, const float& y2, const float& y3) {
  float dy_12 = y2 - y1;
    float dy_23 = y2 - y3;
    float dx_12 = x1 - x2;
    float dx_23 = x2 - x3;

    float diff_dydx= 999;
    if (dx_12!=0 && dx_23!=0) {
        diff_dydx = (dy_12 / dx_12) - (dy_23 / dx_23);
    } else if (dx_12 == 0 && dx_23 == 0) {
      diff_dydx = 0;
    } else if (dx_12 == 0) {
      diff_dydx = ((-dy_23 / dx_23) >= 0) ? 3.402823466e+38F : -3.402823466e+38F;
    } else {
      diff_dydx = ((dy_12 / dx_12) >= 0) ? 3.402823466e+38F : -3.402823466e+38F;
    }
    return diff_dydx;
}

__device__ float DiffDyDx(const float& dy_12, const float& dy_23,
            const float& dx_12, const float& dx_23) {
  float diff_dydx= 999;
    if (dx_12!=0 && dx_23!=0) {
        diff_dydx = (dy_12 / dx_12) - (dy_23 / dx_23);
    } else if (dx_12 == 0 && dx_23 == 0) {
      diff_dydx = 0;
    } else if (dx_12 == 0) {
      diff_dydx = ((-dy_23 / dx_23) >= 0) ? 3.402823466e+38F : -3.402823466e+38F;
    } else {
      diff_dydx = ((dy_12 / dx_12) >= 0) ? 3.402823466e+38F : -3.402823466e+38F;
    }
}
*/