// This file is part of the Acts project.
//
// Copyright (C) 2017-2018 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <cmath>
#include <limits>
#include <TMath.h>

namespace ActsExamples{

// Module coordinates
float Phi(const float& x, const float& y);
float R(const float& x, const float& y);
float Eta(const float& x, const float& y, const float& z);

// Edge features
float Dphi(const float& x1, const float& y1, const float& x2, const float& y2);
float Dr(const float& x1, const float& y1, const float& x2, const float& y2);
float Phi_slope(const float& x1, const float& y1, const float& x2, const float& y2);
float Dz(const float& z1, const float& z2);
float Z0(const float& x1, const float& y1, const float& z1,
          const float& x2, const float& y2, const float& z2);
float Deta(const float& x1, const float& y1, const float& z1,
            const float& x2, const float& y2, const float& z2);
float DiffDzDr(const float& x1, const float& x2, const float& x3,
            const float& y1, const float& y2, const float& y3,
            const float& z1, const float& z2, const float& z3);
float DiffDzDr(const float& dz_12, const float& dz_23,
            const float& dr_12, const float& dr_23);
float DiffDyDx(const float& x1, const float& x2, const float& x3,
            const float& y1, const float& y2, const float& y3);
float DiffDyDx(const float& dy_12, const float& dy_23,
            const float& dx_12, const float& dx_23);

}


inline float ActsExamples::Phi(const float& x, const float& y)
{ 
    return atan2(y,x); 
}

inline float ActsExamples::R(const float& x, const float& y)
{ 
    return sqrt(x*x+y*y); 
}

inline float ActsExamples::Eta(const float& x, const float& y, const float& z)
{
    float r = R(x,y);
    float r3 = sqrt(r*r + z*z);
    float theta = acos(z/r3);

    return -log(tan(theta*0.5)); 
}

// =============
// Edge features
// =============
inline float ActsExamples::Dphi(const float& x1, const float& y1, const float& x2, const float& y2)
{
    float phi1 = Phi(x1,y1);
    float phi2 = Phi(x2,y2);
    float dphi = phi2 - phi1;

    if (dphi>TMath::Pi()) dphi-=2*TMath::Pi();
    if (dphi<-TMath::Pi()) dphi+=2*TMath::Pi();

    return dphi;
}

inline float ActsExamples::Dr(const float& x1, const float& y1, const float& x2, const float& y2)
{
    float r1 = R(x1,y1);
    float r2 = R(x2,y2);
    return r2 - r1;
}

inline float ActsExamples::Phi_slope(const float& x1, const float& y1, const float& x2, const float& y2)
{
    float dphi = Dphi(x1,y1,x2,y2);
    float dr   = Dr(x1,y1,x2,y2);

    float phi_slope = -999;

    if( fabs(dr)>0 ){
        phi_slope = dphi / dr; 
    }
    else{
        // if dr==0 but dphi non null : put float highest value 
        // if dr==0 && dphi == 0 : put 0 (phi do not change) 
        if( dphi>0 ) {
          phi_slope = std::numeric_limits<float>::max();
        }
        else if( dphi<0 ) { 
          phi_slope = -std::numeric_limits<float>::max();
        }
        else {
          phi_slope = 0;
        } 
    }

    return phi_slope;
}

inline float ActsExamples::Dz(const float& z1, const float& z2)
{ 
    return z2 -z1; 
}

inline float ActsExamples::Z0(const float& x1, const float& y1, const float& z1,
          const float& x2, const float& y2, const float& z2)
{
    float dz = Dz(z1,z2);
    float dr = Dr(x1,y1,x2,y2);
    float r1 = R(x1,y1);

    float z0 = -999;

    if( fabs(dr)>0 ){
        z0 = z1 - r1 * dz / dr;
    }
    else{
        // if dr==0, put float highest value instead of "inf"
        // +max() if z2 >= z1
        // -max() if z1 < z2 (we assume that in principle r2>r1)
        z0 = (dz>=0)? std::numeric_limits<float>::max() : -std::numeric_limits<float>::max();
    }

    return z0;
}

inline float ActsExamples::Deta(const float& x1, const float& y1, const float& z1,
            const float& x2, const float& y2, const float& z2)
{
    float eta1 = Eta(x1,y1,z1);
    float eta2 = Eta(x2,y2,z2);
    // TODO: understand why we do 1-2, while for the other (dphi, dz) we do 2-1
    return eta1 - eta2;
}


inline float ActsExamples::DiffDzDr(const float& x1, const float& x2, const float& x3,
            const float& y1, const float& y2, const float& y3,
            const float& z1, const float& z2, const float& z3)
{
    float dz_12 = Dz(z1,z2);
    float dz_23 = Dz(z2,z3);
    float dr_12 = Dr(x1, y1, x2, y2);
    float dr_23 = Dr(x2, y2, x3, y3);

    float diff_dzdr = 999;

    if (dr_12!=0 && dr_23!=0) {
        diff_dzdr = (dz_12 / dr_12) - (dz_23 / dr_23);
    }
    else if (dr_12 == 0 && dr_23 == 0){
        diff_dzdr = 0; 
    }
    else if (dr_12 == 0){
        diff_dzdr= ((-dz_23 / dr_23) >=0 )? std::numeric_limits<float>::max() : -std::numeric_limits<float>::max();
    }else{
        diff_dzdr= ((dz_12 / dr_12) >=0 )? std::numeric_limits<float>::max() : -std::numeric_limits<float>::max();
    }

    return diff_dzdr;
}

inline float ActsExamples::DiffDzDr(const float& dz_12, const float& dz_23,
            const float& dr_12, const float& dr_23)
{

    float diff_dzdr = 999;

    if (dr_12!=0 && dr_23!=0) {
        diff_dzdr = (dz_12 / dr_12) - (dz_23 / dr_23);
    }
    else if (dr_12 == 0 && dr_23 == 0){
        diff_dzdr = 0; 
    }
    else if (dr_12 == 0){
        diff_dzdr= ((-dz_23 / dr_23) >=0 )? std::numeric_limits<float>::max() : -std::numeric_limits<float>::max();
    }else{
        diff_dzdr= ((dz_12 / dr_12) >=0 )? std::numeric_limits<float>::max() : -std::numeric_limits<float>::max();
    }

    return diff_dzdr;
}

inline float ActsExamples::DiffDyDx(const float& x1, const float& x2, const float& x3,
            const float& y1, const float& y2, const float& y3)
{
    float dy_12 = y2 - y1;
    float dy_23 = y2 - y3;
    float dx_12 = x1 - x2;
    float dx_23 = x2 - x3;

    float diff_dydx= 999;
    if (dx_12!=0 && dx_23!=0) {
        diff_dydx = (dy_12 / dx_12) - (dy_23 / dx_23);
    }
    else if (dx_12 == 0 && dx_23 == 0){
        diff_dydx = 0;
    }
    else if (dx_12 == 0){
        diff_dydx= ((-dy_23 / dx_23) >=0 )? std::numeric_limits<float>::max() : -std::numeric_limits<float>::max();
    }else{
        diff_dydx= ((dy_12 / dx_12) >=0 )? std::numeric_limits<float>::max() : -std::numeric_limits<float>::max();
    }

    return diff_dydx;   
}

inline float ActsExamples::DiffDyDx(const float& dy_12, const float& dy_23,
            const float& dx_12, const float& dx_23)
{

    float diff_dydx= 999;
    if (dx_12!=0 && dx_23!=0) {
        diff_dydx = (dy_12 / dx_12) - (dy_23 / dx_23);
    }
    else if (dx_12 == 0 && dx_23 == 0){
        diff_dydx = 0;
    }
    else if (dx_12 == 0){
        diff_dydx= ((-dy_23 / dx_23) >=0 )? std::numeric_limits<float>::max() : -std::numeric_limits<float>::max();
    }else{
        diff_dydx= ((dy_12 / dx_12) >=0 )? std::numeric_limits<float>::max() : -std::numeric_limits<float>::max();
    }

    return diff_dydx;   
}