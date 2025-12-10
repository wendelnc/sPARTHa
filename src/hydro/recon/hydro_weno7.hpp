//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file hydro_weno7.hpp
//  \brief Lax-Friedrichs flux splitting for hydrodynamics
//
// Computes 1D Fluxes using a Finite Difference Lax-Friedrichs Flux Vector Splitting method
// Following Procedure 2.10 from the following paper:
// "Essentially Non-Oscillatory and Weighted Essentially Non-Oscillatory Schemes for Hyperbolic Conservation Laws"
// By: Chi-Wang Shu
// https://www3.nd.edu/~zxu2/acms60790S13/Shu-WENO-notes.pdf


#ifndef HYDRO_WENO7_HPP_
#define HYDRO_WENO7_HPP_

// C headers

// C++ headers
#include <algorithm> // max(), min()
#include <cmath>     // sqrt()
#include <iomanip>   // For std::setprecision
#include <fstream>   // For output

// Athena headers
#include "../../main.hpp"

#include "weno_recon.hpp"

using parthenon::ParArray4D;
using parthenon::Real;

//----------------------------------------------------------------------------------------
//! \fn void Hydro::LaxFriedrichsFlux
//  \brief The Lax-Friedrichs Flux Vector Splitting solver for hydrodynamics (adiabatic)

template <>
struct Reconstruct<Fluid::euler, Reconstruction::weno7> {
  static KOKKOS_INLINE_FUNCTION void
  Solve(parthenon::team_mbr_t const &member, const int k, const int j, const int il,
        const int iu, const int ivx, const parthenon::VariablePack<Real> &q,
        VariableFluxPack<Real> &cons, const AdiabaticHydroEOS &eos) {

    int ivy = IV1 + ((ivx - IV1) + 1) % 3;
    int ivz = IV1 + ((ivx - IV1) + 2) % 3;
    Real gamma;
    gamma = eos.GetGamma();
    Real gm1 = gamma - 1.0;
    Real igm1 = 1.0 / gm1;
    parthenon::par_for_inner(member, il, iu, [&](const int i) {
    
      Real q0[(NHYDRO)], q1[(NHYDRO)], q2[(NHYDRO)], q3[(NHYDRO)], q4[(NHYDRO)], q5[(NHYDRO)], q6[(NHYDRO)], q7[(NHYDRO)];
      Real f0[(NHYDRO)], f1[(NHYDRO)], f2[(NHYDRO)], f3[(NHYDRO)], f4[(NHYDRO)], f5[(NHYDRO)], f6[(NHYDRO)], f7[(NHYDRO)];
      Real eigenvalues[(NHYDRO)], right_eigenmatrix[(NHYDRO)][(NHYDRO)], left_eigenmatrix[(NHYDRO)][(NHYDRO)];
      Real vj0[NHYDRO], vj1[NHYDRO], vj2[NHYDRO], vj3[NHYDRO], vj4[NHYDRO], vj5[NHYDRO], vj6[NHYDRO], vj7[NHYDRO];
      Real gj0[NHYDRO], gj1[NHYDRO], gj2[NHYDRO], gj3[NHYDRO], gj4[NHYDRO], gj5[NHYDRO], gj6[NHYDRO], gj7[NHYDRO];
      Real g_p0[NHYDRO], g_p1[NHYDRO], g_p2[NHYDRO], g_p3[NHYDRO], g_p4[NHYDRO], g_p5[NHYDRO], g_p6[NHYDRO], g_p7[NHYDRO];
      Real g_m0[NHYDRO], g_m1[NHYDRO], g_m2[NHYDRO], g_m3[NHYDRO], g_m4[NHYDRO], g_m5[NHYDRO], g_m6[NHYDRO], g_m7[NHYDRO];
      Real weno_sum[NHYDRO];
      Real f_half[NHYDRO]; 
   
      //--- Step 0.  Load states into local variables:
        
      if (ivx == IV1){
        q0[IDN] = cons(IDN, k, j, i - 4);
        q0[IM1] = cons(ivx, k, j, i - 4);
        q0[IM2] = cons(ivy, k, j, i - 4);
        q0[IM3] = cons(ivz, k, j, i - 4);
        q0[IEN] = cons(IEN, k, j, i - 4);

        q1[IDN] = cons(IDN, k, j, i - 3);
        q1[IM1] = cons(ivx, k, j, i - 3);
        q1[IM2] = cons(ivy, k, j, i - 3);
        q1[IM3] = cons(ivz, k, j, i - 3);
        q1[IEN] = cons(IEN, k, j, i - 3);

        q2[IDN] = cons(IDN, k, j, i - 2);
        q2[IM1] = cons(ivx, k, j, i - 2);
        q2[IM2] = cons(ivy, k, j, i - 2);
        q2[IM3] = cons(ivz, k, j, i - 2);
        q2[IEN] = cons(IEN, k, j, i - 2);

        q3[IDN] = cons(IDN, k, j, i - 1);
        q3[IM1] = cons(ivx, k, j, i - 1);
        q3[IM2] = cons(ivy, k, j, i - 1);
        q3[IM3] = cons(ivz, k, j, i - 1);
        q3[IEN] = cons(IEN, k, j, i - 1);

        q4[IDN] = cons(IDN, k, j, i);
        q4[IM1] = cons(ivx, k, j, i);
        q4[IM2] = cons(ivy, k, j, i);
        q4[IM3] = cons(ivz, k, j, i);
        q4[IEN] = cons(IEN, k, j, i);

        q5[IDN] = cons(IDN, k, j, i + 1);
        q5[IM1] = cons(ivx, k, j, i + 1);
        q5[IM2] = cons(ivy, k, j, i + 1);
        q5[IM3] = cons(ivz, k, j, i + 1);
        q5[IEN] = cons(IEN, k, j, i + 1);  

        q6[IDN] = cons(IDN, k, j, i + 2);
        q6[IM1] = cons(ivx, k, j, i + 2);
        q6[IM2] = cons(ivy, k, j, i + 2);
        q6[IM3] = cons(ivz, k, j, i + 2);
        q6[IEN] = cons(IEN, k, j, i + 2); 
        
        q7[IDN] = cons(IDN, k, j, i + 3);
        q7[IM1] = cons(ivx, k, j, i + 3);
        q7[IM2] = cons(ivy, k, j, i + 3);
        q7[IM3] = cons(ivz, k, j, i + 3);
        q7[IEN] = cons(IEN, k, j, i + 3);             
      }
        
      if (ivx == IV2){
        q0[IDN] = cons(IDN, k, j - 4, i);
        q0[IM1] = cons(ivx, k, j - 4, i);
        q0[IM2] = cons(ivy, k, j - 4, i);
        q0[IM3] = cons(ivz, k, j - 4, i);
        q0[IEN] = cons(IEN, k, j - 4, i);

        q1[IDN] = cons(IDN, k, j - 3, i);
        q1[IM1] = cons(ivx, k, j - 3, i);
        q1[IM2] = cons(ivy, k, j - 3, i);
        q1[IM3] = cons(ivz, k, j - 3, i);
        q1[IEN] = cons(IEN, k, j - 3, i );
    
        q2[IDN] = cons(IDN, k, j - 2, i);
        q2[IM1] = cons(ivx, k, j - 2, i);
        q2[IM2] = cons(ivy, k, j - 2, i);
        q2[IM3] = cons(ivz, k, j - 2, i);
        q2[IEN] = cons(IEN, k, j - 2, i);

        q3[IDN] = cons(IDN, k, j - 1, i);
        q3[IM1] = cons(ivx, k, j - 1, i);
        q3[IM2] = cons(ivy, k, j - 1, i);
        q3[IM3] = cons(ivz, k, j - 1, i);
        q3[IEN] = cons(IEN, k, j - 1, i);

        q4[IDN] = cons(IDN, k, j, i);
        q4[IM1] = cons(ivx, k, j, i);
        q4[IM2] = cons(ivy, k, j, i);
        q4[IM3] = cons(ivz, k, j, i);
        q4[IEN] = cons(IEN, k, j, i);

        q5[IDN] = cons(IDN, k, j + 1, i);
        q5[IM1] = cons(ivx, k, j + 1, i);
        q5[IM2] = cons(ivy, k, j + 1, i);
        q5[IM3] = cons(ivz, k, j + 1, i);
        q5[IEN] = cons(IEN, k, j + 1, i);

        q6[IDN] = cons(IDN, k, j + 2, i);
        q6[IM1] = cons(ivx, k, j + 2, i);
        q6[IM2] = cons(ivy, k, j + 2, i);
        q6[IM3] = cons(ivz, k, j + 2, i);
        q6[IEN] = cons(IEN, k, j + 2, i);
        
        q7[IDN] = cons(IDN, k, j + 3, i);
        q7[IM1] = cons(ivx, k, j + 3, i);
        q7[IM2] = cons(ivy, k, j + 3, i);
        q7[IM3] = cons(ivz, k, j + 3, i);
        q7[IEN] = cons(IEN, k, j + 3, i);     
      }

      if (ivx == IV3){
        q0[IDN] = cons(IDN, k - 4, j, i);
        q0[IM1] = cons(ivx, k - 4, j, i);
        q0[IM2] = cons(ivy, k - 4, j, i);
        q0[IM3] = cons(ivz, k - 4, j, i);
        q0[IEN] = cons(IEN, k - 4, j, i);

        q1[IDN] = cons(IDN, k - 3, j, i);
        q1[IM1] = cons(ivx, k - 3, j, i);
        q1[IM2] = cons(ivy, k - 3, j, i);
        q1[IM3] = cons(ivz, k - 3, j, i);
        q1[IEN] = cons(IEN, k - 3, j, i );

        q2[IDN] = cons(IDN, k - 2, j, i);
        q2[IM1] = cons(ivx, k - 2, j, i);
        q2[IM2] = cons(ivy, k - 2, j, i);
        q2[IM3] = cons(ivz, k - 2, j, i);
        q2[IEN] = cons(IEN, k - 2, j, i);

        q3[IDN] = cons(IDN, k - 1, j, i);
        q3[IM1] = cons(ivx, k - 1, j, i);
        q3[IM2] = cons(ivy, k - 1, j, i);
        q3[IM3] = cons(ivz, k - 1, j, i);
        q3[IEN] = cons(IEN, k - 1, j, i);
   
        q4[IDN] = cons(IDN, k, j, i);
        q4[IM1] = cons(ivx, k, j, i);
        q4[IM2] = cons(ivy, k, j, i);
        q4[IM3] = cons(ivz, k, j, i);
        q4[IEN] = cons(IEN, k, j, i);

        q5[IDN] = cons(IDN, k + 1, j, i);
        q5[IM1] = cons(ivx, k + 1, j, i);
        q5[IM2] = cons(ivy, k + 1, j, i);
        q5[IM3] = cons(ivz, k + 1, j, i);
        q5[IEN] = cons(IEN, k + 1, j, i);

        q6[IDN] = cons(IDN, k + 2, j, i);
        q6[IM1] = cons(ivx, k + 2, j, i);
        q6[IM2] = cons(ivy, k + 2, j, i);
        q6[IM3] = cons(ivz, k + 2, j, i);
        q6[IEN] = cons(IEN, k + 2, j, i);

        q7[IDN] = cons(IDN, k + 3, j, i);
        q7[IM1] = cons(ivx, k + 3, j, i);
        q7[IM2] = cons(ivy, k + 3, j, i);
        q7[IM3] = cons(ivz, k + 3, j, i);
        q7[IEN] = cons(IEN, k + 3, j, i);
      }

      //--- Step 1.  Compute the physical flux at each grid point:

      //--- f0 ---
      Real irho0 = 1.0 / q0[IDN];
      Real vex0 = q0[IM1] * irho0;
      Real vey0 = q0[IM2] * irho0;
      Real vez0 = q0[IM3] * irho0;
      Real vnm0 = vex0 * vex0 + vey0 * vey0 + vez0 * vez0;
      Real pre0 = (gm1) * (q0[IEN] - 0.5*q0[IDN]*vnm0);

      f0[IDN] = q0[IM1];
      f0[IM1] = q0[IM1] * vex0 + pre0;
      f0[IM2] = q0[IM1] * vey0;
      f0[IM3] = q0[IM1] * vez0;
      f0[IEN] = vex0 * (q0[IEN] + pre0);

      //--- f1 ---
      Real irho1 = 1.0 / q1[IDN];
      Real vex1 = q1[IM1] * irho1;
      Real vey1 = q1[IM2] * irho1;
      Real vez1 = q1[IM3] * irho1;
      Real vnm1 = vex1 * vex1 + vey1 * vey1 + vez1 * vez1;
      Real pre1 = (gm1) * (q1[IEN] - 0.5*q1[IDN]*vnm1);

      f1[IDN] = q1[IM1];
      f1[IM1] = q1[IM1] * vex1 + pre1;
      f1[IM2] = q1[IM1] * vey1;
      f1[IM3] = q1[IM1] * vez1;
      f1[IEN] = vex1 * (q1[IEN] + pre1);

      //--- f2 ---
      Real irho2 = 1.0 / q2[IDN];
      Real vex2 = q2[IM1] * irho2;
      Real vey2 = q2[IM2] * irho2;
      Real vez2 = q2[IM3] * irho2;
      Real vnm2 = vex2 * vex2 + vey2 * vey2 + vez2 * vez2;
      Real pre2 = (gm1) * (q2[IEN] - 0.5*q2[IDN]*vnm2);

      f2[IDN] = q2[IM1];
      f2[IM1] = q2[IM1] * vex2 + pre2;
      f2[IM2] = q2[IM1] * vey2;
      f2[IM3] = q2[IM1] * vez2;
      f2[IEN] = vex2 * (q2[IEN] + pre2);

      //--- f3 ---
      Real irho3 = 1.0 / q3[IDN];
      Real vex3 = q3[IM1] * irho3;
      Real vey3 = q3[IM2] * irho3;
      Real vez3 = q3[IM3] * irho3;
      Real vnm3 = vex3 * vex3 + vey3 * vey3 + vez3 * vez3;
      Real pre3 = (gm1) * (q3[IEN] - 0.5*q3[IDN]*vnm3);

      f3[IDN] = q3[IM1];
      f3[IM1] = q3[IM1] * vex3 + pre3;
      f3[IM2] = q3[IM1] * vey3;
      f3[IM3] = q3[IM1] * vez3;
      f3[IEN] = vex3 * (q3[IEN] + pre3);

      //--- f4 ---
      Real irho4 = 1.0 / q4[IDN];
      Real vex4 = q4[IM1] * irho4;
      Real vey4 = q4[IM2] * irho4;
      Real vez4 = q4[IM3] * irho4;
      Real vnm4 = vex4 * vex4 + vey4 * vey4 + vez4 * vez4;
      Real pre4 = (gm1) * (q4[IEN] - 0.5*q4[IDN]*vnm4);

      f4[IDN] = q4[IM1];
      f4[IM1] = q4[IM1] * vex4 + pre4;
      f4[IM2] = q4[IM1] * vey4;
      f4[IM3] = q4[IM1] * vez4;
      f4[IEN] = vex4 * (q4[IEN] + pre4);

      //--- f5 ---
      Real irho5 = 1.0 / q5[IDN];
      Real vex5 = q5[IM1] * irho5;
      Real vey5 = q5[IM2] * irho5;
      Real vez5 = q5[IM3] * irho5;
      Real vnm5 = vex5 * vex5 + vey5 * vey5 + vez5 * vez5;
      Real pre5 = (gm1) * (q5[IEN] - 0.5*q5[IDN]*vnm5);

      f5[IDN] = q5[IM1];
      f5[IM1] = q5[IM1] * vex5 + pre5;
      f5[IM2] = q5[IM1] * vey5;
      f5[IM3] = q5[IM1] * vez5;
      f5[IEN] = vex5 * (q5[IEN] + pre5);

      //--- f6 ---
      Real irho6 = 1.0 / q6[IDN];
      Real vex6 = q6[IM1] * irho6;
      Real vey6 = q6[IM2] * irho6;
      Real vez6 = q6[IM3] * irho6;
      Real vnm6 = vex6 * vex6 + vey6 * vey6 + vez6 * vez6;
      Real pre6 = (gm1) * (q6[IEN] - 0.5*q6[IDN]*vnm6);

      f6[IDN] = q6[IM1];
      f6[IM1] = q6[IM1] * vex6 + pre6;
      f6[IM2] = q6[IM1] * vey6;
      f6[IM3] = q6[IM1] * vez6;
      f6[IEN] = vex6 * (q6[IEN] + pre6);

      //--- f7 ---
      Real irho7 = 1.0 / q7[IDN];
      Real vex7 = q7[IM1] * irho7;
      Real vey7 = q7[IM2] * irho7;
      Real vez7 = q7[IM3] * irho7;
      Real vnm7 = vex7 * vex7 + vey7 * vey7 + vez7 * vez7;
      Real pre7 = (gm1) * (q7[IEN] - 0.5*q7[IDN]*vnm7);

      f7[IDN] = q7[IM1];
      f7[IM1] = q7[IM1] * vex7 + pre7;
      f7[IM2] = q7[IM1] * vey7;
      f7[IM3] = q7[IM1] * vez7;
      f7[IEN] = vex7 * (q7[IEN] + pre7);

      //--- Step 2.  At each x_{i-1/2,j,k}: (using cells i-1 and i)
      //--- (a) Compute the average state w_{i-1/2,j,k} in the primitive variables:

      Real half_den = 0.5 * (q3[IDN] + q4[IDN]);
      Real half_vex = 0.5 * (vex3 + vex4);
      Real half_vey = 0.5 * (vey3 + vey4);
      Real half_vez = 0.5 * (vez3 + vez4);
      Real half_pre = 0.5 * (pre3 + pre4);

      Real half_vsq = half_vex * half_vex + half_vey * half_vey + half_vez * half_vez;
      Real half_E = half_pre / (gm1) + (0.5 * half_den * half_vsq);
      Real half_H = (half_E + half_pre) / half_den;
      Real half_a = std::sqrt((gamma * half_pre) / half_den);

      //--- (b) Compute the right and left eigenvectors of the flux Jacobian matrix, ∂f/∂x, at x = x_{i+1/2,j,k}:

      // Right-eigenvectors, stored as COLUMNS (eq. B3)
      right_eigenmatrix[0][0] = 1.0;
      right_eigenmatrix[1][0] = half_vex - half_a;
      right_eigenmatrix[2][0] = half_vey;
      right_eigenmatrix[3][0] = half_vez;
      right_eigenmatrix[4][0] = half_H - half_vex * half_a;

      right_eigenmatrix[0][1] = 0.0;
      right_eigenmatrix[1][1] = 0.0;
      right_eigenmatrix[2][1] = 1.0;
      right_eigenmatrix[3][1] = 0.0;
      right_eigenmatrix[4][1] = half_vey;

      right_eigenmatrix[0][2] = 0.0;
      right_eigenmatrix[1][2] = 0.0;
      right_eigenmatrix[2][2] = 0.0;
      right_eigenmatrix[3][2] = 1.0;
      right_eigenmatrix[4][2] = half_vez;

      right_eigenmatrix[0][3] = 1.0;
      right_eigenmatrix[1][3] = half_vex;
      right_eigenmatrix[2][3] = half_vey;
      right_eigenmatrix[3][3] = half_vez;
      right_eigenmatrix[4][3] = 0.5 * half_vsq;

      right_eigenmatrix[0][4] = 1.0;
      right_eigenmatrix[1][4] = half_vex + half_a;
      right_eigenmatrix[2][4] = half_vey;
      right_eigenmatrix[3][4] = half_vez;
      right_eigenmatrix[4][4] = half_H + half_vex * half_a;

      // Left-eigenvectors, stored as ROWS (eq. B4)
      Real na = 0.5 / (half_a*half_a);
      left_eigenmatrix[0][0] = na * (0.5 * gm1 * half_vsq + half_vex * half_a);
      left_eigenmatrix[0][1] = -na * (gm1 * half_vex + half_a);
      left_eigenmatrix[0][2] = -na * gm1 * half_vey;
      left_eigenmatrix[0][3] = -na * gm1 * half_vez;
      left_eigenmatrix[0][4] = na * gm1;

      left_eigenmatrix[1][0] = -half_vey;
      left_eigenmatrix[1][1] = 0.0;
      left_eigenmatrix[1][2] = 1.0;
      left_eigenmatrix[1][3] = 0.0;
      left_eigenmatrix[1][4] = 0.0;

      left_eigenmatrix[2][0] = -half_vez;
      left_eigenmatrix[2][1] = 0.0;
      left_eigenmatrix[2][2] = 0.0;
      left_eigenmatrix[2][3] = 1.0;
      left_eigenmatrix[2][4] = 0.0;
        
      Real qa = gm1 / (half_a*half_a);
      left_eigenmatrix[3][0] = 1.0 - na * gm1 * half_vsq;
      left_eigenmatrix[3][1] = qa * half_vex;
      left_eigenmatrix[3][2] = qa * half_vey;
      left_eigenmatrix[3][3] = qa * half_vez;
      left_eigenmatrix[3][4] = -qa;

      left_eigenmatrix[4][0] = na * (0.5 * gm1 * half_vsq - half_vex * half_a);
      left_eigenmatrix[4][1] = -na * (gm1 * half_vex - half_a);
      left_eigenmatrix[4][2] = left_eigenmatrix[0][2];
      left_eigenmatrix[4][3] = left_eigenmatrix[0][3];
      left_eigenmatrix[4][4] = left_eigenmatrix[0][4];

      //--- (c) Project the solution and physical flux into the right eigenvector space:

      for (int ii = 0; ii < NHYDRO; ++ii) {

        vj0[ii] = 0.0;
        vj1[ii] = 0.0;
        vj2[ii] = 0.0;
        vj3[ii] = 0.0;
        vj4[ii] = 0.0;
        vj5[ii] = 0.0;
        vj6[ii] = 0.0;
        vj7[ii] = 0.0;

        gj0[ii] = 0.0;
        gj1[ii] = 0.0;
        gj2[ii] = 0.0;
        gj3[ii] = 0.0;
        gj4[ii] = 0.0;
        gj5[ii] = 0.0;
        gj6[ii] = 0.0;
        gj7[ii] = 0.0;

        for (int jj = 0; jj < NHYDRO; ++jj) {

          vj0[ii] += left_eigenmatrix[ii][jj] * q0[jj];
          vj1[ii] += left_eigenmatrix[ii][jj] * q1[jj];
          vj2[ii] += left_eigenmatrix[ii][jj] * q2[jj];
          vj3[ii] += left_eigenmatrix[ii][jj] * q3[jj];
          vj4[ii] += left_eigenmatrix[ii][jj] * q4[jj];
          vj5[ii] += left_eigenmatrix[ii][jj] * q5[jj];
          vj6[ii] += left_eigenmatrix[ii][jj] * q6[jj];
          vj7[ii] += left_eigenmatrix[ii][jj] * q7[jj];

          gj0[ii] += left_eigenmatrix[ii][jj] * f0[jj];
          gj1[ii] += left_eigenmatrix[ii][jj] * f1[jj];
          gj2[ii] += left_eigenmatrix[ii][jj] * f2[jj];
          gj3[ii] += left_eigenmatrix[ii][jj] * f3[jj];
          gj4[ii] += left_eigenmatrix[ii][jj] * f4[jj];
          gj5[ii] += left_eigenmatrix[ii][jj] * f5[jj];
          gj6[ii] += left_eigenmatrix[ii][jj] * f6[jj];
          gj7[ii] += left_eigenmatrix[ii][jj] * f7[jj];

        }
      }


      //--- (d) Perform a Lax-Friedrichs flux vector splitting for each component of the characteristic variables:
      // Specifically, assume that the mth components of Vj and Gj are vj and gj, respectively, then compute
      // g^{±}_{j}= 0.5 * (g_j ± α^{m} v_j) where α(m) = max_k | λ^{m} q_k | 
      // is the maximal wave speed of the m^{th} component of characteristic variables over all grid points

      Real a0 = std::sqrt((gamma * pre0) / q0[IDN]);
      Real a1 = std::sqrt((gamma * pre1) / q1[IDN]);
      Real a2 = std::sqrt((gamma * pre2) / q2[IDN]);
      Real a3 = std::sqrt((gamma * pre3) / q3[IDN]);
      Real a4 = std::sqrt((gamma * pre4) / q4[IDN]);
      Real a5 = std::sqrt((gamma * pre5) / q5[IDN]);
      Real a6 = std::sqrt((gamma * pre6) / q6[IDN]);
      Real a7 = std::sqrt((gamma * pre7) / q7[IDN]);

      Real max_eig_0 = std::max({std::abs(vex0-a0), std::abs(vex1-a1), std::abs(vex2-a2), std::abs(vex3-a3), std::abs(vex4-a4), std::abs(vex5-a5), std::abs(vex6-a6), std::abs(vex7-a7)});
      Real max_eig_1 = std::max({std::abs(vex0),    std::abs(vex1),    std::abs(vex2),    std::abs(vex3),    std::abs(vex4),    std::abs(vex5),    std::abs(vex6),    std::abs(vex7)});
      Real max_eig_2 = std::max({std::abs(vex0+a0), std::abs(vex1+a1), std::abs(vex2+a2), std::abs(vex3+a3), std::abs(vex4+a4), std::abs(vex5+a5), std::abs(vex6+a6), std::abs(vex7+a7)});

      Real alpha[NHYDRO] = {max_eig_0, max_eig_1, max_eig_1, max_eig_1, max_eig_2};

      for (int iii = 0; iii < NHYDRO; ++iii) {

        g_p0[iii] = 0.5 * (gj0[iii] + 1.1 * alpha[iii] * vj0[iii]);
        g_p1[iii] = 0.5 * (gj1[iii] + 1.1 * alpha[iii] * vj1[iii]);  
        g_p2[iii] = 0.5 * (gj2[iii] + 1.1 * alpha[iii] * vj2[iii]);  
        g_p3[iii] = 0.5 * (gj3[iii] + 1.1 * alpha[iii] * vj3[iii]);  
        g_p4[iii] = 0.5 * (gj4[iii] + 1.1 * alpha[iii] * vj4[iii]);  
        g_p5[iii] = 0.5 * (gj5[iii] + 1.1 * alpha[iii] * vj5[iii]);
        g_p6[iii] = 0.5 * (gj6[iii] + 1.1 * alpha[iii] * vj6[iii]);
        // g_p7[iii] = 0.5 * (gj7[iii] + 1.1 * alpha[iii] * vj7[iii]);  // not needed for WENO

        // g_m0[iii] = 0.5 * (gj0[iii] - 1.1 * alpha[iii] * vj0[iii]);   // not needed for WENO 
        g_m1[iii] = 0.5 * (gj1[iii] - 1.1 * alpha[iii] * vj1[iii]);  
        g_m2[iii] = 0.5 * (gj2[iii] - 1.1 * alpha[iii] * vj2[iii]);  
        g_m3[iii] = 0.5 * (gj3[iii] - 1.1 * alpha[iii] * vj3[iii]);  
        g_m4[iii] = 0.5 * (gj4[iii] - 1.1 * alpha[iii] * vj4[iii]);  
        g_m5[iii] = 0.5 * (gj5[iii] - 1.1 * alpha[iii] * vj5[iii]);  
        g_m6[iii] = 0.5 * (gj6[iii] - 1.1 * alpha[iii] * vj6[iii]);  
        g_m7[iii] = 0.5 * (gj7[iii] - 1.1 * alpha[iii] * vj7[iii]);  

      }

      //--- (e) Perform a WENO reconstruction on each of the computed flux components gj± to obtain 
      // the corresponding component of the numerical flux

      for (int jjj = 0; jjj < NHYDRO; ++jjj) {     

        weno_sum[jjj] = WENO7(g_p0[jjj], g_p1[jjj], g_p2[jjj], g_p3[jjj], g_p4[jjj], g_p5[jjj], g_p6[jjj]) 
                      + WENO7(g_m7[jjj], g_m6[jjj], g_m5[jjj], g_m4[jjj], g_m3[jjj], g_m2[jjj], g_m1[jjj]);  

      }

      //--- (f) Project the numerical flux back to the conserved variables
        
      for (int iiii = 0; iiii < NHYDRO; ++iiii) {
        f_half[iiii] = 0.0;
        for (int jjjj = 0; jjjj < NHYDRO; ++jjjj) {
          f_half[iiii] += right_eigenmatrix[iiii][jjjj] * weno_sum[jjjj];
        }
      }

      //--- Step 3.  Update flux at each x_{i-1/2,j,k}:

      cons.flux(ivx, IDN, k, j, i) = f_half[IDN];
      cons.flux(ivx, ivx, k, j, i) = f_half[IV1];
      cons.flux(ivx, ivy, k, j, i) = f_half[IV2];
      cons.flux(ivx, ivz, k, j, i) = f_half[IV3];
      cons.flux(ivx, IEN, k, j, i) = f_half[IEN];
      
    });
  }
};

#endif // HYDRO_WENO7_HPP_            