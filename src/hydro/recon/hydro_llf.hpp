//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro_llf.hpp
//  \brief Local Lax Friedrichs (LLF) reconstruction for hydro with donor cell recon.
//
//  Computes 1D fluxes using the LLF reconstruction, also known as Rusanov's method.
//  This flux is very diffusive, even more diffusive than HLLE, and so it is not
//  recommended for use in applications.  However, it is useful for testing, or for
//  problems where other Riemann solvers fail.
//  In AthenaPK it is mainly used for first order flux correction.
//
// REFERENCES:
// - E.F. Toro, "Riemann Solvers and numerical methods for fluid dynamics", 2nd ed.,
//   Springer-Verlag, Berlin, (1999) chpt. 10.

#ifndef HYDRO_LLF_HPP_
#define HYDRO_LLF_HPP_

// C++ headers
#include <algorithm> // max(), min()
#include <cmath>     // sqrt()

// Athena headers
#include "../../main.hpp"
#include "../hydro.hpp"
#include "recon.hpp"

using parthenon::ParArray4D;
using parthenon::Real;

template <>
struct Reconstruct<Fluid::euler, Reconstruction::llf> {
  static KOKKOS_INLINE_FUNCTION void Solve(const AdiabaticHydroEOS &eos, const int k,
                                           const int j, const int i, const int ivx,
                                           const VariablePack<Real> &prim,
                                           VariableFluxPack<Real> &cons) {

    const int ivy = IV1 + ((ivx - IV1) + 1) % 3;
    const int ivz = IV1 + ((ivx - IV1) + 2) % 3;

    Real gamma;
    gamma = eos.GetGamma();
    Real gm1 = gamma - 1.0;
    Real igm1 = 1.0 / gm1;
    
    Real q0[(NHYDRO)], q1[(NHYDRO)];
    Real f0[(NHYDRO)], f1[(NHYDRO)];
    Real right_eigenmatrix[(NHYDRO)][(NHYDRO)];
    Real left_eigenmatrix[(NHYDRO)][(NHYDRO)];
    Real v0[(NHYDRO)], v1[(NHYDRO)];
    Real g0[(NHYDRO)], g1[(NHYDRO)];
    Real llf_sum[(NHYDRO)];
    Real f_half[(NHYDRO)]; 

    //--- Step 0.  Load states into local variables:
        
    if (ivx == IV1) {
      q0[IDN] = cons(IDN, k, j, i - 1);
      q0[IM1] = cons(ivx, k, j, i - 1);
      q0[IM2] = cons(ivy, k, j, i - 1);
      q0[IM3] = cons(ivz, k, j, i - 1);
      q0[IEN] = cons(IEN, k, j, i - 1);
    } else if (ivx == IV2) {
      q0[IDN] = cons(IDN, k, j - 1, i);
      q0[IM1] = cons(ivx, k, j - 1, i);
      q0[IM2] = cons(ivy, k, j - 1, i);
      q0[IM3] = cons(ivz, k, j - 1, i);
      q0[IEN] = cons(IEN, k, j - 1, i);
    } else if (ivx == IV3) {
      q0[IDN] = cons(IDN, k - 1, j, i);
      q0[IM1] = cons(ivx, k - 1, j, i);
      q0[IM2] = cons(ivy, k - 1, j, i);
      q0[IM3] = cons(ivz, k - 1, j, i);
      q0[IEN] = cons(IEN, k - 1, j, i);
    }

    q1[IDN] = cons(IDN, k, j, i);
    q1[IM1] = cons(ivx, k, j, i);
    q1[IM2] = cons(ivy, k, j, i);
    q1[IM3] = cons(ivz, k, j, i);
    q1[IEN] = cons(IEN, k, j, i);

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

    //--- Step 2.  At each x_{i-1/2,j,k}: (using cells i-1 and i)
    //--- (a) Compute the average state w_{i-1/2,j,k} in the primitive variables:

    Real half_den = 0.5 * (q0[IDN] + q1[IDN]);
    Real half_vex = 0.5 * (vex0 + vex1);
    Real half_vey = 0.5 * (vey0 + vey1);
    Real half_vez = 0.5 * (vez0 + vez1);
    Real half_pre = 0.5 * (pre0 + pre1);

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

      v0[ii] = 0.0;
      v1[ii] = 0.0;
      
      g0[ii] = 0.0;
      g1[ii] = 0.0;
      
      for (int jj = 0; jj < NHYDRO; ++jj) {

        v0[ii] += left_eigenmatrix[ii][jj] * q0[jj];
        v1[ii] += left_eigenmatrix[ii][jj] * q1[jj];
        
        g0[ii] += left_eigenmatrix[ii][jj] * f0[jj];
        g1[ii] += left_eigenmatrix[ii][jj] * f1[jj];
          
      }
    }

    //--- (d) Perform a Lax-Friedrichs flux vector splitting for the characteristic variables:
    // llf_sum = 0.5 * [ (g0 + g1) + 1.1 * α^{m} * (v1 - v0) ] where α(m) = max_k | λ^{m} q_k | 
    // is the maximal wave speed of the m^{th} component of characteristic variables over all grid points

    Real a0 = std::sqrt((gamma * pre0) / q0[IDN]);
    Real a1 = std::sqrt((gamma * pre1) / q1[IDN]);

    Real max_eig_0 = std::max({std::abs(vex0-a0), std::abs(vex1-a1)});
    Real max_eig_1 = std::max({std::abs(vex0),    std::abs(vex1)});
    Real max_eig_2 = std::max({std::abs(vex0+a0), std::abs(vex1+a1)});

    Real alpha[NHYDRO] = {max_eig_0, max_eig_1, max_eig_1, max_eig_1, max_eig_2};

    
    for (int iii = 0; iii < NHYDRO; ++iii) {

      llf_sum[iii] = 0.5 * ((g0[iii] + g1[iii]) + 1.1 * alpha[iii] * (v1[iii] - v0[iii]));
      
    }

    //--- (e) Project the numerical flux back to the conserved variables
        
    for (int iiii = 0; iiii < NHYDRO; ++iiii) {
      f_half[iiii] = 0.0;
      for (int jjjj = 0; jjjj < NHYDRO; ++jjjj) {
        f_half[iiii] += right_eigenmatrix[iiii][jjjj] * llf_sum[jjjj];
      }
    }

    //--- Step 3.  Update flux at each x_{i-1/2,j,k}:

    cons.flux(ivx, IDN, k, j, i) = f_half[IDN];
    cons.flux(ivx, ivx, k, j, i) = f_half[IV1];
    cons.flux(ivx, ivy, k, j, i) = f_half[IV2];
    cons.flux(ivx, ivz, k, j, i) = f_half[IV3];
    cons.flux(ivx, IEN, k, j, i) = f_half[IEN];

  }
};

#endif // HYDRO_LLF_HPP_
