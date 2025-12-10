//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mhd_weno9.hpp
//  \brief Lax-Friedrichs flux splitting for hydrodynamics
//
// Computes 1D Fluxes using a Finite Difference Lax-Friedrichs Flux Vector Splitting method
// Following Procedure 2.10 from the following paper:
// "Essentially Non-Oscillatory and Weighted Essentially Non-Oscillatory Schemes for Hyperbolic Conservation Laws"
// By: Chi-Wang Shu
// https://www3.nd.edu/~zxu2/acms60790S13/Shu-WENO-notes.pdf


#ifndef MHD_WENO9_HPP_
#define MHD_WENO9_HPP_

// C headers

// C++ headers
#include <algorithm> // max(), min()
#include <cmath>     // sqrt()
#include <iomanip>   // For std::setprecision

// Athena headers
#include "../../main.hpp"

#include "weno_recon.hpp"

using parthenon::ParArray4D;
using parthenon::Real;

//----------------------------------------------------------------------------------------
//! \fn void MHD::LaxFriedrichsFlux
//  \brief The Lax-Friedrichs Flux Vector Splitting solver for magnetohydrodynamics (adiabatic)

template <>
struct Reconstruct<Fluid::mhd, Reconstruction::weno9> {
  static KOKKOS_INLINE_FUNCTION void
  Solve(parthenon::team_mbr_t const &member, const int k, const int j, const int il,
        const int iu, const int ivx, const parthenon::VariablePack<Real> &q,
        VariableFluxPack<Real> &cons, const AdiabaticMHDEOS &eos) {

    const int ivy = IV1 + ((ivx - IV1) + 1) % 3;
    const int ivz = IV1 + ((ivx - IV1) + 2) % 3;
    const int iBx = ivx - 1 + NHYDRO;
    const int iBy = ivy - 1 + NHYDRO;
    const int iBz = ivz - 1 + NHYDRO;

    constexpr int NMHD = 8;

    const auto gamma = eos.GetGamma();
    const auto gm1 = gamma - 1.0;
    const auto igm1 = 1.0 / gm1;
    
    parthenon::par_for_inner(member, il, iu, [&](const int i) {
   
      Real q0[(NMHD)], q1[(NMHD)], q2[(NMHD)], q3[(NMHD)], q4[(NMHD)], q5[(NMHD)], q6[(NMHD)], q7[(NMHD)], q8[(NMHD)], q9[(NMHD)];
      Real f0[(NMHD)], f1[(NMHD)], f2[(NMHD)], f3[(NMHD)], f4[(NMHD)], f5[(NMHD)], f6[(NMHD)], f7[(NMHD)], f8[(NMHD)], f9[(NMHD)];
      Real rr[(NMHD)][(NMHD)], ru[(NMHD)][(NMHD)], lu[(NMHD)][(NMHD)], lq[(NMHD)][(NMHD)];
      Real vj0[NMHD], vj1[NMHD], vj2[NMHD], vj3[NMHD], vj4[NMHD], vj5[NMHD], vj6[(NMHD)], vj7[(NMHD)], vj8[(NMHD)], vj9[(NMHD)];
      Real gj0[NMHD], gj1[NMHD], gj2[NMHD], gj3[NMHD], gj4[NMHD], gj5[NMHD], gj6[(NMHD)], gj7[(NMHD)], gj8[(NMHD)], gj9[(NMHD)];
      Real g_p0[NMHD], g_p1[NMHD], g_p2[NMHD], g_p3[NMHD], g_p4[NMHD], g_p5[NMHD], g_p6[(NMHD)], g_p7[(NMHD)], g_p8[(NMHD)], g_p9[(NMHD)];
      Real g_m0[NMHD], g_m1[NMHD], g_m2[NMHD], g_m3[NMHD], g_m4[NMHD], g_m5[NMHD], g_m6[(NMHD)], g_m7[(NMHD)], g_m8[(NMHD)], g_m9[(NMHD)];
      Real weno_sum[NMHD];
      Real f_half[NMHD];

      //--- Step 0.  Load states into local variables:
        
      if (ivx == IV1){
        q0[IDN] = cons(IDN, k, j, i - 5);
        q0[IM1] = cons(ivx, k, j, i - 5);
        q0[IM2] = cons(ivy, k, j, i - 5);
        q0[IM3] = cons(ivz, k, j, i - 5);
        q0[IEN] = cons(IEN, k, j, i - 5);
        q0[IB1] = cons(iBx, k, j, i - 5);
        q0[IB2] = cons(iBy, k, j, i - 5);
        q0[IB3] = cons(iBz, k, j, i - 5);

        q1[IDN] = cons(IDN, k, j, i - 4);
        q1[IM1] = cons(ivx, k, j, i - 4);
        q1[IM2] = cons(ivy, k, j, i - 4);
        q1[IM3] = cons(ivz, k, j, i - 4);
        q1[IEN] = cons(IEN, k, j, i - 4);
        q1[IB1] = cons(iBx, k, j, i - 4);
        q1[IB2] = cons(iBy, k, j, i - 4);
        q1[IB3] = cons(iBz, k, j, i - 4);

        q2[IDN] = cons(IDN, k, j, i - 3);
        q2[IM1] = cons(ivx, k, j, i - 3);
        q2[IM2] = cons(ivy, k, j, i - 3);
        q2[IM3] = cons(ivz, k, j, i - 3);
        q2[IEN] = cons(IEN, k, j, i - 3);
        q2[IB1] = cons(iBx, k, j, i - 3);
        q2[IB2] = cons(iBy, k, j, i - 3);
        q2[IB3] = cons(iBz, k, j, i - 3);

        q3[IDN] = cons(IDN, k, j, i - 2);
        q3[IM1] = cons(ivx, k, j, i - 2);
        q3[IM2] = cons(ivy, k, j, i - 2);
        q3[IM3] = cons(ivz, k, j, i - 2);
        q3[IEN] = cons(IEN, k, j, i - 2);
        q3[IB1] = cons(iBx, k, j, i - 2);
        q3[IB2] = cons(iBy, k, j, i - 2);
        q3[IB3] = cons(iBz, k, j, i - 2);

        q4[IDN] = cons(IDN, k, j, i - 1);
        q4[IM1] = cons(ivx, k, j, i - 1);
        q4[IM2] = cons(ivy, k, j, i - 1);
        q4[IM3] = cons(ivz, k, j, i - 1);
        q4[IEN] = cons(IEN, k, j, i - 1);
        q4[IB1] = cons(iBx, k, j, i - 1);
        q4[IB2] = cons(iBy, k, j, i - 1);
        q4[IB3] = cons(iBz, k, j, i - 1);

        q5[IDN] = cons(IDN, k, j, i);
        q5[IM1] = cons(ivx, k, j, i);
        q5[IM2] = cons(ivy, k, j, i);
        q5[IM3] = cons(ivz, k, j, i);
        q5[IEN] = cons(IEN, k, j, i);
        q5[IB1] = cons(iBx, k, j, i);
        q5[IB2] = cons(iBy, k, j, i);
        q5[IB3] = cons(iBz, k, j, i);

        q6[IDN] = cons(IDN, k, j, i + 1);
        q6[IM1] = cons(ivx, k, j, i + 1);
        q6[IM2] = cons(ivy, k, j, i + 1);
        q6[IM3] = cons(ivz, k, j, i + 1);
        q6[IEN] = cons(IEN, k, j, i + 1);
        q6[IB1] = cons(iBx, k, j, i + 1);
        q6[IB2] = cons(iBy, k, j, i + 1);
        q6[IB3] = cons(iBz, k, j, i + 1);       
        
        q7[IDN] = cons(IDN, k, j, i + 2);
        q7[IM1] = cons(ivx, k, j, i + 2);
        q7[IM2] = cons(ivy, k, j, i + 2);
        q7[IM3] = cons(ivz, k, j, i + 2);
        q7[IEN] = cons(IEN, k, j, i + 2);
        q7[IB1] = cons(iBx, k, j, i + 2);
        q7[IB2] = cons(iBy, k, j, i + 2);
        q7[IB3] = cons(iBz, k, j, i + 2);

        q8[IDN] = cons(IDN, k, j, i + 3);
        q8[IM1] = cons(ivx, k, j, i + 3);
        q8[IM2] = cons(ivy, k, j, i + 3);
        q8[IM3] = cons(ivz, k, j, i + 3);
        q8[IEN] = cons(IEN, k, j, i + 3);
        q8[IB1] = cons(iBx, k, j, i + 3);
        q8[IB2] = cons(iBy, k, j, i + 3);
        q8[IB3] = cons(iBz, k, j, i + 3);

        q9[IDN] = cons(IDN, k, j, i + 4);
        q9[IM1] = cons(ivx, k, j, i + 4);
        q9[IM2] = cons(ivy, k, j, i + 4);
        q9[IM3] = cons(ivz, k, j, i + 4);
        q9[IEN] = cons(IEN, k, j, i + 4);
        q9[IB1] = cons(iBx, k, j, i + 4);
        q9[IB2] = cons(iBy, k, j, i + 4);
        q9[IB3] = cons(iBz, k, j, i + 4);        
      }
        
      if (ivx == IV2){
        q0[IDN] = cons(IDN, k, j - 5, i);
        q0[IM1] = cons(ivx, k, j - 5, i);
        q0[IM2] = cons(ivy, k, j - 5, i);
        q0[IM3] = cons(ivz, k, j - 5, i);
        q0[IEN] = cons(IEN, k, j - 5, i);
        q0[IB1] = cons(iBx, k, j - 5, i);
        q0[IB2] = cons(iBy, k, j - 5, i);
        q0[IB3] = cons(iBz, k, j - 5, i);

        q1[IDN] = cons(IDN, k, j - 4, i);
        q1[IM1] = cons(ivx, k, j - 4, i);
        q1[IM2] = cons(ivy, k, j - 4, i);
        q1[IM3] = cons(ivz, k, j - 4, i);
        q1[IEN] = cons(IEN, k, j - 4, i);
        q1[IB1] = cons(iBx, k, j - 4, i);
        q1[IB2] = cons(iBy, k, j - 4, i);
        q1[IB3] = cons(iBz, k, j - 4, i);

        q2[IDN] = cons(IDN, k, j - 3, i);
        q2[IM1] = cons(ivx, k, j - 3, i);
        q2[IM2] = cons(ivy, k, j - 3, i);
        q2[IM3] = cons(ivz, k, j - 3, i);
        q2[IEN] = cons(IEN, k, j - 3, i);
        q2[IB1] = cons(iBx, k, j - 3, i);
        q2[IB2] = cons(iBy, k, j - 3, i);
        q2[IB3] = cons(iBz, k, j - 3, i); 

        q3[IDN] = cons(IDN, k, j - 2, i);
        q3[IM1] = cons(ivx, k, j - 2, i);
        q3[IM2] = cons(ivy, k, j - 2, i);
        q3[IM3] = cons(ivz, k, j - 2, i);
        q3[IEN] = cons(IEN, k, j - 2, i);
        q3[IB1] = cons(iBx, k, j - 2, i);
        q3[IB2] = cons(iBy, k, j - 2, i);
        q3[IB3] = cons(iBz, k, j - 2, i);

        q4[IDN] = cons(IDN, k, j - 1, i);
        q4[IM1] = cons(ivx, k, j - 1, i);
        q4[IM2] = cons(ivy, k, j - 1, i);
        q4[IM3] = cons(ivz, k, j - 1, i);
        q4[IEN] = cons(IEN, k, j - 1, i );
        q4[IB1] = cons(iBx, k, j - 1, i);
        q4[IB2] = cons(iBy, k, j - 1, i);
        q4[IB3] = cons(iBz, k, j - 1, i);

        q5[IDN] = cons(IDN, k, j, i);
        q5[IM1] = cons(ivx, k, j, i);
        q5[IM2] = cons(ivy, k, j, i);
        q5[IM3] = cons(ivz, k, j, i);
        q5[IEN] = cons(IEN, k, j, i);
        q5[IB1] = cons(iBx, k, j, i);
        q5[IB2] = cons(iBy, k, j, i);
        q5[IB3] = cons(iBz, k, j, i);

        q6[IDN] = cons(IDN, k, j + 1, i);
        q6[IM1] = cons(ivx, k, j + 1, i);
        q6[IM2] = cons(ivy, k, j + 1, i);
        q6[IM3] = cons(ivz, k, j + 1, i);
        q6[IEN] = cons(IEN, k, j + 1, i);
        q6[IB1] = cons(iBx, k, j + 1, i);
        q6[IB2] = cons(iBy, k, j + 1, i);
        q6[IB3] = cons(iBz, k, j + 1, i);

        q7[IDN] = cons(IDN, k, j + 2, i);
        q7[IM1] = cons(ivx, k, j + 2, i);
        q7[IM2] = cons(ivy, k, j + 2, i);
        q7[IM3] = cons(ivz, k, j + 2, i);
        q7[IEN] = cons(IEN, k, j + 2, i);
        q7[IB1] = cons(iBx, k, j + 2, i);
        q7[IB2] = cons(iBy, k, j + 2, i);
        q7[IB3] = cons(iBz, k, j + 2, i);

        q8[IDN] = cons(IDN, k, j + 3, i);
        q8[IM1] = cons(ivx, k, j + 3, i);
        q8[IM2] = cons(ivy, k, j + 3, i);
        q8[IM3] = cons(ivz, k, j + 3, i);
        q8[IEN] = cons(IEN, k, j + 3, i);
        q8[IB1] = cons(iBx, k, j + 3, i);
        q8[IB2] = cons(iBy, k, j + 3, i);
        q8[IB3] = cons(iBz, k, j + 3, i);

        q9[IDN] = cons(IDN, k, j + 4, i);
        q9[IM1] = cons(ivx, k, j + 4, i);
        q9[IM2] = cons(ivy, k, j + 4, i);
        q9[IM3] = cons(ivz, k, j + 4, i);
        q9[IEN] = cons(IEN, k, j + 4, i);
        q9[IB1] = cons(iBx, k, j + 4, i);
        q9[IB2] = cons(iBy, k, j + 4, i);
        q9[IB3] = cons(iBz, k, j + 4, i);
      }

      if (ivx == IV3){
        q0[IDN] = cons(IDN, k - 5, j, i);
        q0[IM1] = cons(ivx, k - 5, j, i);
        q0[IM2] = cons(ivy, k - 5, j, i);
        q0[IM3] = cons(ivz, k - 5, j, i);
        q0[IEN] = cons(IEN, k - 5, j, i);
        q0[IB1] = cons(iBx, k - 5, j, i);
        q0[IB2] = cons(iBy, k - 5, j, i);
        q0[IB3] = cons(iBz, k - 5, j, i);

        q1[IDN] = cons(IDN, k - 4, j, i);
        q1[IM1] = cons(ivx, k - 4, j, i);
        q1[IM2] = cons(ivy, k - 4, j, i);
        q1[IM3] = cons(ivz, k - 4, j, i);
        q1[IEN] = cons(IEN, k - 4, j, i);
        q1[IB1] = cons(iBx, k - 4, j, i);
        q1[IB2] = cons(iBy, k - 4, j, i);
        q1[IB3] = cons(iBz, k - 4, j, i);

        q2[IDN] = cons(IDN, k - 3, j, i);
        q2[IM1] = cons(ivx, k - 3, j, i);
        q2[IM2] = cons(ivy, k - 3, j, i);
        q2[IM3] = cons(ivz, k - 3, j, i);
        q2[IEN] = cons(IEN, k - 3, j, i);
        q2[IB1] = cons(iBx, k - 3, j, i);
        q2[IB2] = cons(iBy, k - 3, j, i);
        q2[IB3] = cons(iBz, k - 3, j, i);

        q3[IDN] = cons(IDN, k - 2, j, i);
        q3[IM1] = cons(ivx, k - 2, j, i);
        q3[IM2] = cons(ivy, k - 2, j, i);
        q3[IM3] = cons(ivz, k - 2, j, i);
        q3[IEN] = cons(IEN, k - 2, j, i);
        q3[IB1] = cons(iBx, k - 2, j, i);
        q3[IB2] = cons(iBy, k - 2, j, i);
        q3[IB3] = cons(iBz, k - 2, j, i);

        q4[IDN] = cons(IDN, k - 1, j, i);
        q4[IM1] = cons(ivx, k - 1, j, i);
        q4[IM2] = cons(ivy, k - 1, j, i);
        q4[IM3] = cons(ivz, k - 1, j, i);
        q4[IEN] = cons(IEN, k - 1, j, i);
        q4[IB1] = cons(iBx, k - 1, j, i);
        q4[IB2] = cons(iBy, k - 1, j, i);
        q4[IB3] = cons(iBz, k - 1, j, i);

        q5[IDN] = cons(IDN, k, j, i);
        q5[IM1] = cons(ivx, k, j, i);
        q5[IM2] = cons(ivy, k, j, i);
        q5[IM3] = cons(ivz, k, j, i);
        q5[IEN] = cons(IEN, k, j, i);
        q5[IB1] = cons(iBx, k, j, i);
        q5[IB2] = cons(iBy, k, j, i);
        q5[IB3] = cons(iBz, k, j, i);

        q6[IDN] = cons(IDN, k + 1, j, i);
        q6[IM1] = cons(ivx, k + 1, j, i);
        q6[IM2] = cons(ivy, k + 1, j, i);
        q6[IM3] = cons(ivz, k + 1, j, i);
        q6[IEN] = cons(IEN, k + 1, j, i);
        q6[IB1] = cons(iBx, k + 1, j, i);
        q6[IB2] = cons(iBy, k + 1, j, i);
        q6[IB3] = cons(iBz, k + 1, j, i);

        q7[IDN] = cons(IDN, k + 2, j, i);
        q7[IM1] = cons(ivx, k + 2, j, i);
        q7[IM2] = cons(ivy, k + 2, j, i);
        q7[IM3] = cons(ivz, k + 2, j, i);
        q7[IEN] = cons(IEN, k + 2, j, i);
        q7[IB1] = cons(iBx, k + 2, j, i);
        q7[IB2] = cons(iBy, k + 2, j, i);
        q7[IB3] = cons(iBz, k + 2, j, i);

        q8[IDN] = cons(IDN, k + 3, j, i);
        q8[IM1] = cons(ivx, k + 3, j, i);
        q8[IM2] = cons(ivy, k + 3, j, i);
        q8[IM3] = cons(ivz, k + 3, j, i);
        q8[IEN] = cons(IEN, k + 3, j, i);
        q8[IB1] = cons(iBx, k + 3, j, i);
        q8[IB2] = cons(iBy, k + 3, j, i);
        q8[IB3] = cons(iBz, k + 3, j, i);

        q9[IDN] = cons(IDN, k + 4, j, i);
        q9[IM1] = cons(ivx, k + 4, j, i);
        q9[IM2] = cons(ivy, k + 4, j, i);
        q9[IM3] = cons(ivz, k + 4, j, i);
        q9[IEN] = cons(IEN, k + 4, j, i);
        q9[IB1] = cons(iBx, k + 4, j, i);
        q9[IB2] = cons(iBy, k + 4, j, i);
        q9[IB3] = cons(iBz, k + 4, j, i);
      }

      //--- Step 1.  Compute the physical flux at each grid point:

      //--- f0 ---
      Real irho0 = 1.0 / q0[IDN];
      Real vex0 = q0[IM1] * irho0;
      Real vey0 = q0[IM2] * irho0;
      Real vez0 = q0[IM3] * irho0;
      Real Bnm0 = q0[IB1] * q0[IB1] + q0[IB2] * q0[IB2] + q0[IB3] * q0[IB3];
      Real vnm0 = vex0 * vex0 + vey0 * vey0 + vez0 * vez0;
      Real pre0 = (gm1) * (q0[IEN] - 0.5*(q0[IDN]*vnm0 + Bnm0));

      f0[IDN] = q0[IM1];
      f0[IM1] = q0[IM1] * vex0 + pre0 + 0.5 * Bnm0 - q0[IB1] * q0[IB1];
      f0[IM2] = q0[IM1] * vey0 - q0[IB1] * q0[IB2];
      f0[IM3] = q0[IM1] * vez0 - q0[IB1] * q0[IB3];
      f0[IEN] = vex0 * (q0[IEN] + pre0 + 0.5 * Bnm0) - q0[IB1] * (vex0 * q0[IB1] + vey0 * q0[IB2] + vez0 * q0[IB3]);
      f0[IB1] = 0.0;
      f0[IB2] = vex0 * q0[IB2] - vey0 * q0[IB1];
      f0[IB3] = vex0 * q0[IB3] - vez0 * q0[IB1];

      //--- f1 ---
      Real irho1 = 1.0 / q1[IDN];
      Real vex1 = q1[IM1] * irho1;
      Real vey1 = q1[IM2] * irho1;
      Real vez1 = q1[IM3] * irho1;
      Real Bnm1 = q1[IB1] * q1[IB1] + q1[IB2] * q1[IB2] + q1[IB3] * q1[IB3];
      Real vnm1 = vex1 * vex1 + vey1 * vey1 + vez1 * vez1;
      Real pre1 = (gm1) * (q1[IEN] - 0.5*(q1[IDN]*vnm1 + Bnm1));

      f1[IDN] = q1[IM1];
      f1[IM1] = q1[IM1] * vex1 + pre1 + 0.5 * Bnm1 - q1[IB1] * q1[IB1];
      f1[IM2] = q1[IM1] * vey1 - q1[IB1] * q1[IB2];
      f1[IM3] = q1[IM1] * vez1 - q1[IB1] * q1[IB3];
      f1[IEN] = vex1 * (q1[IEN] + pre1 + 0.5 * Bnm1) - q1[IB1] * (vex1 * q1[IB1] + vey1 * q1[IB2] + vez1 * q1[IB3]);
      f1[IB1] = 0.0;
      f1[IB2] = vex1 * q1[IB2] - vey1 * q1[IB1];
      f1[IB3] = vex1 * q1[IB3] - vez1 * q1[IB1];

      //--- f2 ---
      Real irho2 = 1.0 / q2[IDN];
      Real vex2 = q2[IM1] * irho2;
      Real vey2 = q2[IM2] * irho2;
      Real vez2 = q2[IM3] * irho2;
      Real Bnm2 = q2[IB1] * q2[IB1] + q2[IB2] * q2[IB2] + q2[IB3] * q2[IB3];
      Real vnm2 = vex2 * vex2 + vey2 * vey2 + vez2 * vez2;
      Real pre2 = (gm1) * (q2[IEN] - 0.5*(q2[IDN]*vnm2 + Bnm2));

      f2[IDN] = q2[IM1];
      f2[IM1] = q2[IM1] * vex2 + pre2 + 0.5 * Bnm2 - q2[IB1] * q2[IB1];
      f2[IM2] = q2[IM1] * vey2 - q2[IB1] * q2[IB2];
      f2[IM3] = q2[IM1] * vez2 - q2[IB1] * q2[IB3];
      f2[IEN] = vex2 * (q2[IEN] + pre2 + 0.5 * Bnm2) - q2[IB1] * (vex2 * q2[IB1] + vey2 * q2[IB2] + vez2 * q2[IB3]);
      f2[IB1] = 0.0;
      f2[IB2] = vex2 * q2[IB2] - vey2 * q2[IB1];
      f2[IB3] = vex2 * q2[IB3] - vez2 * q2[IB1];

      //--- f3 ---
      Real irho3 = 1.0 / q3[IDN];
      Real vex3 = q3[IM1] * irho3;
      Real vey3 = q3[IM2] * irho3;
      Real vez3 = q3[IM3] * irho3;
      Real Bnm3 = q3[IB1] * q3[IB1] + q3[IB2] * q3[IB2] + q3[IB3] * q3[IB3];
      Real vnm3 = vex3 * vex3 + vey3 * vey3 + vez3 * vez3;
      Real pre3 = (gm1) * (q3[IEN] - 0.5*(q3[IDN]*vnm3 + Bnm3));

      f3[IDN] = q3[IM1];
      f3[IM1] = q3[IM1] * vex3 + pre3 + 0.5 * Bnm3 - q3[IB1] * q3[IB1];
      f3[IM2] = q3[IM1] * vey3 - q3[IB1] * q3[IB2];
      f3[IM3] = q3[IM1] * vez3 - q3[IB1] * q3[IB3];
      f3[IEN] = vex3 * (q3[IEN] + pre3 + 0.5 * Bnm3) - q3[IB1] * (vex3 * q3[IB1] + vey3 * q3[IB2] + vez3 * q3[IB3]);
      f3[IB1] = 0.0;
      f3[IB2] = vex3 * q3[IB2] - vey3 * q3[IB1];
      f3[IB3] = vex3 * q3[IB3] - vez3 * q3[IB1];

      //--- f4 ---
      Real irho4 = 1.0 / q4[IDN];
      Real vex4 = q4[IM1] * irho4;
      Real vey4 = q4[IM2] * irho4;
      Real vez4 = q4[IM3] * irho4;
      Real Bnm4 = q4[IB1] * q4[IB1] + q4[IB2] * q4[IB2] + q4[IB3] * q4[IB3];
      Real vnm4 = vex4 * vex4 + vey4 * vey4 + vez4 * vez4;
      Real pre4 = (gm1) * (q4[IEN] - 0.5*(q4[IDN]*vnm4 + Bnm4));

      f4[IDN] = q4[IM1];
      f4[IM1] = q4[IM1] * vex4 + pre4 + 0.5 * Bnm4 - q4[IB1] * q4[IB1];
      f4[IM2] = q4[IM1] * vey4 - q4[IB1] * q4[IB2];
      f4[IM3] = q4[IM1] * vez4 - q4[IB1] * q4[IB3];
      f4[IEN] = vex4 * (q4[IEN] + pre4 + 0.5 * Bnm4) - q4[IB1] * (vex4 * q4[IB1] + vey4 * q4[IB2] + vez4 * q4[IB3]);
      f4[IB1] = 0.0;
      f4[IB2] = vex4 * q4[IB2] - vey4 * q4[IB1];
      f4[IB3] = vex4 * q4[IB3] - vez4 * q4[IB1];

      //--- f5 ---
      Real irho5 = 1.0 / q5[IDN];
      Real vex5 = q5[IM1] * irho5;
      Real vey5 = q5[IM2] * irho5;
      Real vez5 = q5[IM3] * irho5;
      Real Bnm5 = q5[IB1] * q5[IB1] + q5[IB2] * q5[IB2] + q5[IB3] * q5[IB3];
      Real vnm5 = vex5 * vex5 + vey5 * vey5 + vez5 * vez5;
      Real pre5 = (gm1) * (q5[IEN] - 0.5*(q5[IDN]*vnm5 + Bnm5));

      f5[IDN] = q5[IM1];
      f5[IM1] = q5[IM1] * vex5 + pre5 + 0.5 * Bnm5 - q5[IB1] * q5[IB1];
      f5[IM2] = q5[IM1] * vey5 - q5[IB1] * q5[IB2];
      f5[IM3] = q5[IM1] * vez5 - q5[IB1] * q5[IB3];
      f5[IEN] = vex5 * (q5[IEN] + pre5 + 0.5 * Bnm5) - q5[IB1] * (vex5 * q5[IB1] + vey5 * q5[IB2] + vez5 * q5[IB3]);
      f5[IB1] = 0.0;
      f5[IB2] = vex5 * q5[IB2] - vey5 * q5[IB1];
      f5[IB3] = vex5 * q5[IB3] - vez5 * q5[IB1];

      //--- f6 ---
      Real irho6 = 1.0 / q6[IDN];
      Real vex6 = q6[IM1] * irho6;
      Real vey6 = q6[IM2] * irho6;
      Real vez6 = q6[IM3] * irho6;
      Real Bnm6 = q6[IB1] * q6[IB1] + q6[IB2] * q6[IB2] + q6[IB3] * q6[IB3];
      Real vnm6 = vex6 * vex6 + vey6 * vey6 + vez6 * vez6;
      Real pre6 = (gm1) * (q6[IEN] - 0.5*(q6[IDN]*vnm6 + Bnm6));

      f6[IDN] = q6[IM1];
      f6[IM1] = q6[IM1] * vex6 + pre6 + 0.5 * Bnm6 - q6[IB1] * q6[IB1];
      f6[IM2] = q6[IM1] * vey6 - q6[IB1] * q6[IB2];
      f6[IM3] = q6[IM1] * vez6 - q6[IB1] * q6[IB3];
      f6[IEN] = vex6 * (q6[IEN] + pre6 + 0.5 * Bnm6) - q6[IB1] * (vex6 * q6[IB1] + vey6 * q6[IB2] + vez6 * q6[IB3]);
      f6[IB1] = 0.0;
      f6[IB2] = vex6 * q6[IB2] - vey6 * q6[IB1];
      f6[IB3] = vex6 * q6[IB3] - vez6 * q6[IB1];

      //--- f7 ---
      Real irho7 = 1.0 / q7[IDN];
      Real vex7 = q7[IM1] * irho7;
      Real vey7 = q7[IM2] * irho7;
      Real vez7 = q7[IM3] * irho7;
      Real Bnm7 = q7[IB1] * q7[IB1] + q7[IB2] * q7[IB2] + q7[IB3] * q7[IB3];
      Real vnm7 = vex7 * vex7 + vey7 * vey7 + vez7 * vez7;
      Real pre7 = (gm1) * (q7[IEN] - 0.5*(q7[IDN]*vnm7 + Bnm7));

      f7[IDN] = q7[IM1];
      f7[IM1] = q7[IM1] * vex7 + pre7 + 0.5 * Bnm7 - q7[IB1] * q7[IB1];
      f7[IM2] = q7[IM1] * vey7 - q7[IB1] * q7[IB2];
      f7[IM3] = q7[IM1] * vez7 - q7[IB1] * q7[IB3];
      f7[IEN] = vex7 * (q7[IEN] + pre7 + 0.5 * Bnm7) - q7[IB1] * (vex7 * q7[IB1] + vey7 * q7[IB2] + vez7 * q7[IB3]);
      f7[IB1] = 0.0;
      f7[IB2] = vex7 * q7[IB2] - vey7 * q7[IB1];
      f7[IB3] = vex7 * q7[IB3] - vez7 * q7[IB1];

      //--- f8 ---
      Real irho8 = 1.0 / q8[IDN];
      Real vex8 = q8[IM1] * irho8;
      Real vey8 = q8[IM2] * irho8;
      Real vez8 = q8[IM3] * irho8;
      Real Bnm8 = q8[IB1] * q8[IB1] + q8[IB2] * q8[IB2] + q8[IB3] * q8[IB3];
      Real vnm8 = vex8 * vex8 + vey8 * vey8 + vez8 * vez8;
      Real pre8 = (gm1) * (q8[IEN] - 0.5*(q8[IDN]*vnm8 + Bnm8));

      f8[IDN] = q8[IM1];
      f8[IM1] = q8[IM1] * vex8 + pre8 + 0.5 * Bnm8 - q8[IB1] * q8[IB1];
      f8[IM2] = q8[IM1] * vey8 - q8[IB1] * q8[IB2];
      f8[IM3] = q8[IM1] * vez8 - q8[IB1] * q8[IB3];
      f8[IEN] = vex8 * (q8[IEN] + pre8 + 0.5 * Bnm8) - q8[IB1] * (vex8 * q8[IB1] + vey8 * q8[IB2] + vez8 * q8[IB3]);
      f8[IB1] = 0.0;
      f8[IB2] = vex8 * q8[IB2] - vey8 * q8[IB1];
      f8[IB3] = vex8 * q8[IB3] - vez8 * q8[IB1];

      //--- f9 ---
      Real irho9 = 1.0 / q9[IDN];
      Real vex9 = q9[IM1] * irho9;
      Real vey9 = q9[IM2] * irho9;
      Real vez9 = q9[IM3] * irho9;
      Real Bnm9 = q9[IB1] * q9[IB1] + q9[IB2] * q9[IB2] + q9[IB3] * q9[IB3];
      Real vnm9 = vex9 * vex9 + vey9 * vey9 + vez9 * vez9;
      Real pre9 = (gm1) * (q9[IEN] - 0.5*(q9[IDN]*vnm9 + Bnm9));

      f9[IDN] = q9[IM1];
      f9[IM1] = q9[IM1] * vex9 + pre9 + 0.5 * Bnm9 - q9[IB1] * q9[IB1];
      f9[IM2] = q9[IM1] * vey9 - q9[IB1] * q9[IB2];
      f9[IM3] = q9[IM1] * vez9 - q9[IB1] * q9[IB3];
      f9[IEN] = vex9 * (q9[IEN] + pre9 + 0.5 * Bnm9) - q9[IB1] * (vex9 * q9[IB1] + vey9 * q9[IB2] + vez9 * q9[IB3]);
      f9[IB1] = 0.0;
      f9[IB2] = vex9 * q9[IB2] - vey9 * q9[IB1];
      f9[IB3] = vex9 * q9[IB3] - vez9 * q9[IB1];

      //--- Step 2.  At each x_{i-1/2,j,k}: (using cells i-1 and i)
      //--- (a) Compute the average state w_{i-1/2,j,k} in the primitive variables:

      Real half_den = 0.5 * (q4[IDN] + q5[IDN]);
      Real half_vex = 0.5 * (vex4 + vex5);
      Real half_vey = 0.5 * (vey4 + vey5);
      Real half_vez = 0.5 * (vez4 + vez5);
      Real half_pre = 0.5 * (pre4 + pre5);
      Real half_Bx  = 0.5 * (q4[IB1] + q5[IB1]);
      Real half_By  = 0.5 * (q4[IB2] + q5[IB2]);
      Real half_Bz  = 0.5 * (q4[IB3] + q5[IB3]);


      //--- (b) Compute the right and left eigenvectors of the flux Jacobian matrix, ∂f/∂x, at x = x_{i-1/2,j,k}:

      // See Section 1.5 (Scaling Theorem Example: Magnetohydrodynamic Equations) of:
      // "Numerical Methods for Gasdynamic Systems on Unstructured Meshes"
      // By Timothy J. Barth
      // Entropy Scaled Eigenvectors of the Modified MHD Equations
      // Equations (54) - (57)

      Real t1 = 0.0;
      Real half_Bnm = std::sqrt(half_Bz * half_Bz + half_By * half_By);
      Real t2, t3;

      if (half_Bnm != 0) {
        t2 = half_By / half_Bnm;
        t3 = half_Bz / half_Bnm;
      } else {
        t2 = std::sin(M_PI / 4.0);
        t3 = std::cos(M_PI / 4.0);
      }

      Real n1 = 1.0;
      Real n2 = 0.0;
      Real n3 = 0.0;

      Real rhosq = std::sqrt(half_den); 
      Real presq = std::sqrt(half_pre); 
      Real a2 = (gamma * half_pre) / half_den; 
      Real a = std::sqrt(a2); 
      Real sqg2 = std::sqrt(1.0 / (2.0 * gamma)); 
      Real sq12 = std::sqrt(0.5); 
      Real twosq = std::sqrt(2.0); 
      Real sqpr = std::sqrt(half_pre) / half_den; 
      Real sqpor = std::sqrt(half_pre / half_den); 
      Real sq1og = std::sqrt(1.0 / gamma); 
      Real sqgam = std::sqrt(gm1 / gamma);
      Real b1s = half_Bx / rhosq; 
      Real b2s = half_By / rhosq; 
      Real b3s = half_Bz / rhosq; 
      Real BNs = b1s * n1 + b2s * n2 + b3s * n3; 
      Real BN = half_Bx * n1 + half_By * n2 + half_Bz * n3; 
      Real d = a2 + (b1s * b1s + b2s * b2s + b3s * b3s); 
      Real cf = std::sqrt(0.5 * std::abs(d + std::sqrt(d * d - 4.0 * a2 * (BNs * BNs)))); 
      Real cs = std::sqrt(0.5 * std::abs(d - std::sqrt(d * d - 4.0 * a2 * (BNs * BNs)))); 
      Real cf2 = 0.5 * std::abs(d + std::sqrt(d * d - 4.0 * a2 * (BNs * BNs))); 
      Real cs2 = 0.5 * std::abs(d - std::sqrt(d * d - 4.0 * a2 * (BNs * BNs)));       
      Real beta1 = ((b1s * n1 + b2s * n2 + b3s * n3) >= 0.0) ? 1.0 : -1.0;

      Real alphaf, alphas;
      if (std::abs(cf * cf - cs * cs) <= 1.0e-12) {
        alphaf = std::sin(std::atan(1.0) / 2.0);
        alphas = std::cos(std::atan(1.0) / 2.0);
      } else {
        alphaf = std::sqrt(std::abs(a2 - cs * cs)) / std::sqrt(std::abs(cf * cf - cs * cs));
        alphas = std::sqrt(std::abs(cf * cf - a2)) / std::sqrt(std::abs(cf * cf - cs * cs));
      }

      Real TxN1 = n3 * t2 - n2 * t3; 
      Real TxN2 = n1 * t3 - n3 * t1; 
      Real TxN3 = n2 * t1 - n1 * t2; 

      Real BT = half_Bx * t1 + half_By * t2 + half_Bz * t3; 

      //--- Right Eigenvectors 

      // 1 - Right Eigenvector Entropy Wave
      ru[0][0] = std::sqrt(gm1 / gamma) * rhosq;
      ru[1][0] = 0.0;
      ru[2][0] = 0.0;
      ru[3][0] = 0.0;
      ru[4][0] = 0.0;
      ru[5][0] = 0.0;
      ru[6][0] = 0.0;
      ru[7][0] = 0.0;

      // 2 - Right Eigenvector Divergence Wave
      ru[0][1] = 0.0;
      ru[1][1] = 0.0;
      ru[2][1] = 0.0;
      ru[3][1] = 0.0;
      ru[4][1] = 0.0;
      ru[5][1] = sq1og * a * n1;
      ru[6][1] = sq1og * a * n2;
      ru[7][1] = sq1og * a * n3;

      // 3 - Right Eigenvector Alfven Wave
      ru[0][2] = 0.0;
      ru[1][2] = -sq12 * (sqpr * TxN1);
      ru[2][2] = -sq12 * (sqpr * TxN2);
      ru[3][2] = -sq12 * (sqpr * TxN3);
      ru[4][2] = 0.0;
      ru[5][2] = sq12 * sqpor * TxN1;
      ru[6][2] = sq12 * sqpor * TxN2;
      ru[7][2] = sq12 * sqpor * TxN3;

      // 4 - Right Eigenvector Alfven Wave
      ru[0][3] =  ru[0][2];
      ru[1][3] = -ru[1][2];
      ru[2][3] = -ru[2][2];
      ru[3][3] = -ru[3][2];
      ru[4][3] =  ru[4][2];
      ru[5][3] =  ru[5][2];
      ru[6][3] =  ru[6][2];
      ru[7][3] =  ru[7][2];

      // 5 - Right Eigenvector Fast Magneto-acoustic Wave
      Real bst = b1s * t1 + b2s * t2 + b3s * t3;
      ru[0][4] = sqg2 * alphaf * rhosq;
      ru[1][4] = sqg2 * ((alphaf * a2 * n1 + alphas * a * ((bst) * n1 - (BNs) * t1))) / (rhosq * cf);
      ru[2][4] = sqg2 * ((alphaf * a2 * n2 + alphas * a * ((bst) * n2 - (BNs) * t2))) / (rhosq * cf);
      ru[3][4] = sqg2 * ((alphaf * a2 * n3 + alphas * a * ((bst) * n3 - (BNs) * t3))) / (rhosq * cf);
      ru[4][4] = sqg2 * alphaf * rhosq * a2;
      ru[5][4] = sqg2 * alphas * a * t1;
      ru[6][4] = sqg2 * alphas * a * t2;
      ru[7][4] = sqg2 * alphas * a * t3;

      // 6 - Right Eigenvector Fast Magneto-acoustic Wave
      ru[0][5] =  ru[0][4];
      ru[1][5] = -ru[1][4];
      ru[2][5] = -ru[2][4];
      ru[3][5] = -ru[3][4];
      ru[4][5] =  ru[4][4];
      ru[5][5] =  ru[5][4];
      ru[6][5] =  ru[6][4];
      ru[7][5] =  ru[7][4];

      // 7 - Right Eigenvector Slow Magneto-acoustic Wave
      ru[0][6] = sqg2 * alphas * rhosq;
      ru[1][6] = sqg2 * beta1 * (alphaf * cf * cf * t1 + alphas * a * (BNs) * n1) / (rhosq * cf);
      ru[2][6] = sqg2 * beta1 * (alphaf * cf * cf * t2 + alphas * a * (BNs) * n2) / (rhosq * cf);
      ru[3][6] = sqg2 * beta1 * (alphaf * cf * cf * t3 + alphas * a * (BNs) * n3) / (rhosq * cf);
      ru[4][6] = sqg2 * alphas * rhosq * a2;
      ru[5][6] = -sqg2 * alphaf * a * t1;
      ru[6][6] = -sqg2 * alphaf * a * t2;
      ru[7][6] = -sqg2 * alphaf * a * t3;

      // 8 - Right Eigenvector Slow Magneto-acoustic Wave
      ru[0][7] =  ru[0][6];
      ru[1][7] = -ru[1][6];
      ru[2][7] = -ru[2][6];
      ru[3][7] = -ru[3][6];
      ru[4][7] =  ru[4][6];
      ru[5][7] =  ru[5][6];
      ru[6][7] =  ru[6][6];
      ru[7][7] =  ru[7][6];

      for (int m = 0; m < 8; ++m) {
        rr[0][m] = ru[0][m] / gm1;
        rr[1][m] = (ru[0][m] * half_vex + ru[1][m] * half_den) / gm1;
        rr[2][m] = (ru[0][m] * half_vey + ru[2][m] * half_den) / gm1;
        rr[3][m] = (ru[0][m] * half_vez + ru[3][m] * half_den) / gm1;
        rr[4][m] = (ru[4][m] / gm1 + half_Bx * ru[5][m] + half_By * ru[6][m] + half_Bz * ru[7][m] + 0.5 * ru[0][m] * (half_vex * half_vex + half_vey * half_vey + half_vez * half_vez) + ru[1][m] * half_vex * half_den + ru[2][m] * half_vey * half_den + ru[3][m] * half_vez * half_den) / gm1;
        rr[5][m] = ru[5][m] / gm1;
        rr[6][m] = ru[6][m] / gm1;
        rr[7][m] = ru[7][m] / gm1;
      }

      //--- Left Eigenvectors 
      
      // 1 - Left Eigenvector
      lu[0][0] = 1.0 / (sqgam * rhosq);
      lu[0][1] = 0.0;
      lu[0][2] = 0.0;
      lu[0][3] = 0.0;
      lu[0][4] = -1.0 / (a2 * sqgam * rhosq);
      lu[0][5] = 0.0;
      lu[0][6] = 0.0;
      lu[0][7] = 0.0;
        
      // 2 - Left Eigenvector
      Real nen = n1*n1*(t2*t2+t3*t3) + n2*n2*(t1*t1+t3*t3) + n3*n3*(t1*t1+t2*t2) - 2.0*n2*n3*t2*t3 - 2.0*n1*n3*t1*t3 - 2.0*n1*n2*t1*t2;
      Real nen2 = a * sq1og * nen;
      lu[1][0] = 0.0;
      lu[1][1] = 0.0;
      lu[1][2] = 0.0;
      lu[1][3] = 0.0;
      lu[1][4] = 0.0;
      lu[1][5] = (n1*(t2*t2 + t3*t3) - t1*(n2*t2 + n3*t3))/nen2;
      lu[1][6] = (n2*(t1*t1 + t3*t3) - t2*(n1*t1 + n3*t3))/nen2;
      lu[1][7] = (n3*(t1*t1 + t2*t2) - t3*(n1*t1 + n2*t2))/nen2;

      // 3 - Left Eigenvector
      Real nen3 = twosq * presq * nen;
      Real nen31 = nen3 / rhosq;
      lu[2][0] = 0.0;
      lu[2][1] = half_den * (-TxN1) / nen3;
      lu[2][2] = half_den * (-TxN2) / nen3;
      lu[2][3] = half_den * (-TxN3) / nen3;    
      lu[2][4] = 0.0;
      lu[2][5] = TxN1 / nen31;
      lu[2][6] = TxN2 / nen31;
      lu[2][7] = TxN3 / nen31; 

      // 4 - Left Eigenvector
      lu[3][0] =  lu[2][0];
      lu[3][1] = -lu[2][1];
      lu[3][2] = -lu[2][2];
      lu[3][3] = -lu[2][3];
      lu[3][4] =  lu[2][4];
      lu[3][5] =  lu[2][5];
      lu[3][6] =  lu[2][6];
      lu[3][7] =  lu[2][7];

      // 5 - Left Eigenvector
      Real Term51 = half_den*cf*( rhosq*cf*cf*alphaf*( -t1*(n2*t2 + n3*t3) + n1*(t2*t2+t3*t3) ) - a*BN*alphas*( n2*TxN3 - n3*TxN2 ) );
      Real Term52 = half_den*cf*( rhosq*cf*cf*alphaf*( -t2*(n1*t1 + n3*t3) + n2*(t1*t1+t3*t3) ) - a*BN*alphas*( n3*TxN1 - n1*TxN3 ) );
      Real Term53 = half_den*cf*( rhosq*cf*cf*alphaf*( -t3*(n1*t1 + n2*t2) + n3*(t1*t1+t2*t2) ) - a*BN*alphas*( n1*TxN2 - n2*TxN1 ) );
      
      Real Term54 = alphaf / (twosq * a * a * sq1og * rhosq * (alphaf * alphaf + alphas * alphas));

      Real Term55 = alphas*( n2*TxN3 - n3*TxN2 );
      Real Term56 = alphas*( n3*TxN1 - n1*TxN3 );
      Real Term57 = alphas*( n1*TxN2 - n2*TxN1 );

      Real nen51 = twosq * a * nen * sq1og * (a * BN * BN * alphas * alphas + rhosq * cf * cf * alphaf * (a * rhosq * alphaf + BT * alphas));

      Real nen52 = twosq * a * sq1og * (alphaf * alphaf + alphas * alphas) * nen;

      lu[4][0] = 0.0;
      lu[4][1] = Term51 / nen51;
      lu[4][2] = Term52 / nen51;
      lu[4][3] = Term53 / nen51;
      lu[4][4] = Term54;
      lu[4][5] = Term55 / nen52;
      lu[4][6] = Term56 / nen52;
      lu[4][7] = Term57 / nen52;

      // 6 - Left Eigenvector
      lu[5][0] =  lu[4][0];
      lu[5][1] = -lu[4][1];
      lu[5][2] = -lu[4][2];
      lu[5][3] = -lu[4][3];
      lu[5][4] =  lu[4][4];
      lu[5][5] =  lu[4][5];
      lu[5][6] =  lu[4][6];
      lu[5][7] =  lu[4][7];

      // 7 - Left Eigenvector
      Real Term71 = half_den*cf*( a*rhosq*alphaf*( n2*TxN3 - n3*TxN2 ) + alphas*( ( half_By*TxN2 + half_Bz*TxN3 )*(-TxN1) + half_Bx*( n1*n1*(t2*t2 + t3*t3) + t1*t1*(n2*n2 + n3*n3) - 2.0*n1*t1*(n2*t2 + n3*t3) ) ) );
      Real Term72 = half_den*cf*( a*rhosq*alphaf*( n3*TxN1 - n1*TxN3 ) + alphas*( ( half_Bz*TxN3 + half_Bx*TxN1 )*(-TxN2) + half_By*( n2*n2*(t3*t3 + t1*t1) + t2*t2*(n3*n3 + n1*n1) - 2.0*n2*t2*(n3*t3 + n1*t1) ) ) );
      Real Term73 = half_den*cf*( a*rhosq*alphaf*( n1*TxN2 - n2*TxN1 ) + alphas*( ( half_Bx*TxN1 + half_By*TxN2 )*(-TxN3) + half_Bz*( n3*n3*(t1*t1 + t2*t2) + t3*t3*(n1*n1 + n2*n2) - 2.0*n3*t3*(n1*t1 + n2*t2) ) ) );

      Real Term74 =  alphas / (twosq * a * a * sq1og * rhosq * (alphaf * alphaf + alphas * alphas));

      Real Term75 = -alphaf*( n2*TxN3 - n3*TxN2 );
      Real Term76 = -alphaf*( n3*TxN1 - n1*TxN3 );
      Real Term77 = -alphaf*( n1*TxN2 - n2*TxN1 );

      Real nen71 = twosq * beta1 * nen * sq1og * (a * BN * BN * alphas * alphas + rhosq * cf * cf * alphaf * (a * rhosq * alphaf + BT * alphas));

      Real nen72 = nen52;

      lu[6][0] = 0.0;
      lu[6][1] = Term71 / nen71;
      lu[6][2] = Term72 / nen71;
      lu[6][3] = Term73 / nen71;
      lu[6][4] = Term74;
      lu[6][5] = Term75 / nen72;
      lu[6][6] = Term76 / nen72;
      lu[6][7] = Term77 / nen72;

      // 8 - Left Eigenvector
      lu[7][0] =  lu[6][0];
      lu[7][1] = -lu[6][1];
      lu[7][2] = -lu[6][2];
      lu[7][3] = -lu[6][3];
      lu[7][4] =  lu[6][4];
      lu[7][5] =  lu[6][5];
      lu[7][6] =  lu[6][6];
      lu[7][7] =  lu[6][7];

      for (int mm = 0; mm < 8; ++mm) {
        lq[mm][0] =  lu[mm][0] * gm1 - lu[mm][1] * half_vex * gm1 / half_den - lu[mm][2] * half_vey * gm1 / half_den - lu[mm][3] * half_vez * gm1 / half_den + lu[mm][4] * gm1 * gm1 * (half_vex * half_vex + half_vey * half_vey + half_vez * half_vez) * 0.5;
        lq[mm][1] = -lu[mm][4] * half_vex * gm1 * gm1 + lu[mm][1] * gm1 / half_den;
        lq[mm][2] = -lu[mm][4] * half_vey * gm1 * gm1 + lu[mm][2] * gm1 / half_den;
        lq[mm][3] = -lu[mm][4] * half_vez * gm1 * gm1 + lu[mm][3] * gm1 / half_den;
        lq[mm][4] =  lu[mm][4] * gm1 * gm1;
        lq[mm][5] =  lu[mm][5] * gm1 - half_Bx * lu[mm][4] * gm1 * gm1;
        lq[mm][6] =  lu[mm][6] * gm1 - half_By * lu[mm][4] * gm1 * gm1;
        lq[mm][7] =  lu[mm][7] * gm1 - half_Bz * lu[mm][4] * gm1 * gm1;
      }

      //--- (c) Project the solution and physical flux into the right eigenvector space:

      for (int ii = 0; ii < NMHD; ++ii) {

        vj0[ii] = 0.0;
        vj1[ii] = 0.0;
        vj2[ii] = 0.0;
        vj3[ii] = 0.0;
        vj4[ii] = 0.0;
        vj5[ii] = 0.0;
        vj6[ii] = 0.0;
        vj7[ii] = 0.0;
        vj8[ii] = 0.0;
        vj9[ii] = 0.0;

        gj0[ii] = 0.0;
        gj1[ii] = 0.0;
        gj2[ii] = 0.0;
        gj3[ii] = 0.0;
        gj4[ii] = 0.0;
        gj5[ii] = 0.0;
        gj6[ii] = 0.0;
        gj7[ii] = 0.0;
        gj8[ii] = 0.0;
        gj9[ii] = 0.0;

        for (int jj = 0; jj < NMHD; ++jj) {

          vj0[ii] += lq[ii][jj] * q0[jj];
          vj1[ii] += lq[ii][jj] * q1[jj];
          vj2[ii] += lq[ii][jj] * q2[jj];
          vj3[ii] += lq[ii][jj] * q3[jj];
          vj4[ii] += lq[ii][jj] * q4[jj];
          vj5[ii] += lq[ii][jj] * q5[jj];
          vj6[ii] += lq[ii][jj] * q6[jj];
          vj7[ii] += lq[ii][jj] * q7[jj];
          vj8[ii] += lq[ii][jj] * q8[jj];
          vj9[ii] += lq[ii][jj] * q9[jj];

          gj0[ii] += lq[ii][jj] * f0[jj];
          gj1[ii] += lq[ii][jj] * f1[jj];
          gj2[ii] += lq[ii][jj] * f2[jj];
          gj3[ii] += lq[ii][jj] * f3[jj];
          gj4[ii] += lq[ii][jj] * f4[jj];
          gj5[ii] += lq[ii][jj] * f5[jj];
          gj6[ii] += lq[ii][jj] * f6[jj];
          gj7[ii] += lq[ii][jj] * f7[jj];
          gj8[ii] += lq[ii][jj] * f8[jj];
          gj9[ii] += lq[ii][jj] * f9[jj];

        }
      }

      //--- (d) Perform a Lax-Friedrichs flux vector splitting for each component of the characteristic variables:
      // Specifically, assume that the mth components of Vj and Gj are vj and gj, respectively, then compute
      // g^{±}_{j}= 0.5 * (g_j ± α^{m} v_j) where α(m) = max_k | λ^{m} q_k | 
      // is the maximal wave speed of the m^{th} component of characteristic variables over all grid points

      Real aa0 = std::sqrt(gamma * pre0 * irho0);
      Real aa1 = std::sqrt(gamma * pre1 * irho1);
      Real aa2 = std::sqrt(gamma * pre2 * irho2);
      Real aa3 = std::sqrt(gamma * pre3 * irho3);
      Real aa4 = std::sqrt(gamma * pre4 * irho4);
      Real aa5 = std::sqrt(gamma * pre5 * irho5);
      Real aa6 = std::sqrt(gamma * pre6 * irho6);
      Real aa7 = std::sqrt(gamma * pre7 * irho7);
      Real aa8 = std::sqrt(gamma * pre8 * irho8);
      Real aa9 = std::sqrt(gamma * pre9 * irho9);

      Real ca0 = std::sqrt(Bnm0 * irho0);
      Real ca1 = std::sqrt(Bnm1 * irho1);
      Real ca2 = std::sqrt(Bnm2 * irho2);
      Real ca3 = std::sqrt(Bnm3 * irho3);
      Real ca4 = std::sqrt(Bnm4 * irho4);
      Real ca5 = std::sqrt(Bnm5 * irho5);
      Real ca6 = std::sqrt(Bnm6 * irho6);
      Real ca7 = std::sqrt(Bnm7 * irho7);
      Real ca8 = std::sqrt(Bnm8 * irho8);
      Real ca9 = std::sqrt(Bnm9 * irho9);

      Real cax0 = std::sqrt(q0[IB1] * q0[IB1] * irho0);
      Real cax1 = std::sqrt(q1[IB1] * q1[IB1] * irho1);
      Real cax2 = std::sqrt(q2[IB1] * q2[IB1] * irho2);
      Real cax3 = std::sqrt(q3[IB1] * q3[IB1] * irho3);
      Real cax4 = std::sqrt(q4[IB1] * q4[IB1] * irho4);
      Real cax5 = std::sqrt(q5[IB1] * q5[IB1] * irho5);
      Real cax6 = std::sqrt(q6[IB1] * q6[IB1] * irho6);
      Real cax7 = std::sqrt(q7[IB1] * q7[IB1] * irho7);
      Real cax8 = std::sqrt(q8[IB1] * q8[IB1] * irho8);
      Real cax9 = std::sqrt(q9[IB1] * q9[IB1] * irho9);

      Real cfx0 = std::sqrt(0.5 * std::abs(aa0 * aa0 + ca0 * ca0 + std::sqrt((aa0 * aa0 + ca0 * ca0) * (aa0 * aa0 + ca0 * ca0) - (4 * aa0 * aa0 * cax0 * cax0))));
      Real cfx1 = std::sqrt(0.5 * std::abs(aa1 * aa1 + ca1 * ca1 + std::sqrt((aa1 * aa1 + ca1 * ca1) * (aa1 * aa1 + ca1 * ca1) - (4 * aa1 * aa1 * cax1 * cax1))));
      Real cfx2 = std::sqrt(0.5 * std::abs(aa2 * aa2 + ca2 * ca2 + std::sqrt((aa2 * aa2 + ca2 * ca2) * (aa2 * aa2 + ca2 * ca2) - (4 * aa2 * aa2 * cax2 * cax2))));
      Real cfx3 = std::sqrt(0.5 * std::abs(aa3 * aa3 + ca3 * ca3 + std::sqrt((aa3 * aa3 + ca3 * ca3) * (aa3 * aa3 + ca3 * ca3) - (4 * aa3 * aa3 * cax3 * cax3))));
      Real cfx4 = std::sqrt(0.5 * std::abs(aa4 * aa4 + ca4 * ca4 + std::sqrt((aa4 * aa4 + ca4 * ca4) * (aa4 * aa4 + ca4 * ca4) - (4 * aa4 * aa4 * cax4 * cax4))));
      Real cfx5 = std::sqrt(0.5 * std::abs(aa5 * aa5 + ca5 * ca5 + std::sqrt((aa5 * aa5 + ca5 * ca5) * (aa5 * aa5 + ca5 * ca5) - (4 * aa5 * aa5 * cax5 * cax5))));
      Real cfx6 = std::sqrt(0.5 * std::abs(aa6 * aa6 + ca6 * ca6 + std::sqrt((aa6 * aa6 + ca6 * ca6) * (aa6 * aa6 + ca6 * ca6) - (4 * aa6 * aa6 * cax6 * cax6))));
      Real cfx7 = std::sqrt(0.5 * std::abs(aa7 * aa7 + ca7 * ca7 + std::sqrt((aa7 * aa7 + ca7 * ca7) * (aa7 * aa7 + ca7 * ca7) - (4 * aa7 * aa7 * cax7 * cax7))));
      Real cfx8 = std::sqrt(0.5 * std::abs(aa8 * aa8 + ca8 * ca8 + std::sqrt((aa8 * aa8 + ca8 * ca8) * (aa8 * aa8 + ca8 * ca8) - (4 * aa8 * aa8 * cax8 * cax8))));
      Real cfx9 = std::sqrt(0.5 * std::abs(aa9 * aa9 + ca9 * ca9 + std::sqrt((aa9 * aa9 + ca9 * ca9) * (aa9 * aa9 + ca9 * ca9) - (4 * aa9 * aa9 * cax9 * cax9))));
      
      Real csx0 = std::sqrt(0.5 * std::abs(aa0 * aa0 + ca0 * ca0 - std::sqrt((aa0 * aa0 + ca0 * ca0) * (aa0 * aa0 + ca0 * ca0) - (4 * aa0 * aa0 * cax0 * cax0))));
      Real csx1 = std::sqrt(0.5 * std::abs(aa1 * aa1 + ca1 * ca1 - std::sqrt((aa1 * aa1 + ca1 * ca1) * (aa1 * aa1 + ca1 * ca1) - (4 * aa1 * aa1 * cax1 * cax1))));
      Real csx2 = std::sqrt(0.5 * std::abs(aa2 * aa2 + ca2 * ca2 - std::sqrt((aa2 * aa2 + ca2 * ca2) * (aa2 * aa2 + ca2 * ca2) - (4 * aa2 * aa2 * cax2 * cax2))));
      Real csx3 = std::sqrt(0.5 * std::abs(aa3 * aa3 + ca3 * ca3 - std::sqrt((aa3 * aa3 + ca3 * ca3) * (aa3 * aa3 + ca3 * ca3) - (4 * aa3 * aa3 * cax3 * cax3))));
      Real csx4 = std::sqrt(0.5 * std::abs(aa4 * aa4 + ca4 * ca4 - std::sqrt((aa4 * aa4 + ca4 * ca4) * (aa4 * aa4 + ca4 * ca4) - (4 * aa4 * aa4 * cax4 * cax4))));
      Real csx5 = std::sqrt(0.5 * std::abs(aa5 * aa5 + ca5 * ca5 - std::sqrt((aa5 * aa5 + ca5 * ca5) * (aa5 * aa5 + ca5 * ca5) - (4 * aa5 * aa5 * cax5 * cax5))));
      Real csx6 = std::sqrt(0.5 * std::abs(aa6 * aa6 + ca6 * ca6 - std::sqrt((aa6 * aa6 + ca6 * ca6) * (aa6 * aa6 + ca6 * ca6) - (4 * aa6 * aa6 * cax6 * cax6))));
      Real csx7 = std::sqrt(0.5 * std::abs(aa7 * aa7 + ca7 * ca7 - std::sqrt((aa7 * aa7 + ca7 * ca7) * (aa7 * aa7 + ca7 * ca7) - (4 * aa7 * aa7 * cax7 * cax7))));
      Real csx8 = std::sqrt(0.5 * std::abs(aa8 * aa8 + ca8 * ca8 - std::sqrt((aa8 * aa8 + ca8 * ca8) * (aa8 * aa8 + ca8 * ca8) - (4 * aa8 * aa8 * cax8 * cax8))));
      Real csx9 = std::sqrt(0.5 * std::abs(aa9 * aa9 + ca9 * ca9 - std::sqrt((aa9 * aa9 + ca9 * ca9) * (aa9 * aa9 + ca9 * ca9) - (4 * aa9 * aa9 * cax9 * cax9))));

      Real em = 1.0e-15;

      Real max_eig_1 = std::max({em, std::abs(vex0),      std::abs(vex1),      std::abs(vex2),      std::abs(vex3),      std::abs(vex4),      std::abs(vex5),      std::abs(vex6),      std::abs(vex7),      std::abs(vex8),      std::abs(vex9)});
      Real max_eig_2 = std::max({em, std::abs(vex0),      std::abs(vex1),      std::abs(vex2),      std::abs(vex3),      std::abs(vex4),      std::abs(vex5),      std::abs(vex6),      std::abs(vex7),      std::abs(vex8),      std::abs(vex9)});
      Real max_eig_3 = std::max({em, std::abs(vex0+cax0), std::abs(vex1+cax1), std::abs(vex2+cax2), std::abs(vex3+cax3), std::abs(vex4+cax4), std::abs(vex5+cax5), std::abs(vex6+cax6), std::abs(vex7+cax7), std::abs(vex8+cax8), std::abs(vex9+cax9)});
      Real max_eig_4 = std::max({em, std::abs(vex0-cax0), std::abs(vex1-cax1), std::abs(vex2-cax2), std::abs(vex3-cax3), std::abs(vex4-cax4), std::abs(vex5-cax5), std::abs(vex6-cax6), std::abs(vex7-cax7), std::abs(vex8-cax8), std::abs(vex9-cax9)});
      Real max_eig_5 = std::max({em, std::abs(vex0+cfx0), std::abs(vex1+cfx1), std::abs(vex2+cfx2), std::abs(vex3+cfx3), std::abs(vex4+cfx4), std::abs(vex5+cfx5), std::abs(vex6+cfx6), std::abs(vex7+cfx7), std::abs(vex8+cfx8), std::abs(vex9+cfx9)});
      Real max_eig_6 = std::max({em, std::abs(vex0-cfx0), std::abs(vex1-cfx1), std::abs(vex2-cfx2), std::abs(vex3-cfx3), std::abs(vex4-cfx4), std::abs(vex5-cfx5), std::abs(vex6-cfx6), std::abs(vex7-cfx7), std::abs(vex8-cfx8), std::abs(vex9-cfx9)});
      Real max_eig_7 = std::max({em, std::abs(vex0+csx0), std::abs(vex1+csx1), std::abs(vex2+csx2), std::abs(vex3+csx3), std::abs(vex4+csx4), std::abs(vex5+csx5), std::abs(vex6+csx6), std::abs(vex7+csx7), std::abs(vex8+csx8), std::abs(vex9+csx9)});
      Real max_eig_8 = std::max({em, std::abs(vex0-csx0), std::abs(vex1-csx1), std::abs(vex2-csx2), std::abs(vex3-csx3), std::abs(vex4-csx4), std::abs(vex5-csx5), std::abs(vex6-csx6), std::abs(vex7-csx7), std::abs(vex8-csx8), std::abs(vex9-csx9)});

      Real alpha[NMHD] = {max_eig_1, max_eig_2, max_eig_3, max_eig_4, max_eig_5, max_eig_6, max_eig_7, max_eig_8};
      

      for (int iii = 0; iii < NMHD; ++iii) {

        g_p0[iii] = 0.5 * (gj0[iii] + 1.1 * alpha[iii] * vj0[iii]);
        g_p1[iii] = 0.5 * (gj1[iii] + 1.1 * alpha[iii] * vj1[iii]);  
        g_p2[iii] = 0.5 * (gj2[iii] + 1.1 * alpha[iii] * vj2[iii]);  
        g_p3[iii] = 0.5 * (gj3[iii] + 1.1 * alpha[iii] * vj3[iii]);  
        g_p4[iii] = 0.5 * (gj4[iii] + 1.1 * alpha[iii] * vj4[iii]);  
        g_p5[iii] = 0.5 * (gj5[iii] + 1.1 * alpha[iii] * vj5[iii]);   
        g_p6[iii] = 0.5 * (gj6[iii] + 1.1 * alpha[iii] * vj6[iii]);   
        g_p7[iii] = 0.5 * (gj7[iii] + 1.1 * alpha[iii] * vj7[iii]);
        g_p8[iii] = 0.5 * (gj8[iii] + 1.1 * alpha[iii] * vj8[iii]);
        // g_p9[iii] = 0.5 * (gj9[iii] + 1.1 * alpha[iii] * vj9[iii]);   // not needed for WENO

        // g_m0[iii] = 0.5 * (gj0[iii] - 1.1 * alpha[iii] * vj0[iii]);   // not needed for WENO 
        g_m1[iii] = 0.5 * (gj1[iii] - 1.1 * alpha[iii] * vj1[iii]);  
        g_m2[iii] = 0.5 * (gj2[iii] - 1.1 * alpha[iii] * vj2[iii]);  
        g_m3[iii] = 0.5 * (gj3[iii] - 1.1 * alpha[iii] * vj3[iii]);  
        g_m4[iii] = 0.5 * (gj4[iii] - 1.1 * alpha[iii] * vj4[iii]);  
        g_m5[iii] = 0.5 * (gj5[iii] - 1.1 * alpha[iii] * vj5[iii]);  
        g_m6[iii] = 0.5 * (gj6[iii] - 1.1 * alpha[iii] * vj6[iii]);  
        g_m7[iii] = 0.5 * (gj7[iii] - 1.1 * alpha[iii] * vj7[iii]);  
        g_m8[iii] = 0.5 * (gj8[iii] - 1.1 * alpha[iii] * vj8[iii]);  
        g_m9[iii] = 0.5 * (gj9[iii] - 1.1 * alpha[iii] * vj9[iii]);  

      }

      //--- (e) Perform a WENO reconstruction on each of the computed flux components gj± to obtain 
      // the corresponding component of the numerical flux

      for (int jjj = 0; jjj < NMHD; ++jjj) {     

        weno_sum[jjj] = WENO9(g_p0[jjj], g_p1[jjj], g_p2[jjj], g_p3[jjj], g_p4[jjj], g_p5[jjj], g_p6[jjj], g_p7[jjj], g_p8[jjj]) 
                      + WENO9(g_m9[jjj], g_m8[jjj], g_m7[jjj], g_m6[jjj], g_m5[jjj], g_m4[jjj], g_m3[jjj], g_m2[jjj], g_m1[jjj]);  

      }

      //--- (f) Project the numerical flux back to the conserved variables
        
      for (int iiii = 0; iiii < NMHD; ++iiii) {
        f_half[iiii] = 0.0;
        for (int jjjj = 0; jjjj < NMHD; ++jjjj) {
          f_half[iiii] += rr[iiii][jjjj] * weno_sum[jjjj];
        }
      }

      //--- Step 3.  Update flux at each x_{i-1/2,j,k}:

      cons.flux(ivx, IDN, k, j, i) = f_half[IDN];
      cons.flux(ivx, ivx, k, j, i) = f_half[IV1];
      cons.flux(ivx, ivy, k, j, i) = f_half[IV2];
      cons.flux(ivx, ivz, k, j, i) = f_half[IV3];
      cons.flux(ivx, IEN, k, j, i) = f_half[IEN];
      cons.flux(ivx, iBx, k, j, i) = f_half[IB1];
      cons.flux(ivx, iBy, k, j, i) = f_half[IB2];
      cons.flux(ivx, iBz, k, j, i) = f_half[IB3];

    });
  }
};

#endif // MHD_WENO7_HPP_            