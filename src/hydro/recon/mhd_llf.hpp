//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd_llf.hpp
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

#ifndef MHD_LLF_HPP_
#define MHD_LLF_HPP_

// C++ headers
#include <algorithm> // max(), min()
#include <cmath>     // sqrt()

// Athena headers
#include "../../main.hpp"
#include "../hydro.hpp"
#include "recon.hpp"

using parthenon::ParArray4D;
using parthenon::Real;

constexpr int NMHD = 8;

template <>
struct Reconstruct<Fluid::mhd, Reconstruction::llf> {
  static KOKKOS_INLINE_FUNCTION void Solve(const AdiabaticMHDEOS &eos, const int k,
                                           const int j, const int i, const int ivx,
                                           const VariablePack<Real> &prim,
                                           VariableFluxPack<Real> &cons) {
    
    const int ivy = IV1 + ((ivx - IV1) + 1) % 3;
    const int ivz = IV1 + ((ivx - IV1) + 2) % 3;
    const int iBx = ivx - 1 + NHYDRO;
    const int iBy = ivy - 1 + NHYDRO;
    const int iBz = ivz - 1 + NHYDRO;

    Real gamma;
    gamma = eos.GetGamma();
    Real gm1 = gamma - 1.0;
    Real igm1 = 1.0 / gm1;

    Real q0[(NMHD)], q1[(NMHD)];
    Real f0[(NMHD)], f1[(NMHD)];
    Real rr[(NMHD)][(NMHD)], ru[(NMHD)][(NMHD)];
    Real lu[(NMHD)][(NMHD)], lq[(NMHD)][(NMHD)];
    Real v0[(NMHD)], v1[(NMHD)];
    Real g0[(NMHD)], g1[(NMHD)];
    Real llf_sum[(NMHD)];
    Real f_half[(NMHD)]; 

    //--- Step 0.  Load states into local variables:
        
    if (ivx == IV1){
      q0[IDN] = cons(IDN, k, j, i - 1);
      q0[IM1] = cons(ivx, k, j, i - 1);
      q0[IM2] = cons(ivy, k, j, i - 1);
      q0[IM3] = cons(ivz, k, j, i - 1);
      q0[IEN] = cons(IEN, k, j, i - 1);
      q0[IB1] = cons(iBx, k, j, i - 1);
      q0[IB2] = cons(iBy, k, j, i - 1);
      q0[IB3] = cons(iBz, k, j, i - 1);
    } else if (ivx == IV2) {
      q0[IDN] = cons(IDN, k, j - 1, i);
      q0[IM1] = cons(ivx, k, j - 1, i);
      q0[IM2] = cons(ivy, k, j - 1, i);
      q0[IM3] = cons(ivz, k, j - 1, i);
      q0[IEN] = cons(IEN, k, j - 1, i);
      q0[IB1] = cons(iBx, k, j - 1, i);
      q0[IB2] = cons(iBy, k, j - 1, i);
      q0[IB3] = cons(iBz, k, j - 1, i); 
    } else if (ivx == IV3) {
      q0[IDN] = cons(IDN, k - 1, j, i);
      q0[IM1] = cons(ivx, k - 1, j, i);
      q0[IM2] = cons(ivy, k - 1, j, i);
      q0[IM3] = cons(ivz, k - 1, j, i);
      q0[IEN] = cons(IEN, k - 1, j, i);
      q0[IB1] = cons(iBx, k - 1, j, i);
      q0[IB2] = cons(iBy, k - 1, j, i);
      q0[IB3] = cons(iBz, k - 1, j, i);
    }  

    q1[IDN] = cons(IDN, k, j, i);
    q1[IM1] = cons(ivx, k, j, i);
    q1[IM2] = cons(ivy, k, j, i);
    q1[IM3] = cons(ivz, k, j, i);
    q1[IEN] = cons(IEN, k, j, i);
    q1[IB1] = cons(iBx, k, j, i);
    q1[IB2] = cons(iBy, k, j, i);
    q1[IB3] = cons(iBz, k, j, i);

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

    //--- Step 2.  At each x_{i-1/2,j,k}: (using cells i-1 and i)
    //--- (a) Compute the average state w_{i-1/2,j,k} in the primitive variables:

    Real half_den = 0.5 * (q0[IDN] + q1[IDN]);
    Real half_vex = 0.5 * (vex0 + vex1);
    Real half_vey = 0.5 * (vey0 + vey1);
    Real half_vez = 0.5 * (vez0 + vez1);
    Real half_pre = 0.5 * (pre0 + pre1);
    Real half_Bx  = 0.5 * (q0[IB1] + q1[IB1]);
    Real half_By  = 0.5 * (q0[IB2] + q1[IB2]);
    Real half_Bz  = 0.5 * (q0[IB3] + q1[IB3]);

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

      v0[ii] = 0.0;
      v1[ii] = 0.0;
      
      g0[ii] = 0.0;
      g1[ii] = 0.0;
      
      for (int jj = 0; jj < NMHD; ++jj) {

        v0[ii] += lq[ii][jj] * q0[jj];
        v1[ii] += lq[ii][jj] * q1[jj];
        
        g0[ii] += lq[ii][jj] * f0[jj];
        g1[ii] += lq[ii][jj] * f1[jj];

      }
    }

    //--- (d) Perform a Lax-Friedrichs flux vector splitting for the characteristic variables:
    // llf_sum = 0.5 * [ (g0 + g1) + 1.1 * α^{m} * (v1 - v0) ] where α(m) = max_k | λ^{m} q_k | 
    // is the maximal wave speed of the m^{th} component of characteristic variables over all grid points

    Real aa0 = std::sqrt(gamma * pre0 * irho0);
    Real aa1 = std::sqrt(gamma * pre1 * irho1);
    
    Real ca0 = std::sqrt(Bnm0 * irho0);
    Real ca1 = std::sqrt(Bnm1 * irho1);

    Real cax0 = std::sqrt(q0[IB1] * q0[IB1] * irho0);
    Real cax1 = std::sqrt(q1[IB1] * q1[IB1] * irho1);
  
    Real cfx0 = std::sqrt(0.5 * std::abs(aa0 * aa0 + ca0 * ca0 + std::sqrt((aa0 * aa0 + ca0 * ca0) * (aa0 * aa0 + ca0 * ca0) - (4 * aa0 * aa0 * cax0 * cax0))));
    Real cfx1 = std::sqrt(0.5 * std::abs(aa1 * aa1 + ca1 * ca1 + std::sqrt((aa1 * aa1 + ca1 * ca1) * (aa1 * aa1 + ca1 * ca1) - (4 * aa1 * aa1 * cax1 * cax1))));
      
    Real csx0 = std::sqrt(0.5 * std::abs(aa0 * aa0 + ca0 * ca0 - std::sqrt((aa0 * aa0 + ca0 * ca0) * (aa0 * aa0 + ca0 * ca0) - (4 * aa0 * aa0 * cax0 * cax0))));
    Real csx1 = std::sqrt(0.5 * std::abs(aa1 * aa1 + ca1 * ca1 - std::sqrt((aa1 * aa1 + ca1 * ca1) * (aa1 * aa1 + ca1 * ca1) - (4 * aa1 * aa1 * cax1 * cax1))));

    Real em = 1.0e-15;

    Real max_eig_1 = std::max({em, std::abs(vex0),      std::abs(vex1)});
    Real max_eig_2 = std::max({em, std::abs(vex0),      std::abs(vex1)});
    Real max_eig_3 = std::max({em, std::abs(vex0+cax0), std::abs(vex1+cax1)});
    Real max_eig_4 = std::max({em, std::abs(vex0-cax0), std::abs(vex1-cax1)});
    Real max_eig_5 = std::max({em, std::abs(vex0+cfx0), std::abs(vex1+cfx1)});
    Real max_eig_6 = std::max({em, std::abs(vex0-cfx0), std::abs(vex1-cfx1)});
    Real max_eig_7 = std::max({em, std::abs(vex0+csx0), std::abs(vex1+csx1)});
    Real max_eig_8 = std::max({em, std::abs(vex0-csx0), std::abs(vex1-csx1)});

    Real alpha[NMHD] = {max_eig_1, max_eig_2, max_eig_3, max_eig_4, max_eig_5, max_eig_6, max_eig_7, max_eig_8};

    for (int iii = 0; iii < NMHD; ++iii) {

      llf_sum[iii] = 0.5 * ((g0[iii] + g1[iii]) + 1.1 * alpha[iii] * (v1[iii] - v0[iii]));

    }

    //--- (e) Project the numerical flux back to the conserved variables
      
    for (int iiii = 0; iiii < NMHD; ++iiii) {
      f_half[iiii] = 0.0;
      for (int jjjj = 0; jjjj < NMHD; ++jjjj) {
        f_half[iiii] += rr[iiii][jjjj] * llf_sum[jjjj];
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

  }
};

#endif // MHD_LLF_HPP_

