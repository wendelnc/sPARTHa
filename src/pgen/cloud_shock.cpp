//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file cloud_shock.cpp
//! \brief Problem generator for cloud in wind simulation.
//!

// C++ headers
#include <algorithm> // min, max
#include <cmath>     // log
#include <cstring>   // strcmp()

// Parthenon headers
#include "mesh/mesh.hpp"
#include <basic_types.hpp>
#include <iomanip>
#include <ios>
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
#include <random>
#include <sstream>

// AthenaPK headers
#include "../main.hpp"
#include "../units.hpp"

namespace cloud_shock {
using namespace parthenon::driver::prelude;

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Problem Generator for the cloud in wind setup

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {

  auto hydro_pkg = pmb->packages.Get("Hydro");
  Real gamma = pin->GetReal("hydro","gamma");

  // Bounds
  auto ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  auto jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  auto kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  int nghost = pin->GetOrAddInteger("parthenon/mesh", "nghost", 3);
  int il = ib.s - nghost, iu = ib.e + nghost;
  int jl = jb.s - nghost, ju = jb.e + nghost;

  // If ndim=2, k loops only over one zone
  int kl = kb.s;
  int ku = kb.e;

  // Detect dimensionality
  const bool two_d = (pmb->pmy_mesh->ndim < 3);

  // -------------------------------------------------------------------------------------
  // Read right-of-discontinuity (x<0.05)
  // -------------------------------------------------------------------------------------
  Real rho_r = pin->GetOrAddReal("problem/cloud_shock", "rho_r",   3.86859  );
  Real vex_r = pin->GetOrAddReal("problem/cloud_shock", "vex_r",  11.2536   );
  Real vey_r = pin->GetOrAddReal("problem/cloud_shock", "vey_r",   0.0      );
  Real vez_r = pin->GetOrAddReal("problem/cloud_shock", "vez_r",   0.0      );
  Real pre_r = pin->GetOrAddReal("problem/cloud_shock", "pre_r", 167.345    );
  Real Bx_r  = pin->GetOrAddReal("problem/cloud_shock", "Bx_r",    0.0      );
  Real By_r  = pin->GetOrAddReal("problem/cloud_shock", "By_r",    2.1826182);
  Real Bz_r  = pin->GetOrAddReal("problem/cloud_shock", "Bz_r",   -2.1826182);

  // -------------------------------------------------------------------------------------
  // Read left-of-discontinuity (x>0.05)
  // -------------------------------------------------------------------------------------
  Real rho_l = pin->GetOrAddReal("problem/cloud_shock", "rho_l", 1.0       );
  Real vex_l = pin->GetOrAddReal("problem/cloud_shock", "vex_l", 0.0       );
  Real vey_l = pin->GetOrAddReal("problem/cloud_shock", "vey_l", 0.0       );
  Real vez_l = pin->GetOrAddReal("problem/cloud_shock", "vez_l", 0.0       );
  Real pre_l = pin->GetOrAddReal("problem/cloud_shock", "pre_l", 1.0       );
  Real Bx_l  = pin->GetOrAddReal("problem/cloud_shock", "Bx_l", 0.0        );
  Real By_l  = pin->GetOrAddReal("problem/cloud_shock", "By_l", 0.56418958 );
  Real Bz_l  = pin->GetOrAddReal("problem/cloud_shock", "Bz_l", 0.56418958 );

  // -------------------------------------------------------------------------------------
  // Cloud parameters
  // -------------------------------------------------------------------------------------
  Real rho_c = pin->GetOrAddReal("problem/cloud_shock", "rho_c", 10.0);
  Real rad_c = pin->GetOrAddReal("problem/cloud_shock", "rad_c", 0.15);
  Real xc_c  = pin->GetOrAddReal("problem/cloud_shock", "xc_c", 0.25);
  Real yc_c  = pin->GetOrAddReal("problem/cloud_shock", "yc_c", 0.50);
  Real zc_c  = pin->GetOrAddReal("problem/cloud_shock", "zc_c", 0.50); // Not used in 2D

  // -------------------------------------------------------------------------------------
  // Magnetic potential slopes (from Eq. 3.61)
  // -------------------------------------------------------------------------------------
  Real x0 = 0.05;
  Real Az_slope_left  = -2.1826182;
  Real Az_slope_right = -0.56418958;

  auto &mbd   = pmb->meshblock_data.Get();
  auto &cons  = mbd->Get("cons").data;
  auto &coords = pmb->coords;

  pmb->par_for(
      "ProblemGenerator cloud_shock unified", kl, ku, jl, ju, il, iu,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {

        Real x = coords.Xc<1>(i);
        Real y = coords.Xc<2>(j);
        Real z = two_d ? 0.0 : coords.Xc<3>(k);

        // --------------------------------------------------
        // Shock state
        // --------------------------------------------------
        Real rho, vx, vy, vz, P, Bx, By, Bz;

        if (x < x0) {
          rho = rho_r;  vx = vex_r;  vy = vey_r;  
          vz = two_d ? 0.0 : vez_r;
          P = pre_r;   Bx = Bx_r;   By = By_r;   
          Bz = Bz_r;   // Bz exists even in 2D (Bz is out-of-plane component)
        } else {
          rho = rho_l;  vx = vex_l;  vy = vey_l;
          vz = two_d ? 0.0 : vez_l;
          P = pre_l;    Bx = Bx_l;   By = By_l;
          Bz = Bz_l;
        }

        // --------------------------------------------------
        // Cloud overwrite (circle in 2D, sphere in 3D)
        // --------------------------------------------------
        Real dx = x - xc_c;
        Real dy = y - yc_c;
        Real dz = two_d ? 0.0 : (z - zc_c);

        Real r2 = dx*dx + dy*dy + dz*dz;

        if (r2 < rad_c*rad_c) {
          rho = rho_c;
        }

        // --------------------------------------------------
        // Conserved variables
        // --------------------------------------------------
        cons(IDN,k,j,i) = rho;
        cons(IM1,k,j,i) = rho * vx;
        cons(IM2,k,j,i) = rho * vy;
        cons(IM3,k,j,i) = rho * vz;

        Real kinetic = 0.5 * rho * (vx*vx + vy*vy + vz*vz);
        Real magnetic= 0.5 * (Bx*Bx + By*By + Bz*Bz);
        cons(IEN,k,j,i) = P/(gamma - 1.0) + kinetic + magnetic;

        cons(IB1,k,j,i) = Bx;
        cons(IB2,k,j,i) = By;
        cons(IB3,k,j,i) = Bz;

        // --------------------------------------------------
        // Magnetic potential A^i (unstaggered)
        // 2D: only A^z non-zero (Eq. 3.61)
        // 3D: A^y and A^z non-zero (Eq. 3.64)
        // --------------------------------------------------

        if (two_d) {
          // ------------- 2D case -------------
          Real Az;
          if (x <= x0)
            Az = Az_slope_left * (x - x0);
          else
            Az = Az_slope_right * (x - x0);
        
            cons(IA1,k,j,i) = 0.0;
            cons(IA2,k,j,i) = 0.0;
            cons(IA3,k,j,i) = Az;
          } else {
          // ------------- 3D case -------------

          Real Ay;
          Real Az;
          
          if (x <= x0) {
            Ay = Az_slope_left * (x - x0);
            Az = Az_slope_left * (x - x0);
          } else{ 
            Ay =     Az_slope_right * (x - x0);
            Az = -1 * Az_slope_right * (x - x0);
          }

          cons(IA1,k,j,i) = 0.0; // A^x
          cons(IA2,k,j,i) = Ay;  // A^y
          cons(IA3,k,j,i) = Az;  // A^z
        }
        
      });

}

void InflowX1(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  
  auto pmb = mbd->GetBlockPointer();

  auto cons = mbd->PackVariables(std::vector<std::string>{"cons"}, coarse);

  const auto nb = IndexRange{0, 0};
  const bool fine = false;

  // --- Inflow state (use your left-of-discontinuity values) ---
  const Real rho0 = 1.0;   
  const Real vx0  = 0.0;   
  const Real vy0  = 0.0;   
  const Real vz0  = 0.0;    
  const Real P0   = 1.0;    

  const Real Bx0 = 0.0;
  const Real By0 = 0.56418958;
  const Real Bz0 = 0.56418958;

  const Real gamma = 1.666666666666667;

  pmb->par_for_bndry(
      "InflowX1", nb, IndexDomain::inner_x1, parthenon::TopologicalElement::CC,
      coarse, fine,
      KOKKOS_LAMBDA(const int &, const int &k, const int &j, const int &i) {
        // fill conserved vars
        cons(IDN, k, j, i) = rho0;
        cons(IM1, k, j, i) = rho0 * vx0;
        cons(IM2, k, j, i) = rho0 * vy0;
        cons(IM3, k, j, i) = rho0 * vz0;

        Real kinetic  = 0.5 * rho0 * (vx0*vx0 + vy0*vy0 + vz0*vz0);
        Real magnetic = 0.5 * (Bx0*Bx0 + By0*By0 + Bz0*Bz0);

        cons(IEN, k, j, i) = P0/(gamma - 1.0) + kinetic + magnetic;

        cons(IB1, k, j, i) = Bx0;
        cons(IB2, k, j, i) = By0;
        cons(IB3, k, j, i) = Bz0;
      });
}

} // namespace cloud_shock