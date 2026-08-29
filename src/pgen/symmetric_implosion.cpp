//========================================================================================
// AthenaPK - a performance portable block structured AMR MHD code
// Copyright (c) 2021-2023, Athena Parthenon Collaboration. All rights reserved.
// Licensed under the 3-Clause License (the "LICENSE")
//========================================================================================
//! \file symmetric_implosion.cpp
//! \brief Problem generator for 4.7 Symmetric Implosion Problem 
//!
//! REFERENCE: Comparison of Several Different Schemes on 
//! 1D and 2D Test Problems for the Euler Equations
//! by Kiska and Wendroff (2003) SIAM J. Sci. Comput Vol. 25, No. 3, pp. 995-1017
//========================================================================================

#include <iomanip>   // For std::setprecision

// Parthenon headers
#include "mesh/mesh.hpp"
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../main.hpp"

namespace symmetric_implosion {
using namespace parthenon::driver::prelude;

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  Real nghost = pin->GetOrAddReal("parthenon/mesh", "nghost", 3);

  int il, iu, jl, ju, kl, ku;
  il = ib.s - nghost, iu = ib.e + nghost;
  jl = jb.s - nghost, ju = jb.e + nghost;
  kl = kb.s, ku = kb.e;

  auto &mbd = pmb->meshblock_data.Get();
  auto &u = mbd->Get("cons").data;
  Real gamma = pin->GetReal("hydro", "gamma");

  Real gm1 =  gamma - 1.0;

  Real rho0 = 1.0;
  Real rhoi = 0.125;

  Real pre0 = 1.0;
  Real prei = 0.14;

  Real vex = 0.0;
  Real vey = 0.0;

  auto &coords = pmb->coords;

  pmb->par_for(
      "ProblemGenerator: symmetric_implosion", kl, ku, jl, ju, il, iu,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        
        Real rho, pre; 

        const Real x = coords.Xc<1>(i);
        const Real y = coords.Xc<2>(j);

        if ((x + y) < 0.15) { // inside rotated square
          rho = rhoi;
          pre = prei;
        } else { // outside rotated square
          rho = rho0;
          pre = pre0;
        }

        u(IDN, k, j, i) = rho;    


        u(IM1, k, j, i) = rho * vex;
        u(IM2, k, j, i) = rho * vey; 

        u(IEN, k, j, i) = pre / gm1 + 0.5 * rho * (vex * vex + vey * vey);

      });
}
} // namespace symmetric_implosion
