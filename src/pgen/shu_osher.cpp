
// AthenaPK - a performance portable block structured AMR MHD code
// Copyright (c) 2020-2021, Athena Parthenon Collaboration. All rights reserved.
// Licensed under the 3-Clause License (the "LICENSE")

#include <cmath>

// Parthenon headers
#include "mesh/mesh.hpp"
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

// Athena headers
#include "../main.hpp"

namespace shu_osher {
using namespace parthenon::driver::prelude;

void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin) {
  auto hydro_pkg = pmb->packages.Get("Hydro");
  auto ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  auto jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  auto kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  Real rho_l = pin->GetOrAddReal("problem/shu_osher", "rho_l", 3.857143);
  Real vex_l = pin->GetOrAddReal("problem/shu_osher", "vex_l", 2.629369);
  Real pre_l = pin->GetOrAddReal("problem/shu_osher", "pre_l", 10.33333);

  Real rho0_r = pin->GetOrAddReal("problem/shu_osher", "rho0_r", 1.0);
  Real amp_r  = pin->GetOrAddReal("problem/shu_osher", "amp_r", 0.2);
  Real k_r    = pin->GetOrAddReal("problem/shu_osher", "k_r", 5.0);
  Real vex_r  = pin->GetOrAddReal("problem/shu_osher", "vex_r", 0.0);
  Real pre_r  = pin->GetOrAddReal("problem/shu_osher", "pre_r", 1.0);

  Real x_discont = pin->GetOrAddReal("problem/shu_osher", "x_discont", -0.8);

  Real gamma = pin->GetReal("hydro", "gamma");

  // initialize conserved variables
  auto &mbd = pmb->meshblock_data.Get();
  auto &cons = mbd->Get("cons").data;
  auto &coords = pmb->coords;

  pmb->par_for(
    "Init Shu-Osher", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
    KOKKOS_LAMBDA(const int k, const int j,
                  const int i) {

      Real rho, vx, vy = 0.0, vz = 0.0;
      Real pre;

      Real x = coords.Xc<1>(i);

      if (x < x_discont) {

        rho = rho_l;
        vx  = vex_l;
        pre = pre_l;

      } else {

        rho = rho0_r + amp_r * sin(k_r * M_PI * xgrid[i]);
        vx  = vex_r;
        pre = pre_r;

      }

      cons(IDN,k,j,i) = rho;
      cons(IM1,k,j,i) = rho * vx;
      cons(IM2,k,j,i) = 0.0;
      cons(IM3,k,j,i) = 0.0;

      cons(IEN,k,j,i) =
          pre/(gamma-1.0)
        + 0.5*rho*vx*vx;

  });

}
} // namespace shu_osher