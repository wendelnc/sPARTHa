//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//
// Edited by Katie Schram 2018
// Edited by Forrrest Glines 2019
//========================================================================================
//! \file taylor_green.cpp
//  \brief Problem generator for Taylor-Green vortex problem.
//
// REFERENCE: Brachet, M. E., Bustamante, M. D., Krstulovic, G., et al. 2013,
// Physical Review E, 87 Ideal evolution of MHD turbulence when imposing
// Taylor-Green symmetries
//========================================================================================

// Parthenon headers
#include "mesh/mesh.hpp"
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../main.hpp"

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Linear wave problem generator for 1D/2D/3D problems.
//========================================================================================

namespace taylorgreen {
using namespace parthenon::driver::prelude;

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  Real nghost = pin->GetOrAddReal("parthenon/mesh", "nghost", 3);

  int il, iu, jl, ju, kl, ku;
  il = ib.s - nghost, iu = ib.e + nghost;
  jl = jb.s - nghost, ju = jb.e + nghost;
  kl = kb.s - nghost, ku = kb.e + nghost;

  auto &mbd = pmb->meshblock_data.Get();
  auto &u = mbd->Get("cons").data;
  
  //Get user defined variables

  Real M0   = pin->GetReal("problem/taylorgreen",   "M0");
  Real rho0 = pin->GetReal("problem/taylorgreen", "rho0");
  Real pre0 = pin->GetReal("problem/taylorgreen",   "p0");
  Real b0   = pin->GetReal("problem/taylorgreen",   "b0");

  Real two_pi = 6.28318530717958647692528676655900576839433879875021164194988918461563281257;

  Real L = (pin->GetReal("parthenon/mesh","x1max") - pin->GetReal("parthenon/mesh","x1min") ) / (two_pi);

  std::cout << L << std::endl;

  Real gamma = pin->GetReal("hydro","gamma");
  Real gm1 =  gamma - 1.0;

  Real c = std::sqrt(gamma*pre0/rho0);
  Real v0 = c*M0;

  std::cout << M0  << std::endl;
  std::cout << std::sqrt( 16. / (6. * gamma)) << std::endl; 

  if (M0 > std::sqrt(16.0 / (6.0 * gamma))) {
  std::cout << "### FATAL ERROR in taylor_green.cpp ProblemGenerator\n"
            << "Mach number " << M0
            << " and adiabatic index " << gamma
            << " leads to negative pressures\n";
  std::exit(EXIT_FAILURE);
  }

  auto &coords = pmb->coords;

  pmb->par_for(
      "ProblemGenerator: Taylor Green 3D", kl, ku, jl, ju, il, iu,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {

        Real xtmp = coords.Xc<1>(i);
        Real ytmp = coords.Xc<2>(j);
        Real ztmp = coords.Xc<3>(k);

        Real pre = pre0 + (rho0*v0*v0/16) * (std::cos(2*xtmp/L) + std::cos(2*ytmp/L)) * (std::cos(2*ztmp/L) + 2);
        
        // Real rho = pre*rho0/pre0;
        Real rho = rho0;

        u(IDN, k, j, i) = rho;

        Real vex =   v0 * std::sin(xtmp/L) * std::cos(ytmp/L) * std::cos(ztmp/L); 
        Real vey = - v0 * std::cos(xtmp/L) * std::sin(ytmp/L) * std::cos(ztmp/L); 
        Real vez = 0.0;

        u(IM1, k, j, i) = rho * vex;
        u(IM2, k, j, i) = rho * vey;
        u(IM3, k, j, i) = rho * vez; 

        Real Bx =      b0 * std::cos(xtmp/L) * std::sin(ytmp/L) * std::sin(ztmp/L);
        Real By =      b0 * std::sin(xtmp/L) * std::cos(ytmp/L) * std::sin(ztmp/L);
        Real Bz = -2 * b0 * std::sin(xtmp/L) * std::sin(ytmp/L) * std::cos(ztmp/L);

        u(IB1, k, j, i) = Bx;
        u(IB2, k, j, i) = By;
        u(IB3, k, j, i) = Bz;

        Real Ax = -b0 * std::sin(xtmp/L) * std::cos(ytmp/L) * std::cos(ztmp/L);
        Real Ay =  b0 * std::cos(xtmp/L) * std::sin(ytmp/L) * std::cos(ztmp/L);
        Real Az = 0.0;        

        u(IA1, k, j, i) = Ax;
        u(IA2, k, j, i) = Ay;
        u(IA3, k, j, i) = Az;

        u(IEN, k, j, i) = pre / gm1 + 0.5 * rho * (vex * vex + vey * vey + vez * vez) + 0.5 * (Bx * Bx + By * By + Bz * Bz);

      });    

}
} // namespace taylorgreen

