//========================================================================================
// parthenon-hydro - a performance portable block-structured AMR compr. hydro miniapp
// Copyright (c) 2020-2023, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================

#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// Parthenon header
#include <amr_criteria/refinement_package.hpp>
#include <parthenon/parthenon.hpp>
#include <prolong_restrict/prolong_restrict.hpp>

// parthenon-hydro headers
#include "../eos/adiabatic_hydro.hpp"
#include "../eos/adiabatic_mhd.hpp"
#include "hydro.hpp"
#include "hydro_driver.hpp"
// #include "recon/mp_weno3.hpp"
// #include "recon/mp_weno5.hpp"
// #include "recon/mp_weno7.hpp"
// #include "recon/mp_weno9.hpp"

using namespace parthenon::driver::prelude;

namespace Hydro {

template <>
TaskStatus UpdateWithFluxDivergenceCA(MeshData<Real> *u0_data, MeshData<Real> *u1_data,
                                    const Real gam0, const Real gam1,
                                    const Real beta_dt) {
  const IndexDomain interior = IndexDomain::interior;

  std::vector<parthenon::MetadataFlag> flags({Metadata::WithFluxes});
  auto u0_pack = u0_data->PackVariablesAndFluxes(flags);
  const auto &u1_pack = u1_data->PackVariables(flags);
  const IndexRange ib = u0_data->GetBoundsI(interior);
  const IndexRange jb = u0_data->GetBoundsJ(interior);
  const IndexRange kb = u0_data->GetBoundsK(interior);
  const int ndim = u0_pack.GetNdim();
  // std::cout << "UpdateWithFluxDivergenceCA" << std::endl;

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "UpdateWithFluxDivergenceMesh", DevExecSpace(), 0,
      u0_pack.GetDim(5) - 1, 0, u0_pack.GetDim(4) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int m, const int l, const int k, const int j, const int i) {
        if (u0_pack.IsAllocated(m, l) && u1_pack.IsAllocated(m, l)) {
          const auto &coords = u0_pack.GetCoords(m);
          const auto &u0 = u0_pack(m);
          if (l < 8) {
            u0_pack(m, l, k, j, i) = gam0 * u0(l, k, j, i) + gam1 * u1_pack(m, l, k, j, i) +
                                   beta_dt * parthenon::Update::FluxDivHelper(l, k, j, i, ndim, coords, u0);
          } else {
            u0_pack(m, l, k, j, i) = gam0 * u0(l, k, j, i) + gam1 * u1_pack(m, l, k, j, i) -
                                   beta_dt * u0.flux(X1DIR, l, k, j, i);
          }
        }
      });
  return TaskStatus::complete;
}  

HydroDriver::HydroDriver(ParameterInput *pin, ApplicationInput *app_in, Mesh *pm)
    : MultiStageDriver(pin, app_in, pm) {
  // fail if these are not specified in the input file
  pin->CheckRequired("hydro", "gamma");

  // warn if these fields aren't specified in the input file
  pin->CheckDesired("parthenon/time", "cfl");
}

// See the advection.hpp declaration for a description of how this function gets called.
TaskCollection HydroDriver::MakeTaskCollection(BlockList_t &blocks, int stage) {
  TaskCollection tc;
  const auto &stage_name = integrator->stage_name;
  auto hydro_pkg = blocks[0]->packages.Get("Hydro");

  auto hj_flux_2d =
      hydro_pkg->Param<HJFlux2DFunc>("hj_flux_2d");

  auto hj_flux_3d =
      hydro_pkg->Param<HJFlux3DFunc>("hj_flux_3d");

  auto hj_afterstep_2d =
      hydro_pkg->Param<HJAfterstep2DFunc>("hj_afterstep_2d");

  auto hj_afterstep_3d =
      hydro_pkg->Param<HJAfterstep3DFunc>("hj_afterstep_3d");

  TaskID none(0);
  // Number of task lists that can be executed indepenently and thus *may*
  // be executed in parallel and asynchronous.
  // Being extra verbose here in this example to highlight that this is not
  // required to be 1 or blocks.size() but could also only apply to a subset of blocks.
  auto num_task_lists_executed_independently = blocks.size();

  TaskRegion &async_region_1 = tc.AddRegion(num_task_lists_executed_independently);
  for (int i = 0; i < blocks.size(); i++) {
    auto &pmb = blocks[i];
    auto &tl = async_region_1[i];
    auto &u0 = pmb->meshblock_data.Get();

    // init u1, see (11) in Athena++ method paper
    if (stage == 1) {
      pmb->meshblock_data.Add("u1", u0);

      auto &u1 = pmb->meshblock_data.Get("u1");
      auto init_u1 = tl.AddTask(
          none,
          [](MeshBlockData<Real> *u0, MeshBlockData<Real> *u1, bool copy_prim) {
            u1->Get("cons").data.DeepCopy(u0->Get("cons").data);
            if (copy_prim) {
              u1->Get("prim").data.DeepCopy(u0->Get("prim").data);
            }
            return TaskStatus::complete;
          },
          // First order flux correction needs the original prim variables in the
          // during the correction.          
          u0.get(), u1.get(), hydro_pkg->Param<bool>("first_order_flux_correct"));
    }
  }

  const int num_partitions = pmesh->DefaultNumPartitions();

  // note that task within this region that contains one tasklist per pack
  // could still be executed in parallel
  TaskRegion &single_tasklist_per_pack_region = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = single_tasklist_per_pack_region[i];
    auto &mu0 = pmesh->mesh_data.GetOrAdd("base", i);
    auto &mu1 = pmesh->mesh_data.GetOrAdd("u1", i);

    const auto any = parthenon::BoundaryType::any;
  
    auto start_bnd = tl.AddTask(none, parthenon::StartReceiveBoundBufs<any>, mu0);
    auto start_flxcor_recv = tl.AddTask(none, parthenon::StartReceiveFluxCorrections, mu0);

    // Calculate fluxes (will be stored in the x1, x2, x3 flux arrays of each var)
    const auto flux_str = (stage == 1) ? "flux_first_stage" : "flux_other_stage";
    FluxFun_t *calc_flux_fun = hydro_pkg->Param<FluxFun_t *>(flux_str);
    auto calc_flux = tl.AddTask(none, calc_flux_fun, mu0);

    // TODO(pgrete) figure out what to do about the sources from the first stage
    // that are potentially disregarded when the (m)hd fluxes are corrected in the second
    // stage.
    TaskID first_order_flux_correct = calc_flux;
    if (hydro_pkg->Param<bool>("first_order_flux_correct")) {
      auto *first_order_flux_correct_fun =
          hydro_pkg->Param<FirstOrderFluxCorrectFun_t *>("first_order_flux_correct_fun");
      first_order_flux_correct =
          tl.AddTask(calc_flux, first_order_flux_correct_fun, mu0.get(), mu1.get(),
                     integrator->gam0[stage - 1], integrator->gam1[stage - 1],
                     integrator->beta[stage - 1] * integrator->dt);
    }

    // Unstaggered Constrained Transport Step
    Fluid fluid = hydro_pkg->Param<Fluid>("fluid");
    auto pmb = mu0->GetBlockData(0)->GetBlockPointer();
    auto calc_mp_flux = first_order_flux_correct;
    if (fluid == Fluid::mhd) {
      if (pmb->pmy_mesh->ndim == 2) {
        calc_mp_flux = tl.AddTask(first_order_flux_correct, hj_flux_2d, mu0);
      } else if (pmb->pmy_mesh->ndim == 3) {
        calc_mp_flux = tl.AddTask(first_order_flux_correct, hj_flux_3d, mu0, integrator->dt);
      }
    }

    // Correct for fluxes across levels (to maintain conservative nature of update)
    auto send_flx = tl.AddTask(calc_mp_flux, parthenon::LoadAndSendFluxCorrections, mu0);
    auto recv_flx = tl.AddTask(start_flxcor_recv, parthenon::ReceiveFluxCorrections, mu0);
    auto set_flx  = tl.AddTask(recv_flx | calc_mp_flux, parthenon::SetFluxCorrections, mu0);

    // Compute the divergence of fluxes of conserved variables
    auto update = tl.AddTask(
        set_flx, UpdateWithFluxDivergenceCA<MeshData<Real>>, mu0.get(),
        mu1.get(), integrator->gam0[stage - 1], integrator->gam1[stage - 1],
        integrator->beta[stage - 1] * integrator->dt);

    // Turbulence Driver Force
    auto source_split_first_order = update;
    if (stage == integrator->nstages) {
      source_split_first_order =
          tl.AddTask(update, AddSplitSourcesFirstOrder, mu0.get(), tm);
    } 

    // Update ghost cells (local and non local)
    // Note that Parthenon also support to add those tasks manually for more fine-grained
    // control.
    auto bcstep1 = parthenon::AddBoundaryExchangeTasks(source_split_first_order | start_bnd, tl, mu0, pmesh->multilevel); 


    // Unstaggered Constrained Transport Afterstep
    if (fluid == Fluid::mhd) {
      if (pmb->pmy_mesh->ndim == 2) {

        auto afterstep_flux = tl.AddTask(bcstep1, hj_afterstep_2d, mu0);
        parthenon::AddBoundaryExchangeTasks(afterstep_flux | start_bnd, tl, mu0, pmesh->multilevel);

      } else if (pmb->pmy_mesh->ndim == 3) {

        auto afterstep_flux = tl.AddTask(bcstep1, hj_afterstep_3d, mu0);
        parthenon::AddBoundaryExchangeTasks(afterstep_flux | start_bnd, tl, mu0, pmesh->multilevel);
      }
    }


  } // single_tasklist_per_pack_region
  
  TaskRegion &single_tasklist_per_pack_region_3 = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = single_tasklist_per_pack_region_3[i];
    auto &mu0 = pmesh->mesh_data.GetOrAdd("base", i);
    auto fill_derived =
        tl.AddTask(none, parthenon::Update::FillDerived<MeshData<Real>>, mu0.get());
  }
  
  if (stage == integrator->nstages) {
    TaskRegion &tr = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; i++) {
      auto &tl = tr[i];
      auto &mu0 = pmesh->mesh_data.GetOrAdd("base", i);
      auto new_dt = tl.AddTask(none, parthenon::Update::EstimateTimestep<MeshData<Real>>,
                               mu0.get());
    }
  }

  

  if (stage == integrator->nstages && pmesh->adaptive) {
    TaskRegion &async_region_4 = tc.AddRegion(num_task_lists_executed_independently);
    for (int i = 0; i < blocks.size(); i++) {
      auto &tl = async_region_4[i];
      auto &u0 = blocks[i]->meshblock_data.Get("base");
      auto tag_refine =
          tl.AddTask(none, parthenon::Refinement::Tag<MeshBlockData<Real>>, u0.get());
    }
  }

  return tc;
}
} // namespace Hydro