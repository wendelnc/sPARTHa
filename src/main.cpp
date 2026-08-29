// AthenaPK - a performance portable block structured AMR MHD code
// Copyright (c) 2020-2021, Athena Parthenon Collaboration. All rights reserved.
// Licensed under the 3-Clause License (the "LICENSE");

// Parthenon headers
#include "bvals/boundary_conditions_generic.hpp"
#include "defs.hpp"
#include "globals.hpp"
#include "parthenon_manager.hpp"

// AthenaPK headers
#include "bvals/boundary_conditions_apk.hpp"
// #include "hydro/srcterms/constant_accel.hpp"
#include "hydro/hydro.hpp"
#include "hydro/hydro_driver.hpp"
#include "main.hpp"
#include "pgen/pgen.hpp"

namespace Hydro {
InitPackageDataFun_t ProblemInitPackageData = nullptr;
SourceFun_t ProblemSourceFirstOrder = nullptr;
SourceFun_t ProblemSourceStrangSplit = nullptr;
SourceFun_t ProblemSourceUnsplit = nullptr;
EstimateTimestepFun_t ProblemEstimateTimestep = nullptr;
std::function<AmrTag(MeshBlockData<Real> *mbd)> ProblemCheckRefinementBlock = nullptr;
} // namespace Hydro

int main(int argc, char *argv[]) {
  using parthenon::ParthenonManager;
  using parthenon::ParthenonStatus;
  using BF = parthenon::BoundaryFace;
  using parthenon::BoundaryFunction::BCSide;
  ParthenonManager pman;

  // call ParthenonInit to initialize MPI and Kokkos, parse the input deck, and set up
  auto manager_status = pman.ParthenonInitEnv(argc, argv);
  if (manager_status == ParthenonStatus::complete) {
    pman.ParthenonFinalize();
    return 0;
  }
  if (manager_status == ParthenonStatus::error) {
    pman.ParthenonFinalize();
    return 1;
  }
  // Now that ParthenonInit has been called and setup succeeded, the code can now
  // make use of MPI and Kokkos

  // Redefine defaults
  pman.app_input->ProcessPackages = Hydro::ProcessPackages;
  const auto problem = pman.pinput->GetOrAddString("job", "problem_id", "unset");
  if (problem == "linear_wave") {
    pman.app_input->InitUserMeshData = linear_wave::InitUserMeshData;
    pman.app_input->ProblemGenerator = linear_wave::ProblemGenerator;
    pman.app_input->UserWorkAfterLoop = linear_wave::UserWorkAfterLoop;
  } else if (problem == "blast") {
    pman.app_input->ProblemGenerator = blast::ProblemGenerator;
  } else if (problem == "kh") {
    pman.app_input->ProblemGenerator = kh::ProblemGenerator;
  } else if (problem == "sod") {
    pman.app_input->ProblemGenerator = sod::ProblemGenerator;
  } else if (problem == "riemann_2d") {
    pman.app_input->ProblemGenerator = riemann_2d::ProblemGenerator;
  } else if (problem == "brio_wu") {
    pman.app_input->ProblemGenerator = brio_wu::ProblemGenerator;
  } else if (problem == "orszag_tang_2d") {
    pman.app_input->ProblemGenerator = orszag_tang_2d::ProblemGenerator;
  } else if (problem == "orszag_tang_3d") {
    pman.app_input->ProblemGenerator = orszag_tang_3d::ProblemGenerator;
  } else if (problem == "turbulence") {
    pman.app_input->MeshProblemGenerator = turbulence::ProblemGenerator;
    Hydro::ProblemInitPackageData = turbulence::ProblemInitPackageData;
    Hydro::ProblemSourceFirstOrder = turbulence::Driving;
    pman.app_input->InitMeshBlockUserData = turbulence::SetPhases;
    pman.app_input->MeshBlockUserWorkBeforeOutput = turbulence::UserWorkBeforeOutput;
  } else if (problem == "field_loop") {
    pman.app_input->ProblemGenerator = field_loop::ProblemGenerator;
    Hydro::ProblemInitPackageData = field_loop::ProblemInitPackageData;
  // } else if (problem == "smooth_alfven_2d") {
  //   pman.app_input->ProblemGenerator = smooth_alfven_2d::ProblemGenerator;
  } else if (problem == "cloud_shock") {
    pman.app_input->ProblemGenerator = cloud_shock::ProblemGenerator;
    pman.app_input->RegisterBoundaryCondition(parthenon::BoundaryFace::inner_x1,
                                              "inflow_x1", cloud_shock::InflowX1);
  } else if (problem == "taylorgreen") {
    pman.app_input->ProblemGenerator = taylorgreen::ProblemGenerator;
  
  // } else if (problem == "rt") {
  //   Hydro::ProblemInitPackageData = rt::ProblemInitPackageData;
  //   pman.app_input->ProblemGenerator = rt::ProblemGenerator;
  //   pman.app_input->RegisterBoundaryCondition(
  //       BF::inner_x2, "project_pressure",
  //       Hydro::BoundaryFunction::ProjectPressure<X2DIR, BCSide::Inner>);
  //   pman.app_input->RegisterBoundaryCondition(
  //       BF::outer_x2, "project_pressure",
  //       Hydro::BoundaryFunction::ProjectPressure<X2DIR, BCSide::Outer>);
  //   pman.app_input->RegisterBoundaryCondition(
  //       BF::inner_x3, "project_pressure",
  //       Hydro::BoundaryFunction::ProjectPressure<X3DIR, BCSide::Inner>);
  //   pman.app_input->RegisterBoundaryCondition(
  //       BF::outer_x3, "project_pressure",
  //       Hydro::BoundaryFunction::ProjectPressure<X3DIR, BCSide::Outer>);
  //   Hydro::ProblemSourceFirstOrder = const_accel::ConstantAccelSrcTerm;
  
  } else if (problem == "symmetric_implosion") {
    pman.app_input->ProblemGenerator = symmetric_implosion::ProblemGenerator;
  } else if (problem == "shu_osher") {
    pman.app_input->ProblemGenerator = shu_osher::ProblemGenerator;
  } else {
    // parthenon throw error message for the invalid problem
    std::stringstream msg;
    msg << "Problem ID '" << problem << "' is not implemented yet.";
    PARTHENON_THROW(msg);
  }

  const std::string REFLECTING = "reflecting";
  using Hydro::BoundaryFunction::ReflectBC;
  pman.app_input->RegisterBoundaryCondition(BF::inner_x1, REFLECTING,
                                            ReflectBC<X1DIR, BCSide::Inner>);
  pman.app_input->RegisterBoundaryCondition(BF::outer_x1, REFLECTING,
                                            ReflectBC<X1DIR, BCSide::Outer>);
  pman.app_input->RegisterBoundaryCondition(BF::inner_x2, REFLECTING,
                                            ReflectBC<X2DIR, BCSide::Inner>);
  pman.app_input->RegisterBoundaryCondition(BF::outer_x2, REFLECTING,
                                            ReflectBC<X2DIR, BCSide::Outer>);
  pman.app_input->RegisterBoundaryCondition(BF::inner_x3, REFLECTING,
                                            ReflectBC<X3DIR, BCSide::Inner>);
  pman.app_input->RegisterBoundaryCondition(BF::outer_x3, REFLECTING,
                                            ReflectBC<X3DIR, BCSide::Outer>);

  pman.ParthenonInitPackagesAndMesh();

  // Startup the corresponding driver for the integrator
  if (parthenon::Globals::my_rank == 0) {
    std::cout << "Starting up hydro driver" << std::endl;
  }

  Hydro::HydroDriver driver(pman.pinput.get(), pman.app_input.get(), pman.pmesh.get());

  // This line actually runs the simulation
  auto driver_status = driver.Execute();

  // call MPI_Finalize and Kokkos::finalize if necessary
  pman.ParthenonFinalize();

  // MPI and Kokkos can no longer be used

  return (0);
}
