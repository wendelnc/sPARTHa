#ifndef PGEN_PGEN_HPP_
#define PGEN_PGEN_HPP_
//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2020-2022, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

namespace linear_wave {
using namespace parthenon::driver::prelude;

void InitUserMeshData(Mesh *, ParameterInput *pin);
void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
void UserWorkAfterLoop(Mesh *mesh, parthenon::ParameterInput *pin,
                       parthenon::SimTime &tm);
} // namespace linear_wave

namespace blast {
using namespace parthenon::driver::prelude;

void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
} // namespace blast

namespace kh {
using namespace parthenon::driver::prelude;

void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
} // namespace kh

namespace sod {
using namespace parthenon::driver::prelude;

void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
} // namespace sod

namespace riemann_2d {
using namespace parthenon::driver::prelude;

void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
} // namespace riemann_2d

namespace brio_wu {
using namespace parthenon::driver::prelude;

void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
} // namespace brio_wu


namespace orszag_tang_2d {
using namespace parthenon::driver::prelude;

void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
} // namespace orszag_tang_2d

namespace orszag_tang_3d {
using namespace parthenon::driver::prelude;

void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
} // namespace orszag_tang_3d

namespace turbulence {
using namespace parthenon::driver::prelude;

void ProblemGenerator(Mesh *pm, parthenon::ParameterInput *pin, MeshData<Real> *md);
void ProblemInitPackageData(ParameterInput *pin, parthenon::StateDescriptor *pkg);
void Driving(MeshData<Real> *md, const parthenon::SimTime &tm, const Real dt);
void SetPhases(MeshBlock *pmb, ParameterInput *pin);
void UserWorkBeforeOutput(MeshBlock *pmb, ParameterInput *pin,
                          const parthenon::SimTime &tm);
void Cleanup();
} // namespace turbulence

namespace field_loop {
using namespace parthenon::driver::prelude;

void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
void ProblemInitPackageData(ParameterInput *pin, parthenon::StateDescriptor *pkg);
} // namespace field_loop

namespace smooth_alfven_2d {
using namespace parthenon::driver::prelude;

void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
} // namespace smooth_alfven_2d

namespace cloud_shock {
using namespace parthenon::driver::prelude;

void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
void InflowX1(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse);

} // namespace cloud

namespace taylorgreen {
using namespace parthenon::driver::prelude;

void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
} // namespace taylorgreen

// namespace rt {
// using namespace parthenon::driver::prelude;

// void ProblemInitPackageData(ParameterInput *pin, parthenon::StateDescriptor *hydro_pkg);
// void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
// } // namespace rt

namespace symmetric_implosion {
using namespace parthenon::driver::prelude;

void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
} // namespace symmetric_implosion

namespace shu_osher {
using namespace parthenon::driver::prelude;

void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
} // namespace shu_osher

#endif // PGEN_PGEN_HPP_
