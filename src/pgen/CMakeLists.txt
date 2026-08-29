# Copyright (c) 2020, Athena Parthenon Collaboration. All rights reserved.
# Licensed under the 3-Clause License (the "LICENSE");

target_sources(spartha PRIVATE
    linear_wave.cpp
    blast.cpp
    kh.cpp
    sod.cpp
    riemann_2d.cpp
    brio_wu.cpp
    orszag_tang_2d.cpp
    orszag_tang_3d.cpp
    turbulence.cpp
    field_loop.cpp
    smooth_alfven_2d.cpp
    cloud_shock.cpp
    taylor_green.cpp
    # rt.cpp
    symmetric_implosion.cpp
    shu_osher.cpp
    )
