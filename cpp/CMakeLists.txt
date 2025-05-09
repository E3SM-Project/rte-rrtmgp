###############################################################################
# It assumes you've enabled CXX and C as languages
###############################################################################

# Add RRTMGP source files.
set(CXX_SRC
  rte/expand_and_transpose.cpp
  rte/kernels/mo_fluxes_broadband_kernels.cpp
  rte/kernels/mo_optical_props_kernels.cpp
  rte/kernels/mo_rte_solver_kernels.cpp
)

set (CMAKE_CXX_STANDARD 17)

# Add RRTMGP as a library and make sure we link against kokkos
add_library(rrtmgp STATIC ${CXX_SRC})
target_link_libraries(rrtmgp Kokkos::kokkos)

# Special compile flags for C++ source
target_include_directories(rrtmgp PUBLIC ./)
target_include_directories(rrtmgp PUBLIC rte)
target_include_directories(rrtmgp PUBLIC rte/kernels)
target_include_directories(rrtmgp PUBLIC rrtmgp)
target_include_directories(rrtmgp PUBLIC rrtmgp/kernels)
