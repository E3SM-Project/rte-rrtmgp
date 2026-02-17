###############################################################################
## This assumes you have already run add_subdirectory(${YAKL_HOME} ${YAKL_BIN})
## It also assumes you've enabled CXX and C as languages
###############################################################################

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake-modules)
set(CMAKE_PREFIX_PATH ${NETCDF_DIR} ${CMAKE_PREFIX_PATH})

# Compatibility with E3SM
if (NOT TARGET netcdf)
  find_package(NetCDF REQUIRED)
endif()

# Add RRTMGP source files.
set(CXX_SRC
  rrtmgp/kernels/mo_gas_optics_kernels.cpp
  rrtmgp/mo_rrtmgp_util_reorder.cpp
  rte/expand_and_transpose.cpp
  rte/kernels/mo_fluxes_broadband_kernels.cpp
  rte/kernels/mo_optical_props_kernels.cpp
  rte/kernels/mo_rte_solver_kernels.cpp
)

set (CMAKE_CXX_STANDARD 17)

# Add RRTMGP as a library and make sure we link against YAKL
add_library(rrtmgp STATIC ${CXX_SRC})
if (RRTMGP_ENABLE_YAKL)
  include(${YAKL_HOME}/yakl_utils.cmake)
  yakl_process_target(rrtmgp)
  target_include_directories(rrtmgp PUBLIC ${YAKL_HOME})
  message(STATUS "YAKL Flags: ${YAKL_COMPILER_FLAGS}")
endif()

if (RRTMGP_ENABLE_KOKKOS)
  target_link_libraries(rrtmgp Kokkos::kokkos)
endif()

if (NOT TARGET netcdf)
  target_include_directories(rrtmgp PUBLIC $<BUILD_INTERFACE:${NETCDF_INCLUDES}>)
  target_link_libraries(rrtmgp $<BUILD_INTERFACE:${NETCDF_LIBRARIES}>)
endif()

# Special compile flags for C++ source
target_include_directories(rrtmgp PUBLIC ./)
target_include_directories(rrtmgp PUBLIC rte)
target_include_directories(rrtmgp PUBLIC rte/kernels)
target_include_directories(rrtmgp PUBLIC rrtmgp)
target_include_directories(rrtmgp PUBLIC rrtmgp/kernels)
