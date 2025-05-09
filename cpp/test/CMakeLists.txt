cmake_minimum_required(VERSION 3.0)
project(standalone C Fortran CXX)

enable_testing()

# TODO REMOVE
add_definitions("-DRRTMGP_ENABLE_KOKKOS")

set (CMAKE_CXX_STANDARD 17)

# Build RRTMGP, Kokkos, and the standalone allsky code (which depends on both)
# NOTE: ideally, we would build RRTMGP as a library and link allsky against it,
# but that does not seem to be working for GPU builds on summit right now for
# some reason (leads to segfaults). So instead, the allsky build will just pull
# in all the RRTMGP source and build and link all at once.
add_subdirectory(${Kokkos_DIR} ./kokkos)
add_subdirectory(../ ./rrtmgp)
add_subdirectory(../examples/all-sky ./allsky)
add_subdirectory(./allsky_fortran ./allsky_fortran)
add_subdirectory(./allsky_fortran_openacc ./allsky_fortran_openacc)

# Add unit test script commands
add_test(NAME allsky_lw COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/build/test_lw.sh)
add_test(NAME allsky_sw COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/build/test_sw.sh)
add_test(NAME allsky_lw_r COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/build/test_lw_r.sh)
add_test(NAME allsky_sw_r COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/build/test_sw_r.sh)
