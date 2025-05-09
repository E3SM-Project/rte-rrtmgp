###############################################################################
# It assumes you've enabled CXX and C as languages
###############################################################################

set (CMAKE_CXX_STANDARD 17)

# Add RRTMGP as a library and make sure we link against kokkos
add_library(rrtmgp INTERFACE)
target_link_libraries(rrtmgp INTERFACE Kokkos::kokkos)

# Special compile flags for C++ source
target_include_directories(rrtmgp INTERFACE ./)
target_include_directories(rrtmgp INTERFACE rte)
target_include_directories(rrtmgp INTERFACE rte/kernels)
target_include_directories(rrtmgp INTERFACE rrtmgp)
target_include_directories(rrtmgp INTERFACE rrtmgp/kernels)

execute_process(COMMAND nc-config --includedir OUTPUT_VARIABLE NCINCL OUTPUT_STRIP_TRAILING_WHITESPACE)
target_include_directories(rrtmgp INTERFACE ${NCINCL})
