cmake_minimum_required(VERSION 3.0)
project(standalone)

set(CPP_SRC
    rrtmgp_allsky.cpp
)
add_executable(allsky ${CPP_SRC})

target_link_libraries(allsky rrtmgp "${CXX_LINK}")
target_include_directories(allsky PUBLIC ../)
target_include_directories(allsky PUBLIC ../../)
target_include_directories(allsky PUBLIC ../../extensions/cloud_optics)
target_include_directories(allsky PUBLIC ../../extensions/fluxes_byband)
target_include_directories(allsky PUBLIC ../../rte)
target_include_directories(allsky PUBLIC ../../rte/kernels)
target_include_directories(allsky PUBLIC ../../rrtmgp)
target_include_directories(allsky PUBLIC ../../rrtmgp/kernels)
