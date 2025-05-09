#!/bin/bash -ex

this_dir=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)

# Clean previous build
$this_dir/cmakeclean.sh

# Configure new build. This will pass along arguments to this script
# to CMake so you can set things like CMAKE_BUILD_TYPE easily.
#
# Example: Enable Kokkos and debug build type:
#  $path_to_build/cmakescript.sh -DCMAKE_BUILD_TYPE=Debug
#
# Example: Enable Kokkos with release build type:
#  $path_to_build/cmakescript.sh -DCMAKE_BUILD_TYPE=Release
#
# Example: Enable Kokkos with release build type and machine setup
#  $path_to_build/cmakescript.sh -C $KOKKOSHOME/../../cmake/machine-files/mappy.cmake -DCMAKE_BUILD_TYPE=Release
#
# Example: Enable Kokkos on weaver
#  $path_to_build/cmakescript.sh -C $KOKKOSHOME/../../cmake/machine-files/weaver.cmake -DKokkos_ENABLE_CUDA_CONSTEXPR=On -DCMAKE_BUILD_TYPE=Release
cmake                                          \
  $@                                           \
  -DKokkos_DIR="$KOKKOSHOME"                   \
  ${KOKKOS_CONFIG}                             \
  -DCXX_LINK="$CXX_LINK"                       \
  -DF90_LINK="$F90_LINK"                       \
  -DCMAKE_Fortran_MODULE_DIRECTORY="`pwd`/fortran_module_files" \
  -DCMAKE_CUDA_ARCHITECTURES=OFF               \
  $this_dir/..
