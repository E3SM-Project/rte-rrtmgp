#!/bin/bash

unset ARCH
unset CUDA_ARCH
unset CUBHOME

# Just use the scream env

this_dir=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)

export CC=gcc
export CXX=g++
export FC=gfortran
export CXX_LINK="`nc-config --libs`"
export F90_LINK="`nf-config --flibs`"
export CXXFLAGS="-I`nc-config --includedir` -Wno-psabi"
E3SM_ROOT=$this_dir/../../../../../../../../../..
export KOKKOSHOME=$E3SM_ROOT/externals/ekat/extern/kokkos
export KOKKOS_CONFIG="-C $E3SM_ROOT/externals/ekat/cmake/machine-files/weaver.cmake"
