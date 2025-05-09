#!/bin/bash

source $MODULESHOME/init/bash
module load PrgEnv-amd cray-hdf5 cray-netcdf cmake craype-accel-amd-gfx90a

unset CXXFLAGS

export CC=gcc
export CXX=hipcc
export FC=gfortran
export CXX_LINK="-L/opt/cray/pe/netcdf/4.8.1.3/amd/4.3/lib -lnetcdf --rocm-path=${ROCM_PATH} -L${ROCM_PATH}/lib -lamdhip64"
export F90_LINK="-L/opt/cray/pe/netcdf/4.8.1.3/amd/4.3/lib -lnetcdff -lnetcdf"
