#!/bin/bash

source $MODULESHOME/init/bash
module purge
module load DefApps gcc netcdf-c netcdf-fortran cmake cuda/11.5.2 nco git

unset CXXFLAGS

export CC=gcc
export CXX=g++
export FC=gfortran
export CXX_LINK="`nc-config --libs`"
export F90_LINK="`nf-config --flibs`"

