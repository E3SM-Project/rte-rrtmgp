#!/bin/bash

source $MODULESHOME/init/bash
module purge
module load DefApps gcc/11.2.0 netcdf-c netcdf-fortran cmake nco git

unset CXXFLAGS

export CC=gcc
export CXX=g++
export FC=gfortran
export CXX_LINK="`nc-config --libs`"
export F90_LINK="`nf-config --flibs`"

