#!/bin/bash

source $MODULESHOME/init/bash
module purge
module load DefApps gcc cmake cuda netcdf-cxx4 netcdf

export NCFLAGS="`ncxx4-config --libs`"
export CC=gcc
export CXX=g++
export CXXFLAGS="-O3"
export ARCH="CUDA"
export CUDA_ARCH="-arch sm_70 --std=c++14 --use_fast_math -O3"
export CUBHOME="/ccs/home/$USER/cub"
export YAKLHOME="/ccs/home/$USER/YAKL"
