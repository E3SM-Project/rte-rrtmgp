#!/bin/bash

unset CXXFLAGS

export CC=gcc
export CXX=g++
export CXX_LINK="`nc-config --libs`"
export F90_LINK="`nf-config --flibs`"

