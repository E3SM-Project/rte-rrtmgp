#!/bin/bash

unset CXXFLAGS

export CC=gcc-11
export CXX=g++-11
export CXX_LINK="`nc-config --libs`"
export F90_LINK="`nf-config --flibs`"

