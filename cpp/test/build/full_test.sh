#!/bin/bash -ex

this_dir=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)

#export RUNCMD=valgrind

# Just Kokkos
$this_dir/cmakescript.sh -DCMAKE_BUILD_TYPE=Debug
make -j8
$this_dir/test_lw.sh
$this_dir/test_sw.sh
