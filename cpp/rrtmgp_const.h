
#pragma once

#include <iostream>
#include <cmath>
#include <vector>
#include <cstdlib>

#include <Kokkos_Core.hpp>

using DefaultDevice =
  Kokkos::Device<Kokkos::DefaultExecutionSpace, Kokkos::DefaultExecutionSpace::memory_space>;
using HostDevice =
  Kokkos::Device<Kokkos::DefaultHostExecutionSpace, Kokkos::DefaultHostExecutionSpace::memory_space>;

using real = double;
using string1dv = std::vector<std::string>;

inline void stoprun( std::string str ) {
  std::cout << "FATAL ERROR:\n";
  std::cout << str << "\n" << std::endl;
  throw str;
}

inline
std::ostream& operator<<(std::ostream& out, const string1dv& names)
{
  for (const auto& name : names) {
    out << name << " ";
  }
  out << std::endl;
  return out;
}
