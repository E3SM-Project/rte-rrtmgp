
#pragma once

#include "rrtmgp_const.h"

inline std::string lower_case( std::string in ) {
  std::for_each( in.begin() , in.end() , [] (char & c) { c = ::tolower(c); } );
  return in;
}

inline bool string_in_array(std::string str, string1dv const &arr) {
  auto lstr = lower_case(str);
  for (const auto& item : arr) {
    if ( lstr == lower_case(item) ) { return true; }
  }
  return false;
}

inline int string_loc_in_array(std::string str, string1dv const &arr) {
  auto lstr = lower_case(str);
  for (int i=0; i < arr.size(); i++) {
    if ( lstr == lower_case(arr[i]) ) { return i; } // Note: 0-based
  }
  return -1;
}

template <typename InT>
inline string1dv char2d_to_string1d( const InT &in , std::string label="") {
  int nstr  = in.extent(1);
  string1dv out(nstr, "");
  for (int j=0 ; j < nstr ; ++j) {
    for (int i=0 ; i < in.extent(0); ++i) {
      if ( ! isspace(in(i,j)) ) { out[j] += in(i,j); }
    }
  }
  return out;
}
