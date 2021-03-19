
#pragma once

#include "const.h"
#include "mo_optical_props.h"

// Expand from band to g-point dimension, transpose dimensions (nband, ncol) -> (ncol,ngpt)
void expand_and_transpose(OpticalProps const &ops, real2d const &arr_in, real2d &arr_out);
// Expand from band to g-point dimension, do NOT transpose dimensions
void expand(OpticalProps const &ops, real2d const &arr_in, real2d &arr_out);


