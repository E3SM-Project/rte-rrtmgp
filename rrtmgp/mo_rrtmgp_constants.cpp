
#include "mo_rrtmgp_constants.h"


real m_dry;
real grav;
real cp_dry;


void init_constants(real gravity, real mol_weight_dry_air, real heat_capacity_dry_air) {
  grav   = gravity;
  m_dry  = mol_weight_dry_air;
  cp_dry = heat_capacity_dry_air;
}



