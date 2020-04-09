//
// Contacts: Robert Pincus and Eli Mlawer
// email:  rrtmgp@aer.com
//
// Copyright 2015-2018,  Atmospheric and Environmental Research and
// Regents of the University of Colorado.  All right reserved.
//
// Use and duplication is permitted under the terms of the
//    BSD 3-clause license, see http://opensource.org/licenses/BSD-3-Clause
// -------------------------------------------------------------------------------------------------
//
// Compute output quantities from RTE based on spectrally-resolved flux profiles
//    This module contains an abstract class and a broadband implmentation that sums over all spectral points
//    The abstract base class defines the routines that extenstions must implement: reduce() and are_desired()
//    The intent is for users to extend it as required, using mo_flxues_broadband as an example
//
// -------------------------------------------------------------------------------------------------

class FluxesBroadband {
public:
  real2d flux_up;
  real2d flux_net;
  real2d flux_dn_dir;


  void reduce(real3d const &gpt_flux_up, const real3d &gpt_flux_dn, OpticalProps const &spectral_disc,
              bool top_at_1, real3d const &gpt_flux_dn_dir=real3d()) {
    int ncol = size(gpt_flux_up,1);
    int nlev = size(gpt_flux_up,2);
    int ngpt = size(gpt_flux_up,3);
    // Check array sizes
    //  Input arrays
    if (size(gpt_flux_dn,1) != ncol || size(gpt_flux_dn,2) != nlev || size(gpt_flux_dn,3) != ngpt) {
      stoprun("reduce: gpt_flux_dn array incorrectly sized");
    }
    if (allocated(gpt_flux_dn_dir)) {
      if (size(gpt_flux_dn_dir,1) != ncol || size(gpt_flux_dn_dir,2) != nlev || size(gpt_flux_dn_dir,3) != ngpt) {
        stoprun("reduce: gpt_flux_dn_dir array incorrectly sized");
      }
    }
    // Output arrays
    if (allocated(this->flux_up)) {
      if (size(this->flux_up,1) != ncol || size(this->flux_up,2) != nlev) { stoprun("reduce: flux_up array incorrectly sized"); }
    }
    if (allocated(this->flux_dn)) {
      if (size(this->flux_dn,1) != ncol || size(this->flux_dn,2) != nlev) { stoprun("reduce: flux_dn array incorrectly sized"); }
    }
    if (allocated(this->flux_net)) {
      if (size(this->flux_net,1) != ncol || size(this->flux_net,2) != nlev) { stoprun("reduce: flux_net array incorrectly sized"); }
    }
    if (allocated(this->flux_dn_dir)) {
      if (size(this->flux_dn_dir,1) != ncol || size(this->flux_dn_dir,2) != nlev) { stoprun("reduce: flux_dn_dir array incorrectly sized"); }
    }
    ! Self-consistency -- shouldn't be asking for direct beam flux if it isn't supplied
    if (allocated(this->flux_dn_dir) && ! allocated(gpt_flux_dn_dir)) {
      stoprun("reduce: requesting direct downward flux but this hasn't been supplied");
    }

    // Broadband fluxes - call the kernels
    if (allocated(this->flux_up    )) { sum_broadband(ncol, nlev, ngpt, gpt_flux_up,     this->flux_up    ); }
    if (allocated(this->flux_dn    )) { sum_broadband(ncol, nlev, ngpt, gpt_flux_dn,     this->flux_dn    ); }
    if (allocated(this->flux_dn_dir)) { sum_broadband(ncol, nlev, ngpt, gpt_flux_dn_dir, this->flux_dn_dir); }
    if (allocated(this->flux_net   )) {
      // Reuse down and up results if possible
      if (allocated(this->flux_dn) .and. allocated(this->flux_up)) {
        call net_broadband(ncol, nlev,      this->flux_dn, this->flux_up, this->flux_net);
      } else {
        call net_broadband(ncol, nlev, ngpt,  gpt_flux_dn,   gpt_flux_up, this->flux_net);
      }
    }
  }


  bool are_desired() {
    return allocated(this->flux_up) || allocated(this->flux_dn) || allocated(this->flux_dn_dir) || allocated(this->flux_net);
  }
};


