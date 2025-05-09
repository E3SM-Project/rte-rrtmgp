#pragma once

#include "rrtmgp_const.h"
#include "mo_fluxes_broadband_kernels.h"
#include <iomanip>

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


template <typename RealT=double, typename LayoutT=Kokkos::LayoutLeft, typename DeviceT=DefaultDevice>
class FluxesBroadbandK {
public:

  using real2d_t = Kokkos::View<RealT**,  LayoutT, DeviceT>;
  using real3d_t = Kokkos::View<RealT***, LayoutT, DeviceT>;

  real2d_t flux_up;
  real2d_t flux_dn;
  real2d_t flux_net;
  real2d_t flux_dn_dir;

  template <typename FluxUpT, typename FluxDnT, typename FluxDnDirT = real3d_t>
  void reduce(FluxUpT const &gpt_flux_up, const FluxDnT &gpt_flux_dn,
              OpticalPropsK<RealT, LayoutT, DeviceT> const &spectral_disc,
              bool top_at_1, FluxDnDirT const &gpt_flux_dn_dir=FluxDnDirT()) {
    int ncol = gpt_flux_up.extent(0);
    int nlev = gpt_flux_up.extent(1);
    int ngpt = gpt_flux_up.extent(2);

    // Self-consistency -- shouldn't be asking for direct beam flux if it isn't supplied
    if (this->flux_dn_dir.is_allocated() && ! gpt_flux_dn_dir.is_allocated()) {
      stoprun("reduce: requesting direct downward flux but this hasn't been supplied");
    }

    // Broadband fluxes - call the kernels
    if (this->flux_up.is_allocated()    ) { sum_broadband(ncol, nlev, ngpt, gpt_flux_up,     this->flux_up    ); }
    if (this->flux_dn.is_allocated()    ) { sum_broadband(ncol, nlev, ngpt, gpt_flux_dn,     this->flux_dn    ); }
    if (this->flux_dn_dir.is_allocated()) { sum_broadband(ncol, nlev, ngpt, gpt_flux_dn_dir, this->flux_dn_dir); }
    if (this->flux_net.is_allocated()   ) {
      // Reuse down and up results if possible
      if (this->flux_dn.is_allocated() && this->flux_up.is_allocated()) {
        net_broadband(ncol, nlev,      this->flux_dn, this->flux_up, this->flux_net);
      } else {
        net_broadband(ncol, nlev, ngpt,  gpt_flux_dn,   gpt_flux_up, this->flux_net);
      }
    }
  }

  bool are_desired() const {
    return this->flux_up.is_allocated() || this->flux_dn.is_allocated() || this->flux_dn_dir.is_allocated() || this->flux_net.is_allocated();
  }

  void print_norms(const bool print_prefix=false) const {
    std::string prefix = print_prefix ? "JGFK" : "";
    if (flux_up.is_allocated()    ) { std::cout << prefix << std::setprecision(16) << "flux_up    : " << conv::sum(flux_up    ) << "\n"; }
    if (flux_dn.is_allocated()    ) { std::cout << prefix << std::setprecision(16) << "flux_dn    : " << conv::sum(flux_dn    ) << "\n"; }
    if (flux_net.is_allocated()   ) { std::cout << prefix << std::setprecision(16) << "flux_net   : " << conv::sum(flux_net   ) << "\n"; }
    if (flux_dn_dir.is_allocated()) { std::cout << prefix << std::setprecision(16) << "flux_dn_dir: " << conv::sum(flux_dn_dir) << "\n"; }
  }

};
