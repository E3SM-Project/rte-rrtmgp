#pragma once
#include "mo_fluxes.h"
#include "mo_fluxes_byband_kernels.h"
#include "mo_optical_props.h"
#include "rrtmgp_const.h"
//#include "mo_fluxes_broadband_kernels.h"
#include <iomanip>

template <typename RealT=double, typename LayoutT=Kokkos::LayoutLeft, typename DeviceT=DefaultDevice>
class FluxesBybandK : public FluxesBroadbandK<RealT, LayoutT, DeviceT> {
public:

  using parent_t = FluxesBroadbandK<RealT, LayoutT, DeviceT>;

  using real3d_t = Kokkos::View<RealT***, LayoutT, DeviceT>;

  real3d_t bnd_flux_up;
  real3d_t bnd_flux_dn;
  real3d_t bnd_flux_dn_dir;
  real3d_t bnd_flux_net;

  template <typename FluxUpT, typename FluxDnT, typename FluxDnDirT=real3d_t>
  void reduce(FluxUpT const &gpt_flux_up, const FluxDnT &gpt_flux_dn,
              OpticalPropsK<RealT, LayoutT, DeviceT> const &spectral_disc,
              bool top_at_1, FluxDnDirT const &gpt_flux_dn_dir=FluxDnDirT()) {
    int ncol = gpt_flux_up.extent(0);
    int nlev = gpt_flux_up.extent(1);
    int ngpt = gpt_flux_up.extent(2);
    int nbnd = this->bnd_flux_up.extent(2);

    // Base clase reduce
    parent_t::reduce(gpt_flux_up, gpt_flux_dn, spectral_disc, top_at_1, gpt_flux_dn_dir);
    // Reduce byband fluxes
    auto band2gpt = spectral_disc.band2gpt;
    if (this->bnd_flux_up.is_allocated()    ) { sum_byband(ncol, nlev, ngpt, nbnd, band2gpt, gpt_flux_up,     this->bnd_flux_up    ); }
    if (this->bnd_flux_dn.is_allocated()    ) { sum_byband(ncol, nlev, ngpt, nbnd, band2gpt, gpt_flux_dn,     this->bnd_flux_dn    ); }
    if (this->bnd_flux_dn_dir.is_allocated()) { sum_byband(ncol, nlev, ngpt, nbnd, band2gpt, gpt_flux_dn_dir, this->bnd_flux_dn_dir); }
    // Compute net fluxes
    if (this->bnd_flux_net.is_allocated()) {
      if (this->bnd_flux_dn.is_allocated() && this->bnd_flux_up.is_allocated()) {
        net_byband(ncol, nlev, nbnd, this->bnd_flux_dn, this->bnd_flux_up, this->bnd_flux_net);
      } else {
        stoprun("reduce: bnd_flux_net requested but bnd_flux_dn or bnd_flux_up not allocated.");
      }
    }
  }

};
