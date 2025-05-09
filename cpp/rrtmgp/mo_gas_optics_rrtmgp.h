
#pragma once

#include "mo_optical_props.h"
#include "mo_source_functions.h"
#include "mo_rrtmgp_util_string.h"
#include "mo_gas_optics_kernels.h"
#include "mo_rrtmgp_constants.h"
#include "mo_rrtmgp_util_reorder.h"
#include "mo_gas_concentrations.h"

#include <iomanip>

// This code is part of RRTM for GCM Applications - Parallel (RRTMGP)
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
// Class for computing spectrally-resolved gas optical properties and source functions
//   given atmopsheric physical properties (profiles of temperature, pressure, and gas concentrations)
//   The class must be initialized with data (provided as a netCDF file) before being used.
//
// Two variants apply to internal Planck sources (longwave radiation in the Earth's atmosphere) and to
//   external stellar radiation (shortwave radiation in the Earth's atmosphere).
//   The variant is chosen based on what information is supplied during initialization.
//   (It might make more sense to define two sub-classes)
//
// -------------------------------------------------------------------------------------------------

template <typename RealT=double, typename LayoutT=Kokkos::LayoutLeft, typename DeviceT=DefaultDevice>
class GasOpticsRRTMGPK : public OpticalPropsK<RealT, LayoutT, DeviceT> {
 public:

  using parent_t = OpticalPropsK<RealT, LayoutT, DeviceT>;
  using hparent_t = OpticalPropsK<RealT, LayoutT, HostDevice>;
  using pool_t = conv::MemPoolSingleton<RealT, LayoutT, DeviceT>;
  using const_t = rrtmgp_constants<RealT>;
  using mdrp_t = typename conv::MDRP<LayoutT, DeviceT>;

  template <typename T>
  using view_t = typename parent_t::template view_t<T>;

  template <typename T>
  using uview_t = conv::Unmanaged<view_t<T>>;

  template <typename T>
  using hview_t = typename hparent_t::template view_t<T>;

  // RRTMGP computes absorption in each band arising from
  //   two major species in each band, which are combined to make
  //     a relative mixing ratio eta and a total column amount (col_mix)
  //   contributions from zero or more minor species whose concentrations
  //     may be scaled by other components of the atmosphere
  // Absorption coefficients are interpolated from tables on a pressure/temperature/(eta) grid
  // Interpolation variables: Temperature and pressure grids
  view_t<RealT*> press_ref;
  view_t<RealT*> press_ref_log;
  view_t<RealT*> temp_ref;

  // Derived and stored for convenience:
  //   Min and max for temperature and pressure intepolation grids
  //   difference in ln pressure between consecutive reference levels
  //   log of reference pressure separating the lower and upper atmosphere
  RealT press_ref_min;
  RealT press_ref_max;
  RealT temp_ref_min;
  RealT temp_ref_max;
  RealT press_ref_log_delta;
  RealT temp_ref_delta;
  RealT press_ref_trop_log;

  int max_gpt_diff_lower;
  int max_gpt_diff_upper;

  // Major absorbers ("key species")
  //   Each unique set of major species is called a flavor.
  // Names  and reference volume mixing ratios of major gases
  string1dv gas_names; // gas names
  view_t<RealT***> vmr_ref;  // vmr_ref(lower or upper atmosphere, gas, temp)

  // Which two gases are in each flavor? By index
  view_t<int**> flavor;        // major species pair; (2,nflav)

  // Which flavor for each g-point? One each for lower, upper atmosphere
  view_t<int**> gpoint_flavor; // flavor = gpoint_flavor(2, g-point)

  // Major gas absorption coefficients
  view_t<RealT****> kmajor;       //  kmajor(g-point,eta,pressure,temperature)

  // Minor species, independently for upper and lower atmospheres
  //   Array extents in the n_minor dimension will differ between upper and lower atmospheres
  //   Each contribution has starting and ending g-points
  view_t<int**> minor_limits_gpt_lower;
  view_t<int**> minor_limits_gpt_upper;

  // Minor gas contributions might be scaled by other gas amounts; if so we need to know
  //   the total density and whether the contribution is scaled by the partner gas
  //   or its complement (i.e. all other gases)
  // Water vapor self- and foreign continua work like this, as do
  //   all collision-induced abosption pairs
  view_t<bool*> minor_scales_with_density_lower;
  view_t<bool*> minor_scales_with_density_upper;
  view_t<bool*> scale_by_complement_lower;
  view_t<bool*> scale_by_complement_upper;
  view_t<int*> idx_minor_lower;
  view_t<int*> idx_minor_upper;
  view_t<int*> idx_minor_scaling_lower;
  view_t<int*> idx_minor_scaling_upper;

  // Index into table of absorption coefficients
  view_t<int*> kminor_start_lower;
  view_t<int*> kminor_start_upper;

  // The absorption coefficients themselves
  view_t<RealT***> kminor_lower; // kminor_lower(n_minor,eta,temperature)
  view_t<RealT***> kminor_upper; // kminor_upper(n_minor,eta,temperature)

  // Rayleigh scattering coefficients
  view_t<RealT****> krayl; // krayl(g-point,eta,temperature,upper/lower atmosphere)

  // Planck function spectral mapping
  //   Allocated only when gas optics object is internal-source
  view_t<RealT****> planck_frac;   // stored fraction of Planck irradiance in band for given g-point
                        // planck_frac(g-point, eta, pressure, temperature)
  view_t<RealT**> totplnk;       // integrated Planck irradiance by band; (Planck temperatures,band)
  RealT   totplnk_delta; // temperature steps in totplnk

  // Solar source function spectral mapping
  //   Allocated only when gas optics object is external-source
  view_t<RealT*> solar_src; // incoming solar irradiance(g-point)

  // Ancillary
  // Index into %gas_names -- is this a key species in any band?
  view_t<bool*> is_key;

  void finalize () {
      press_ref = decltype(press_ref)();
      press_ref_log = decltype(press_ref_log)();
      temp_ref = decltype(temp_ref)();
      gas_names = decltype(gas_names)();  // gas names
      vmr_ref = decltype(vmr_ref)();      // vmr_ref(lower or upper atmosphere, gas, temp)
      flavor = decltype(flavor)();        // major species pair; (2,nflav)
      gpoint_flavor = decltype(gpoint_flavor)(); // flavor = gpoint_flavor(2, g-point)
      kmajor = decltype(kmajor)();       //  kmajor(g-point,eta,pressure,temperature)
      minor_limits_gpt_lower = decltype(minor_limits_gpt_lower)();
      minor_limits_gpt_upper = decltype(minor_limits_gpt_upper)();
      minor_scales_with_density_lower = decltype(minor_scales_with_density_lower)();
      minor_scales_with_density_upper = decltype(minor_scales_with_density_upper)();
      scale_by_complement_lower = decltype(scale_by_complement_lower)();
      scale_by_complement_upper = decltype(scale_by_complement_upper)();
      idx_minor_lower = decltype(idx_minor_lower)();
      idx_minor_upper = decltype(idx_minor_upper)();
      idx_minor_scaling_lower = decltype(idx_minor_scaling_lower)();
      idx_minor_scaling_upper = decltype(idx_minor_scaling_upper)();
      kminor_start_lower = decltype(kminor_start_lower)();
      kminor_start_upper = decltype(kminor_start_upper)();
      kminor_lower = decltype(kminor_lower)(); // kminor_lower(n_minor,eta,temperature)
      kminor_upper = decltype(kminor_upper)(); // kminor_upper(n_minor,eta,temperature)
      krayl = decltype(krayl)(); // krayl(g-point,eta,temperature,upper/lower atmosphere)
      planck_frac = decltype(planck_frac)();   // stored fraction of Planck irradiance in band for given g-point
      totplnk = decltype(totplnk)();       // integrated Planck irradiance by band; (Planck temperatures,band)
      solar_src = decltype(solar_src)(); // incoming solar irradiance(g-point)
      is_key = decltype(is_key)();

      // Free memory in base class
      parent_t::finalize();
  }

  // Everything except GasConcs is on the host by default, and the available_gases.gas_name is on the host as well
  // Things will be copied to the GPU outside of this routine and stored into class variables
  template <typename KminorAtmT, typename MinorLimitsT, typename MinorScalesT, typename ScaleByT,
            typename KminorStartT, typename KminorAtmRedT, typename MinorLimitsRedT,
            typename MinorScalesRedT, typename ScaleByRedT, typename KminorStartRedT>
  void reduce_minor_arrays(GasConcsK<RealT, LayoutT, DeviceT> const &available_gases,
                           string1dv    const &gas_names,
                           string1dv    const &gas_minor,
                           string1dv    const &identifier_minor,
                           KminorAtmT   const &kminor_atm,
                           string1dv    const &minor_gases_atm,
                           MinorLimitsT const &minor_limits_gpt_atm,
                           MinorScalesT const &minor_scales_with_density_atm,
                           string1dv    const &scaling_gas_atm,
                           ScaleByT     const &scale_by_complement_atm,
                           KminorStartT const &kminor_start_atm,
                           KminorAtmRedT      &kminor_atm_red,
                           string1dv          &minor_gases_atm_red,
                           MinorLimitsRedT    &minor_limits_gpt_atm_red,
                           MinorScalesRedT    &minor_scales_with_density_atm_red,
                           string1dv          &scaling_gas_atm_red,
                           ScaleByRedT        &scale_by_complement_atm_red,
                           KminorStartRedT    &kminor_start_atm_red) {
    int nm = minor_gases_atm.size();  // Size of the larger list of minor gases
    int tot_g = 0;
    int red_nm = 0;                    // Reduced number of minor gasses (only the ones we need)
    hview_t<bool*> gas_is_present("gas_is_present",nm);   // Determines whether a gas in the list is needed
    // Determine the gasses needed
    for (int i=0; i < nm; i++) {
      int idx_mnr = string_loc_in_array(minor_gases_atm[i], identifier_minor);
      gas_is_present(i) = string_in_array(gas_minor[idx_mnr], available_gases.gas_name);
      if (gas_is_present(i)) {
        tot_g = tot_g + (minor_limits_gpt_atm(1,i)-minor_limits_gpt_atm(0,i)+1);
        red_nm = red_nm + 1;
      }
    }

    // Allocate reduced arrays
    minor_gases_atm_red               = string1dv  (red_nm);
    minor_scales_with_density_atm_red = hview_t<bool*>("minor_scales_with_density_atm_red"  ,red_nm);
    scaling_gas_atm_red               = string1dv  (red_nm);
    scale_by_complement_atm_red       = hview_t<bool*>("scale_by_complement_atm_red      "  ,red_nm);
    kminor_start_atm_red              = hview_t<int*> ("kminor_start_atm_red             "  ,red_nm);
    minor_limits_gpt_atm_red          = hview_t<int**> ("minor_limits_gpt_atm_red         ",2,red_nm);
    kminor_atm_red                    = hview_t<RealT***>("kminor_atm_red                   ",tot_g , kminor_atm.extent(1), kminor_atm.extent(2));

    if (red_nm == nm) {
      // If the gasses listed exactly matches the gasses needed, just copy it
      minor_gases_atm_red = minor_gases_atm;
      scaling_gas_atm_red = scaling_gas_atm;
      Kokkos::deep_copy(kminor_atm_red, kminor_atm);
      Kokkos::deep_copy(minor_limits_gpt_atm_red, minor_limits_gpt_atm);
      Kokkos::deep_copy(minor_scales_with_density_atm_red, minor_scales_with_density_atm);
      Kokkos::deep_copy(scale_by_complement_atm_red, scale_by_complement_atm);
      Kokkos::deep_copy(kminor_start_atm_red, kminor_start_atm);
    } else {
      // Otherwise, pack into reduced arrays
      int slot = 0;
      for (int i=0; i < nm; i++) {
        if (gas_is_present(i)) {
          minor_gases_atm_red              [slot] = minor_gases_atm              [i];
          scaling_gas_atm_red              [slot] = scaling_gas_atm              [i];
          minor_scales_with_density_atm_red(slot) = minor_scales_with_density_atm(i);
          scale_by_complement_atm_red      (slot) = scale_by_complement_atm      (i);
          kminor_start_atm_red             (slot) = kminor_start_atm             (i);
          slot++;
        }
      }

      slot = -1;
      int n_elim = 0;
      for (int i=0 ; i < nm ; i++) {
        int ng = minor_limits_gpt_atm(1,i)-minor_limits_gpt_atm(0,i)+1;
        if (gas_is_present(i)) {
          ++slot;
          minor_limits_gpt_atm_red   (0,slot) = minor_limits_gpt_atm(0,i);
          minor_limits_gpt_atm_red   (1,slot) = minor_limits_gpt_atm(1,i);
          kminor_start_atm_red         (slot) = kminor_start_atm(i)-n_elim;
          for (int l=0 ; l < kminor_atm.extent(2); l++ ) {
            for (int k=0 ; k < kminor_atm.extent(1) ; k++ ) {
              for (int j=0 ; j < ng ; j++) {
                kminor_atm_red(kminor_start_atm_red(slot)+j,k,l) = kminor_atm(kminor_start_atm(i)+j,k,l);
              }
            }
          }
        } else {
          n_elim = n_elim + ng;
        }
      }
    }
  }

  // create index list for extracting col_gas needed for minor gas optical depth calculations
  void create_idx_minor(string1dv const &gas_names, string1dv const &gas_minor, string1dv const &identifier_minor,
                        string1dv const &minor_gases_atm, hview_t<int*> &idx_minor_atm) {
    idx_minor_atm = hview_t<int*>("idx_minor_atm", minor_gases_atm.size());
    for (size_t imnr=0 ; imnr < minor_gases_atm.size() ; imnr++) {
      // Find identifying string for minor species in list of possible identifiers (e.g. h2o_slf)
      int idx_mnr     = string_loc_in_array(minor_gases_atm[imnr], identifier_minor);
      // Find name of gas associated with minor species identifier (e.g. h2o)
      idx_minor_atm(imnr) = string_loc_in_array(gas_minor[idx_mnr], gas_names);
    }
  }

  // create index for special treatment in density scaling of minor gases
  void create_idx_minor_scaling(string1dv const &gas_names, string1dv const &scaling_gas_atm,
                                hview_t<int*> &idx_minor_scaling_atm) {
    idx_minor_scaling_atm = hview_t<int*>("idx_minor_scaling_atm", scaling_gas_atm.size());
    for (auto imnr=0 ; imnr < scaling_gas_atm.size() ; imnr++) {
      // This will be -1 if there's no interacting gas
      idx_minor_scaling_atm(imnr) = string_loc_in_array(scaling_gas_atm[imnr], gas_names);
    }
  }

  template <typename KeySpeciesT>
  void create_key_species_reduce(string1dv const &gas_names, string1dv const &gas_names_red, KeySpeciesT const &key_species,
                                 hview_t<int***> &key_species_red, hview_t<bool*> &key_species_present_init) {
    int np = key_species.extent(0);
    int na = key_species.extent(1);
    int nt = key_species.extent(2);
    key_species_red = hview_t<int***>("key_species_red",np,na,nt);
    key_species_present_init = hview_t<bool*>("key_species_present_init", gas_names.size());
    Kokkos::deep_copy(key_species_present_init, true);

    for (int ip=0 ; ip < np ; ip++) {
      for (int ia=0 ; ia < na ; ia++) {
        for (int it=0 ; it < nt ; it++) {
          if (key_species(ip,ia,it) != -1) {
            key_species_red(ip,ia,it) = string_loc_in_array(gas_names[key_species(ip,ia,it)],gas_names_red);
            if (key_species_red(ip,ia,it) == -1) {
              key_species_present_init(key_species(ip,ia,it)) = false;
            }
          } else {
            key_species_red(ip,ia,it) = -1;
          }
        }
      }
    }
  }

  // Create flavor list
  // An unordered array of extent (2,:) containing all possible pairs of key species used in either upper or lower atmos
  template <typename KeySpeciesT>
  void create_flavor(KeySpeciesT const &key_species, hview_t<int**> &flavor) {
    // prepare list of key_species
    int i = 0;
    hview_t<int**> key_species_list("key_species_list", 2, key_species.extent(2)*2);
    for (int ibnd=0 ; ibnd < key_species.extent(2) ; ibnd++) {
      for (int iatm=0 ; iatm < key_species.extent(0) ; iatm++, i++) {
        key_species_list(0,i) = key_species(0,iatm,ibnd);
        key_species_list(1,i) = key_species(1,iatm,ibnd);
      }
    }
    // rewrite single key_species pairs
    for (int i=0 ; i < key_species_list.extent(1) ; i++) {
      if (key_species_list(0,i) == -1 && key_species_list(1,i) == -1) {
        key_species_list(0,i) = 1;
        key_species_list(1,i) = 1;
      }
    }
    // count unique key species pairs
    int iflavor = 0;
    for (int i=0; i < key_species_list.extent(1) ; i++) {
      // Loop through previous pairs. Only increment iflavor if we haven't seen this pair before
      bool unique = true;
      for (int j=0; j <= i-1 ; j++) {
        if ( key_species_list(0,j) == key_species_list(0,i) && key_species_list(1,j) == key_species_list(1,i) ) {
          unique = false;
          break;
        }
      }
      if (unique) { ++iflavor; }
    }
    // fill flavors
    flavor = hview_t<int**>("flavor",2,iflavor);
    iflavor = 0;
    for (int i=0 ; i < key_species_list.extent(1) ; i++) {
      bool unique = true;
      for (int j=0; j <= i-1 ; j++) {
        if ( key_species_list(0,j) == key_species_list(0,i) && key_species_list(1,j) == key_species_list(1,i) ) {
          unique = false;
          break;
        }
      }
      if (unique) {
        flavor(0,iflavor) = key_species_list(0,i);
        flavor(1,iflavor) = key_species_list(1,i);
        ++iflavor;
      }
    }
  }

  // create gpoint_flavor list: a map pointing from each g-point to the corresponding entry in the "flavor list"
  template <typename KeySpeciesT, typename Gpt2T, typename FlavorT>
  void create_gpoint_flavor(KeySpeciesT const &key_species, Gpt2T const &gpt2band, FlavorT const &flavor,
                            hview_t<int**> &gpoint_flavor) {
    int ngpt = gpt2band.extent(0);
    gpoint_flavor = hview_t<int**>("gpoint_flavor",2,ngpt);
    for (int igpt=0 ; igpt < ngpt ; igpt++) {
      for (int iatm = 0 ; iatm < 2 ; iatm++) {
        int key_species_pair2flavor = -1;
        for (int iflav=0 ; iflav < flavor.extent(1) ; iflav++) {
          int key_species1 = key_species(0,iatm,gpt2band(igpt));
          int key_species2 = key_species(1,iatm,gpt2band(igpt));
          if (key_species1 == -1 && key_species2 == -1) {
            key_species1 = 1;
            key_species2 = 1;
          }
          if ( flavor(0,iflav) == key_species1 && flavor(1,iflav) == key_species2 ) {
            key_species_pair2flavor = iflav;
          }
        }
        gpoint_flavor(iatm,igpt) = key_species_pair2flavor;
      }
    }
  }

  // Initialize absorption coefficient arrays,
  //   including Rayleigh scattering tables if provided (allocated)
  template <typename KeySpeciesT, typename Band2gptT, typename BandLimsT, typename PressT,
            typename TempT, typename VmrT, typename KmajorT, typename KminorLowerT,
            typename KminorUpperT, typename MinorLimitsLowerT, typename MinorLimitsUpperT,
            typename MinorScalesLowerT, typename MinorScalesUpperT, typename ScaleCompLowerT,
            typename ScaleCompUpperT, typename KminorStartLowerT, typename KminorStartUpperT,
            typename RaylLowerT, typename RaylUpperT>
  void init_abs_coeffs(GasConcsK<RealT, LayoutT, DeviceT> const &available_gases,
                       string1dv   const        &gas_names,
                       KeySpeciesT const        &key_species,
                       Band2gptT   const        &band2gpt,
                       BandLimsT   const        &band_lims_wavenum,
                       PressT      const        &press_ref,
                       TempT       const        &temp_ref,
                       RealT                     press_ref_trop,
                       RealT                     temp_ref_p,
                       RealT                     temp_ref_t,
                       VmrT              const &vmr_ref,
                       KmajorT           const &kmajor,
                       KminorLowerT      const &kminor_lower,
                       KminorUpperT      const &kminor_upper,
                       string1dv         const &gas_minor,
                       string1dv         const &identifier_minor,
                       string1dv         const &minor_gases_lower,
                       string1dv         const &minor_gases_upper,
                       MinorLimitsLowerT const &minor_limits_gpt_lower,
                       MinorLimitsUpperT const &minor_limits_gpt_upper,
                       MinorScalesLowerT const &minor_scales_with_density_lower,
                       MinorScalesUpperT const &minor_scales_with_density_upper,
                       string1dv         const &scaling_gas_lower,
                       string1dv         const &scaling_gas_upper,
                       ScaleCompLowerT   const &scale_by_complement_lower,
                       ScaleCompUpperT   const &scale_by_complement_upper,
                       KminorStartLowerT const &kminor_start_lower,
                       KminorStartUpperT const &kminor_start_upper,
                       RaylLowerT        const &rayl_lower,
                       RaylUpperT        const &rayl_upper) {
    auto band_lims_wavenum_h = Kokkos::create_mirror_view_and_copy(DeviceT(), band_lims_wavenum);
    auto band2gpt_h = Kokkos::create_mirror_view_and_copy(DeviceT(), band2gpt);
    parent_t::init(band_lims_wavenum_h, band2gpt_h);

    // Which gases known to the gas optics are present in the host model (available_gases)?
    for (auto item : gas_names) {
      if (string_in_array(item, available_gases.gas_name)) {
        this->gas_names.push_back(item);
      }
    }
    // Now the number of gases is the union of those known to the k-distribution and provided by the host model

    // Initialize the gas optics object, keeping only those gases known to the gas optics and also present in the host model
    const int ngas = this->gas_names.size();
    const int vmr_e0 = vmr_ref.extent(0);
    const int vmr_e2 = vmr_ref.extent(2);

    hview_t<RealT***> vmr_ref_red("vmr_ref_red", vmr_e0, ngas+1, vmr_e2);
    // Gas 0 is used in single-key species method, set to 1.0 (col_dry)
    for (int k=0 ; k < vmr_e2 ; k++) {
      for (int j=0 ; j < vmr_e0 ; j++) {
        vmr_ref_red(j,0,k) = vmr_ref(j,0,k);
      }
    }
    for (int i=0 ; i < ngas ; i++) {
      int idx = string_loc_in_array(this->gas_names[i], gas_names);
      for (int k=0 ; k < vmr_e2 ; k++) {
        for (int j=0 ; j < vmr_e0 ; j++) {
          vmr_ref_red(j,i+1,k) = vmr_ref(j,idx+1,k);
        }
      }
    }
    // Allocate class copy, and deep copy to the class data member
    this->vmr_ref = Kokkos::create_mirror_view_and_copy(DeviceT(), vmr_ref_red);

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // REDUCE MINOR ARRAYS SO VARIABLES ONLY CONTAIN MINOR GASES THAT ARE AVAILABLE
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // LOWER MINOR GASSES
    string1dv         minor_gases_lower_red;
    string1dv         scaling_gas_lower_red;
    hview_t<RealT***> kminor_lower_red;
    hview_t<int**>    minor_limits_gpt_lower_red;
    hview_t<bool*>    minor_scales_with_density_lower_red;
    hview_t<bool*>    scale_by_complement_lower_red;
    hview_t<int*>     kminor_start_lower_red;

    reduce_minor_arrays(available_gases, gas_names, gas_minor, identifier_minor, kminor_lower, minor_gases_lower,
                        minor_limits_gpt_lower, minor_scales_with_density_lower, scaling_gas_lower,
                        scale_by_complement_lower, kminor_start_lower, kminor_lower_red, minor_gases_lower_red,
                        minor_limits_gpt_lower_red, minor_scales_with_density_lower_red, scaling_gas_lower_red,
                        scale_by_complement_lower_red, kminor_start_lower_red);

    this->kminor_lower                    = Kokkos::create_mirror_view_and_copy(DeviceT(), kminor_lower_red);
    this->minor_limits_gpt_lower          = Kokkos::create_mirror_view_and_copy(DeviceT(), minor_limits_gpt_lower_red);
    this->minor_scales_with_density_lower = Kokkos::create_mirror_view_and_copy(DeviceT(), minor_scales_with_density_lower_red);
    this->scale_by_complement_lower       = Kokkos::create_mirror_view_and_copy(DeviceT(), scale_by_complement_lower_red);
    this->kminor_start_lower              = Kokkos::create_mirror_view_and_copy(DeviceT(), kminor_start_lower_red);

    // Find the largest number of g-points per band
    this->max_gpt_diff_lower = std::numeric_limits<int>::lowest();
    for (int i=0; i<minor_limits_gpt_lower_red.extent(1); i++) {
      this->max_gpt_diff_lower = std::max( this->max_gpt_diff_lower , minor_limits_gpt_lower_red(1,i) - minor_limits_gpt_lower_red(0,i) );
    }

    // UPPER MINOR GASSES
    string1dv         minor_gases_upper_red;
    string1dv         scaling_gas_upper_red;
    hview_t<RealT***> kminor_upper_red;
    hview_t<int**>    minor_limits_gpt_upper_red;
    hview_t<bool*>    minor_scales_with_density_upper_red;
    hview_t<bool*>    scale_by_complement_upper_red;
    hview_t<int*>     kminor_start_upper_red;

    reduce_minor_arrays(available_gases, gas_names, gas_minor, identifier_minor, kminor_upper, minor_gases_upper,
                        minor_limits_gpt_upper, minor_scales_with_density_upper, scaling_gas_upper,
                        scale_by_complement_upper, kminor_start_upper, kminor_upper_red, minor_gases_upper_red,
                        minor_limits_gpt_upper_red, minor_scales_with_density_upper_red, scaling_gas_upper_red,
                        scale_by_complement_upper_red, kminor_start_upper_red);

    this->kminor_upper                    = Kokkos::create_mirror_view_and_copy(DeviceT(), kminor_upper_red);
    this->minor_limits_gpt_upper          = Kokkos::create_mirror_view_and_copy(DeviceT(), minor_limits_gpt_upper_red);
    this->minor_scales_with_density_upper = Kokkos::create_mirror_view_and_copy(DeviceT(), minor_scales_with_density_upper_red);
    this->scale_by_complement_upper       = Kokkos::create_mirror_view_and_copy(DeviceT(), scale_by_complement_upper_red);
    this->kminor_start_upper              = Kokkos::create_mirror_view_and_copy(DeviceT(), kminor_start_upper_red);

    // Find the largest number of g-points per band
    this->max_gpt_diff_upper = std::numeric_limits<int>::lowest();
    for (int i=0; i<minor_limits_gpt_upper_red.extent(1); i++) {
      this->max_gpt_diff_upper = std::max( this->max_gpt_diff_upper , minor_limits_gpt_upper_red(1,i) - minor_limits_gpt_upper_red(0,i) );
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // HANDLE ARRAYS NOT REDUCED BY THE PRESENCE, OR LACK THEREOF, OF A GAS
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    this->press_ref = Kokkos::create_mirror_view_and_copy(DeviceT(), press_ref);
    this->temp_ref  = Kokkos::create_mirror_view_and_copy(DeviceT(), temp_ref);
    this->kmajor    = Kokkos::create_mirror_view_and_copy(DeviceT(), kmajor);

    // Process rayl_lower and rayl_upper into a combined this->krayl
    if (rayl_lower.is_allocated() != rayl_upper.is_allocated()) {
      stoprun("rayl_lower and rayl_upper must have the same allocation status");
    }
    if (rayl_lower.is_allocated()) {
      hview_t<RealT****> krayltmp("krayltmp",rayl_lower.extent(0),rayl_lower.extent(1),rayl_lower.extent(2),2);
      for (int k=0 ; k < rayl_lower.extent(2) ; k++ ) {
        for (int j=0 ; j < rayl_lower.extent(1) ; j++ ) {
          for (int i=0 ; i < rayl_lower.extent(0) ; i++ ) {
            krayltmp(i,j,k,0) = rayl_lower(i,j,k);
            krayltmp(i,j,k,1) = rayl_upper(i,j,k);
          }
        }
      }
      this->krayl = Kokkos::create_mirror_view_and_copy(DeviceT(), krayltmp);
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // POST PROCESSING
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // creates log reference pressure
    this->press_ref_log = view_t<RealT*>("press_ref_log", this->press_ref.extent(0));
    // Running a kernel because it's more convenient in this case
    auto this_press_ref_log = this->press_ref_log;
    auto this_press_ref = this->press_ref;
    TIMED_KERNEL(Kokkos::parallel_for( this->press_ref.extent(0) , KOKKOS_LAMBDA (int i) {
       this_press_ref_log(i) = log(this_press_ref(i));
    }));

    // log scale of reference pressure (this is a scalar, not an array)
    this->press_ref_trop_log = log(press_ref_trop);

    // Get index of gas (if present) for determining col_gas
    hview_t<int*> idx_minor_lower_tmp;
    hview_t<int*> idx_minor_upper_tmp;
    create_idx_minor(this->gas_names, gas_minor, identifier_minor, minor_gases_lower_red, idx_minor_lower_tmp);
    create_idx_minor(this->gas_names, gas_minor, identifier_minor, minor_gases_upper_red, idx_minor_upper_tmp);
    this->idx_minor_lower = Kokkos::create_mirror_view_and_copy(DeviceT(), idx_minor_lower_tmp);
    this->idx_minor_upper = Kokkos::create_mirror_view_and_copy(DeviceT(), idx_minor_upper_tmp);
    // Get index of gas (if present) that has special treatment in density scaling
    hview_t<int*> idx_minor_scaling_lower_tmp;
    hview_t<int*> idx_minor_scaling_upper_tmp;
    create_idx_minor_scaling(this->gas_names, scaling_gas_lower_red, idx_minor_scaling_lower_tmp);
    create_idx_minor_scaling(this->gas_names, scaling_gas_upper_red, idx_minor_scaling_upper_tmp);
    this->idx_minor_scaling_lower = Kokkos::create_mirror_view_and_copy(DeviceT(), idx_minor_scaling_lower_tmp);
    this->idx_minor_scaling_upper = Kokkos::create_mirror_view_and_copy(DeviceT(), idx_minor_scaling_upper_tmp);

    // create flavor list
    // Reduce (remap) key_species list; checks that all key gases are present in incoming
    hview_t<bool*> key_species_present_init;
    hview_t<int***>  key_species_red;
    create_key_species_reduce(gas_names, this->gas_names, key_species, key_species_red, key_species_present_init);
    // create flavor and gpoint_flavor lists
    hview_t<int**> flavor_tmp;
    hview_t<int**> gpoint_flavor_tmp;
    auto gpoint_bands_tmp = Kokkos::create_mirror_view_and_copy(HostDevice(), this->get_gpoint_bands());
    create_flavor       (key_species_red, flavor_tmp);
    create_gpoint_flavor(key_species_red, gpoint_bands_tmp, flavor_tmp, gpoint_flavor_tmp);
    this->flavor        = Kokkos::create_mirror_view_and_copy(DeviceT(), flavor_tmp);
    this->gpoint_flavor = Kokkos::create_mirror_view_and_copy(DeviceT(), gpoint_flavor_tmp);

    // minimum, maximum reference temperature, pressure -- assumes low-to-high ordering
    //   for T, high-to-low ordering for p
    this->temp_ref_min  = temp_ref (0);
    this->temp_ref_max  = temp_ref (temp_ref.extent(0)-1);
    this->press_ref_min = press_ref(press_ref.extent(0)-1);
    this->press_ref_max = press_ref(0);

    // creates press_ref_log, temp_ref_delta
    this->press_ref_log_delta = (log(this->press_ref_min)-log(this->press_ref_max))/(press_ref.extent(0)-1);
    this->temp_ref_delta      = (this->temp_ref_max-this->temp_ref_min)/(temp_ref.extent(0)-1);

    // Which species are key in one or more bands?
    //   this->flavor is an index into this->gas_names
    this->is_key = view_t<bool*>("is_key",this->get_ngas());
    // do j = 1, size(this%flavor, 2)
    //   do i = 1, size(this%flavor, 1) ! extents should be 2
    auto this_flavor = this->flavor;
    auto this_is_key = this->is_key;
    const int dim0 = this->flavor.extent(0);
    const int dim1 = this->flavor.extent(1);
    TIMED_KERNEL(FLATTEN_MD_KERNEL2(dim0, dim1, i, j,
      if (this_flavor(i,j) != -1) { this_is_key(this_flavor(i,j)) = true; }
    ));
  }

  // Initialize object based on data read from netCDF file however the user desires.
  //  Rayleigh scattering tables may or may not be present; this is indicated with allocation status
  // This interface is for the internal-sources object -- includes Plank functions and fractions
  template <typename KeySpeciesT, typename Band2gptT, typename BandLimsT, typename PressT,
            typename TempT, typename VmrT, typename KmajorT, typename KminorLowerT,
            typename KminorUpperT, typename MinorLimitsLowerT, typename MinorLimitsUpperT,
            typename MinorScalesLowerT, typename MinorScalesUpperT, typename ScaleCompLowerT,
            typename ScaleCompUpperT, typename KminorStartLowerT, typename KminorStartUpperT,
            typename TotplnkT, typename PlanckT, typename RaylLowerT, typename RaylUpperT>
  void load(GasConcsK<RealT, LayoutT, DeviceT> const &available_gases,
            string1dv   const        &gas_names,
            KeySpeciesT  const       &key_species,
            Band2gptT  const         &band2gpt,
            BandLimsT const          &band_lims_wavenum,
            PressT const             &press_ref,
            RealT                     press_ref_trop,
            TempT const              &temp_ref,
            RealT                     temp_ref_p,
            RealT                     temp_ref_t,
            VmrT const               &vmr_ref,
            KmajorT const            &kmajor,
            KminorLowerT const       &kminor_lower,
            KminorUpperT const       &kminor_upper,
            string1dv   const        &gas_minor,
            string1dv   const        &identifier_minor,
            string1dv   const        &minor_gases_lower,
            string1dv   const        &minor_gases_upper,
            MinorLimitsLowerT  const &minor_limits_gpt_lower,
            MinorLimitsUpperT  const &minor_limits_gpt_upper,
            MinorScalesLowerT const  &minor_scales_with_density_lower,
            MinorScalesUpperT const  &minor_scales_with_density_upper,
            string1dv   const        &scaling_gas_lower,
            string1dv   const        &scaling_gas_upper,
            ScaleCompLowerT const    &scale_by_complement_lower,
            ScaleCompUpperT const    &scale_by_complement_upper,
            KminorStartLowerT  const &kminor_start_lower,
            KminorStartUpperT  const &kminor_start_upper,
            TotplnkT const           &totplnk,
            PlanckT const            &planck_frac,
            RaylLowerT const         &rayl_lower,
            RaylUpperT const &rayl_upper) {
    init_abs_coeffs(available_gases, gas_names, key_species, band2gpt, band_lims_wavenum, press_ref, temp_ref,
                    press_ref_trop, temp_ref_p, temp_ref_t, vmr_ref, kmajor, kminor_lower, kminor_upper,
                    gas_minor, identifier_minor, minor_gases_lower, minor_gases_upper, minor_limits_gpt_lower,
                    minor_limits_gpt_upper, minor_scales_with_density_lower, minor_scales_with_density_upper,
                    scaling_gas_lower, scaling_gas_upper, scale_by_complement_lower, scale_by_complement_upper,
                    kminor_start_lower, kminor_start_upper, rayl_lower, rayl_upper);

    // Planck function tables
    this->totplnk = Kokkos::create_mirror_view_and_copy(DeviceT(), totplnk);
    this->planck_frac = Kokkos::create_mirror_view_and_copy(DeviceT(), planck_frac);
    // Temperature steps for Planck function interpolation
    //   Assumes that temperature minimum and max are the same for the absorption coefficient grid and the
    //   Planck grid and the Planck grid is equally spaced
    this->totplnk_delta = (this->temp_ref_max - this->temp_ref_min) / (this->totplnk.extent(0)-1);
  }

  // Initialize object based on data read from netCDF file however the user desires.
  //  Rayleigh scattering tables may or may not be present; this is indicated with allocation status
  // This interface is for the external-sources object -- includes TOA source function table
  template <typename KeySpeciesT, typename Band2gptT, typename BandLimsT, typename PressT,
            typename TempT, typename VmrT, typename KmajorT, typename KminorLowerT,
            typename KminorUpperT, typename MinorLimitsLowerT, typename MinorLimitsUpperT,
            typename MinorScalesLowerT, typename MinorScalesUpperT, typename ScaleCompLowerT,
            typename ScaleCompUpperT, typename KminorStartLowerT, typename KminorStartUpperT,
            typename SolarT, typename RaylLowerT, typename RaylUpperT>
  void load(GasConcsK<RealT, LayoutT, DeviceT> const &available_gases,
            string1dv   const        &gas_names,
            KeySpeciesT  const       &key_species,
            Band2gptT  const         &band2gpt,
            BandLimsT const          &band_lims_wavenum,
            PressT const             &press_ref,
            RealT                     press_ref_trop,
            TempT const              &temp_ref,
            RealT                     temp_ref_p,
            RealT                     temp_ref_t,
            VmrT const               &vmr_ref,
            KmajorT const            &kmajor,
            KminorLowerT const       &kminor_lower,
            KminorUpperT const       &kminor_upper,
            string1dv   const        &gas_minor,
            string1dv   const        &identifier_minor,
            string1dv   const        &minor_gases_lower,
            string1dv   const        &minor_gases_upper,
            MinorLimitsLowerT  const &minor_limits_gpt_lower,
            MinorLimitsUpperT  const &minor_limits_gpt_upper,
            MinorScalesLowerT const  &minor_scales_with_density_lower,
            MinorScalesUpperT const  &minor_scales_with_density_upper,
            string1dv   const        &scaling_gas_lower,
            string1dv   const        &scaling_gas_upper,
            ScaleCompLowerT const    &scale_by_complement_lower,
            ScaleCompUpperT const    &scale_by_complement_upper,
            KminorStartLowerT  const &kminor_start_lower,
            KminorStartUpperT  const &kminor_start_upper,
            SolarT const &solar_src,
            RaylLowerT const &rayl_lower,
            RaylUpperT const &rayl_upper) {
    init_abs_coeffs(available_gases,  gas_names, key_species, band2gpt, band_lims_wavenum, press_ref, temp_ref,
                    press_ref_trop, temp_ref_p, temp_ref_t, vmr_ref, kmajor, kminor_lower, kminor_upper,
                    gas_minor, identifier_minor, minor_gases_lower, minor_gases_upper, minor_limits_gpt_lower,
                    minor_limits_gpt_upper, minor_scales_with_density_lower, minor_scales_with_density_upper,
                    scaling_gas_lower, scaling_gas_upper, scale_by_complement_lower, scale_by_complement_upper,
                    kminor_start_lower, kminor_start_upper, rayl_lower, rayl_upper);

    // Solar source table init
    this->solar_src = Kokkos::create_mirror_view_and_copy(DeviceT(), solar_src);
    this->totplnk_delta = 0.;
  }

  // Two functions to define array sizes needed by gas_optics()
  int get_ngas() const { return this->gas_names.size(); }

  // return the number of distinct major gas pairs in the spectral bands (referred to as
  // "flavors" - all bands have a flavor even if there is one or no major gas)
  int get_nflav() const { return this->flavor.extent(1); }

  string1dv get_gases() const { return this->gas_names; }

  // return the minimum pressure on the interpolation grids
  RealT get_press_min() const { return this->press_ref_min; }

  // return the maximum pressure on the interpolation grids
  RealT get_press_max() const { return this->press_ref_max; }

  // return the minimum temparature on the interpolation grids
  RealT get_temp_min()const  { return this->temp_ref_min; }

  // return the maximum temparature on the interpolation grids
  RealT get_temp_max() const { return this->temp_ref_max; }

  int get_neta() const { return this->kmajor.extent(1); }

  // return the number of pressures in reference profile
  //   absorption coefficient table is one bigger since a pressure is repeated in upper/lower atmos
  int get_npres() const { return this->kmajor.extent(2)-1; }

  int get_ntemp() const { return this->kmajor.extent(3); }

  // return the number of temperatures for Planck function
  int get_nPlanckTemp() const { return this->totplnk.extent(0); }


  // Function to define names of key and minor gases to be used by gas_optics().
  // The final list gases includes those that are defined in gas_optics_specification
  // and are provided in ty_gas_concs.
  string1dv get_minor_list(GasConcsK<RealT, LayoutT, DeviceT> const &gas_desc, int ngas, string1dv const &name_spec) const {
    // List of minor gases to be used in gas_optics()
    string1dv rv;
    for (int igas=0 ; igas < this->get_ngas() ; igas++) {
      if (string_in_array(name_spec[igas], gas_desc.gas_name)) {
        rv.push_back(this->gas_names[igas]);
      }
    }
    return rv;
  }

  // return true if initialized for internal sources, false otherwise
  bool source_is_internal() const { return this->totplnk.is_allocated() && this->planck_frac.is_allocated(); }

  // return true if initialized for external sources, false otherwise
  bool source_is_external() const { return this->solar_src.is_allocated(); }

  // Ensure that every key gas required by the k-distribution is present in the gas concentration object
  void check_key_species_present(GasConcsK<RealT, LayoutT, DeviceT> const &gas_desc) const {
    string1dv key_gas_names;
    auto is_key_h = Kokkos::create_mirror_view_and_copy(HostDevice(), this->is_key);
    for (auto i = 0; i < is_key_h.extent(0); ++i) {
      if (is_key_h(i)) {
        key_gas_names.push_back(this->gas_names[i]);
      }
    }
    for (auto igas=0 ; igas < key_gas_names.size() ; igas++) {
      if (! string_in_array(key_gas_names[igas], gas_desc.gas_name)) {
        stoprun("gas required by k-distribution is not present in the GasConcs object");
      }
    }
  }

  // Compute gas optical depth and Planck source functions, given temperature, pressure, and composition
  template <typename PlayT, typename PlevT, typename TlayT, typename TsfcT, typename ColGasT,
            class OpticalPropsT, typename ColDryT=view_t<RealT**>, typename TlevT=view_t<RealT**> >
  void gas_optics(const int ncol, const int nlay,
                  bool top_at_1, PlayT const &play, PlevT const &plev, TlayT const &tlay,
                  TsfcT const &tsfc,
                  GasConcsK<RealT, LayoutT, DeviceT> const &gas_desc,
                  ColGasT const& col_gas, OpticalPropsT &optical_props,
                  SourceFuncLWK<RealT, LayoutT, DeviceT> &sources,
                  ColDryT const &col_dry=ColDryT(), TlevT const &tlev=TlevT()) {
    int ngpt  = this->get_ngpt();
    int nband = this->get_nband();
    // Interpolation coefficients for use in source function
    auto jtemp  = pool_t::template alloc<int>(ncol, nlay);
    auto jpress = pool_t::template alloc<int>(ncol, nlay);
    auto jeta   = pool_t::template alloc<int>(2, this->get_nflav(), ncol, nlay);
    auto tropo  = pool_t::template alloc<bool>(ncol, nlay);
    auto fmajor = pool_t::template alloc<RealT>(2,2,2,this->get_nflav(),ncol,nlay);
    // Gas optics
    compute_gas_taus(top_at_1, ncol, nlay, ngpt, nband, play, plev, tlay, gas_desc, col_gas, optical_props, jtemp, jpress,
                     jeta, tropo, fmajor, col_dry);

    // External source -- check arrays sizes and values
    // input data sizes and values
    if (tsfc.extent(0) != ncol) { stoprun("gas_optics(): array tsfc has wrong size"); }
    #ifdef RRTMGP_EXPENSIVE_CHECKS
      if (any(tsfc < this->temp_ref_min) || any(tsfc > this->temp_ref_max)) {
        stoprun("gas_optics(): array tsfc has values outside range");
      }
    #endif

    if (tlev.is_allocated()) {
      #ifdef RRTMGP_EXPENSIVE_CHECKS
        if (any(tlev < this->temp_ref_min) || any(tlev > this->temp_ref_max)) {
          stoprun("gas_optics(): array tlev has values outside range");
        }
      #endif
    }

    // output extents
    if (sources.get_ncol() != ncol || sources.get_nlay() != nlay || sources.get_ngpt() != ngpt) {
      stoprun("gas_optics%gas_optics: source function arrays inconsistently sized");
    }

    // Interpolate source function
    this->source(top_at_1, ncol, nlay, nband, ngpt, play, plev, tlay, tsfc, jtemp, jpress, jeta, tropo, fmajor, sources, tlev);

    pool_t::dealloc(jtemp);
    pool_t::dealloc(jpress);
    pool_t::dealloc(tropo);
    pool_t::dealloc(fmajor);
    pool_t::dealloc(jeta);
  }

  // Compute gas optical depth given temperature, pressure, and composition
  template <typename PlayT, typename PlevT, typename TlayT, typename ColGasT,
            class OpticalPropsT, typename ToaT, typename ColDryT=view_t<RealT**> >
  void gas_optics(const int ncol, const int nlay,
                  bool top_at_1, PlayT const &play, PlevT const &plev, TlayT const &tlay,
                  GasConcsK<RealT, LayoutT, DeviceT> const &gas_desc,
                  ColGasT const& col_gas, OpticalPropsT &optical_props, ToaT &toa_src, ColDryT const &col_dry=ColDryT()) {
    const int ngpt  = this->get_ngpt();
    const int nband = this->get_nband();
    const int nflav = get_nflav();

    // Interpolation coefficients for use in source function
    auto jtemp  = pool_t::template alloc<int>(ncol,nlay);
    auto jpress = pool_t::template alloc<int>(ncol,nlay);
    auto tropo  = pool_t::template alloc<bool>(ncol,nlay);
    auto fmajor = pool_t::template alloc<RealT>(2,2,2,nflav,ncol,nlay);
    auto jeta   = pool_t::template alloc<int>(2,nflav,ncol,nlay);
    // Gas optics
    compute_gas_taus(top_at_1, ncol, nlay, ngpt, nband, play, plev, tlay, gas_desc, col_gas, optical_props, jtemp, jpress, jeta,
                     tropo, fmajor, col_dry);

    // External source function is constant
    if (toa_src.extent(0) != ncol || toa_src.extent(1) != ngpt) { stoprun("gas_optics(): array toa_src has wrong size"); }

    auto this_solar_src = this->solar_src;
    TIMED_KERNEL(FLATTEN_MD_KERNEL2(ncol, ngpt, icol, igpt,
      toa_src(icol,igpt) = this_solar_src(igpt);
    ));

    pool_t::dealloc(jtemp);
    pool_t::dealloc(jpress);
    pool_t::dealloc(tropo);
    pool_t::dealloc(fmajor);
    pool_t::dealloc(jeta);
  }

  // Returns optical properties and interpolation coefficients
  template <typename PlayT, typename PlevT, typename TlayT, typename ColGasT,
            class OpticalPropsT, typename JtempT, typename JpressT, typename JetaT, typename TropoT,
            typename FmajorT, typename ColDryT=view_t<RealT**> >
  void compute_gas_taus(bool top_at_1, int ncol, int nlay, int ngpt, int nband,
                        PlayT const &play, PlevT const &plev, TlayT const &tlay,
                        GasConcsK<RealT, LayoutT, DeviceT> const &gas_desc,
                        ColGasT const& col_gas, OpticalPropsT &optical_props,
                        JtempT const &jtemp, JpressT const &jpress, JetaT const &jeta,
                        TropoT const &tropo, FmajorT const &fmajor, ColDryT const &col_dry=ColDryT() ) {
    // Number of molecules per cm^2
    const int nlev = plev.extent(1);
    auto tau         = pool_t::template alloc<RealT>(ngpt,nlay,ncol);
    auto tau_rayleigh= pool_t::template alloc<RealT>(ngpt,nlay,ncol);
    // Interpolation variables used in major gas but not elsewhere, so don't need exporting
    auto vmr         = pool_t::template alloc<RealT>(ncol,nlay,this->get_ngas());
    auto col_mix     = pool_t::template alloc<RealT>(2,this->get_nflav(),ncol,nlay); // combination of major species's column amounts
                                                                               // index(1) : reference temperature level
                                                                               // index(2) : flavor
                                                                               // index(3) : layer
    auto fminor      = pool_t::template alloc<RealT>(2,2,this->get_nflav(),ncol,nlay); // interpolation fractions for minor species
                                                                                 // index(1) : reference eta level (temperature dependent)
                                                                                 // index(2) : reference temperature level
                                                                                 // index(3) : flavor
                                                                                 // index(4) : layer
    auto g0 = pool_t::template alloc<RealT>( ncol);
    auto col_dry_tmp= pool_t::template alloc<RealT>( ncol, nlev-1);

    // Error checking
    // Check for initialization
    if (! this->is_initialized()) { stoprun("ERROR: spectral configuration not loaded"); }
    // Check for presence of key species in ty_gas_concs; return error if any key species are not present
    this->check_key_species_present(gas_desc);
    #ifdef RRTMGP_EXPENSIVE_CHECKS
      if ( any(play < this->press_ref_min) || any(play > this->press_ref_max) ) {
        stoprun("gas_optics(): array play has values outside range");
      }
      if ( any(plev < this->press_ref_min) || any(plev > this->press_ref_max) ) {
        stoprun("gas_optics(): array plev has values outside range");
      }
      if ( any(tlay < this->temp_ref_min) || any(tlay > this->temp_ref_max) ) {
        stoprun("gas_optics(): array tlay has values outside range");
      }
    #endif
    if (col_dry.is_allocated()) {
      if (col_dry.extent(0) != ncol || col_dry.extent(1) != nlay) { stoprun("gas_optics(): array col_dry has wrong size"); }
      #ifdef RRTMGP_EXPENSIVE_CHECKS
        if (any(col_dry < 0.)) { stoprun("gas_optics(): array col_dry has values outside range"); }
      #endif
    }

    int ngas  = this->get_ngas();
    int nflav = this->get_nflav();
    int neta  = this->get_neta();
    int npres = this->get_npres();
    int ntemp = this->get_ntemp();
    // number of minor contributors, total num absorption coeffs
    int nminorlower  = this->minor_scales_with_density_lower.extent(0);
    int nminorklower = this->kminor_lower.extent(0);
    int nminorupper  = this->minor_scales_with_density_upper.extent(0);
    int nminorkupper = this->kminor_upper.extent(0);
    // Fill out the array of volume mixing ratios
    for (int igas = 0 ; igas < ngas ; igas++) {
      // Get vmr if  gas is provided in ty_gas_concs
      for (size_t igas2 = 0 ; igas2 < gas_desc.gas_name.size() ; igas2++) {
        if ( lower_case(this->gas_names[igas]) == lower_case(gas_desc.gas_name[igas2]) ) {
          auto vmr_slice = Kokkos::subview(vmr, Kokkos::ALL, Kokkos::ALL, igas);
          gas_desc.get_vmr(this->gas_names[igas], vmr_slice);
        }
      }
    }

    // Compute dry air column amounts (number of molecule per cm^2) if user hasn't provided them
    int idx_h2o = string_loc_in_array("h2o", this->gas_names);
    uview_t<RealT**> col_dry_wk;
    if (col_dry.is_allocated()) {
      col_dry_wk = col_dry;
    } else {
      this->get_col_dry(Kokkos::subview(vmr, Kokkos::ALL, Kokkos::ALL, idx_h2o),plev,g0,col_dry_tmp); // dry air column amounts computation
      col_dry_wk = col_dry_tmp;
    }
    // compute column gas amounts [molec/cm^2]
    // do ilay = 1, nlay
    //   do icol = 1, ncol
    TIMED_KERNEL(FLATTEN_MD_KERNEL2(ncol, nlay, icol, ilay,
      col_gas(icol,ilay,0) = col_dry_wk(icol,ilay);
    ));
    // do igas = 1, ngas
    //   do ilay = 1, nlay
    //     do icol = 1, ncol
    TIMED_KERNEL(FLATTEN_MD_KERNEL3(ncol, nlay, ngas, icol, ilay, igas,
      col_gas(icol,ilay,igas+1) = vmr(icol,ilay,igas) * col_dry_wk(icol,ilay);
    ));
    // ---- calculate gas optical depths ----
    Kokkos::deep_copy(tau, 0);

    interpolation(ncol, nlay, ngas, nflav, neta, npres, ntemp, this->flavor, this->press_ref_log, this->temp_ref,
                  this->press_ref_log_delta, this->temp_ref_min, this->temp_ref_delta, this->press_ref_trop_log,
                  this->vmr_ref, play, tlay, col_gas, jtemp, fmajor, fminor, col_mix, tropo, jeta, jpress);

    compute_tau_absorption(this->max_gpt_diff_lower, this->max_gpt_diff_upper, ncol, nlay, nband, ngpt, ngas, nflav, neta, npres, ntemp, nminorlower, nminorklower,
                           nminorupper, nminorkupper, idx_h2o, this->gpoint_flavor, this->get_band_lims_gpoint(),
                           this->kmajor, this->kminor_lower, this->kminor_upper, this->minor_limits_gpt_lower,
                           this->minor_limits_gpt_upper, this->minor_scales_with_density_lower,
                           this->minor_scales_with_density_upper, this->scale_by_complement_lower,
                           this->scale_by_complement_upper, this->idx_minor_lower, this->idx_minor_upper,
                           this->idx_minor_scaling_lower, this->idx_minor_scaling_upper, this->kminor_start_lower,
                           this->kminor_start_upper, tropo, col_mix, fmajor, fminor, play, tlay, col_gas,
                           jeta, jtemp, jpress, tau, top_at_1);

    if (this->krayl.is_allocated()) {
      compute_tau_rayleigh( ncol, nlay, nband, ngpt, ngas, nflav, neta, npres, ntemp, this->gpoint_flavor,
                            this->get_band_lims_gpoint(), this->krayl, idx_h2o, col_dry_wk, col_gas,
                            fminor, jeta, tropo, jtemp, tau_rayleigh);
    }
    combine_and_reorder(tau, tau_rayleigh, this->krayl.is_allocated(), optical_props);

    pool_t::dealloc(tau);
    pool_t::dealloc(tau_rayleigh);
    pool_t::dealloc(vmr);
    pool_t::dealloc(col_mix);
    pool_t::dealloc(fminor);
    pool_t::dealloc(g0);
    pool_t::dealloc(col_dry_tmp);
  }

  // Compute Planck source functions at layer centers and levels
  template <typename PlayT, typename PlevT, typename TlayT, typename TsfcT,
            typename JtempT, typename JpressT, typename JetaT, typename TropoT,
            typename FmajorT, typename TlevT=view_t<RealT**> >
  void source(bool top_at_1, int ncol, int nlay, int nbnd, int ngpt,
              PlayT const &play, PlevT const &plev, TlayT const &tlay,
              TsfcT const &tsfc, JtempT const &jtemp, JpressT const &jpress, JetaT const &jeta,
              TropoT const &tropo, FmajorT const &fmajor, SourceFuncLWK<RealT, LayoutT, DeviceT> &sources,
              TlevT const &tlev=TlevT()) {
    auto lay_source_t     = pool_t::template alloc<RealT>(ngpt,nlay,ncol);
    auto lev_source_inc_t = pool_t::template alloc<RealT>(ngpt,nlay,ncol);
    auto lev_source_dec_t = pool_t::template alloc<RealT>(ngpt,nlay,ncol);
    auto sfc_source_t     = pool_t::template alloc<RealT>(ngpt     ,ncol);
    // Variables for temperature at layer edges [K] (ncol, nlay+1)
    auto tlev_arr         = pool_t::template alloc<RealT>(ncol,nlay+1);

    // Source function needs temperature at interfaces/levels and at layer centers
    auto tlev_wk_pool = pool_t::template alloc<RealT>(ncol,nlay+1);
    uview_t<RealT**> tlev_wk;
    if (tlev.is_allocated()) {
      //   Users might have provided these
      tlev_wk = tlev;
    } else {
      tlev_wk = tlev_wk_pool;
      // Interpolate temperature to levels if not provided
      //   Interpolation and extrapolation at boundaries is weighted by pressure
      // do ilay = 1, nlay+1
      //   do icol = 1, ncol
      TIMED_KERNEL(FLATTEN_MD_KERNEL2(ncol, nlay+1, icol, ilay,
        if (ilay == 0) {
          tlev_wk(icol,0) = tlay(icol,0) + (plev(icol,0)-play(icol,0))*(tlay(icol,1)-tlay(icol,0)) / (play(icol,1)-play(icol,0));
        }
        else if (ilay == nlay) {
          tlev_wk(icol,ilay) = tlay(icol,ilay-1) + (plev(icol,ilay)-play(icol,ilay-1))*(tlay(icol,ilay-1)-tlay(icol,ilay)) /
            (play(icol,nlay)-play(icol,ilay));
        }
        else {
          tlev_wk(icol,ilay) = ( play(icol,ilay-1)*tlay(icol,ilay-1)*(plev(icol,ilay  )-play(icol,ilay))  +
                                 play(icol,ilay  )*tlay(icol,ilay  )*(play(icol,ilay-1)-plev(icol,ilay)) ) /
            (plev(icol,ilay)*(play(icol,ilay-1) - play(icol,ilay)));
        }
      ));
    }
    // Compute internal (Planck) source functions at layers and levels,
    //  which depend on mapping from spectral space that creates k-distribution.
    int nlayTmp = conv::merge( nlay-1 , 0 , top_at_1 );
    compute_Planck_source(ncol, nlay, nbnd, ngpt, this->get_nflav(), this->get_neta(), this->get_npres(), this->get_ntemp(),
                          this->get_nPlanckTemp(), tlay, tlev_wk, tsfc, nlayTmp, fmajor, jeta, tropo, jtemp, jpress,
                          this->get_gpoint_bands(), this->get_band_lims_gpoint(), this->planck_frac, this->temp_ref_min,
                          this->totplnk_delta, this->totplnk, this->gpoint_flavor, sfc_source_t, lay_source_t, lev_source_inc_t,
                          lev_source_dec_t);

    auto &sources_sfc_source = sources.sfc_source;
    // do igpt = 1, ngpt
    //   do icol = 1, ncol
    TIMED_KERNEL(FLATTEN_MD_KERNEL2(ncol, ngpt, icol, igpt,
      sources_sfc_source(icol,igpt) = sfc_source_t(igpt,icol);
    ));
    reorder123x321(ngpt, nlay, ncol, lay_source_t    , sources.lay_source    );
    reorder123x321(ngpt, nlay, ncol, lev_source_inc_t, sources.lev_source_inc);
    reorder123x321(ngpt, nlay, ncol, lev_source_dec_t, sources.lev_source_dec);

    pool_t::dealloc(lay_source_t);
    pool_t::dealloc(lev_source_inc_t);
    pool_t::dealloc(lev_source_dec_t);
    pool_t::dealloc(sfc_source_t);
    pool_t::dealloc(tlev_arr);
    pool_t::dealloc(tlev_wk_pool);
  }

  // Utility function, provided for user convenience
  // computes column amounts of dry air using hydrostatic equation
  template <typename VmrT, typename PlevT, typename G0T, typename ColDryT, typename LatT=view_t<RealT*> >
  void get_col_dry(VmrT const &vmr_h2o, PlevT const &plev, G0T const& g0, ColDryT const& col_dry, LatT const &latitude=LatT()) {
    // first and second term of Helmert formula
    RealT constexpr helmert1 = 9.80665;
    RealT constexpr helmert2 = 0.02586;
    int ncol = plev.extent(0);
    int nlev = plev.extent(1);
    if (latitude.is_allocated()) {
      // A purely OpenACC implementation would probably compute g0 within the kernel below
      // do icol = 1, ncol
      TIMED_KERNEL(Kokkos::parallel_for( ncol , KOKKOS_LAMBDA (int icol) {
        g0(icol) = helmert1 - helmert2 * cos(2.0 * M_PI * latitude(icol) / 180.0); // acceleration due to gravity [m/s^2]
      }));
    } else {
      // do icol = 1, ncol
      const auto grav = const_t::grav;
      TIMED_KERNEL(Kokkos::parallel_for( ncol, KOKKOS_LAMBDA (int icol) {
        g0(icol) = grav;
      }));
    }

    // do ilev = 1, nlev-1
    //   do icol = 1, ncol
    const auto m_dry = const_t::m_dry;
    const auto m_h2o = const_t::m_h2o;
    const auto avogad = const_t::avogad;
    TIMED_KERNEL(FLATTEN_MD_KERNEL2(ncol, nlev-1, icol, ilev,
      RealT delta_plev = Kokkos::fabs(plev(icol,ilev) - plev(icol,ilev+1));
      // Get average mass of moist air per mole of moist air
      RealT fact = 1. / (1.+vmr_h2o(icol,ilev));
      RealT m_air = (m_dry + m_h2o * vmr_h2o(icol,ilev)) * fact;
      col_dry(icol,ilev) = 10. * delta_plev * avogad * fact/(1000.*m_air*100.*g0(icol));
    ));
  }

  // Utility function to combine optical depths from gas absorption and Rayleigh scattering
  //   (and reorder them for convenience, while we're at it)
  template <typename TauT, typename TauRayT>
  void combine_and_reorder(TauT const &tau, TauRayT const &tau_rayleigh, bool has_rayleigh,
                           OpticalProps1sclK<RealT, LayoutT, DeviceT> &optical_props) {
    int ncol = tau.extent(2);
    int nlay = tau.extent(1);
    int ngpt = tau.extent(0);
    reorder123x321(ngpt, nlay, ncol, tau, optical_props.tau);
  }

  // Utility function to combine optical depths from gas absorption and Rayleigh scattering
  //   (and reorder them for convenience, while we're at it)
  template <typename TauT, typename TauRayT>
  void combine_and_reorder(TauT const &tau, TauRayT const &tau_rayleigh, bool has_rayleigh,
                           OpticalProps2strK<RealT, LayoutT, DeviceT> &optical_props) {
    int ncol = tau.extent(2);
    int nlay = tau.extent(1);
    int ngpt = tau.extent(0);
    if (has_rayleigh) {
      // combine optical depth and rayleigh scattering
      combine_and_reorder_2str(ncol, nlay, ngpt, tau, tau_rayleigh, optical_props.tau, optical_props.ssa, optical_props.g);
    } else {
      // index reorder (ngpt, nlay, ncol) -> (ncol,nlay,gpt)
      reorder123x321(ngpt, nlay, ncol, tau, optical_props.tau);
      Kokkos::deep_copy(optical_props.ssa, 0);
      Kokkos::deep_copy(optical_props.g,   0);
    }
  }

  void print_norms(const bool print_prefix=false) const {
    std::string prefix = print_prefix ? "JGFK" : "";
                                                      std::cout << prefix << "name                                  : " << std::setw(20) << this->name                                   << "\n";
                                                      std::cout << prefix << "totplnk_delta                         : " << std::setw(20) << this->totplnk_delta                          << "\n";
                                                      std::cout << prefix << "press_ref_min                         : " << std::setw(20) << this->press_ref_min                          << "\n";
                                                      std::cout << prefix << "press_ref_max                         : " << std::setw(20) << this->press_ref_max                          << "\n";
                                                      std::cout << prefix << "temp_ref_min                          : " << std::setw(20) << this->temp_ref_min                           << "\n";
                                                      std::cout << prefix << "temp_ref_max                          : " << std::setw(20) << this->temp_ref_max                           << "\n";
                                                      std::cout << prefix << "press_ref_log_delta                   : " << std::setw(20) << this->press_ref_log_delta                    << "\n";
                                                      std::cout << prefix << "temp_ref_delta                        : " << std::setw(20) << this->temp_ref_delta                         << "\n";
                                                      std::cout << prefix << "press_ref_trop_log                    : " << std::setw(20) << this->press_ref_trop_log                     << "\n";
    if (this->band2gpt.is_allocated()                       ) { std::cout << prefix << "sum(band2gpt     )                    : " << std::setw(20) << conv::sum(this->band2gpt     )                     << "\n"; }
    if (this->gpt2band.is_allocated()                       ) { std::cout << prefix << "sum(gpt2band     )                    : " << std::setw(20) << conv::sum(this->gpt2band     )                     << "\n"; }
    if (this->band_lims_wvn.is_allocated()                  ) { std::cout << prefix << "sum(band_lims_wvn)                    : " << std::setw(20) << conv::sum(this->band_lims_wvn)                     << "\n"; }
    if (this->press_ref.is_allocated()                      ) { std::cout << prefix << "sum(press_ref    )                    : " << std::setw(20) << conv::sum(this->press_ref    )                     << "\n"; }
    if (this->press_ref_log.is_allocated()                  ) { std::cout << prefix << "sum(press_ref_log)                    : " << std::setw(20) << conv::sum(this->press_ref_log)                     << "\n"; }
    if (this->temp_ref.is_allocated()                       ) { std::cout << prefix << "sum(temp_ref     )                    : " << std::setw(20) << conv::sum(this->temp_ref     )                     << "\n"; }
    if (this->vmr_ref.is_allocated()                        ) { std::cout << prefix << "sum(vmr_ref                )          : " << std::setw(20) << conv::sum(this->vmr_ref                )           << "\n"; }
    if (this->flavor.is_allocated()                         ) { std::cout << prefix << "sum(flavor                 )          : " << std::setw(20) << conv::sum(this->flavor                 )           << "\n"; }
    if (this->gpoint_flavor.is_allocated()                  ) { std::cout << prefix << "sum(gpoint_flavor          )          : " << std::setw(20) << conv::sum(this->gpoint_flavor          )           << "\n"; }
    if (this->kmajor.is_allocated()                         ) { std::cout << prefix << "sum(kmajor                 )          : " << std::setw(20) << conv::sum(this->kmajor                 )           << "\n"; }
    if (this->minor_limits_gpt_lower.is_allocated()         ) { std::cout << prefix << "sum(minor_limits_gpt_lower )          : " << std::setw(20) << conv::sum(this->minor_limits_gpt_lower )           << "\n"; }
    if (this->minor_limits_gpt_upper.is_allocated()         ) { std::cout << prefix << "sum(minor_limits_gpt_upper )          : " << std::setw(20) << conv::sum(this->minor_limits_gpt_upper )           << "\n"; }
    if (this->idx_minor_lower.is_allocated()                ) { std::cout << prefix << "sum(idx_minor_lower        )          : " << std::setw(20) << conv::sum(this->idx_minor_lower        )           << "\n"; }
    if (this->idx_minor_upper.is_allocated()                ) { std::cout << prefix << "sum(idx_minor_upper        )          : " << std::setw(20) << conv::sum(this->idx_minor_upper        )           << "\n"; }
    if (this->idx_minor_scaling_lower.is_allocated()        ) { std::cout << prefix << "sum(idx_minor_scaling_lower)          : " << std::setw(20) << conv::sum(this->idx_minor_scaling_lower)           << "\n"; }
    if (this->idx_minor_scaling_upper.is_allocated()        ) { std::cout << prefix << "sum(idx_minor_scaling_upper)          : " << std::setw(20) << conv::sum(this->idx_minor_scaling_upper)           << "\n"; }
    if (this->kminor_start_lower.is_allocated()             ) { std::cout << prefix << "sum(kminor_start_lower     )          : " << std::setw(20) << conv::sum(this->kminor_start_lower     )           << "\n"; }
    if (this->kminor_start_upper.is_allocated()             ) { std::cout << prefix << "sum(kminor_start_upper     )          : " << std::setw(20) << conv::sum(this->kminor_start_upper     )           << "\n"; }
    if (this->kminor_lower.is_allocated()                   ) { std::cout << prefix << "sum(kminor_lower           )          : " << std::setw(20) << conv::sum(this->kminor_lower           )           << "\n"; }
    if (this->kminor_upper.is_allocated()                   ) { std::cout << prefix << "sum(kminor_upper           )          : " << std::setw(20) << conv::sum(this->kminor_upper           )           << "\n"; }
    if (this->krayl.is_allocated()                          ) { std::cout << prefix << "sum(krayl                  )          : " << std::setw(20) << conv::sum(this->krayl                  )           << "\n"; }
    if (this->planck_frac.is_allocated()                    ) { std::cout << prefix << "sum(planck_frac            )          : " << std::setw(20) << conv::sum(this->planck_frac            )           << "\n"; }
    if (this->totplnk.is_allocated()                        ) { std::cout << prefix << "sum(totplnk                )          : " << std::setw(20) << conv::sum(this->totplnk                )           << "\n"; }
    if (this->solar_src.is_allocated()                      ) { std::cout << prefix << "sum(solar_src              )          : " << std::setw(20) << conv::sum(this->solar_src              )           << "\n"; }
    if (this->minor_scales_with_density_lower.is_allocated()) { std::cout << prefix << "count(minor_scales_with_density_lower): " << std::setw(20) << conv::sum(this->minor_scales_with_density_lower) << "\n"; }
    if (this->minor_scales_with_density_upper.is_allocated()) { std::cout << prefix << "count(minor_scales_with_density_upper): " << std::setw(20) << conv::sum(this->minor_scales_with_density_upper) << "\n"; }
    if (this->scale_by_complement_lower.is_allocated()      ) { std::cout << prefix << "count(scale_by_complement_lower      ): " << std::setw(20) << conv::sum(this->scale_by_complement_lower      ) << "\n"; }
    if (this->scale_by_complement_upper.is_allocated()      ) { std::cout << prefix << "count(scale_by_complement_upper      ): " << std::setw(20) << conv::sum(this->scale_by_complement_upper      ) << "\n"; }
    if (this->is_key.is_allocated()                         ) { std::cout << prefix << "count(is_key                         ): " << std::setw(20) << conv::sum(this->is_key                         ) << "\n"; }
  }

};
