
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include "mo_gas_concentrations.h"
#include "mo_garand_atmos_io.h"
#include "mo_gas_optics_rrtmgp.h"
#include "rrtmgp_const.h"
#include "mo_load_coefficients.h"
#include "mo_load_cloud_coefficients.h"
#include "mo_fluxes.h"
#include "mo_fluxes_byband.h"
#include "mo_rte_lw.h"
#include "mo_rte_sw.h"

bool constexpr verbose      = false;
bool constexpr write_fluxes = false;
bool constexpr print_norms  = true;

int main(int argc , char **argv) {

  Kokkos::initialize(argc, argv);

  using DeviceT = DefaultDevice;
  using LayoutT = Kokkos::LayoutLeft;
  using MDRP = conv::MDRP<LayoutT>;
  using real = double;
  using real1d_t = Kokkos::View<real*,   LayoutT, DeviceT>;
  using real2d_t = Kokkos::View<real**,  LayoutT, DeviceT>;
  using real3d_t = Kokkos::View<real***, LayoutT, DeviceT>;
  using bool2d_t = Kokkos::View<bool**,  LayoutT, DeviceT>;
  using hreal2d_t = Kokkos::View<real**, LayoutT, HostDevice>;
  using pool_t = conv::MemPoolSingleton<real, LayoutT, DeviceT>;

  {
    using conv::merge;

    bool constexpr use_luts = true;

    if (argc < 5) { stoprun("Error: Fewer than 4 command line arguments provided"); }
    std::string input_file        =      argv[1];
    std::string k_dist_file       =      argv[2];
    std::string cloud_optics_file =      argv[3];
    const int ncol                = std::atoi(argv[4]);
    int nloops = 1;
    if (argc >= 6) { nloops       = std::atoi(argv[5]); }
    if (ncol   <= 0) { stoprun("Error: Number of columns must be > 0"); }
    if (nloops <= 0) { stoprun("Error: Number of loops must be > 0"); }
    if (argc > 6) { std::cout << "WARNING: Using only 5 parameters. Ignoring the rest\n"; }
    if (input_file == "-h" || input_file == "--help") {
      std::cout << "./rrtmgp_allsky  input_file  absorption_coefficients_file  cloud_optics_file  ncol  [nloops]\n\n";
      std::exit(0);
    }

    if (verbose) {
      std::cout << "Parameters: \n";
      std::cout << "    Input file:        " << input_file        << "\n";
      std::cout << "    k_dist file:       " << k_dist_file       << "\n";
      std::cout << "    Cloud Optics file: " << cloud_optics_file << "\n";
      std::cout << "    ncol:              " << ncol              << "\n";
      std::cout << "    nloops:            " << nloops            << "\n\n";
    }

    // Read temperature, pressure, gas concentrations. Arrays are allocated as they are read
    real2d_t p_lay_k;
    real2d_t t_lay_k;
    real2d_t p_lev_k;
    real2d_t t_lev_k;
    GasConcsK<real, LayoutT> gas_concs_k;
    real2d_t col_dry_k;

    // Read data from the input file
    if (verbose) std::cout << "Reading input file\n\n";
    read_atmos(input_file, p_lay_k, t_lay_k, p_lev_k, t_lev_k, gas_concs_k, col_dry_k, ncol);

    const int nlay = p_lay_k.extent(1);
    const int nlev = p_lev_k.extent(1);

    // load data into classes
    if (verbose) std::cout << "Reading k_dist file\n\n";
    GasOpticsRRTMGPK<real, LayoutT> k_dist_k;
    load_and_init(k_dist_k, k_dist_file, gas_concs_k);

    bool is_sw = k_dist_k.source_is_external();

    if (verbose) std::cout << "Reading cloud optics file\n\n";
    CloudOpticsK<real, LayoutT> cloud_optics_k;
    if (use_luts) {
      load_cld_lutcoeff (cloud_optics_k, cloud_optics_file);
    } else {
      load_cld_padecoeff(cloud_optics_k, cloud_optics_file);
    }
    cloud_optics_k.set_ice_roughness(2);

    // Problem sizes
    int nbnd = k_dist_k.get_nband();
    int ngpt = k_dist_k.get_ngpt();
    auto p_lay_host_k = Kokkos::create_mirror_view_and_copy(HostDevice(), p_lay_k);
    bool top_at_1 = p_lay_host_k(0, 0) < p_lay_host_k(0, nlay-1);

    // LW calculations neglect scattering; SW calculations use the 2-stream approximation
    if (is_sw) {  // Shortwave

      if (verbose) std::cout << "This is a shortwave simulation\n\n";
      OpticalProps2strK<real, LayoutT> atmos_k;
      OpticalProps2strK<real, LayoutT> clouds_k;

      // Clouds optical props are defined by band
      clouds_k.init(k_dist_k.get_band_lims_wavenumber());

      // Allocate arrays for the optical properties themselves.
      atmos_k .alloc_2str(ncol, nlay, k_dist_k);
      clouds_k.alloc_2str(ncol, nlay);

      //  Boundary conditions depending on whether the k-distribution being supplied
      real2d_t toa_flux_k   ("toa_flux"   ,ncol,ngpt);
      real2d_t sfc_alb_dir_k("sfc_alb_dir",nbnd,ncol);
      real2d_t sfc_alb_dif_k("sfc_alb_dif",nbnd,ncol);
      real1d_t mu0_k        ("mu0"        ,ncol);
      // Ocean-ish values for no particular reason
      Kokkos::deep_copy(sfc_alb_dir_k, 0.06);
      Kokkos::deep_copy(sfc_alb_dif_k, 0.06);
      Kokkos::deep_copy(mu0_k        , 0.86);

      // Fluxes
      real2d_t flux_up_k ("flux_up" ,ncol,nlay+1);
      real2d_t flux_dn_k ("flux_dn" ,ncol,nlay+1);
      real2d_t flux_dir_k("flux_dir",ncol,nlay+1);
      real2d_t flux_net_k("flux_net",ncol,nlay+1);
      real3d_t bnd_flux_up_k ("bnd_flux_up" ,ncol,nlay+1,nbnd);
      real3d_t bnd_flux_dn_k ("bnd_flux_dn" ,ncol,nlay+1,nbnd);
      real3d_t bnd_flux_dir_k("bnd_flux_dir",ncol,nlay+1,nbnd);
      real3d_t bnd_flux_net_k("bnd_flux_net",ncol,nlay+1,nbnd);

      // Clouds
      real2d_t lwp_k("lwp",ncol,nlay);
      real2d_t iwp_k("iwp",ncol,nlay);
      real2d_t rel_k("rel",ncol,nlay);
      real2d_t rei_k("rei",ncol,nlay);
      bool2d_t cloud_mask_k("cloud_mask",ncol,nlay);

      // Restrict clouds to troposphere (> 100 hPa = 100*100 Pa) and not very close to the ground (< 900 hPa), and
      // put them in 2/3 of the columns since that's roughly the total cloudiness of earth
      real rel_val = 0.5 * (cloud_optics_k.get_min_radius_liq() + cloud_optics_k.get_max_radius_liq());
      real rei_val = 0.5 * (cloud_optics_k.get_min_radius_ice() + cloud_optics_k.get_max_radius_ice());

      // do ilay=1,nlay
      //   do icol=1,ncol
      Kokkos::parallel_for( MDRP::template get<2>({ncol, nlay}) , KOKKOS_LAMBDA (int icol, int ilay) {
        cloud_mask_k(icol,ilay) = p_lay_k(icol,ilay) > 100. * 100. && p_lay_k(icol,ilay) < 900. * 100. && ((icol+1) % 3) != 0;
        // Ice and liquid will overlap in a few layers
        lwp_k(icol,ilay) = merge(10.,  0., cloud_mask_k(icol,ilay) && t_lay_k(icol,ilay) > 263.);
        iwp_k(icol,ilay) = merge(10.,  0., cloud_mask_k(icol,ilay) && t_lay_k(icol,ilay) < 273.);
        rel_k(icol,ilay) = merge(rel_val, 0., lwp_k(icol,ilay) > 0.);
        rei_k(icol,ilay) = merge(rei_val, 0., iwp_k(icol,ilay) > 0.);
      });

      const size_t base_ref = 18000;
      const size_t my_size_ref = ncol * nlay * nlev;
      pool_t::init(2e6 * (float(my_size_ref) / base_ref));
      real3d_t col_gas("col_gas", ncol, nlay, k_dist_k.get_ngas()+1);

      if (verbose) std::cout << "Running the main loop\n\n";
      auto start_t = std::chrono::high_resolution_clock::now();

      for (int iloop = 1 ; iloop <= nloops ; iloop++) {

        cloud_optics_k.cloud_optics(ncol, nlay, lwp_k, iwp_k, rel_k, rei_k, clouds_k);

        // Solvers
        FluxesBybandK<real, LayoutT> fluxes_k;
        fluxes_k.flux_up     = flux_up_k ;
        fluxes_k.flux_dn     = flux_dn_k ;
        fluxes_k.flux_dn_dir = flux_dir_k;
        fluxes_k.flux_net    = flux_net_k;
        fluxes_k.bnd_flux_up     = bnd_flux_up_k ;
        fluxes_k.bnd_flux_dn     = bnd_flux_dn_k ;
        fluxes_k.bnd_flux_dn_dir = bnd_flux_dir_k;
        fluxes_k.bnd_flux_net = bnd_flux_net_k;
        k_dist_k.gas_optics(ncol, nlay, top_at_1, p_lay_k, p_lev_k, t_lay_k, gas_concs_k, col_gas, atmos_k, toa_flux_k);

        clouds_k.delta_scale();

        clouds_k.increment(atmos_k);

        rte_sw(atmos_k, top_at_1, mu0_k, toa_flux_k, sfc_alb_dir_k, sfc_alb_dif_k, fluxes_k);

        if (print_norms) fluxes_k.print_norms();
      }

      auto stop_t = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_t - start_t);
      std::cout << "Shortwave did " << nloops << " loops of " << ncol << " cols and " << nlay << " layers in " <<  duration.count() / 1000000.0 << " s" << std::endl;

      pool_t::finalize();

      if (verbose) std::cout << "Writing fluxes\n\n";
      if (write_fluxes) write_sw_fluxes(input_file, flux_up_k, flux_dn_k, flux_dir_k, ncol);

      // Hacky "unit" test against pre-computed reference fluxes
      if (ncol == 1 && nloops == 1) {
        if (std::abs(conv::sum(flux_up_k )-19104.862129836212)/(19104.862129836212)          > 1.e-10) std::exit(-1);
        if (std::abs(conv::sum(flux_dn_k )-38046.157649700355)/(38046.157649700355)          > 1.e-10) std::exit(-1);
        if (std::abs(conv::sum(flux_dir_k)-24998.593939345046)/(24998.593939345046)          > 1.e-10) std::exit(-1);
        // And test to make sure our broadband and byband fluxes are consistent
        if (std::abs(conv::sum(flux_up_k )-conv::sum(bnd_flux_up_k ) )/conv::sum(flux_up_k ) > 1.e-10) std::exit(-1);
        if (std::abs(conv::sum(flux_dn_k )-conv::sum(bnd_flux_dn_k ) )/conv::sum(flux_dn_k ) > 1.e-10) std::exit(-1);
        if (std::abs(conv::sum(flux_dir_k)-conv::sum(bnd_flux_dir_k) )/conv::sum(flux_dir_k) > 1.e-10) std::exit(-1);
        if (std::abs(conv::sum(flux_net_k)-conv::sum(bnd_flux_net_k) )/conv::sum(flux_net_k) > 1.e-10) std::exit(-1);
      }

    } else {  // Longwave

      if (verbose) std::cout << "This is a longwave simulation\n\n";

      // Weights and angle secants for first order (k=1) Gaussian quadrature.
      //   Values from Table 2, Clough et al, 1992, doi:10.1029/92JD01419
      //   after Abramowitz & Stegun 1972, page 921
      int constexpr max_gauss_pts = 4;
      hreal2d_t gauss_Ds_host_k ("gauss_Ds" ,max_gauss_pts,max_gauss_pts);
      gauss_Ds_host_k(0,0) = 1.66      ; gauss_Ds_host_k(1,0) =         0.; gauss_Ds_host_k(2,0) =         0.; gauss_Ds_host_k(3,0) =         0.;
      gauss_Ds_host_k(0,1) = 1.18350343; gauss_Ds_host_k(1,1) = 2.81649655; gauss_Ds_host_k(2,1) =         0.; gauss_Ds_host_k(3,1) =         0.;
      gauss_Ds_host_k(0,2) = 1.09719858; gauss_Ds_host_k(1,2) = 1.69338507; gauss_Ds_host_k(2,2) = 4.70941630; gauss_Ds_host_k(3,2) =         0.;
      gauss_Ds_host_k(0,3) = 1.06056257; gauss_Ds_host_k(1,3) = 1.38282560; gauss_Ds_host_k(2,3) = 2.40148179; gauss_Ds_host_k(3,3) = 7.15513024;

      hreal2d_t gauss_wts_host_k("gauss_wts",max_gauss_pts,max_gauss_pts);
      gauss_wts_host_k(0,0) = 0.5         ; gauss_wts_host_k(1,0) = 0.          ; gauss_wts_host_k(2,0) = 0.          ; gauss_wts_host_k(3,0) = 0.          ;
      gauss_wts_host_k(0,1) = 0.3180413817; gauss_wts_host_k(1,1) = 0.1819586183; gauss_wts_host_k(2,1) = 0.          ; gauss_wts_host_k(3,1) = 0.          ;
      gauss_wts_host_k(0,2) = 0.2009319137; gauss_wts_host_k(1,2) = 0.2292411064; gauss_wts_host_k(2,2) = 0.0698269799; gauss_wts_host_k(3,2) = 0.          ;
      gauss_wts_host_k(0,3) = 0.1355069134; gauss_wts_host_k(1,3) = 0.2034645680; gauss_wts_host_k(2,3) = 0.1298475476; gauss_wts_host_k(3,3) = 0.0311809710;

      real2d_t gauss_Ds_k ("gauss_Ds" ,max_gauss_pts,max_gauss_pts);
      real2d_t gauss_wts_k("gauss_wts",max_gauss_pts,max_gauss_pts);
      Kokkos::deep_copy(gauss_Ds_k, gauss_Ds_host_k);
      Kokkos::deep_copy(gauss_wts_k, gauss_wts_host_k);

      OpticalProps1sclK<real, LayoutT> atmos_k;
      OpticalProps1sclK<real, LayoutT> clouds_k;

      // Clouds optical props are defined by band
      clouds_k.init(k_dist_k.get_band_lims_wavenumber());

      // Allocate arrays for the optical properties themselves.
      atmos_k .alloc_1scl(ncol, nlay, k_dist_k);
      clouds_k.alloc_1scl(ncol, nlay);

      //  Boundary conditions depending on whether the k-distribution being supplied
      //   is LW or SW
      SourceFuncLWK<real, LayoutT> lw_sources_k;
      lw_sources_k.alloc(ncol, nlay, k_dist_k);

      real1d_t t_sfc_k   ("t_sfc"        ,ncol);
      real2d_t emis_sfc_k("emis_sfc",nbnd,ncol);
      // Surface temperature
      auto t_lev_host_k = Kokkos::create_mirror_view_and_copy(HostDevice(), t_lev_k);
      Kokkos::deep_copy(t_sfc_k, t_lev_host_k(0, merge(nlay, 0, top_at_1)));
      Kokkos::deep_copy(emis_sfc_k, 0.98);

      // Fluxes
      real2d_t flux_up_k ( "flux_up" ,ncol,nlay+1);
      real2d_t flux_dn_k ( "flux_dn" ,ncol,nlay+1);
      real2d_t flux_net_k("flux_net" ,ncol,nlay+1);
      real3d_t bnd_flux_up_k ("bnd_flux_up" ,ncol,nlay+1,nbnd);
      real3d_t bnd_flux_dn_k ("bnd_flux_dn" ,ncol,nlay+1,nbnd);
      real3d_t bnd_flux_net_k("bnd_flux_net" ,ncol,nlay+1,nbnd);

      // Clouds
      real2d_t lwp_k("lwp",ncol,nlay);
      real2d_t iwp_k("iwp",ncol,nlay);
      real2d_t rel_k("rel",ncol,nlay);
      real2d_t rei_k("rei",ncol,nlay);
      bool2d_t cloud_mask_k("cloud_mask",ncol,nlay);

      // Restrict clouds to troposphere (> 100 hPa = 100*100 Pa)
      //   and not very close to the ground (< 900 hPa), and
      //   put them in 2/3 of the columns since that's roughly the
      //   total cloudiness of earth
      real rel_val = 0.5 * (cloud_optics_k.get_min_radius_liq() + cloud_optics_k.get_max_radius_liq());
      real rei_val = 0.5 * (cloud_optics_k.get_min_radius_ice() + cloud_optics_k.get_max_radius_ice());

      // do ilay=1,nlay
      //   do icol=1,ncol
      Kokkos::parallel_for( MDRP::template get<2>({ncol, nlay}) , KOKKOS_LAMBDA (int icol, int ilay) {
        cloud_mask_k(icol,ilay) = p_lay_k(icol,ilay) > 100. * 100. && p_lay_k(icol,ilay) < 900. * 100. && ((icol+1) % 3) != 0;
        // Ice and liquid will overlap in a few layers
        lwp_k(icol,ilay) = merge(10.,  0., cloud_mask_k(icol,ilay) && t_lay_k(icol,ilay) > 263.);
        iwp_k(icol,ilay) = merge(10.,  0., cloud_mask_k(icol,ilay) && t_lay_k(icol,ilay) < 273.);
        rel_k(icol,ilay) = merge(rel_val, 0., lwp_k(icol,ilay) > 0.);
        rei_k(icol,ilay) = merge(rei_val, 0., iwp_k(icol,ilay) > 0.);
      });

      const size_t base_ref = 18000;
      const size_t my_size_ref = ncol * nlay * nlev;
      pool_t::init(2e6 * (float(my_size_ref) / base_ref));
      real3d_t col_gas("col_gas", ncol, nlay, k_dist_k.get_ngas()+1);

      // Multiple iterations for big problem sizes, and to help identify data movement
      //   For CPUs we can introduce OpenMP threading over loop iterations
      if (verbose) std::cout << "Running the main loop\n\n";
      auto start_t = std::chrono::high_resolution_clock::now();

      for (int iloop = 1 ; iloop <= nloops ; iloop++) {
        cloud_optics_k.cloud_optics(ncol, nlay, lwp_k, iwp_k, rel_k, rei_k, clouds_k);

        // Solvers
        FluxesBybandK<real, LayoutT> fluxes_k;
        fluxes_k.flux_up = flux_up_k;
        fluxes_k.flux_dn = flux_dn_k;
        fluxes_k.flux_net= flux_net_k;
        fluxes_k.bnd_flux_up = bnd_flux_up_k;
        fluxes_k.bnd_flux_dn = bnd_flux_dn_k;
        fluxes_k.bnd_flux_net= bnd_flux_net_k;

        // Calling with an empty col_dry parameter
        k_dist_k.gas_optics(ncol, nlay, top_at_1, p_lay_k, p_lev_k, t_lay_k, t_sfc_k, gas_concs_k, col_gas, atmos_k, lw_sources_k, real2d_t(), t_lev_k);

        clouds_k.increment(atmos_k);

        rte_lw(max_gauss_pts, gauss_Ds_k, gauss_wts_k, atmos_k, top_at_1, lw_sources_k, emis_sfc_k, fluxes_k);
        if (print_norms) fluxes_k.print_norms();
      }

      auto stop_t = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_t - start_t);
      std::cout << "Longwave did " << nloops << " loops of " << ncol << " cols and " << nlay << " layers in " <<  duration.count() / 1000000.0 << " s" << std::endl;

      pool_t::finalize();

      if (verbose) std::cout << "Writing fluxes\n\n";
      if (write_fluxes) write_lw_fluxes(input_file, flux_up_k, flux_dn_k, ncol);

      // Hacky "unit" test against pre-computed reference fluxes
      if (ncol == 1 && nloops == 1) {
        if (std::abs(conv::sum(flux_up_k )-10264.518998579415)/(10264.518998579415)          > 1.e-10) std::exit(-1);
        if (std::abs(conv::sum(flux_dn_k )-6853.2350138542843)/(6853.2350138542843)          > 1.e-10) std::exit(-1);
        // And test to make sure our broadband and byband fluxes are consistent
        if (std::abs(conv::sum(flux_up_k )-conv::sum(bnd_flux_up_k ) )/conv::sum(flux_up_k ) > 1.e-10) std::exit(-1);
        if (std::abs(conv::sum(flux_dn_k )-conv::sum(bnd_flux_dn_k ) )/conv::sum(flux_dn_k ) > 1.e-10) std::exit(-1);
        if (std::abs(conv::sum(flux_net_k)-conv::sum(bnd_flux_net_k) )/conv::sum(flux_net_k) > 1.e-10) std::exit(-1);
      }

    }  // if (is_sw)

  }

  Kokkos::finalize();
}
