

#include "mo_fluxes_broadband_kernels.h"


// Spectral reduction over all points
extern "C" void sum_broadband(int ncol, int nlev, int ngpt, real *spectral_flux_p, real *broadband_flux_p) {
  umgReal3d spectral_flux ("spectral_flux" ,spectral_flux_p ,ncol,nlev,ngpt);
  umgReal2d broadband_flux("broadband_flux",broadband_flux_p,ncol,nlev);

  // do ilev = 1, nlev
  //   do icol = 1, ncol
  parallel_for_cpu_serial( Bounds<2>(nlev,ncol) , YAKL_LAMBDA (int ilev, int icol) {
    real bb_flux_s = 0.0_wp;
    for (int igpt=1; igpt<=ngpt; igpt++) {
      bb_flux_s += spectral_flux(icol, ilev, igpt);
    }
    broadband_flux(icol, ilev) = bb_flux_s;
  });
}



// Net flux: Spectral reduction over all points
extern "C" void net_broadband_full(int ncol, int nlev, int ngpt, real *spectral_flux_dn_p, real *spectral_flux_up_p, real *broadband_flux_net_p) {
  umgReal3d spectral_flux_dn  ("spectral_flux_dn"  ,spectral_flux_dn_p  ,ncol,nlev,ngpt);
  umgReal3d spectral_flux_up  ("spectral_flux_up"  ,spectral_flux_up_p  ,ncol,nlev,ngpt);
  umgReal2d broadband_flux_net("broadband_flux_net",broadband_flux_net_p,ncol,nlev     );

  // do ilev = 1, nlev
  //   do icol = 1, ncol
  parallel_for_cpu_serial( Bounds<2>(nlev,ncol) , YAKL_LAMBDA (int ilev, int icol) {
    real diff = spectral_flux_dn(icol, ilev, 1   ) - spectral_flux_up(icol, ilev,     1);
    broadband_flux_net(icol, ilev) = diff;
  });

  // do igpt = 2, ngpt
  //   do ilev = 1, nlev
  //     do icol = 1, ncol
  parallel_for_cpu_serial( Bounds<3>({2,ngpt},nlev,ncol) , YAKL_LAMBDA (int igpt, int ilev, int icol) {
    real diff = spectral_flux_dn(icol, ilev, igpt) - spectral_flux_up(icol, ilev, igpt);
    yakl::atomicAdd( broadband_flux_net(icol,ilev) , diff );
  });
  std::cout << "WARNING: THIS ISN'T TESTED!\n";
  std::cout << __FILE__ << ": " << __LINE__ << std::endl;
}



// Net flux when bradband flux up and down are already available
extern "C" void net_broadband_precalc(int ncol, int nlev, real *flux_dn_p, real *flux_up_p, real *broadband_flux_net_p) {
  umgReal2d flux_dn           ("flux_dn"           ,flux_dn_p           ,ncol,nlev);
  umgReal2d flux_up           ("flux_up"           ,flux_up_p           ,ncol,nlev);
  umgReal2d broadband_flux_net("broadband_flux_net",broadband_flux_net_p,ncol,nlev);

  // do ilev = 1, nlev
  //   do icol = 1, ncol
  parallel_for_cpu_serial( Bounds<2>(nlev,ncol) , YAKL_LAMBDA (int ilev, int icol) {
     broadband_flux_net(icol,ilev) = flux_dn(icol,ilev) - flux_up(icol,ilev);
  });
  std::cout << "WARNING: THIS ISN'T TESTED!\n";
  std::cout << __FILE__ << ": " << __LINE__ << std::endl;
}

