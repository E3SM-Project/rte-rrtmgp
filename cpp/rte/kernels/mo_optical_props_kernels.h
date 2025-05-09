
#pragma once

#include "rrtmgp_const.h"
#include "rrtmgp_conversion.h"

// increment 2stream by 2stream
template <typename Tau1T, typename Ssa1T, typename G1T,
          typename Tau2T, typename Ssa2T, typename G2T,
          typename GptLimsT>
void inc_2stream_by_2stream_bybnd(int ncol, int nlay, int ngpt,
                                  Tau1T const &tau1, Ssa1T const &ssa1, G1T const &g1,
                                  Tau2T const &tau2, Ssa2T const &ssa2, G2T const &g2,
                                  int nbnd, GptLimsT const &gpt_lims) {
  using RealT = typename Tau1T::non_const_value_type;
  using DeviceT = typename Tau1T::device_type;
  using LayoutT = typename Tau1T::array_layout;
  using mdrp_t  = typename conv::MDRP<LayoutT, DeviceT>;

  constexpr RealT eps = 3*std::numeric_limits<RealT>::min();

  // do igpt = 1 , ngpt
  //   do ilay = 1, nlay
  //     do icol = 1, ncol
  TIMED_KERNEL(FLATTEN_MD_KERNEL4(ncol,nlay,ngpt,nbnd, icol, ilay, igpt, ibnd,
    if (igpt >= gpt_lims(0,ibnd) && igpt <= gpt_lims(1,ibnd) ) {
      // t=tau1 + tau2
      RealT tau12 = tau1(icol,ilay,igpt) + tau2(icol,ilay,ibnd);
      // w=(tau1*ssa1 + tau2*ssa2) / t
      RealT tauscat12 = tau1(icol,ilay,igpt) * ssa1(icol,ilay,igpt) +
                       tau2(icol,ilay,ibnd) * ssa2(icol,ilay,ibnd);
      g1(icol,ilay,igpt) = (tau1(icol,ilay,igpt) * ssa1(icol,ilay,igpt) * g1(icol,ilay,igpt) +
                            tau2(icol,ilay,ibnd) * ssa2(icol,ilay,ibnd) * g2(icol,ilay,ibnd)) / Kokkos::fmax(eps,tauscat12);
      ssa1(icol,ilay,igpt) = tauscat12 / Kokkos::fmax(eps,tau12);
      tau1(icol,ilay,igpt) = tau12;
    }
  ));
}

// Addition of optical properties: the first set are incremented by the second set.
//
//   There are three possible representations of optical properties (scalar = optical depth only;
//   two-stream = tau, single-scattering albedo, and asymmetry factor g, and
//   n-stream = tau, ssa, and phase function moments p.) Thus we need nine routines, three for
//   each choice of representation on the left hand side times three representations of the
//   optical properties to be added.
//
//   There are two sets of these nine routines. In the first the two sets of optical
//   properties are defined at the same spectral resolution. There is also a set of routines
//   to add properties defined at lower spectral resolution to a set defined at higher spectral
//   resolution (adding properties defined by band to those defined by g-point)
template <typename Tau1T, typename Tau2T>
void increment_1scalar_by_1scalar(int ncol, int nlay, int ngpt, Tau1T const &tau1, Tau2T const &tau2) {
  using DeviceT = typename Tau1T::device_type;
  using LayoutT = typename Tau1T::array_layout;
  using mdrp_t  = typename conv::MDRP<LayoutT, DeviceT>;

  // do igpt = 1, ngpt
  //   do ilay = 1, nlay
  //     do icol = 1, ncol
  TIMED_KERNEL(FLATTEN_MD_KERNEL3(ncol,nlay,ngpt, icol, ilay, igpt,
    tau1(icol,ilay,igpt) = tau1(icol,ilay,igpt) + tau2(icol,ilay,igpt);
  ));
  //std::cout << "WARNING: THIS ISN'T TESTED: " << __FILE__ << ": " << __LINE__ << "\n";
}

// Incrementing when the second set of optical properties is defined at lower spectral resolution
//   (e.g. by band instead of by gpoint)
template <typename Tau1T, typename Tau2T, typename GptLimsT>
void inc_1scalar_by_1scalar_bybnd(int ncol, int nlay, int ngpt, Tau1T const &tau1, Tau2T const &tau2,
                                  int nbnd, GptLimsT const &gpt_lims) {
  using DeviceT = typename Tau1T::device_type;
  using LayoutT = typename Tau1T::array_layout;
  using mdrp_t  = typename conv::MDRP<LayoutT, DeviceT>;

  // do igpt = 1 , ngpt
  //   do ilay = 1 , nlay
  //     do icol = 1 , ncol
  TIMED_KERNEL(FLATTEN_MD_KERNEL3(ncol,nlay,ngpt, icol, ilay, igpt,
    for (int ibnd=0; ibnd<nbnd; ibnd++) {
      if (igpt >= gpt_lims(0,ibnd) && igpt <= gpt_lims(1,ibnd) ) {
        tau1(icol,ilay,igpt) += tau2(icol,ilay,ibnd);
      }
    }
  ));
}

// Delta-scale
//   f = g*g
template <typename TauT, typename SsaT, typename GT>
void delta_scale_2str_kernel(int ncol, int nlay, int ngpt, TauT const &tau, SsaT const &ssa, GT const &g) {
  using RealT = typename TauT::non_const_value_type;
  using DeviceT = typename TauT::device_type;
  using LayoutT = typename TauT::array_layout;
  using mdrp_t  = typename conv::MDRP<LayoutT, DeviceT>;

  constexpr RealT eps = 3*std::numeric_limits<RealT>::min();

  // do igpt = 1, ngpt
  //   do ilay = 1, nlay
  //     do icol = 1, ncol
  TIMED_KERNEL(FLATTEN_MD_KERNEL3(ncol,nlay,ngpt, icol, ilay, igpt,
    if (tau(icol,ilay,igpt) > eps) {
      RealT f  = g  (icol,ilay,igpt) * g  (icol,ilay,igpt);
      RealT wf = ssa(icol,ilay,igpt) * f;
      tau(icol,ilay,igpt) = (1. - wf) * tau(icol,ilay,igpt);
      ssa(icol,ilay,igpt) = (ssa(icol,ilay,igpt) - wf) / (1.0 - wf);
      g  (icol,ilay,igpt) = (g  (icol,ilay,igpt) -  f) / (1.0 -  f);
    }
  ));
}


// Delta-scaling, provided only for two-stream properties at present
// -------------------------------------------------------------------------------------------------
// Delta-scale
//   user-provided value of f (forward scattering)
template <typename TauT, typename SsaT, typename GT, typename FT>
void delta_scale_2str_kernel(int ncol, int nlay, int ngpt, TauT const &tau, SsaT const &ssa, GT const &g, FT const &f) {
  using RealT = typename TauT::non_const_value_type;
  using DeviceT = typename TauT::device_type;
  using LayoutT = typename TauT::array_layout;
  using mdrp_t  = typename conv::MDRP<LayoutT, DeviceT>;
  constexpr RealT eps = 3*std::numeric_limits<RealT>::min();

  // do igpt = 1, ngpt
  //   do ilay = 1, nlay
  //     do icol = 1, ncol
  TIMED_KERNEL(FLATTEN_MD_KERNEL3(ncol, nlay, ngpt, icol, ilay, igpt,
    if (tau(icol,ilay,igpt) > eps) {
      RealT wf = ssa(icol,ilay,igpt) * f(icol,ilay,igpt);
      tau(icol,ilay,igpt) = (1. - wf) * tau(icol,ilay,igpt);
      ssa(icol,ilay,igpt) = (ssa(icol,ilay,igpt) - wf) /  (1.0 - wf);
      g  (icol,ilay,igpt) = (g  (icol,ilay,igpt) - f(icol,ilay,igpt)) / (1. - f(icol,ilay,igpt));
    }
  ));
  std::cout << "WARNING: THIS ISN'T TESTED: " << __FILE__ << ": " << __LINE__ << "\n";
}

// increment 2stream by 2stream
template <typename Tau1T, typename Ssa1T, typename G1T,
          typename Tau2T, typename Ssa2T, typename G2T>
void increment_2stream_by_2stream(int ncol, int nlay, int ngpt, Tau1T const &tau1, Ssa1T const &ssa1, G1T const &g1,
                                  Tau2T const &tau2, Ssa2T const &ssa2, G2T const &g2) {
  using RealT = typename Tau1T::non_const_value_type;
  using DeviceT = typename Tau1T::device_type;
  using LayoutT = typename Tau1T::array_layout;
  using mdrp_t  = typename conv::MDRP<LayoutT, DeviceT>;
  constexpr RealT eps = 3*std::numeric_limits<RealT>::min();

  // do igpt = 1, ngpt
  //   do ilay = 1, nlay
  //     do icol = 1, ncol
  TIMED_KERNEL(FLATTEN_MD_KERNEL3(ncol,nlay,ngpt, icol, ilay, igpt,
    // t=tau1 + tau2
    RealT tau12 = tau1(icol,ilay,igpt) + tau2(icol,ilay,igpt);
    // w=(tau1*ssa1 + tau2*ssa2) / t
    RealT tauscat12 = tau1(icol,ilay,igpt) * ssa1(icol,ilay,igpt) +
                     tau2(icol,ilay,igpt) * ssa2(icol,ilay,igpt);
    g1(icol,ilay,igpt) = (tau1(icol,ilay,igpt) * ssa1(icol,ilay,igpt) * g1(icol,ilay,igpt) +
                          tau2(icol,ilay,igpt) * ssa2(icol,ilay,igpt) * g2(icol,ilay,igpt))
                           / Kokkos::fmax(eps,tauscat12);
    ssa1(icol,ilay,igpt) = tauscat12 / Kokkos::fmax(eps,tau12);
    tau1(icol,ilay,igpt) = tau12;
  ));
  //std::cout << "WARNING: THIS ISN'T TESTED: " << __FILE__ << ": " << __LINE__ << "\n";
}
