
#pragma once

#include "rrtmgp_const.h"
#include "rrtmgp_conversion.h"
#include "mo_optical_props_kernels.h"

// Base class for optical properties
//   Describes the spectral discretization including the wavenumber limits
//   of each band (spectral region) and the mapping between g-points and bands
template <typename RealT=double, typename LayoutT=Kokkos::LayoutLeft, typename DeviceT=DefaultDevice>
class OpticalPropsK {
public:

  template <typename T>
  using view_t = typename Kokkos::View<T, LayoutT, DeviceT>;

  using self_t = OpticalPropsK<RealT, LayoutT, DeviceT>;

  using mdrp_t = typename conv::MDRP<LayoutT, DeviceT>;

  view_t<int**> band2gpt;       // (begin g-point, end g-point) = band2gpt(2,band)
  view_t<int*>  gpt2band;       // band = gpt2band(g-point)
  int     ngpt;
  view_t<RealT**> band_lims_wvn;  // (upper and lower wavenumber by band) = band_lims_wvn(2,band)
  std::string name;

  template <typename BandLimsT>
  static inline void init_band_lims(const BandLimsT& band_lims)
  {
    // Assume that values are defined by band, one g-point per band
    TIMED_KERNEL(Kokkos::parallel_for( band_lims.extent(1), KOKKOS_LAMBDA (size_t iband) {
      band_lims(1,iband) = iband;
      band_lims(0,iband) = iband;
    }));
  }

  template <typename BandLimsT, typename Gpt2BandT>
  static inline void set_gpt2band(const BandLimsT& band_lims, const Gpt2BandT& gpt2band)
  {
    // TODO: I didn't want to bother with race conditions at the moment, so it's an entirely serialized kernel for now
    TIMED_KERNEL(Kokkos::parallel_for(1, KOKKOS_LAMBDA(int dummy) {
      for (int iband=0; iband < band_lims.extent(1); iband++) {
        for (int i=band_lims(0,iband); i <= band_lims(1,iband); i++) {
          gpt2band(i) = iband;
        }
      }
    }));
  }

  template <typename BandLimsWvnT, typename BandLimsGptT=view_t<int**>,
            typename std::enable_if<conv::is_view_v<BandLimsWvnT>>::type* = nullptr >
  void init( BandLimsWvnT const &band_lims_wvn , BandLimsGptT const &band_lims_gpt=view_t<int**>() , std::string name="" ) {
    view_t<int**> band_lims_gpt_lcl("band_lims_gpt_lcl", 2, band_lims_wvn.extent(1)); // ALLOC!
    if (band_lims_wvn.extent(0) != 2) { stoprun("optical_props::init(): band_lims_wvn 1st dim should be 2"); }
    #ifdef RRTMGP_EXPENSIVE_CHECKS
    if (conv::any(band_lims_wvn, conv::LTFunc<RealT>(0.))) { stoprun("optical_props::init(): band_lims_wvn has values <  0."); }
    #endif
    if (band_lims_gpt.is_allocated()) {
      if (band_lims_gpt.extent(1) != band_lims_wvn.extent(1)) {
        stoprun("optical_props::init(): band_lims_gpt size inconsistent with band_lims_wvn");
      }
      #ifdef RRTMGP_EXPENSIVE_CHECKS
      if (conv::any(band_lims_gpt, conv::LTFunc<int>(1)) ) { stoprun("optical_props::init(): band_lims_gpt has values < 1"); }
      #endif
      Kokkos::deep_copy(band_lims_gpt_lcl, band_lims_gpt);
    } else {
      init_band_lims(band_lims_gpt_lcl);
    }
    // Assignment
    this->band2gpt       = band_lims_gpt_lcl;
    this->band_lims_wvn  = band_lims_wvn;
    this->name           = name;
    this->ngpt           = conv::maxval(this->band2gpt) + 1;

    // Make a map between g-points and bands
    //   Efficient only when g-point indexes start at 1 and are contiguous.
    this->gpt2band = view_t<int*>("gpt2band", this->ngpt); // ALLOC
    set_gpt2band(band_lims_gpt_lcl, this->gpt2band);
  }

  // This function does the same thing as the one above, except takes Views as arguments
  // instead of allocating new ones. Presumably, these views would come from the pool
  // allocator in order to avoid cudaMalloc (hurts performance).
  template <typename BandLimsWvnT, typename Band2Gpt, typename Gpt2Band>
  void init_no_alloc( BandLimsWvnT const &band_lims_wvn ,
                      Band2Gpt const& band2gpt_mem,
                      Gpt2Band const& gpt2band_mem,
                      std::string name="" ) {
    if (band_lims_wvn.extent(0) != 2) { stoprun("optical_props::init(): band_lims_wvn 1st dim should be 2"); }
    #ifdef RRTMGP_EXPENSIVE_CHECKS
    if (conv::any(band_lims_wvn, conv::LTFunc<RealT>(0.))) { stoprun("optical_props::init(): band_lims_wvn has values <  0."); }
    #endif
    init_band_lims(band2gpt_mem);

    // Assignment
    this->band2gpt       = band2gpt_mem;
    this->band_lims_wvn  = band_lims_wvn;
    this->name           = name;
    this->ngpt           = this->band2gpt.extent(1);
    assert(this->ngpt == conv::maxval(this->band2gpt) + 1);
    assert(gpt2band_mem.extent(0) == this->ngpt);

    // Make a map between g-points and bands
    //   Efficient only when g-point indexes start at 1 and are contiguous.
    this->gpt2band = gpt2band_mem;
    set_gpt2band(this->band2gpt, this->gpt2band);
  }

  template <typename BandLimsWvnT, typename BandLimsGptT, typename Band2Gpt, typename Gpt2Band>
  void init_no_alloc( BandLimsWvnT const &band_lims_wvn,
                      BandLimsGptT const &band_lims_gpt,
                      Band2Gpt const& band2gpt_mem,
                      Gpt2Band const& gpt2band_mem,
                      std::string name="" ) {
    if (band_lims_wvn.extent(0) != 2) { stoprun("optical_props::init(): band_lims_wvn 1st dim should be 2"); }
    #ifdef RRTMGP_EXPENSIVE_CHECKS
    if (conv::any(band_lims_wvn, conv::LTFunc<RealT>(0.))) { stoprun("optical_props::init(): band_lims_wvn has values <  0."); }
    #endif
    if (band_lims_gpt.is_allocated()) {
      if (band_lims_gpt.extent(1) != band_lims_wvn.extent(1)) {
        stoprun("optical_props::init(): band_lims_gpt size inconsistent with band_lims_wvn");
      }
      #ifdef RRTMGP_EXPENSIVE_CHECKS
      if (conv::any(band_lims_gpt, conv::LTFunc<int>(1)) ) { stoprun("optical_props::init(): band_lims_gpt has values < 1"); }
      #endif
      Kokkos::deep_copy(band2gpt_mem, band_lims_gpt);
    } else {
      init_band_lims(band2gpt_mem);
    }
    // Assignment
    this->band2gpt       = band2gpt_mem;
    this->band_lims_wvn  = band_lims_wvn;
    this->name           = name;
    this->ngpt           = conv::maxval(this->band2gpt) + 1;
    assert(this->ngpt == gpt2band_mem.extent(0));

    // Make a map between g-points and bands
    //   Efficient only when g-point indexes start at 1 and are contiguous.
    this->gpt2band = gpt2band_mem;
    set_gpt2band(band2gpt_mem, this->gpt2band);
  }

  void init(self_t const &in) {
    if ( ! in.is_initialized() ) {
      stoprun("optical_props::init(): can't initialize based on un-initialized input");
    } else {
      this->init( in.get_band_lims_wavenumber() , in.get_band_lims_gpoint() );
    }
  }

  bool is_initialized() const { return this->band2gpt.is_allocated(); }

  // Base class: finalize (deallocate memory)
  void finalize() {
    this->band2gpt      = view_t<int**>();
    this->gpt2band      = view_t<int*>();
    this->band_lims_wvn = view_t<RealT**>();
    this->name          = "";
  }

  // Number of bands
  int get_nband() const {
    if (this->is_initialized()) { return this->band2gpt.extent(1); }
    return 0;
  }


  // Number of g-points
  int get_ngpt() const {
    if (this->is_initialized()) { return this->ngpt; }
    return 0;
  }

  // Bands for all the g-points at once;  dimension (ngpt)
  const view_t<int*>& get_gpoint_bands() const { return gpt2band; }

  // The first and last g-point of all bands at once;  dimension (2, nbands)
  const view_t<int**>& get_band_lims_gpoint() const { return this->band2gpt; }

  // Lower and upper wavenumber of all bands
  // (upper and lower wavenumber by band) = band_lims_wvn(2,band)
  const view_t<RealT**>& get_band_lims_wavenumber() const { return this->band_lims_wvn; }

  // Lower and upper wavelength of all bands
  const view_t<RealT**>& get_band_lims_wavelength() const {
    view_t<RealT**> ret("band_lim_wavelength", band_lims_wvn.extent(0), band_lims_wvn.extent(1)); // ALLOC
    // for (int j = 1; j <= size(band_lims_wvn,2); j++) {
    //   for (int i = 1; i <= size(band_lims_wvn,1); i++) {
    auto this_band_lims_wvn = this->band_lims_wvn;
    if (this->is_initialized()) {
      TIMED_KERNEL(FLATTEN_MD_KERNEL2(band_lims_wvn.extent(0) , band_lims_wvn.extent(1), i, j,
        ret(i,j) = 1. / this_band_lims_wvn(i,j);
      ));
    } else {
      TIMED_KERNEL(FLATTEN_MD_KERNEL2(band_lims_wvn.extent(0) , band_lims_wvn.extent(1), i, j,
        ret(i,j) = 0.;
      ));
    }
    return ret;
  }

  template <typename WavelengthBounds>
  void get_band_lims_wavelength(WavelengthBounds const& ret) const {
    auto this_band_lims_wvn = this->band_lims_wvn;
    if (this->is_initialized()) {
      TIMED_KERNEL(FLATTEN_MD_KERNEL2(band_lims_wvn.extent(1) , band_lims_wvn.extent(0), j, i,
        ret(i,j) = 1. / this_band_lims_wvn(i,j);
      ));
    } else {
      TIMED_KERNEL(FLATTEN_MD_KERNEL2(band_lims_wvn.extent(1) , band_lims_wvn.extent(0), j, i,
        ret(i,j) = 0.;
      ));
    }
  }

  // Are the bands of two objects the same? (same number, same wavelength limits)
  bool bands_are_equal(self_t const &rhs) const {
    // This is working around an issue that arises in E3SM's rrtmgpxx integration.
    // Previously the code failed in the creation of the ScalarLiveOut variable, but only for higher optimizations
    bool ret = true;
    auto this_band_lims_wvn = Kokkos::create_mirror_view_and_copy(HostDevice(), this->band_lims_wvn);
    auto rhs_band_lims_wvn  = Kokkos::create_mirror_view_and_copy(HostDevice(), rhs.band_lims_wvn);
    for (int j=0 ; j < this->band_lims_wvn.extent(1); j++) {
      for (int i=0 ; i < this->band_lims_wvn.extent(0); i++) {
        if ( std::abs( this_band_lims_wvn(i,j) - rhs_band_lims_wvn(i,j) ) > 5*conv::epsilon(this_band_lims_wvn) ) {
          ret = false;
        }
      }
    }
    return ret;
  }

  // Is the g-point structure of two objects the same?
  //   (same bands, same number of g-points, same mapping between bands and g-points)
  bool gpoints_are_equal(self_t const &rhs) const {
    if ( ! this->bands_are_equal(rhs) || this->get_ngpt() != rhs.get_ngpt() ) { return false; }

    // This is working around an issue that arises in E3SM's rrtmgpxx integration.
    // Previously the code failed in the creation of the ScalarLiveOut variable, but only for higher optimizations
    bool ret = true;
    auto this_gpt2band = Kokkos::create_mirror_view_and_copy(HostDevice(), this->gpt2band);
    auto rhs_gpt2band  = Kokkos::create_mirror_view_and_copy(HostDevice(), rhs.gpt2band);
    for (int i=0; i < this->gpt2band.extent(0); i++) {
      if ( this_gpt2band(i) != rhs_gpt2band(i) ) { ret = false; }
    }
    return ret;
  }

  void set_name( std::string name ) { this->name = name; }

  std::string get_name() const { return this->name; }

};

template <typename RealT=double, typename LayoutT=Kokkos::LayoutLeft, typename DeviceT=DefaultDevice>
class OpticalPropsArryK : public OpticalPropsK<RealT, LayoutT, DeviceT> {
public:
  using parent_t = OpticalPropsK<RealT, LayoutT, DeviceT>;
  template <typename T>
  using view_t = typename parent_t::template view_t<T>;

  view_t<RealT***> tau; // optical depth (ncol, nlay, ngpt)

  int get_ncol() const { if (tau.is_allocated()) { return this->tau.extent(0); } else { return 0; } }
  int get_nlay() const { if (tau.is_allocated()) { return this->tau.extent(1); } else { return 0; } }

};

// We need to know about 2str's existence because it is referenced in 1scl
template <typename RealT, typename LayoutT, typename DeviceT>
class OpticalProps2strK;

template <typename RealT=double, typename LayoutT=Kokkos::LayoutLeft, typename DeviceT=DefaultDevice>
class OpticalProps1sclK : public OpticalPropsArryK<RealT, LayoutT, DeviceT> {
public:

  using parent_t = OpticalPropsArryK<RealT, LayoutT, DeviceT>;
  template <typename T>
  using view_t = typename parent_t::template view_t<T>;

  void validate() const {
    if (!this->tau.is_allocated()) { stoprun("validate: tau not allocated/initialized"); }
    #ifdef RRTMGP_EXPENSIVE_CHECKS
    if (conv::any(this->tau, conv::LTFunc<RealT>(0.))) { stoprun("validate: tau values out of range"); }
    #endif
  }

  template <typename DummyT>
  void delta_scale(DummyT const &dummy) const { }

  void alloc_1scl(int ncol, int nlay) {
    if (! this->is_initialized()) { stoprun("OpticalProps1scl::alloc_1scl: spectral discretization hasn't been provided"); }
    if (ncol <= 0 || nlay <= 0) { stoprun("OpticalProps1scl::alloc_1scl: must provide > 0 extents for ncol, nlay"); }
    this->tau = view_t<RealT***>("tau",ncol,nlay,this->get_ngpt()); // ALLOC
  }

  template <typename TauMem>
  void alloc_1scl_no_alloc(int ncol, int nlay, TauMem const& tau_mem) {
    if (! this->is_initialized()) { stoprun("OpticalProps1scl::alloc_1scl: spectral discretization hasn't been provided"); }
    if (ncol <= 0 || nlay <= 0) { stoprun("OpticalProps1scl::alloc_1scl: must provide > 0 extents for ncol, nlay"); }
    assert(tau_mem.extent(0) == ncol);
    assert(tau_mem.extent(1) == nlay);
    assert(tau_mem.extent(2) == this->get_ngpt());
    this->tau = tau_mem;
  }

  // Initialization by specifying band limits and possibly g-point/band mapping
  template <typename BandLimsWvnT, typename BandLimsGptT=view_t<int**>,
            typename std::enable_if<conv::is_view_v<BandLimsWvnT>>::type* = nullptr >
  void alloc_1scl(int ncol, int nlay, BandLimsWvnT const &band_lims_wvn, BandLimsGptT const &band_lims_gpt=view_t<int**>(), std::string name="") {
    this->init(band_lims_wvn, band_lims_gpt, name);
    this->alloc_1scl(ncol, nlay);
  }

  void alloc_1scl(int ncol, int nlay, OpticalPropsK<RealT, LayoutT, DeviceT> const &opIn, std::string name="") {
    if (this->is_initialized()) { this->finalize(); }
    this->init(opIn.get_band_lims_wavenumber(), opIn.get_band_lims_gpoint(), name);
    this->alloc_1scl(ncol, nlay);
  }

  template <typename Band2Gpt, typename Gpt2Band, typename TauMem>
  void alloc_1scl_no_alloc(int ncol, int nlay, OpticalPropsK<RealT, LayoutT, DeviceT> const &opIn, Band2Gpt const& band2gpt_mem, Gpt2Band const& gpt2band_mem, TauMem const& tau_mem, std::string name="") {
    if (this->is_initialized()) { this->finalize(); }
    this->init_no_alloc(opIn.get_band_lims_wavenumber(), opIn.get_band_lims_gpoint(), band2gpt_mem, gpt2band_mem, name);
    this->alloc_1scl_no_alloc(ncol, nlay, tau_mem);
  }

  void increment(OpticalProps1sclK<RealT, LayoutT, DeviceT> &that) {
    if (! this->bands_are_equal(that)) { stoprun("OpticalProps::increment: optical properties objects have different band structures"); }
    int ncol = that.get_ncol();
    int nlay = that.get_nlay();
    int ngpt = that.get_ngpt();
    if (this->gpoints_are_equal(that)) {
      increment_1scalar_by_1scalar(ncol, nlay, ngpt, that.tau, this->tau);
    } else {
      if (this->get_ngpt() != that.get_nband()) {
        stoprun("OpticalProps::increment: optical properties objects have incompatible g-point structures");
      }
      inc_1scalar_by_1scalar_bybnd(ncol, nlay, ngpt, that.tau, this->tau, that.get_nband(), that.get_band_lims_gpoint());
    }
  }

  // Implemented later because OpticalProps2str hasn't been created yet
  inline void increment(OpticalProps2strK<RealT, LayoutT, DeviceT> &that);

  void print_norms(const bool print_prefix=false) const {
    std::string prefix = print_prefix ? "JGFK" : "";
                                        std::cout << prefix << "name         : " << this->name               << "\n";
    if (this->band2gpt.is_allocated()     ) { std::cout << prefix << "band2gpt     : " << conv::sum(this->band2gpt     ) << "\n"; }
    if (this->gpt2band.is_allocated()     ) { std::cout << prefix << "gpt2band     : " << conv::sum(this->gpt2band     ) << "\n"; }
    if (this->band_lims_wvn.is_allocated()) { std::cout << prefix << "band_lims_wvn: " << conv::sum(this->band_lims_wvn) << "\n"; }
    if (this->tau.is_allocated()          ) { std::cout << prefix << "tau          : " << conv::sum(this->tau          ) << "\n"; }
  }
};

template <typename RealT=double, typename LayoutT=Kokkos::LayoutLeft, typename DeviceT=DefaultDevice>
class OpticalProps2strK : public OpticalPropsArryK<RealT, LayoutT, DeviceT> {
 public:

  using parent_t = OpticalPropsArryK<RealT, LayoutT, DeviceT>;
  template <typename T>
  using view_t = typename parent_t::template view_t<T>;

  view_t<RealT***> ssa; // single-scattering albedo (ncol, nlay, ngpt)
  view_t<RealT***> g;   // asymmetry parameter (ncol, nlay, ngpt)

  void validate() const {
    if ( ! this->tau.is_allocated() || ! this->ssa.is_allocated() || ! this->g.is_allocated() ) {
      stoprun("validate: arrays not allocated/initialized");
    }
    int d1 = this->tau.extent(0);
    int d2 = this->tau.extent(1);
    int d3 = this->tau.extent(2);
    if ( d1 != this->ssa.extent(0) || d2 != this->ssa.extent(1) || d3 != this->ssa.extent(2) ||
         d1 != this->g.extent(0) || d2 != this->g.extent(1) || d3 != this->g.extent(2) ) {
      stoprun("validate: arrays not sized consistently");
    }
    #ifdef RRTMGP_EXPENSIVE_CHECKS
      if (any(this->tau <  0)                      ) { stoprun("validate: tau values out of range"); }
      if (any(this->ssa <  0) || any(this->ssa > 1)) { stoprun("validate: ssa values out of range"); }
      if (any(this->g   < -1) || any(this->g   > 1)) { stoprun("validate: g   values out of range"); }
    #endif
  }

  template <typename ForwardT=view_t<RealT***> >
  void delta_scale(ForwardT const &forward=view_t<RealT***>()) {

    // Forward scattering fraction; g**2 if not provided
    int ncol = this->get_ncol();
    int nlay = this->get_nlay();
    int ngpt = this->get_ngpt();
    if (forward.is_allocated()) {
      if (forward.extent(0) != ncol || forward.extent(1) != nlay || forward.extent(2) != ngpt) {
        stoprun("delta_scale: dimension of 'forward' don't match optical properties arrays");
      }
      #ifdef RRTMGP_EXPENSIVE_CHECKS
        if (any(forward < 0) || any(forward > 1)) { stoprun("delta_scale: values of 'forward' out of bounds [0,1]"); }
      #endif
      delta_scale_2str_kernel(ncol, nlay, ngpt, this->tau, this->ssa, this->g, forward);
    } else {
      delta_scale_2str_kernel(ncol, nlay, ngpt, this->tau, this->ssa, this->g);
    }
  }

  void alloc_2str(int ncol, int nlay) {
    if (! this->is_initialized()) { stoprun("optical_props::alloc: spectral discretization hasn't been provided"); }
    if (ncol <= 0 || nlay <= 0) { stoprun("optical_props::alloc: must provide positive extents for ncol, nlay"); }
    this->tau = view_t<RealT***>("tau",ncol,nlay,this->get_ngpt()); // ALLOC
    this->ssa = view_t<RealT***>("ssa",ncol,nlay,this->get_ngpt()); // ALLOC
    this->g   = view_t<RealT***>("g  ",ncol,nlay,this->get_ngpt()); // ALLOC
  }

  template <typename TauMem, typename SsaMem, typename GMem>
  void alloc_2str_no_alloc(int ncol, int nlay, TauMem const& tau_mem, SsaMem const& ssa_mem, GMem const& g_mem) {
    if (! this->is_initialized()) { stoprun("optical_props::alloc: spectral discretization hasn't been provided"); }
    if (ncol <= 0 || nlay <= 0) { stoprun("optical_props::alloc: must provide positive extents for ncol, nlay"); }

    assert(tau_mem.extent(0) == ncol);
    assert(tau_mem.extent(1) == nlay);
    assert(tau_mem.extent(2) == this->get_ngpt());
    assert(ssa_mem.extent(0) == ncol);
    assert(ssa_mem.extent(1) == nlay);
    assert(ssa_mem.extent(2) == this->get_ngpt());
    assert(g_mem.extent(0) == ncol);
    assert(g_mem.extent(1) == nlay);
    assert(g_mem.extent(2) == this->get_ngpt());

    this->tau = tau_mem;
    this->ssa = ssa_mem;
    this->g   = g_mem;
  }

  template <typename BandLimsWvnT, typename BandLimsGptT=view_t<int**>,
            typename std::enable_if<conv::is_view_v<BandLimsWvnT>>::type* = nullptr>
  void alloc_2str(int ncol, int nlay, BandLimsWvnT const &band_lims_wvn, BandLimsGptT const &band_lims_gpt=view_t<int**>(), std::string name="") {
    this->init(band_lims_wvn, band_lims_gpt, name);
    this->alloc_2str(ncol, nlay);
  }

  void alloc_2str(int ncol, int nlay, OpticalPropsK<RealT, LayoutT, DeviceT> const &opIn, std::string name="") {
    if (this->is_initialized()) { this->finalize(); }
    this->init(opIn.get_band_lims_wavenumber(), opIn.get_band_lims_gpoint(), name);
    this->alloc_2str(ncol, nlay);
  }

  template <typename Band2Gpt, typename Gpt2Band, typename TauMem, typename SsaMem, typename GMem>
  void alloc_2str_no_alloc(int ncol, int nlay, OpticalPropsK<RealT, LayoutT, DeviceT> const &opIn,
                           Band2Gpt const& band2gpt_mem, Gpt2Band const& gpt2band_mem,
                           TauMem const& tau_mem, SsaMem const& ssa_mem, GMem const& g_mem,
                           std::string name="") {
    if (this->is_initialized()) { this->finalize(); }
    this->init_no_alloc(opIn.get_band_lims_wavenumber(), opIn.get_band_lims_gpoint(), band2gpt_mem, gpt2band_mem, name);
    this->alloc_2str_no_alloc(ncol, nlay, tau_mem, ssa_mem, g_mem);
  }

  void increment(OpticalProps1sclK<RealT, LayoutT, DeviceT> &that) {
    if (! this->bands_are_equal(that)) { stoprun("OpticalProps::increment: optical properties objects have different band structures"); }
    int ncol = that.get_ncol();
    int nlay = that.get_nlay();
    int ngpt = that.get_ngpt();
    if (this->gpoints_are_equal(that)) {
      increment_1scalar_by_2stream(ncol, nlay, ngpt, that.tau, this->tau, this->ssa);
    } else {
      if (this->get_ngpt() != that.get_nband()) {
        stoprun("OpticalProps::increment: optical properties objects have incompatible g-point structures");
      }
      inc_1scalar_by_2stream_bybnd(ncol, nlay, ngpt, that.tau, this->tau, this->ssa, that.get_nband(), that.get_band_lims_gpoint());
    }
  }


  void increment(OpticalProps2strK<RealT, LayoutT, DeviceT> &that) {
    if (! this->bands_are_equal(that)) { stoprun("OpticalProps::increment: optical properties objects have different band structures"); }
    int ncol = that.get_ncol();
    int nlay = that.get_nlay();
    int ngpt = that.get_ngpt();
    if (this->gpoints_are_equal(that)) {
      increment_2stream_by_2stream(ncol, nlay, ngpt, that.tau, that.ssa, that.g, this->tau, this->ssa, this->g);
    } else {
      if (this->get_ngpt() != that.get_nband()) {
        stoprun("OpticalProps::increment: optical properties objects have incompatible g-point structures");
      }
      inc_2stream_by_2stream_bybnd(ncol, nlay, ngpt, that.tau, that.ssa, that.g, this->tau, this->ssa, this->g, that.get_nband(), that.get_band_lims_gpoint());
    }
  }


  void print_norms(const bool print_prefix=false) const {
    std::string prefix = print_prefix ? "JGFK" : "";
                                        std::cout << prefix << "name         : " << this->name                     << "\n";
    if (this->band2gpt.is_allocated()     ) { std::cout << prefix << "band2gpt     : " << conv::sum(this->band2gpt     ) << "\n"; }
    if (this->gpt2band.is_allocated()     ) { std::cout << prefix << "gpt2band     : " << conv::sum(this->gpt2band     ) << "\n"; }
    if (this->band_lims_wvn.is_allocated()) { std::cout << prefix << "band_lims_wvn: " << conv::sum(this->band_lims_wvn) << "\n"; }
    if (this->tau.is_allocated()          ) { std::cout << prefix << "tau          : " << conv::sum(this->tau          ) << "\n"; }
    if (this->ssa.is_allocated()          ) { std::cout << prefix << "ssa          : " << conv::sum(this->ssa          ) << "\n"; }
    if (this->g.is_allocated()            ) { std::cout << prefix << "g            : " << conv::sum(this->g            ) << "\n"; }
  }

};

template <typename RealT, typename LayoutT, typename DeviceT>
inline void OpticalProps1sclK<RealT, LayoutT, DeviceT>::increment(OpticalProps2strK<RealT, LayoutT, DeviceT> &that) {
  if (! this->bands_are_equal(that)) { stoprun("OpticalProps::increment: optical properties objects have different band structures"); }
  int ncol = that.get_ncol();
  int nlay = that.get_nlay();
  int ngpt = that.get_ngpt();
  if (this->gpoints_are_equal(that)) {
    increment_2stream_by_1scalar(ncol, nlay, ngpt, that.tau, that.ssa, this->tau);
  } else {
    if (this->get_ngpt() != that.get_nband()) {
      stoprun("OpticalProps::increment: optical properties objects have incompatible g-point structures");
    }
    inc_2stream_by_1scalar_bybnd(ncol, nlay, ngpt, that.tau, that.ssa, this->tau, that.get_nband(), that.get_band_lims_gpoint());
  }
}
