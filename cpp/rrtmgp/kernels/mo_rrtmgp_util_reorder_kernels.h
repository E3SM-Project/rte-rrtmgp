
#pragma once

#include "rrtmgp_const.h"
#include "rrtmgp_conversion.h"

template <typename ArrayInT, typename ArrayOutT>
inline void reorder_123x321_kernel(int d1, int d2, int d3, ArrayInT const &array_in, ArrayOutT const &array_out) {

  using DeviceT = typename ArrayInT::device_type;
  using LayoutT = typename ArrayInT::array_layout;
  using mdrp_t  = typename conv::MDRP<LayoutT, DeviceT>;
  // int constexpr TILE_SIZE = 8;
  // int ntiles1 = (d1-1) / TILE_SIZE + 1;
  // int ntiles3 = (d3-1) / TILE_SIZE + 1;

  // for (int i2=1; i2<=d2; i2++) {
  //   for (int t3=1; t3<=ntiles3; t3++) {
  //     for (int t1=1; t1<=ntiles1; t1++) {
  //       for (int it3=1; it3<=TILE_SIZE; it3++) {
  //         for (int it1=1; it1<=TILE_SIZE; it1++) {
  TIMED_KERNEL(Kokkos::parallel_for( mdrp_t::template getrl<3>({d1,d2,d3}) , KOKKOS_LAMBDA (int i1, int i2, int i3) {
    array_out(i3,i2,i1) = array_in(i1,i2,i3);
  }));
}

template <typename ArrayInT, typename ArrayOutT>
inline void reorder_123x312_kernel(int d1, int d2, int d3, ArrayInT const &array_in, ArrayOutT const &array_out) {
  using DeviceT = typename ArrayInT::device_type;
  using LayoutT = typename ArrayInT::array_layout;
  using mdrp_t  = typename conv::MDRP<LayoutT, DeviceT>;

  // do i3 = 1 , d3
  //   do i2 = 1 , d2
  //     do i1 = 1 , d1
  TIMED_KERNEL(FLATTEN_MD_KERNEL3(d3,d2,d1, i3, i2, i1,
    array_out(i3,i1,i2) = array_in(i1,i2,i3);
  ));
}
