cmake_minimum_required(VERSION 3.13.2)
project(gtpsa)

add_library(gtpsa SHARED
  code/mad_bit.c
  code/mad_cst.c
  code/mad_ctpsa.c
  code/mad_ctpsa_comp.c
  code/mad_ctpsa_conv.c
  code/mad_ctpsa_fun.c
  code/mad_ctpsa_io.c
  code/mad_ctpsa_minv.c
  code/mad_ctpsa_mops.c
  code/mad_ctpsa_ops.c
  code/mad_desc.c
  code/mad_erfw.c
  code/mad_log.c
  code/mad_mat.c
  code/mad_mem.c
  code/mad_mono.c
  code/mad_num.c
  code/mad_poly.c
  code/mad_rad.c
  code/mad_str.c
  code/mad_tpsa.c
  code/mad_tpsa_comp.c
  code/mad_tpsa_fun.c
  code/mad_tpsa_io.c
  code/mad_tpsa_minv.c
  code/mad_tpsa_mops.c
  code/mad_tpsa_ops.c
  code/mad_vec.c
)

set_target_properties(gtpsa PROPERTIES 
  PUBLIC_HEADER include/gtpsa.h)

set(CFLAGS "-std=c99 -W -Wall -Wextra -pedantic -ffast-math -ftree-vectorize -O3 -ftree-vectorize -fPIC -flto -Waggregate-return -Wcast-align -Wdisabled-optimization -Wpointer-arith -Wsign-compare -Wwrite-strings -Wbad-function-cast -Wmissing-prototypes -Wnested-externs -Wstrict-prototypes -Wunreachable-code -Wno-vla-parameter -fno-math-errno -fno-builtin-cproj -ffast-math")
# set(CMAKE_C_FLAGS "${CFLAGS}")

include(GNUInstallDirs)
install(TARGETS gtpsa
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})