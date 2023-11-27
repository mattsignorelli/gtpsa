cmake_minimum_required(VERSION 3.13.2)
project(mad_tpsa)

add_library(mad_tpsa SHARED
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

set_target_properties(mad_tpsa PROPERTIES 
  LIBRARY_OUTPUT_DIRECTORY ./build/lib 
  PUBLIC_HEADER ./build/include/mad_tpsa.h)

set(BLA_PREFER_PKGCONFIG ON)
# Try to find LAPACK:
find_package(LAPACK)
# lapack
 # HINTS ENV LD_LIBRARY_PATH             # Linux
  #HINTS ENV DYLD_FALLBACK_LIBRARY_PATH  # Mac
  #PATH_SUFFIXES lapack)

if (NOT LAPACK_FOUND)
  message(FATAL_ERROR "LAPACK library not found")
endif()

target_link_libraries(mad_tpsa PUBLIC ${LAPACK_LIBRARIES})

set(CFLAGS "-std=c99 -W -Wall -Wextra -pedantic -ffast-math -ftree-vectorize -O3 \
-ftree-vectorize -fPIC -flto -Waggregate-return -Wcast-align -Wdisabled-optimization \
-Wpointer-arith -Wsign-compare -Wwrite-strings -Wbad-function-cast -Wmissing-prototypes \
-Wnested-externs -Wstrict-prototypes -Wunreachable-code -Wno-vla-parameter -fno-math-errno \
-fno-builtin-cproj -ffast-math -llapack")

#include(GNUInstallDirs)
install(TARGETS mad_tpsa 
  LIBRARY DESTINATION /usr/local/lib 
  PUBLIC_HEADER DESTINATION /usr/local/include)

