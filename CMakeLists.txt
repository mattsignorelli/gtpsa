cmake_minimum_required(VERSION 3.13.2)
set (CMAKE_C_COMPILER "gcc") # Prefer GCC when available
project(gtpsa)
set(CFLAGS "-std=c99 -W -Wall -Wextra -pedantic -ffast-math -ftree-vectorize -O3 \
-ftree-vectorize -fPIC -flto -Waggregate-return -Wcast-align -Wdisabled-optimization \
-Wpointer-arith -Wsign-compare -Wwrite-strings -Wbad-function-cast -Wmissing-prototypes \
-Wnested-externs -Wstrict-prototypes -Wunreachable-code -Wno-vla-parameter -fno-math-errno \
-fno-builtin-cproj -ffast-math -llapack")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

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

# Find LAPACK or LIBBLASTRAMPOLINE
set(BLA_PREFER_PKGCONFIG ON)
find_package(LAPACK QUIET)

if(NOT LAPACK_FOUND)
    find_library(LIBBLASTRAMPOLINE NAMES blastrampoline-5)
    if(NOT LIBBLASTRAMPOLINE)
      message(FATAL_ERROR "Unable to find LAPACK")
    endif()
endif()

if(LAPACK_FOUND)
  target_link_libraries(gtpsa PUBLIC ${LAPACK_LIBRARIES})
else()
  target_link_libraries(gtpsa PUBLIC ${LIBBLASTRAMPOLINE})
endif()

install(TARGETS gtpsa LIBRARY DESTINATION lib)
