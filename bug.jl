using GTPSA
import GTPSA: Desc, RTPSA, CTPSA
using Printf

const NV = Cint(4);
const MO = UInt8(5);
const MAD_TPSA = GTPSA.MAD_TPSA

const d = @ccall MAD_TPSA.mad_desc_newv(NV::Cint, MO::Cint)::Ptr{Desc}

x = @ccall MAD_TPSA.mad_tpsa_newd(d::Ptr{Desc}, 255::UInt8)::Ptr{RTPSA}
y = @ccall MAD_TPSA.mad_tpsa_newd(d::Ptr{Desc}, 255::UInt8)::Ptr{RTPSA}
z = @ccall MAD_TPSA.mad_tpsa_newd(d::Ptr{Desc}, 255::UInt8)::Ptr{RTPSA}

m = UInt8[0x5]

@ccall MAD_TPSA.mad_tpsa_seti(x::Ptr{RTPSA}, 0::Cint, 0.::Cdouble, 1.::Cdouble)::Cvoid;
@ccall MAD_TPSA.mad_tpsa_setm(y::Ptr{RTPSA}, 1::Cint, m::Ptr{UInt8}, 0.::Cdouble, 1.::Cdouble)::Cvoid;
@ccall MAD_TPSA.mad_tpsa_add(x::Ptr{RTPSA}, y::Ptr{RTPSA}, z::Ptr{RTPSA})::Cvoid;

@ccall MAD_TPSA.mad_tpsa_print(z::Ptr{RTPSA}, ""::Cstring, 0.::Cdouble, 0::Cint, C_NULL::Ptr{Cvoid})::Cvoid
  
@ccall MAD_TPSA.mad_tpsa_del(x::Ptr{RTPSA})::Cvoid;
@ccall MAD_TPSA.mad_tpsa_del(y::Ptr{RTPSA})::Cvoid;
@ccall MAD_TPSA.mad_tpsa_del(z::Ptr{RTPSA})::Cvoid;
@ccall MAD_TPSA.mad_desc_del(C_NULL::Ptr{Cvoid})::Cvoid;
