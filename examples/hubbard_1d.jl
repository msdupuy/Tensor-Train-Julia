using TensorTrains

function hubbard_1D_dmrg(L;rmax=64,U=1.0,pbc=false)
  h,V = hubbard_1D(L;U,pbc)
  dims = ntuple(x->2,2L)
  H_tto = hV_to_mpo(Matrix(h),V,dims,tol=1e-5)
  ψ_0 = tt_up_rks(half_filling(L),16;ϵ_wn = 1e-1)
  schedule = dmrg_schedule(2,ceil(Int64,rmax/4),rmax,3;N=2)
  E,ψ_tt,dmrg_info = dmrg_eigsolv(H_tto,ψ_0;schedule)
  return  E,ψ_tt,dmrg_info,H_tto
end