using TensorTrains

function hubbard_2D_dmrg(w,L)
  h,V = hubbard_2D(w,L;U=1)
  H_tto = hV_to_mpo(h,V,tol=1e-5)
  ψ_0 = tt_up_rks(half_filling(w*L),16;ϵ_wn = 1e-1)
  E,ψ_tt,r_dmrg = dmrg_eigsolv(H_tto,ψ_0,sweep_schedule=[2,4,7],rmax_schedule=[16,32,64])
  return  E,ψ_tt,r_dmrg
end