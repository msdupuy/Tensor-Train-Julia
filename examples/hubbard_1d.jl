using TensorTrains

L=10
h,V = hubbard_1D(L)
H_tto = hV_to_mpo(h,V,tol=1e-5)
ψ_0 = half_filling(L)
E,ψ_tt,r_dmrg = dmrg_eigsolv(H_tto,ψ_0,sweep_schedule=[2,4,7],rmax_schedule=[16,32,64])