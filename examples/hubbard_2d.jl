using TensorTrains

w=2
L=5
h,V = hubbard_2D(w,L;U=4)
H_tto = hV_to_mpo(h,V,tol=1e-5)
ψ_0 = half_filling(w*L)
E_als,ψ_tt = als_eigsolv(H_tto,ψ_0,sweep_schedule=[2,5],rmax_schedule=[16,32])
E_mals,ψ_tt,r = mals_eigsolv(H_tto,ψ_tt,tol=1e-6,sweep_schedule=[2,5],rmax_schedule=[32,48])