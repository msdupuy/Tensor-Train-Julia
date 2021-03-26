using TensorTrains

w=2
L=5
h,V = hubbard_2D(w,L;U=4,w_pbc=false,L_pbc=false)
H_tto = hV_to_mpo(h,V,tol=1e-5)
ψ_0 = half_filling(w*L)
E_als,ψ_r = als_eigsolv(H_tto,ψ_0,sweep_schedule=[2,5],rmax_schedule=[16,32])
E_mals,ψ_tt,r = mals_eigsolv(H_tto,ψ_r,tol=1e-6,sweep_schedule=[2,5],rmax_schedule=[32,64])
#Fiedler order
σ_fiedler = fiedler_order(ψ_r)
h_f = h[σ_fiedler,σ_fiedler]
V_f = V[σ_fiedler,σ_fiedler,σ_fiedler,σ_fiedler]
H_tto = hV_to_mpo(h_f,V_f,tol=1e-5)
E_fiedler,ψ_fiedler,r = mals_eigsolv(H_tto,ψ_r,tol=1e-6,sweep_schedule=[3],rmax_schedule=[64])
#BWPO order
σ_bwpo = bwpo_order(ψ_r)
h_b = h[σ_bwpo,σ_bwpo]
V_b = V[σ_bwpo,σ_bwpo,σ_bwpo,σ_bwpo]
H_tto = hV_to_mpo(h_b,V_b,tol=1e-5)
E_bwpo,ψ_bwpo,r = mals_eigsolv(H_tto,ψ_r,tol=1e-6,sweep_schedule=[3],rmax_schedule=[64])