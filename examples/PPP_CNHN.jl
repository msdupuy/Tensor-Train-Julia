using .TensorTrains

"""
Example of the PPP Hamiltonian for N=6
E_mals ≃ -0.46752535023652453
"""

N = 6
H_tto = PPP_C_NH_N(N)
ψ_0 = tt_up_rks(half_filling(N),32;ϵ_wn=1e-2)
ψ_0 = orthogonalize(ψ_0)
E_mals, ψ_tt = mals_eigsolv(H_tto,ψ_0)