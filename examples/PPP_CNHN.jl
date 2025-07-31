using TensorTrains

"""
Example of the PPP Hamiltonian for N=6
E_dmrg ≃ -0.46752535023652453
"""

N = 6
H_tto = PPP_C_NH_N(N)
ψ_0 = tt_up_rks(half_filling(N),32;ϵ_wn=1e-1)
ψ_0 = orthogonalize(ψ_0)
schedule = dmrg_schedule_default(;N=2,rmax=64,nsweeps=2,it_solver=true)
E_dmrg, ψ_tt, dmrg_info = dmrg_eigsolv(H_tto,ψ_0;schedule)