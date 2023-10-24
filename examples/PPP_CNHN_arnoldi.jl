using TensorTrains

"""
Example of the PPP Hamiltonian for N=6
E_dmrg ≃ -0.46752535023652453
"""

function PPP_CNHN_arnoldi(N;m=3,ε=1e-6,ε_tt=1e-4,rmax=256,Imax=50,σ=0.0)
  H_tto = PPP_C_NH_N(N)
  ψ_0 = rand_tt(H_tto.tto_dims,ones(Int64,H_tto.N+1);normalise=true)
  E_dmrg, ψ_tt,hist = eig_arnoldi(H_tto,m,ψ_0;ε=ε,ε_tt=ε_tt,rmax=rmax,Imax=Imax,which=:SM,σ=σ)
  return E_dmrg,ψ_tt
end
