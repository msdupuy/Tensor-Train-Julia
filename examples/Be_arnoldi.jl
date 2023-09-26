"""
Transcorrelated Beryllium using Arnoldi
"""

function Be_arnoldi(Imax;m=3,rmax=64,ε=1e-5,ε_tt=1e-5,σ=15.0,r=2)
  F = read_electron_integral_tensors("./examples/Be.ezfio.FCIDUMP")
  int_1e,int_2e = F[4],F[5]
  v = rand_tt(2*ones(Int64,2F[2]),vcat(1,r*ones(Int64,2F[2]-1),1);normalise=true)
  h,V = one_e_two_e_integrals_to_hV(int_1e,int_2e)
  H_tto = hV_to_mpo(h,V,tol=1e-5)
  E,ψ_tt =eig_arnoldi(H_tto,m,v;Imax=Imax,ε=ε,ε_tt=ε_tt,rmax=rmax,which=:SR,σ=σ) 
  return E,ψ_tt
end