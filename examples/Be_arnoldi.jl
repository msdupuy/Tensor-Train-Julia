"""
Transcorrelated Beryllium using Arnoldi
"""

function Be_arnoldi(Imax;m=3,rmax=64,ε=1e-5,ε_tt=1e-5,σ=15.0,r=2)
  F = read_electron_integral_tensors("./examples/Be.ezfio.FCIDUMP")
  int_1e,int_2e = F[4],F[5]
  v = slater(F[3],2F[2])
  h,V = one_e_two_e_integrals_to_hV(int_1e,int_2e)
  H_tto = hV_to_mpo(h,V,ntuple(x->2,2F[2]),tol=1e-5)
  E,ψ_tt =eig_arnoldi(H_tto,m,v;Imax=Imax,ε=ε,ε_tt=ε_tt,rmax=rmax,which=:SR,σ=σ) 
  return E,ψ_tt
end