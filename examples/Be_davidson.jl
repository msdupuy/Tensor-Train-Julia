"""
Transcorrelated Beryllium using Arnoldi
valeur Giner : -14.6251920
"""

using TensorTrains
using JSON3
using LinearAlgebra
using Dates

function Be_davidson(Imax;m=3,rmax=64,ε=1e-5,ε_tt=1e-5,σ=16.0)
  F = read_electron_integral_tensors_nosymmetry("./examples/Be.ezfio.FCIDUMP")
  int_1e,int_2e = F[4],F[5]
  v = slater(F[3],2F[2])
  h,V = one_e_two_e_integrals_to_hV(int_1e,int_2e)
  H_tto = TensorTrains.hV_no_to_mpo(h,V,ntuple(x->2,2F[2]),tol=1e-10) #,n=F[3]
  prec = one_body_diagonal(diag(h),σ=σ)
  λ,ψ,res = davidson(H_tto,m,v;Imax=Imax,ε=ε,ε_tt=ε_tt,rmax=rmax,which=:SR,σ=σ,prec=prec)
  return λ,ψ,res
end