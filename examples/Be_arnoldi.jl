"""
Transcorrelated Beryllium using Arnoldi
valeur Giner : -14.6251920
"""
using JSON3

function Be_arnoldi(Imax;m=3,rmax=64,ε=1e-5,ε_tt=1e-5,σ=20.0)
  F = read_electron_integral_tensors("./examples/Be.ezfio.FCIDUMP")
  int_1e,int_2e = F[4],F[5]
  v = slater(F[3],2F[2])
  h,V = one_e_two_e_integrals_to_hV(int_1e,int_2e)
  H_tto = TensorTrains.hV_no_to_mpo(h,V,ntuple(x->2,2F[2]),tol=1e-10) #,n=F[3]
  E,ψ_tt =eig_arnoldi(H_tto,m,v;Imax=Imax,ε=ε,ε_tt=ε_tt,rmax=rmax,which=:SR,σ=σ) 
  open("H_tto.json","w") do io 
    JSON3.pretty(io,H_tto)
  end
  open("psi.json","w") do io 
    JSON3.pretty(io,ψ_tt)
  end
  open("E.json","w") do io 
    JSON3.pretty(io,E)
  end
  return E,ψ_tt
end