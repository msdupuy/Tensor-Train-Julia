"""
Transcorrelated Beryllium using Arnoldi
valeur Giner : -14.6251920
"""

using TensorTrains
using JSON3
using Dates

function Be_arnoldi(Imax;m=3,rmax=64,ε=1e-5,ε_tt=1e-5,σ=20.0,Imaxmax=10,fiedler=false,history=true,print=true)
  F = read_electron_integral_tensors_nosymmetry("./examples/Be.ezfio.FCIDUMP")
  int_1e,int_2e = F[4],F[5]
  v = slater(F[3],2F[2])
  h,V = one_e_two_e_integrals_to_hV(int_1e,int_2e)
  H_tto = TensorTrains.hV_no_to_mpo(h,V,ntuple(x->2,2F[2]),tol=1e-10) #,n=F[3]
  E,ψ_tt,hist =eig_arnoldi(H_tto,m,v;Imax=Imax,ε=ε,ε_tt=ε_tt,rmax=min(24,rmax),which=:SR,σ=σ,history=history) 
  if print
    date = now()
    open("H_tto_$(date).json","w") do io 
      JSON3.pretty(io,H_tto)
    end
    open("psi_$(date).json","w") do io 
      JSON3.pretty(io,ψ_tt)
    end
    open("E_$(date).json","w") do io 
      JSON3.pretty(io,E)
    end
    open("hist_$(date).json","w") do io
      JSON3.pretty(io,hist)
    end
  end
  imaxmax = 0
  while imaxmax<Imaxmax && norm(H_tto*ψ_tt-E*ψ_tt) > ε
    imaxmax += 1
    perm = fiedler_order(ψ_tt)
    date = now()
    if print
      open("sigma_fiedler_$(date).json","w") do io 
        JSON3.pretty(io,perm)
      end
    end
    v = slater(F[3],2F[2];σ=invperm(perm)[1:F[3]])
    h,V = h[perm,perm],V[perm,perm,perm,perm]
    H_tto = TensorTrains.hV_no_to_mpo(h,V,ntuple(x->2,2F[2]),tol=1e-10)
    E,ψ_tt,hist =eig_arnoldi(H_tto,m,v;Imax=Imax,ε=ε,ε_tt=ε_tt,rmax=rmax,which=:SR,σ=σ,history=history) 
    if print
      open("H_tto_$(date).json","w") do io 
        JSON3.pretty(io,H_tto)
      end
      open("psi_$(date).json","w") do io 
        JSON3.pretty(io,ψ_tt)
      end
      open("E_$(date).json","w") do io 
        JSON3.pretty(io,E)
      end
      open("hist_$(date).json","w") do io
        JSON3.pretty(io,hist)
      end
    end
  end
  return E,ψ_tt
end