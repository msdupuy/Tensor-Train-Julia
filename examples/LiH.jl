using TensorTrains
using Dates
using JSON3

function lih(;n_cas = 10, rmax=64,ε=1e-5,ε_tt=1e-5,fiedler=false,history=true,print=true)
  F = read_electron_integral_tensors_nosymmetry("./examples/lih_fcidump.txt")
  int_1e,int_2e = F[4],F[5]
  v = slater(F[3],2n_cas)
  v = tt_up_rks(v,64;ϵ_wn=1e-2)
  h,V = one_e_two_e_integrals_to_hV(int_1e[1:n_cas,1:n_cas],int_2e[1:n_cas,1:n_cas,1:n_cas,1:n_cas])
  H_tto = TensorTrains.hV_no_to_mpo(h,V,ntuple(x->2,2n_cas),tol=1e-10) #,n=F[3]
  E,ψ_tt,r_hist =dmrg_eigsolv(H_tto,v;sweep_schedule=[4],rmax_schedule=[64],it_solver=true) 
  if print
    date = now()
    open("psi_LiH_canonical_cas=$(n_cas)_$(date).json","w") do io 
      JSON3.pretty(io,ψ_tt)
    end
    open("E_LiH_canonical_cas=$(n_cas)_$(date).json","w") do io 
      JSON3.pretty(io,E)
    end
    open("r_LiH_canonical_cas=$(n_cas)_$(date).json","w") do io
      JSON3.pretty(io,r_hist)
    end
  end
  if fiedler
    imaxmax += 1
    perm = fiedler_order(ψ_tt)
    date = now()
    if print
      open("sigma_LiH_fiedler_cas=$(n_cas)_$(date).json","w") do io 
        JSON3.pretty(io,perm)
      end
    end
    v = slater(F[3],2n_cas;σ=invperm(perm)[1:F[3]])
    v = tt_up_rks(v,64;ϵ_wn=1e-2)
    h,V = h[perm,perm], V[perm,perm,perm,perm]
    H_tto = TensorTrains.hV_no_to_mpo(h,V,ntuple(x->2,2n_cas),tol=1e-10)
    E,ψ_tt,r_hist =dmrg_eigsolv(H_tto,v;sweep_schedule=[4],rmax_schedule=[128],it_solver=true) 
    if print
      open("psi_cas=$(n_cas)_$(date).json","w") do io 
        JSON3.pretty(io,ψ_tt)
      end
      open("E_cas=$(n_cas)_$(date).json","w") do io 
        JSON3.pretty(io,E)
      end
      open("hist_cas=$(n_cas)_$(date).json","w") do io
        JSON3.pretty(io,r_hist)
      end
    end
  end
  return E,ψ_tt
end
