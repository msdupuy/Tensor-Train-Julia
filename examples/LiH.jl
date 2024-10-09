using TensorTrains
using Dates
using LinearAlgebra
using JSON3

function lih(;n_cas = 10,canonical=true,fiedler=false,print=true,sweep_schedule=[4,8,12,16,20,24,28],rmax_schedule=[16,24,32,40,48,56,64],μ=0.2)
  F = read_electron_integral_tensors("./examples/lih_fcidump.txt")
  int_1e,int_2e = F[4],F[5]
  v = slater(F[3],2n_cas)
  h,V = one_e_two_e_integrals_to_hV(int_1e[1:n_cas,1:n_cas],int_2e[1:n_cas,1:n_cas,1:n_cas,1:n_cas])
  H_tto = hV_to_mpo(h+μ*I,V,v.ttv_dims;tol=1e-8,chemistry=true) #,n=F[3]
  println(dot(v,H_tto*v)+F[1]-F[3]*μ) # =-7.9836
  if canonical
    v = tt_up_rks(v,rmax_schedule[1];ϵ_wn=1e-2)
    E,ψ_tt,r_hist =dmrg_eigsolv(H_tto,v;sweep_schedule=sweep_schedule,rmax_schedule=rmax_schedule,it_solver=true) 
    if print
      date = now()
      open("psi_LiH_canonical_cas=$(n_cas)_$(date).json","w") do io 
        JSON3.pretty(io,ψ_tt)
      end
      open("E_LiH_canonical_cas=$(n_cas)_$(date).json","w") do io 
        JSON3.pretty(io,E.-μ*F[3])
      end
      open("r_LiH_canonical_cas=$(n_cas)_$(date).json","w") do io
        JSON3.pretty(io,r_hist)
      end
    end
  end
  if fiedler
    v = slater(F[3],2n_cas)
    v = tt_up_rks(v,rmax_schedule[1];ϵ_wn=1e-2)
    E,ψ_tt,r_hist =dmrg_eigsolv(H_tto,v;sweep_schedule=[4],rmax_schedule=[rmax_schedule[1]]) 
    perm = fiedler_order(ψ_tt)
    date = now()
    if print
      open("sigma_LiH_fiedler_cas=$(n_cas)_$(date).json","w") do io 
        JSON3.pretty(io,perm)
      end
    end
    v = slater(F[3],2n_cas;σ=invperm(perm)[1:F[3]])
    v = tt_up_rks(v,rmax_schedule[1];ϵ_wn=1e-2)
    h,V = one_e_two_e_integrals_to_hV(int_1e[1:n_cas,1:n_cas],int_2e[1:n_cas,1:n_cas,1:n_cas,1:n_cas])
    h,V = h[perm,perm], V[perm,perm,perm,perm]
    h = h+μ*I
    H_tto = hV_to_mpo(h,V,ntuple(x->2,2n_cas),tol=1e-8,chemistry=true)
    E,ψ_tt,r_hist =dmrg_eigsolv(H_tto,v;sweep_schedule=sweep_schedule,rmax_schedule=rmax_schedule,it_solver=true) 
    if print
      open("psi_LiH_fiedler_cas=$(n_cas)_$(date).json","w") do io 
        JSON3.pretty(io,ψ_tt)
      end
      open("E_LiH_fiedler_cas=$(n_cas)_$(date).json","w") do io 
        JSON3.pretty(io,E.-μ*F[3])
      end
      open("r_LiH_fiedler_cas=$(n_cas)_$(date).json","w") do io
        JSON3.pretty(io,r_hist)
      end
    end
  end
end
