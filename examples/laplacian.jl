using TensorTrains
using LinearAlgebra

"""
test run
"""
const n = 4
const d = 4

#b_tt = b_ttv(n,d)
const r = 2
const b_tt = rand_tt(n*ones(Int64,d),vcat(1,r*ones(Int64,d-1),1))
const H_mpo = Δ_tto(n,d)

const α_opt = 2/(d*eigmax(Δ(n))+d*eigmin(Δ(n)))
const x_tt, res = gradient_fixed_step(H_mpo,b_tt,α_opt,Imax=4,i_trunc=2)

"""
Real run
"""
const n = 10
const d = 10
#b_tt = b_ttv(n,d)
const r = 2
const b_tt = rand_tt(n*ones(Int64,d),vcat(1,r*ones(Int64,d-1),1))
const H_mpo = Δ_tto(n,d)

const Imax = 100
const α_opt = 2/(d*eigmax(Δ(n))+d*eigmin(Δ(n)))
const i_trunc = 1
#eps_list = [1e-2]
const eps_list = [1e-2,1e-3,1e-4,1e-5,1e-6]
const res_list = zeros(length(eps_list),Imax+1)
const sol_hsvd = Array{Array{Array{Float64,1}}}(undef,length(eps_list))

for i_ε in eachindex(eps_list)
  x_ϵ_tt, res_ϵ = gradient_fixed_step(H_mpo,b_tt,α_opt,Imax=Imax,i_trunc=1,eps_tt=eps_list[i_ε])
  println(x_ϵ_tt.ttv_rks)
  Σ = tt_svdvals(x_ϵ_tt)
  sol_hsvd[i_ε] = Σ
  res_list[i_ε,1] = eps_list[i_ε]
  res_list[i_ε,2:length(res_ϵ)+1] = res_ϵ/sqrt(n^d)
end