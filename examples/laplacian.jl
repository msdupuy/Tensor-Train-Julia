using TensorTrains
using LinearAlgebra
using Plots

"""
test run
"""
const n = 4
const d = 4

#b_tt = b_ttv(n,d)
const r = 2
const b_tt = rand_tt(ntuple(x->n,d),vcat(1,r*ones(Int64,d-1),1))
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
const b2_tt = rand_tt(ntuple(x->n,d),vcat(1,r*ones(Int64,d-1),1),normalise=true)
const H2_mpo = Δ_tto(n,d)

const Imax = 50
const α_opt = 2/(d*eigmax(Δ(n))+d*eigmin(Δ(n)))
const i_trunc = 1
#eps_list = [1e-2]
eps_list = [1e-2,1e-3,1e-4]
const res_list = zeros(length(eps_list),Imax+1)
const sol_hsvd = Array{Array{Array{Float64,1}}}(undef,length(eps_list))

for i_ε in eachindex(eps_list)
  @time x_ϵ_tt, res_ϵ = gradient_fixed_step(H2_mpo,b2_tt,α_opt,Imax=Imax,i_trunc=i_trunc,eps_tt=eps_list[i_ε],r_tt=50)
  println(x_ϵ_tt.rks)
  Σ = tt_svdvals(x_ϵ_tt)
  sol_hsvd[i_ε] = Σ
  res_list[i_ε,1] = eps_list[i_ε]
  res_list[i_ε,2:length(res_ϵ)+1] = res_ϵ
end

scatter(res_list[1,2:end][res_list[1,2:end].>0],label="$(eps_list[1])",yscale=:log10)
scatter!(res_list[2,2:end][res_list[2,2:end].>0],label="$(eps_list[2])",yscale=:log10)
scatter!(res_list[3,2:end][res_list[3,2:end].>0],label="$(eps_list[3])",yscale=:log10)

"""
Randomised version
"""

const res_rand_list = zeros(length(eps_list),Imax+1)
const sol_rand_hsvd = Array{Array{Array{Float64,1}}}(undef,length(eps_list))

for i_ε in eachindex(eps_list)
  @time x_ϵ_tt, res_ϵ = gradient_fixed_step(H2_mpo,b2_tt,α_opt,Imax=Imax,i_trunc=i_trunc,eps_tt=eps_list[i_ε],rand_rounding=true,r_tt=50,ℓ=10)
  println(x_ϵ_tt.rks)
  Σ = tt_svdvals(x_ϵ_tt)
  sol_rand_hsvd[i_ε] = Σ
  res_rand_list[i_ε,1] = eps_list[i_ε]
  res_rand_list[i_ε,2:length(res_ϵ)+1] = res_ϵ
end

# influence of the TT truncation parameter
scatter(res_rand_list[1,2:end][res_rand_list[1,2:end].>0],label="$(eps_list[1])",yscale=:log10)
scatter!(res_rand_list[2,2:end][res_rand_list[2,2:end].>0],label="$(eps_list[2])",yscale=:log10)
scatter!(res_rand_list[3,2:end][res_rand_list[3,2:end].>0],label="$(eps_list[3])",yscale=:log10)

# Comparison deterministic vs randomised rounding
scatter(res_rand_list[2,2:end][res_rand_list[2,2:end].>0],label="ε=$(eps_list[2]), RandRound",yscale=:log10)
scatter!(res_list[2,2:end][res_list[2,2:end].>0],label="ε=$(eps_list[2]), HSVD",yscale=:log10)