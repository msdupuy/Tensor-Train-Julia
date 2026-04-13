include("tt_randrounding.jl")
using Base.Threads

d_list = 8:2:20
rks = 10
δrks_list = -5:5
ℓ_list = [0,5]
n = 5
ε_list = [1e-2,1e-3,1e-4]
@threads for ℓ in ℓ_list
  for ε in ε_list
    modes_sensitivity(d_list=d_list,δrks_list=δrks_list,ℓ=ℓ,ε=ε, n=n,rks=rks)
  end
end