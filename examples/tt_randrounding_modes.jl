include("tt_randrounding.jl")

d_list = 4:2:10
rks = 30
δrks_list = -10:10
ℓ_list = [0,5,10]
ε_list = [1e-1,1e-2,1e-3]
for ℓ in ℓ_list
  for ε in ε_list
    modes_sensitivity(d_list=d_list,δrks_list=δrks_list,ℓ=ℓ,ε=ε)
  end
end