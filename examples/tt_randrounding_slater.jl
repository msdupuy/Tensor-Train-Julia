include("tt_randrounding.jl")
using JSON3

slater_mode(
  N_list = 6:10,
  rks_list = [32,48,64,80,96],
  ℓ_in = 0.5,
  n_samples = 20
  )