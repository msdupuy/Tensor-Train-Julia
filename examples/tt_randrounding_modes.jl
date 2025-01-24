include("tt_randrounding.jl")
using JSON3

out = modes_sensitivity()
data = Dict{String,Any}()
data["N_list"] = 4:2:10
data["rks_list"] = -10:10 .+40
data["n"] = 50
data["ε"] = 1e-2
data["exact_error"] = out[1]
data["randrounding_error"] = out[2]
data["randorth_error"] = out[3]
data["stta_error"] = out[4]
data["n_samples"] = 10
open(io -> JSON3.write(io, data, allow_inf=true), "out/rand-rounding/perturbed_ℓ=0.5rmax.json", "w")
