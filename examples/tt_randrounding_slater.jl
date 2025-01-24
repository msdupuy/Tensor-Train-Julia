include("tt_randrounding.jl")
using JSON3

out = slater_mode()
data = Dict{String,Any}()
data["N_list"] = 6:10
data["rks_list"] = [32,64,128]
data["exact_error"] = out[1]
data["randrounding_error"] = out[2]
data["randorth_error"] = out[3]
data["stta_error"] = out[4]
data["n_samples"] = 10
open(io -> JSON3.write(io, data, allow_inf=true), "out/rand-rounding/slater_ℓ=10.json", "w")