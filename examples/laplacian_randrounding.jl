using TensorTrains
using LinearAlgebra
using Profile
using ProfileView
using Random

function test_gradient(n,d,r;rand_rounding=false,Imax=10)
  Random.seed!(1234)
  b_tt = rand_tt(n*ones(Int64,d),vcat(1,r*ones(Int64,d-1),1))
  H_mpo = Δ_tto(n,d)
  α_opt = 2/(d*eigmax(Δ(n))+d*eigmin(Δ(n)))
  x_tt, res = gradient_fixed_step(H_mpo,b_tt,α_opt,Imax=Imax,i_trunc=3,rand_rounding=rand_rounding)
  return x_tt,res
end

function test_arnoldi(n,d,r;m=3,Imax=10,which=:LM)
  Random.seed!(1234)
  v = rand_tt(n*ones(Int64,d),vcat(1,r*ones(Int64,d-1),1);normalise=true)
  H_mpo = Δ_tto(n,d)
  λ,v = eig_arnoldi(H_mpo,m,v;Imax=Imax,ε=1e-6,ε_tt=1e-4,rmax=256,which=which)
  return λ,v
end

#@profview gradient_fixed_step(H_mpo,b_tt,α_opt,Imax=10,i_trunc=3,rand_rounding=false)