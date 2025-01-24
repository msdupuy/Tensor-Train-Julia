using TensorTrains
using Plots
using Base.Threads

"""
Sensitivity wrt the number of modes
"""
function modes_sensitivity()
  d_list = 4:2:10
  rks = 40
  δrks_list = -10:10
  rks_list = δrks_list .+ rks
  norm_list_exact = zeros(length(d_list),length(δrks_list))
  norm_list = zeros(length(d_list),length(δrks_list))
  norm_list_orth = zeros(length(d_list),length(δrks_list))
  norm_list_stta = zeros(length(d_list),length(δrks_list))
  n_samples = 10
  n = 50
  ε = 1e-2
  for i_d in eachindex(d_list)
    println(d_list[i_d])
    dims = ntuple(x->n,d_list[i_d])
    A_tt = rand_tt(dims,rks;normalise=true)
    B_tt = rand_tt(dims,rks;normalise=true)
    C_tt = rand_tt(dims,5*rks;normalise=true)
    for (i_rks,δrks) in enumerate(δrks_list)
      A_ttsvd = tt_rounding(A_tt+ε*B_tt+ε^2*C_tt;rmax=rks+δrks)
      norm_list_exact[i_d,i_rks] = norm(A_ttsvd-A_tt+ε*B_tt+ε^2*C_tt)
      @threads for _ in 1:n_samples
        A_rand = ttrand_rounding(A_tt+ε*B_tt+ε^2*C_tt;rmax=rks+δrks,orthogonal=false,ℓ=round(Int,0.5*rks_list[i_rks]))
        A_orth = ttrand_rounding(A_tt+ε*B_tt+ε^2*C_tt;rmax=rks+δrks,ℓ=round(Int,0.5*rks_list[i_rks]))
        A_ttstta = stta(A_tt+ε*B_tt+ε^2*C_tt,rmax=rks_list[i_rks],ℓ=round(Int,0.5*rks_list[i_rks]))
        norm_list[i_d,i_rks] += norm(A_tt+ε*B_tt+ε^2*C_tt-A_rand)
        norm_list_orth[i_d,i_rks] += norm(A_tt+ε*B_tt+ε^2*C_tt-A_orth)
        norm_list_stta[i_d,i_rks] += norm(A_tt+ε*B_tt+ε^2*C_tt-A_ttstta)
      end
    end
  end
  return norm_list_exact,norm_list/n_samples,norm_list_orth/n_samples,norm_list_stta/n_samples
end

#rks_list = -10:10 .+ 40
#heatmap(rks_list,d_list,log10.(norm_list/n_samples),yflip=true)

"""
Error vs rks

d_list = 4:2:10
rks_list = -10:10 #oversampling
norm_list = zeros(length(d_list),length(rks_list))
norm_list_rand = zeros(length(d_list),length(rks_list))
norm_list_orth = zeros(length(d_list),length(rks_list))
n_samples = 10
rks = 40
n = 50
ε = 1e-4
for i_d in eachindex(d_list)
  println(d_list[i_d])
  dims = ntuple(x->n,d_list[i_d])
  A_tt = rand_tt(dims,rks;normalise=true)
  B_tt = rand_tt(dims,5*rks;normalise=true)
  for i_p in eachindex(rks_list)
    A_ttsvd = tt_rounding(A_tt+ε*B_tt;rmax=rks+rks_list[i_p])
    norm_list[i_d,i_p] += norm(A_tt+ε*B_tt-A_ttsvd)
    for _ in 1:n_samples
      A_rand = ttrand_rounding(A_tt+ε*B_tt;rmax=rks+rks_list[i_p],orthogonal=false)
      A_orth = ttrand_rounding(A_tt+ε*B_tt;rmax=rks+rks_list[i_p])
      norm_list_rand[i_d,i_p] += norm(A_tt+ε*B_tt-A_rand)
      norm_list_orth[i_d,i_p] += norm(A_tt+ε*B_tt-A_orth)
    end
  end
end

Sensitivity wrt the oversampling parameter and the TT ranks

d = 10
rks_list = 10:5:50
p_list = 5:5:30 #oversampling
norm_list = zeros(length(rks_list),length(p_list))
n_samples = 10
n = 50
ε = 1e-2
for i_rks in eachindex(rks_list)
  rks = rks_list[i_rks]
  println(rks_list[i_rks])
  dims = ntuple(x->n,d)
  A_tt = rand_tt(dims,rks;normalise=true)
  B_tt = rand_tt(dims,rks;normalise=true)
  C_tt = rand_tt(dims,20*rks;normalise=true)
  A_ttsvd = tt_rounding(A_tt+ε*B_tt+ε^2*C_tt;tol=ε^2)
  for i_p in eachindex(p_list)
    for _ in 1:n_samples
      A_rand = ttrand_rounding(A_tt+ε*B_tt+ε^2*C_tt;rmax=2rks+p_list[i_p])
      norm_list[i_rks,i_p] += norm(A_ttsvd-A_rand)
    end
  end
end
heatmap(p_list,rks_list,log10.(norm_list/n_samples),yflip=true)

Test avec un Slater random
"""

#N = 10
#Ψ = TensorTrains.random_slater(N,2N)
#ψ_tt = ttv_decomp(Ψ)
#rks_list = 50:50:500
#n_samples = 10
#norm_list_exact = zeros(length(rks_list))
#norm_list_rand = zeros(length(rks_list))
#norm_list_orth = zeros(length(rks_list))
#norm_list_stta = zeros(length(rks_list))
#for i_rks in eachindex(rks_list)
#  println(rks_list[i_rks])
#  ϕ_tt = tt_rounding(ψ_tt,rmax=rks_list[i_rks])
#  norm_list_exact[i_rks] = norm(ψ_tt-ϕ_tt)
#  for _ in 1:n_samples
#    ϕ_ttrand = ttrand_rounding(ψ_tt,rmax=rks_list[i_rks],orthogonal=false)
#    ϕ_ttorth = ttrand_rounding(ψ_tt,rmax=rks_list[i_rks])
#    ϕ_ttstta = stta(ψ_tt,rmax=rks_list[i_rks])
#    norm_list_rand[i_rks] += norm(ψ_tt-ϕ_ttrand)
#    norm_list_orth[i_rks] += norm(ψ_tt-ϕ_ttorth)
#    norm_list_stta[i_rks] += norm(ψ_tt-ϕ_ttstta)
#  end
#end
#norm_list_rand/=n_samples
#norm_list_orth/=n_samples
#norm_list_stta/=n_samples

function slater_mode()
  N_list = 6:10
  rks_list = [32,64,128] 
  n_samples = 10
  norm_list_exact = zeros(length(N_list),length(rks_list))
  norm_list_rand = zeros(length(N_list),length(rks_list))
  norm_list_orth = zeros(length(N_list),length(rks_list))
  norm_list_stta = zeros(length(N_list),length(rks_list))
  @threads for _ in 1:n_samples
    for i_N in eachindex(N_list)
      Ψ = TensorTrains.random_slater(N_list[i_N],2N_list[i_N])
      ψ_tt = ttv_decomp(Ψ)
      for i_rks in eachindex(rks_list)
        println(rks_list[i_rks])
        ϕ_tt = tt_rounding(ψ_tt,rmax=rks_list[i_rks])
        norm_list_exact[i_N,i_rks] = norm(ψ_tt-ϕ_tt)
        for _ in 1:n_samples
          ϕ_ttrand = ttrand_rounding(ψ_tt,rmax=rks_list[i_rks],orthogonal=false,ℓ=10)
          ϕ_ttorth = ttrand_rounding(ψ_tt,rmax=rks_list[i_rks],ℓ=10)
          ϕ_ttstta = stta(ψ_tt,rmax=rks_list[i_rks],ℓ=10)
          norm_list_rand[i_N,i_rks] += norm(ψ_tt-ϕ_ttrand)
          norm_list_orth[i_N,i_rks] += norm(ψ_tt-ϕ_ttorth)
          norm_list_stta[i_N,i_rks] += norm(ψ_tt-ϕ_ttstta)
        end
      end
    end
  end
  norm_list_exact/=n_samples
  norm_list_rand/=n_samples^2
  norm_list_orth/=n_samples^2
  norm_list_stta/=n_samples^2
  return norm_list_exact,norm_list_rand,norm_list_orth,norm_list_stta
end

"""
Plot Slater modes
out = slater_mode()
plot(2*(6:10),out[1][:,1],yscale=:log10,legend=:bottomright,label="TT-SVD (rks = 32)",xlabel="Number of modes",ylabel="Error")
plot!(2*(6:10),out[1][:,2],yscale=:log10,legend=:bottomright,label="TT-SVD (rks = 64)",xlabel="Number of modes",ylabel="Error")
plot!(2*(6:10),out[1][:,3],yscale=:log10,legend=:bottomright,label="TT-SVD (rks = 128)",xlabel="Number of modes",ylabel="Error")
plot!(2*(6:10),out[2][:,1],linestyle=:dash,yscale=:log10,legend=:bottomright,label="Rand-rounding (rks = 32)",xlabel="Number of modes",ylabel="Error")
plot!(2*(6:10),out[2][:,2],linestyle=:dash,yscale=:log10,legend=:bottomright,label="Rand-rounding (rks = 64)",xlabel="Number of modes",ylabel="Error")
plot!(2*(6:10),out[2][:,3],linestyle=:dash,yscale=:log10,legend=:bottomright,label="Rand-rounding (rks = 128)",xlabel="Number of modes",ylabel="Error")
plot!(2*(6:10),out[3][:,1],linestyle=:dot,yscale=:log10,legend=:bottomright,label="RandOrth-rounding (rks = 32)",xlabel="Number of modes",ylabel="Error")
plot!(2*(6:10),out[3][:,2],linestyle=:dot,yscale=:log10,legend=:bottomright,label="RandOrth-rounding (rks = 64)",xlabel="Number of modes",ylabel="Error")
plot!(2*(6:10),out[3][:,3],linestyle=:dot,yscale=:log10,legend=:bottomright,label="RandOrth-rounding (rks = 128)",xlabel="Number of modes",ylabel="Error")
plot!(2*(6:10),out[4][:,1],linestyle=:dashdot,yscale=:log10,legend=:bottomright,label="STTA (rks = 32)",xlabel="Number of modes",ylabel="Error")
plot!(2*(6:10),out[4][:,2],linestyle=:dashdot,yscale=:log10,legend=:bottomright,label="STTA (rks = 64)",xlabel="Number of modes",ylabel="Error")
plot!(2*(6:10),out[4][:,3],linestyle=:dashdot,yscale=:log10,legend=:bottomright,label="STTA (rks = 128)",xlabel="Number of modes",ylabel="Error")
"""


"""
#=
N_list = 10:10
eps_list = [1e-1,5e-2,1e-2,5e-3,1e-3]
rks_list = zeros(Int,length(eps_list))
n_samples = 50
norm_list_rand = zeros(length(N_list),length(rks_list))
norm_list_rand_orth = zeros(length(N_list),length(rks_list))
norm_list_exact = zeros(length(N_list),length(rks_list))
for i_N in eachindex(N_list)
  println(N_list[i_N])
Ψ = TensorTrains.random_slater(N_list[i_N],2N_list[i_N])
ψ_tt = ttv_decomp(Ψ)
  for i_eps in eachindex(eps_list)
    println(eps_list[i_eps])
    ϕ_tt = tt_rounding(ψ_tt;tol=eps_list[i_eps]/sqrt(N_list[i_N]))
    rks_list[i_eps] = maximum(ϕ_tt.ttv_rks)
    norm_list_exact[i_N,i_eps] = norm(ψ_tt-ϕ_tt)
    for _ in 1:n_samples
      ϕ_ttrand = ttrand_rounding(ψ_tt,rmax=rks_list[i_eps],orthogonal=false)
      norm_list_rand[i_N,i_eps] += norm(ψ_tt-ϕ_ttrand)
      ϕ_ttrand = ttrand_rounding(ψ_tt,rmax=rks_list[i_eps])
      norm_list_rand_orth[i_N,i_eps] += norm(ψ_tt-ϕ_ttrand)
    end
  end
end
norm_list_rand/=n_samples
norm_list_rand_orth/=n_samples
=#

Test contractions 

#=
ε_list = [1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4,5e-5]
N_list = 6:10
n_samples = 10
opnorm_list = zeros(length(ε_list),length(N_list))
for i_N in eachindex(N_list)
  Ψ = TensorTrains.random_slater(N_list[i_N],2N_list[i_N])
  ψ_tt = ttv_decomp(Ψ)
  for i_ε in eachindex(ε_list)
    println(ε_list[i_ε])
    for _ in 1:n_samples
      ψ_eps = rand_tt(ψ_tt;ε=ε_list[i_ε])
      W = TensorTrains.partial_contraction(ψ_tt,orthogonalize(ψ_eps;i=1))
      opnorm_list[i_ε,i_N] += maximum([opnorm(M*M'-I) for M in W])
    end
  end
end
opnorm_list/=n_samples
=#
"""