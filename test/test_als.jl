include("discrete_laplacian.jl")

using LinearAlgebra
using Test

@testset "ALS linsolv" begin
    n = 10
    L = Lap(n,3)
    b = ones(n,n,n)
    x0 = randn(n,n,n)
    L_tt = tto_decomp(reshape(L,n,n,n,n,n,n),1)
    b_tt = ttv_decomp(b,1)
    x0_tt = ttv_decomp(x0,1)
    x = L\b[:]
    x_tt = als_linsolv(L_tt,b_tt,x0_tt)
    y = ttv_to_tensor(x_tt)
    @test isapprox(y[:],x)
end

@testset "ALS eigsolv" begin
    n = 10
    L = Lap(n,3)
    E = eigmin(L)
    x0 = randn(n,n,n)
    L_tt = tto_decomp(reshape(L,n,n,n,n,n,n),1)
    x0_tt = ttv_decomp(x0,1)
    E_tt,x_tt = als_eigsolv(L_tt,x0_tt)
    y = ttv_to_tensor(x_tt)
    @test isapprox(E,E_tt[end])
    #with the iterative solver
    E_tt,x_tt = als_eigsolv(L_tt,x0_tt,it_solver=true)
    @test isapprox(E,E_tt[end],atol=1e-8)
    #with scheduler
    x0_tt = ttv_decomp(ones(n,n,n),1) 
    E_tt,x_tt = als_eigsolv(L_tt,x0_tt;sweep_schedule=[2,3,4],rmax_schedule=[5,7,10],noise_schedule=[1e-3,1e-4,0.0])
    y = ttv_to_tensor(x_tt)
    @test isapprox(E,E_tt[end])
end

@testset "ALS Generalized eigsolv" begin
    n = 5
    L = randn(n^3,n^3)
    L = L*L'+I
    S = randn(n^3,n^3)
    S = S*S'+I
    F = eigen(L,S)
    x0 = randn(n,n,n)
    L_tt = tto_decomp(reshape(L,n,n,n,n,n,n),1)
    S_tt = tto_decomp(reshape(S,n,n,n,n,n,n),1)
    x0_tt = ttv_decomp(x0,1)
    E_tt,x_tt = als_gen_eigsolv(L_tt,S_tt,x0_tt)
    y = ttv_to_tensor(x_tt)
    @test isapprox(real(F.values[1]),E_tt[end])
    E_tt,x_tt = als_gen_eigsolv(L_tt,S_tt,x0_tt,it_solver=true)
    @test isapprox(real(F.values[1]),E_tt[end],atol=1e-8)
end