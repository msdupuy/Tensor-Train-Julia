include("discrete_laplacian.jl")

using Test

@testset "MALS linsolv" begin
    n = 10
    L = Lap(n,3)
    b = ones(n,n,n)
    x0 = randn(n,n,n)
    L_tt = tto_decomp(reshape(L,n,n,n,n,n,n),1)
    b_tt = ttv_decomp(b,1)
    x0_tt = ttv_decomp(x0,1)
    x = L\b[:]
    x_tt = mals_linsolv(L_tt,b_tt,x0_tt;tol=0.0)
    y = ttv_to_tensor(x_tt)
    @test isapprox(y[:],x)
end

@testset "MALS eigsolv" begin
    n = 10
    L = Lap(n,3)
    E = eigmin(L)
    x0 = randn(n,n,n)
    L_tt = tto_decomp(reshape(L,n,n,n,n,n,n),1)
    x0_tt = ttv_decomp(x0,1)
    E_tt,x_tt,r_hist = mals_eigsolv(L_tt,x0_tt;tol=0.0)
    y = ttv_to_tensor(x_tt)
    @test isapprox(E,E_tt[end])
    E_tt,x_tt,r_hist = mals_eigsolv(L_tt,x0_tt,it_solver=true)
    @test isapprox(E,E_tt[end],atol=1e-8)
end