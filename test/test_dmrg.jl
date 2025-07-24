include("discrete_laplacian.jl")

using Test

@testset "DMRG linsolv" begin
    A = Matrix(Lap(5,4))
    Atto = tto_decomp(reshape(A,5,5,5,5,5,5,5,5))
    btt = rand_tt(Atto.tto_dims,[1,5,25,5,1])
    ytt = dmrg_linsolv(Atto,btt,btt,sweep_count=2,N=2,it_solver=false)
    ysol = A\(ttv_to_tensor(btt)[:])
    ydmrg = ttv_to_tensor(ytt)[:]
    @test isapprox(ysol,ydmrg,rtol=1e-5)
    btt = rand_tt(Atto.tto_dims,[1,5,25,5,1])
    ysol = A\(ttv_to_tensor(btt)[:])
    ytt = dmrg_linsolv(Atto,btt,btt,sweep_count=2,N=2,it_solver=true)
    ydmrg = ttv_to_tensor(ytt)[:]
    @test isapprox(ysol,ydmrg,rtol=1e-5)
    btt = rand_tt(Atto.tto_dims,[1,5,25,5,1])
    ysol = A\(ttv_to_tensor(btt)[:])
    ytt = dmrg_linsolv(Atto,btt,btt,sweep_schedule=[2],N=1,it_solver=true)
    ydmrg = ttv_to_tensor(ytt)[:]
    @test isapprox(ysol,ydmrg,rtol=1e-5)
end

@testset "DMRG eigsolv" begin
    N = 6
    H_tto = PPP_C_NH_N(N)
    ψ_0 = tt_up_rks(half_filling(N),32;ϵ_wn=1e-2)
    ψ_0 = orthogonalize(ψ_0)
    E_dmrg, ψ_tt, r_dmrg = dmrg_eigsolv(H_tto,ψ_0)
    A = Matrix(Lap(5,4))
    @test isapprox(norm(H_tto*ψ_tt-E_dmrg[end]*ψ_tt),0.0,atol=1e-5)
end

# Define a test case for K_eigmin
function test_K_eigmin()
    # Create a random matrix G
    G = rand(10, 10,10)

    # Create a random matrix H
    H = rand(10, 10,10)

    # Create a random vector V0
    V0 = rand(10,10,10)

    # Create a random matrix Amid
    Amid = rand(10, 10, 10, 10)

    # Call K_eigmin
    λ, V = K_eigmin(G, H, V0, Amid)

    # Check that λ is a real number
    @test isreal(λ)

    # Check that V is a vector of the correct length
    @test length(V) == 10

    # Check that the eigenvalue equation is satisfied
    @test norm(G * V - λ * V) < 1e-6
end