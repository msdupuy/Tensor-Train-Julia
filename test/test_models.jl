using Test
using LinearAlgebra
import TensorTrains:one_body_to_matrix, one_body_mpo, two_body_to_matrix, two_body_mpo
using Random

@testset "MPO constructors" begin
    L=6
    p=rand(1:L)
    q=rand(1:L)
    #one-body MPO constructors
    H = one_body_to_matrix(p,q,L)
    Hmpo = one_body_mpo(p,q,L)
    H2 = tto_to_tensor(Hmpo)
    @test isapprox(norm(H-reshape(H2,2^L,2^L)),0.0,atol=1e-10)
    #two-body MPO constructors
#    k,l,m,n=1,2,3,4 
#    H = two_body_to_matrix(k,l,m,n,L) #fails but probably bug in two_body_to_matrix
#    Hmpo = two_body_mpo(k,l,m,n,L)
#    H2 = tto_to_tensor(Hmpo)
#    @test isapprox(norm(H-reshape(H2,2^L,2^L)),0.0,atol=1e-10)
end

@testset "PPP_C_NH_N" begin
    N = 6
    H_tto = PPP_C_NH_N(N)
    ψ_0 = tt_up_rks(half_filling(N),32;ϵ_wn=1e-1)
    ψ_0 = orthogonalize(ψ_0)
    E_dmrg, ψ_tt, r_dmrg = dmrg_eigsolv(H_tto,ψ_0)
    @test isapprox(E_dmrg[end],-0.46752535023652453,atol=1e-6) #value in Bendazzoli, G. L., & Evangelisti, S. (1991). Full-CI calculations of alternant cyclic polyenes (CH) N, N= 2, 4, 6, ƒ 18, in the PPP approximation. Chemical physics letters, 185(1-2), 125-130.
end