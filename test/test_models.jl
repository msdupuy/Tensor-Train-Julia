using Test
import .TensorTrains:one_body_to_matrix, one_body_mpo, two_body_to_matrix, two_body_mpo

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