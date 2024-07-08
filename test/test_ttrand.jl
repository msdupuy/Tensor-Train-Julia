using Test
using TensorTrains

@testset "partial contraction" begin
  dims = (2,2,2,2)
  rks = [1,4,4,4,1]
  A = rand_tt(dims,rks)
  B = rand_tt(dims,rks)
  W = TensorTrains.partial_contraction(A,B)
  @test isapprox(W[1][1],dot(A,B))
end

@testset "ttrand_rounding" begin
  dims = (2,2,2,2,2,2,2,2)
  rks = [1,2,4,4,4,4,4,2,1]
  A_tt = rand_tt(dims,rks)
  A_pert = tt_up_rks(A_tt,20,ϵ_wn=1e-8)
  A = ttv_to_tensor(A_pert)
  A_ttrand = ttrand_rounding(A_pert;rks=[1,2,4,6,6,6,4,2,1])
  A_rand = ttv_to_tensor(A_ttrand)
  @test isapprox(A,A_rand)
end

@testset "TT sum rand_rounding" begin
  d = 10
  n = 50
  rks = 5
  dims = ntuple(x->n,d)
  A_tt = rand_tt(dims,rks;normalise=true)
  B_tt = rand_tt(dims,20*rks;normalise=true)
  ε = 1e-4
  @time A_ttsvd = tt_rounding(A_tt+ε*B_tt;tol=ε)
  @time A_rand = ttrand_rounding(A_tt+ε*B_tt;rmax=10)
  @test(norm(A_tt-A_rand)<1e-3*norm(A_tt))
end