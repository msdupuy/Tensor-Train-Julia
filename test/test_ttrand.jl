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
