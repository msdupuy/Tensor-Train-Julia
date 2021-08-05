include("discrete_laplacian.jl")

using Test

@testset "DMRG linsolv" begin
    A = Lap(5,4)
    Atto = tto_decomp(reshape(A,5,5,5,5,5,5,5,5))
    btt = rand_tt(Atto.tto_dims,[1,5,25,5,1])
    ytt = dmrg_linsolv(Atto,btt,btt,sweep_count=2,N=3)
    ysol = A\(ttv_to_tensor(btt)[:])
    ydmrg = ttv_to_tensor(ytt)[:]
    @test isapprox(ysol,ydmrg,rtol=1e-5)
end

#@testset "DMRG eigsolv" begin
#    
#end