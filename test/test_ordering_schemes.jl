using Test
import .TensorTrains:one_rdm,two_rdm

@testset "Orbital RDM" begin
    C = randn(2,2,2,2)
    C = 1/norm(C)*C
    C_tt =ttv_decomp(C)
    #one_rdm
    γ1 = one_rdm(C_tt)
    i = rand(1:4)
    A = zeros(2,2)
    A[1,1] = dot(selectdim(C,i,1),selectdim(C,i,1))
    A[2,2] = 1-A[1,1]
    @test(isapprox(γ1[i,:,:],A,atol=1e-12))
    #two_rdm
    γ2 = two_rdm(C_tt;fermion=false)
    i = rand(1:4)
    j = rand(setdiff(1:4,i))
    i,j = min(i,j),max(i,j)
    A = zeros(2,2,2,2)
    A[1,1,1,1] = norm(selectdim(selectdim(C,j,1),i,1))^2
    A[1,2,1,2] = norm(selectdim(selectdim(C,j,2),i,1))^2
    A[2,1,2,1] = norm(selectdim(selectdim(C,j,1),i,2))^2
    A[2,2,2,2] = norm(selectdim(selectdim(C,j,2),i,2))^2
    A[1,2,2,1] = dot(selectdim(selectdim(C,j,2),i,1),selectdim(selectdim(C,j,1),i,2))
    A[2,1,1,2] = A[1,2,2,1]
    @test(isapprox(γ2[i,j,:,:,:,:],A,atol=1e-12))
end 