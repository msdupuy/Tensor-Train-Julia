include("tt_tools.jl")
include("models.jl")
using StatsBase
using Random

"""
ordering schemes for QC-DMRG or 2D statistical models
"""

#returns the list of one-orbital reduced density matrix
function one_rdm(x_tt::ttvector)
    d = length(x_tt.ttv_dims)
    @assert(2*ones(Int,d)==x_tt.ttv_dims)
    γ = zeros(d,2,2)
    for i in 1:d
        y_tt = mult(one_body_mpo(i,i,d),x_tt)
        γ[i,2,2] = tt_dot(x_tt,y_tt)
        γ[i,1,1] = 1-γ[i,2,2]
    end
    return γ
end

function test_one_rdm()
    C = randn(2,2,2,2)
    C = 1/norm(C)*C
    C_tt =ttv_decomp(C,1)
    γ = one_rdm(C_tt)
    i = rand(1:4)
    A = zeros(2,2)
    A[1,1] = dot(selectdim(C,i,1),selectdim(C,i,1))
    A[2,2] = 1-A[1,1]
    @test(isapprox(γ[i,:,:],A,atol=1e-12))
end

#returns the list of two-orbitals reduced density matrix
function two_rdm(x_tt::ttvector;fermion=true)
    d = length(x_tt.ttv_dims)
    @assert(2*ones(Int,d)==x_tt.ttv_dims)
    γ = zeros(d,d,2,2,2,2) #(i,j;i,j) occupancy
    for i in 1:d-1
        for j in i+1:d
            γ[i,j,2,2,2,2] = -tt_dot(x_tt,mult(two_body_mpo(i,j,i,j,d),x_tt))
            γ[i,j,1,2,1,2] = -γ[i,j,2,2,2,2] + tt_dot(x_tt,mult(one_body_mpo(j,j,d),x_tt))
            γ[i,j,2,1,2,1] = -γ[i,j,2,2,2,2] + tt_dot(x_tt,mult(one_body_mpo(i,i,d),x_tt))
            γ[i,j,1,1,1,1] = 1.0 -γ[i,j,2,2,2,2] -γ[i,j,2,1,2,1] -γ[i,j,1,2,1,2] 
            γ[i,j,2,1,1,2] = tt_dot(x_tt,mult(one_body_mpo(i,j,d;fermion=fermion),x_tt))
            γ[i,j,1,2,2,1] = γ[i,j,2,1,1,2]
        end
    end
    return γ
end

function test_two_rdm()
    C = randn(2,2,2,2)
    C = 1/norm(C)*C
    C_tt =ttv_decomp(C,1)
    γ = two_rdm(C_tt;fermion=false)
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
    #return A,γ[i,j,:,:,:,:]
    @test(isapprox(γ[i,j,:,:,:,:],A,atol=1e-12))
end

"""
returns the entropy of a matrix M
if a==1 : von Neumann entropy, else Renyi entropy
"""
function entropy(a,M)
   x = eigvals(M)
   x = x[x.>1e-15]
   if a==Inf
      return -log(maximum(x))
   elseif isapprox(a,1.)
      return -sum(x.*log.(x))
   else
      return 1/(1-a)*log.(sum(x.^a))
   end
end

#γ_1 = one orbital RDM, γ_2 = two orbital RDM
function mutual_information(γ1,γ2;a=1) #a=1 von neumann entropy
    d = size(γ1,1)
    IM = zeros(d,d)
    s1 = [entropy(a,γ1[i,:,:]) for i in 1:d]
    for i in 1:d-1
        for j in i+1:d
            IM[i,j] = s1[i]+s1[j]-entropy(a,reshape(γ2[i,j,:,:,:,:],4,4))
        end
    end
    return Symmetric(IM)
end

#returns the fiedler order of a mutual information matrix IM
function fiedler(IM)
   L = size(IM)[1]
   Lap = Diagonal([sum(IM[i,:]) for i=1:L]) - IM
   F = eigen(Lap)
   @test isapprox(F.values[1],0.,atol=1e-14)
   return sortperm(F.vectors[:,2]) #to get 2nd eigenvector
end

function fiedler_order(x_tt::ttvector;a=1) #a=1 : von Neumann entropy
    γ1 = one_rdm(x_tt)
    γ2 = two_rdm(x_tt)
    IM = mutual_information(γ1,γ2;a=a)
    return fiedler(IM)
end

#returns the one particle reduced density matrix of a state encoded in the TT x_tt
function one_prdm(x_tt::ttvector)
    d = length(x_tt.ttv_dims)
    γ = zeros(d,d)
    for i in 1:d
        γ[i,i] = tt_dot(x_tt,mult(one_body_mpo(i,i,d),x_tt))
        for j in i+1:d
            γ[i,j] = tt_dot(x_tt,mult(one_body_mpo(i,j,d),x_tt))
        end
    end
    return Symmetric(γ)
end

function cost(x;tol=1e-10)
    return -sum(log10.((x.+tol).^2.0 .*((1+tol).-x.^2.0)))
end

#best weighted prefactor order
#Warning: V needs to be given in a row-"occupancy" way i.e. V[i,:] represents the coefficients of the natural orbital ψ_i in the basis of the ϕ_j, 1 ≤ j ≤ L
function bwpo_entropy(N,L,V;imax=1000,sigma_current=collect(1:L),CAS=[collect(1:N)],i_cuts=[N],tol=1e-10,temp=1.0)
   iter = 0
   x_N = sigma_current
   cost_max = sum([cost(ones(min(i,L-N,N)),tol=tol) for i in i_cuts])*length(CAS)
   prefactor = sum([cost(svdvals(V[i_cas,x_N[1:i]]),tol=tol) for i in i_cuts for i_cas in CAS]) 
   while iter < imax && temp*prefactor/(imax*cost_max) < rand()
      #nouveau voisin
      j = rand(1:N)
      k = rand(N+1:L)
      x_temp = copy(x_N) 
      x_temp[k],x_temp[j] = x_N[j],x_N[k]
      new_prefactor = sum([cost(svdvals(V[i_cas,x_temp[1:i]]),tol=tol) for i in i_cuts for i_cas in CAS])
      if new_prefactor > prefactor
         x_N = x_temp
         prefactor = new_prefactor
      end
      iter = iter+1
   end
   return x_N,prefactor
end
