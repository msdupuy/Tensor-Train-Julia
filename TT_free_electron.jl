"""
TT of Hartree product for 1D free electron gas
"""

using Test
using Combinatorics
using LinearAlgebra
using QuadGK

#one particle basis function
function one_particle_basis(x,k)
   return sqrt(2)*sin(2pi*k*x)
end

@test isapprox(quadgk(x -> one_particle_basis(x,3.)^2,0.,1.)[1],1)

function random_orthogonal(n)
   A = rand(n,n)
   Q,R = qr(A)
   return Q
end

function tuple_to_index(t,L)
   A = ones(Integer,L)
   for x=t
      A[x]=A[x]+1
   end
   return A
end

@test tuple_to_index([2,3,5],5) == [1,2,2,1,2]

#full tensor of Hartree product in Fock space picture
function full_tensor(N,L;V=random_orthogonal(L))
   C = zeros(Float64, 2*ones(Integer,L)...)
   tuple_collection = collect(combinations(1:L,N))
   for x=1:size(tuple_collection)[1]
      C[tuple_to_index(tuple_collection[x],L)...] = det(V[1:N,tuple_collection[x]])
   end
   return C
end

"test for N=2,L=4"
tensor_example = zeros(2,2,2,2)
tensor_example[2,2,1,1] = 1.0
@test full_tensor(2,4;V=Matrix{Float64}(I,4,4)) == tensor_example

"TT decomposition  L == even"

function TT_decomposition(tensor,L)
   TT_matrix = zeros(L,2,Int(2^(L/2)),Int(2^(L/2)))
   TT_sigma = zeros(L-1,Int(2^(L/2)))
   reshaped_tensor = tensor
   rank_now = 2
   rank_prev = 1
   for k=1:(L-1)
      reshaped_tensor = reshape(reshaped_tensor,2*rank_prev,:)
      u,s,v = svd(reshaped_tensor) #thin svd u,s,v (see doc)
      rank_now = size(s)[1]
      rk = rank_prev #dim A[μ_k] = rank_prev × rank_now
      rk_next = rank_now
      TT_matrix[k,1,1:rk,1:rk_next] = u[1:rk,:]
      TT_matrix[k,2,1:rk,1:rk_next] = u[(rk+1):2*rk,:]
      TT_sigma[k,1:rank_now] = s
      reshaped_tensor = Diagonal(s)*Transpose(v)
      rank_prev = rank_now
   end
   TT_matrix[L,1,1:2,1] = reshaped_tensor[:,1]
   TT_matrix[L,2,1:2,1] = reshaped_tensor[:,2]
   return TT_matrix, TT_sigma
end

function index_to_tuple(k,L)
   return digits(k,base=2,pad=L).+1
end

function TT_to_tensor(TT_matrix,L)
   tensor = zeros(Float64, 2*ones(Integer,L)...)
   for k=0:(2^L-1)
      x = index_to_tuple(k,L)
      A = Transpose(TT_matrix[1,x[1],1,1:2])
      for j=2:L
         A = A*TT_matrix[j,x[j],1:Int(min(2^(j-1),2^(L-j+1))),1:Int(min(2^(j),2^(L-j)))]
      end
      tensor[x...] = A[1,1]
   end
   return tensor
end

V = full_tensor(3,6)
TT_matrix, TT_sigma = TT_decomposition(V,6)
@test isapprox(V,TT_to_tensor(TT_matrix,6))

"""
some examples
"""

function pertub_identity(L,x)
   L2 = Int(L/2)
   V = random_orthogonal(L2)
   A = zeros(L,L)
   A[1:L2,1:L2] = sqrt(1-x^2)*Matrix{Float64}(I, L2, L2)
   A[(L2+1):L,1:L2] = x*V
   A[1:L2,(L2+1):L] = -x*V
   A[(L2+1):L,(L2+1):L] = sqrt(1-x^2)*Matrix{Float64}(I, L2, L2)
   return A
end

U = pertub_identity(4,0.2)
@test isapprox(U*Transpose(U),Matrix{Float64}(I,4,4))

"""
Hierarchical SVD keeping singular values sigma_k^2 > tol
"""

function cut_off(x,tol)
   res = 0.
   j=0
   while res <= tol
      res = res + x[end-j]^2
      j=j+1
   end
   return x[1:(end-j+1)]
end

@test cut_off([2,1,0.2,0.1],1) == [2,1]

function HSVD(tensor,L,tol)
   TT_matrix = zeros(L,2,Int(2^(L/2)),Int(2^(L/2)))
   TT_sigma = zeros(L-1,Int(2^(L/2)))
   reshaped_tensor = tensor
   rank_now = 2
   rank_prev = 1
   for k=1:(L-1)
      reshaped_tensor = reshape(reshaped_tensor,2*rank_prev,:)
      u,s,v = svd(reshaped_tensor) #thin svd u,s,v (see doc)
#      s = s[s .> sqrt(tol)]
      s = cut_off(s,tol)
      rank_now = size(s)[1]
      rk = rank_prev #dim A[μ_k] = rank_prev × rank_now
      rk_next = rank_now
      TT_matrix[k,1,1:rk,1:rk_next] = u[1:rk,1:rk_next]
      TT_matrix[k,2,1:rk,1:rk_next] = u[(rk+1):2*rk,1:rk_next]
      TT_sigma[k,1:rk_next] = s
      reshaped_tensor = Diagonal(s)*Transpose(v[:,1:rank_now])
      rank_prev = rank_now
   end
   TT_matrix[L,1,1:2,1] = reshaped_tensor[:,1]
   TT_matrix[L,2,1:2,1] = reshaped_tensor[:,2]
   return TT_matrix, TT_sigma
end

TT1, s1 = TT_decomposition(V,6)
TT2, s2= HSVD(V,6,0.)
@test isapprox(TT1,TT2)
@test isapprox(s1,s2)
