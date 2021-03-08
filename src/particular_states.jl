using Primes

#returns a random orthogonal matrix of size n
function random_orthogonal(n::Int)
   A = rand(n,n)
   Q,R = qr(A)
   return Q
end

#returns the array A of length L such that A[i] = 1 if i ∉ t and A[i] = 2 else
function tuple_to_index(t::Array{Int},L)
   A = ones(Integer,L)
   for x=t
      A[x]=A[x]+1
   end
   return A
end

#random Slater determinant ψ_1 ∧ ⋯ ∧ ψ_N where (ψ_1,⋯,ψ_N)^T = V (ϕ_1,⋯,ϕ_L)^T for V ∈ R^{N × L} orthogonal
function random_slater(N,L;V=random_orthogonal(L))
   C = zeros(Float64, 2*ones(Integer,L)...)
   tuple_collection = collect(combinations(1:L,N))
   for x=1:size(tuple_collection,1)
      C[tuple_to_index(tuple_collection[x],L)...] = det(V[1:N,tuple_collection[x]])
   end
   return C
end

#returns a sum of random Slater determinants 
# C = ∑_{(i_1,…,i_N) ∈ CAS) a_i1…i_N ψ_i1 ∧ ⋯ ∧ ψ_iN
function random_static_correlation_tensor(N,L;V=random_orthogonal(L),n=2,a=randn(n),
   CAS=collect(combinations(1:L,N))[vcat(1,rand(2:binomial(L,N),n-1))])
   a = a/norm(a)
   C = zeros(Float64, 2*ones(Integer,L)...)
   tuple_collection = collect(combinations(1:L,N))
   for x=1:size(tuple_collection,1)
      for k=1:n
         C[tuple_to_index(tuple_collection[x],L)...] += a[k]*det(V[CAS[k],tuple_collection[x]])
      end
   end
   return C
end

#returns the state C[μ_1,⋯,μ_L] = √(prime) iff ∑ μ_i = N with different primes
function random_prime_tensor(N,L)
   C = zeros(Float64, 2*ones(Integer,L)...)
   tuple_collection = collect(combinations(1:L,N))
   prime_collection = shuffle!(primes(2^(N+L))) #works for N+L moderate
   for x=1:size(tuple_collection,1)
      C[tuple_to_index(tuple_collection[x],L)...] = sqrt(prime_collection[x])
   end
   return C./norm(C)
end
