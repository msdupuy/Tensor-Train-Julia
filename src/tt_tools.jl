using Random
using LinearAlgebra
using Base.Threads
using IterativeSolvers
using TensorOperations
import Base.isempty
import Base.eltype
import Base.copy
import Base.complex

"""
TT constructor of a tensor
``C[\\mu_1,…,\\mu_d] = A_1[1][\\mu_1]*⋯*A_d[d][\\mu_d] \\in \\mathbb{K}^{n_1 \\times ... \\times n_d}``
The following properties are stored
	* ttv_vec: the TT cores A_k as a list of 3-order tensors ``(A_1,...,A_d)`` where ``A_k = A_k[\\mu_k,\\alpha_{k-1},\\alpha_k]``, ``1 \\leq \\alpha_{k-1} \\leq r_{k-1}``, ``1 \\leq \\alpha_{k} \\leq r_{k}``, ``1 \\leq \\mu_k \\leq n_k``
	* ttv_dims: the dimension of the tensor along each mode
	* ttv_rks: the TT ranks ``(r_0,...,r_d)`` where ``r_0=r_d=1``
	* ttv_ot: the orthogonality of the TT where 
		* ttv_ot[i] = 1 iff ``A_i`` is left-orthogonal *i.e.* ``\\sum_{\\mu_i} A_i[\\mu_i]^T A_i[\\mu_i] = I_{r_i}``
		* ttv_ot[i] = -1 iff ``A_i`` is right-orthogonal *i.e.* ``\\sum_{\\mu_i} A_i[\\mu_i] A_i[\\mu_i]^T = I_{r_{i-1}}``
		* ttv_ot[i] = 0 if nothing is known
"""
abstract type AbstractTTvector end
struct TTvector{T<:Number,M} <: AbstractTTvector
	N :: Int64
	ttv_vec :: Vector{Array{T,3}}
	ttv_dims :: NTuple{M,Int64}
	ttv_rks :: Vector{Int64}
	ttv_ot :: Vector{Int64}
	function TTvector{T,M}(N,vec,dims,rks,ot) where {T,M}
		@assert M isa Int64
		new{T,M}(N,vec,dims,rks,ot)
	end
end

Base.eltype(::TTvector{T,N}) where {T<:Number,N} = T 

"""
Vidal representation of TT vector
"""
struct TT_vidal{T<:Number,M} <: AbstractTTvector
	#Vidal representation of a higher-order tensor 
	#C[μ_1,…,μ_L] = core[μ_1] * Diagonal(Σ_1) * … * Diagonal(Σ_{L-1}) * core[μ_L]
	#Cores are orthogonal and Σ are the higher-order singular values
	N :: Int64
	core :: Array{Array{T,3},1}
	Σ :: Array{Array{Float64,1},1}
	dims :: NTuple{M,Int64}
	rks :: Array{Int64,1}
end

"""	
TT constructor of a matrix ``M[i_1,..,i_d;j_1,..,j_d] \\in \\mathbb{K}^{n_1 \\cdots n_d \\times n_1 \\cdots n_d}`` where 
``M[i_1,..,i_d;j_1,..,j_d] = A_1[i_1,j_1] ... A_L[i_d,j_d]``
The following properties are stored
	* tto_vec: the TT cores A_k as a list of 4-order tensors ``(A_1,...,A_d)`` of dimensions A_k \\in \\mathbb{K}^{n_k × n_k × r_{k-1} × r_k}
	* ttv_dims: the dimension of the tensor along each mode
	* ttv_rks: the TT ranks ``(r_0,...,r_d)`` where ``r_0=r_d=1``
"""
abstract type AbstractTToperator end
struct TToperator{T<:Number,M} <: AbstractTToperator
	N :: Int64
	tto_vec :: Array{Array{T,4},1}
	tto_dims :: NTuple{M,Int64}
	tto_rks :: Array{Int64,1}
	tto_ot :: Array{Int64,1}
	function TToperator{T,M}(N,vec,dims,rks,ot) where {T,M}
		@assert M isa Int64
		new{T,M}(N,vec,dims,rks,ot)
	end
end

Base.eltype(::TToperator{T,M}) where {T,M} = T 
function Base.complex(A::TToperator{T,M}) where {T,M}
	return TToperator{Complex{T},M}(A.N,complex(A.tto_vec),A.tto_dims,A.tto_rks,A.tto_ot)
end

function Base.complex(v::TTvector{T,N}) where {T,N}
	return TTvector{Complex{T},N}(v.N,complex(v.ttv_vec),v.ttv_dims,v.ttv_rks,v.ttv_ot)
end

"""
returns a zero TTvector with dimensions `dims` and ranks `rks`
"""
function zeros_tt(dims,rks;ot=zeros(Int64,length(dims)))
	return zeros_tt(Float64,dims,rks;ot=ot)
end

function zeros_tt(::Type{T},dims::NTuple{N,Int64},rks;ot=zeros(Int64,length(dims))) where {T,N}
	#@assert length(dims)+1==length(rks) "Dimensions and ranks are not compatible"
	tt_vec = [zeros(T,dims[i],rks[i],rks[i+1]) for i in eachindex(dims)]
	return TTvector{T,N}(N,tt_vec,dims,deepcopy(rks),deepcopy(ot))
end

function zeros_tt(n::Integer,d::Integer,r;ot=zeros(Int64,d),r_and_d=true)
	dims = ntuple(x->n,d)
	if r_and_d
		rks = r_and_d_to_rks(r*ones(Int64,d+1),dims)
	else 
		rks = r*ones(Int64,d+1)
		rks[1],rks[end] = 1,1
	end
	return zeros_tt(Float64,dims,rks;ot=ot)
end

"""
returns the ones tensor in TT format
"""
function ones_tt(dims)
	return ones_tt(Float64,dims)
end

function ones_tt(::Type{T},dims) where T
	N = length(dims)
	vec = [ones(T,n,1,1) for n in dims]
	rks = ones(Int64,N+1)
	ot = zeros(Int64,N)
	return TTvector{T,N}(N,vec,dims,rks,ot)
end

function ones_tt(n::Integer,d::Integer)
	dims = ntuple(x->n,d)
	return ones_tt(dims)
end

"""
returns a zero TToperator with dimensions `dims` and ranks `rks`
"""
function zeros_tto(dims,rks)
	return zeros_tto(Float64,dims,rks)
end

function zeros_tto(n,d,r)
	dims = ntuple(x->n,d)
	rks = r*ones(Int64,d+1)
	rks = r_and_d_to_rks(rks,dims.^2;rmax=r)
	return zeros_tto(Float64,dims,rks)
end

function zeros_tto(::Type{T},dims::NTuple{N,Int64},rks)  where {T,N}
	@assert length(dims)+1==length(rks) "Dimensions and ranks are not compatible"
	vec = [zeros(T,dims[i],dims[i],rks[i],rks[i+1]) for i in eachindex(dims)]
	return TToperator{T,N}(N,vec,dims,rks,zeros(Int64,N))
end

#returns partial isometry Q ∈ R^{n x m}
function rand_orthogonal(n,m;T=Float64)
    N = max(n,m)
    q,r = qr(rand(T,N,N))
    return Matrix(q)[1:n,1:m]
end

"""
Returns a random TTvector with dimensions `dims` and ranks `rks`
"""
function rand_tt(dims,rks;normalise=false,orthogonal=false, right=true,stable=false)
	return rand_tt(Float64,dims,rks;normalise=normalise,orthogonal=orthogonal,right=right,stable=stable)
end

function rand_tt(::Type{T},dims,rks;normalise=false,orthogonal=false,right=true,stable=false) where T
	y = zeros_tt(T,dims,rks)
	@simd for i in eachindex(y.ttv_vec)
		y.ttv_vec[i] = randn(T,dims[i],rks[i],rks[i+1])
		if normalise
			if right
				y.ttv_vec[i] *= 1/sqrt(dims[i]*rks[i+1])
				if orthogonal || (stable && rks[i]>=dims[i]*rks[i+1])
					q,_ = qr(reshape(permutedims(y.ttv_vec[i],(1,3,2)),dims[i]*rks[i+1],rks[i]))
					y.ttv_vec[i] = permutedims(reshape(Matrix(q),dims[i],rks[i+1],rks[i]),(1,3,2))
				end
			else
				y.ttv_vec[i] *= 1/sqrt(dims[i]*rks[i])
				if orthogonal || (stable && rks[i+1]>=dims[i]*rks[i])
					q,_ = qr(reshape(y.ttv_vec[i],dims[i]*rks[i],rks[i+1]))
					y.ttv_vec[i] = reshape(Matrix(q),dims[i],rks[i],rks[i+1])
				end
			end
		end
	end
	return y
end

"""
Returns a random TTvector with dimensions `dims` and maximal rank `rmax`
"""
function rand_tt(dims,rmax::Int;normalise=false,orthogonal=false,right=true,stable=false)
	d = length(dims)
	rks = rmax*ones(Int,d+1)
	rks = r_and_d_to_rks(rks,dims;rmax=rmax)
	return rand_tt(dims,rks;normalise=normalise,orthogonal=orthogonal,right=right,stable=stable)
end

function rand_tt(x_tt::TTvector{T,N};ε=convert(T,1e-3)) where {T,N}
	tt_vec = copy(x_tt.ttv_vec)
	for i in eachindex(x_tt.ttv_vec)
		tt_vec[i] += ε*randn(x_tt.ttv_dims[i],x_tt.ttv_rks[i],x_tt.ttv_rks[i+1])
	end
	return TTvector{T,N}(N,tt_vec,x_tt.ttv_dims,x_tt.ttv_rks,zeros(Int,N))
end

function Base.copy(x_tt::TTvector{T,N}) where {T<:Number,N}
	y_tt = zeros_tt(T,x_tt.ttv_dims,x_tt.ttv_rks;ot=x_tt.ttv_ot)
	@threads for i in eachindex(x_tt.ttv_dims)
		y_tt.ttv_vec[i] = copy(x_tt.ttv_vec[i])
	end
	return y_tt
end

"""
TT decomposition by the Hierarchical SVD algorithm 
	* Oseledets, I. V. (2011). Tensor-train decomposition. *SIAM Journal on Scientific Computing*, 33(5), 2295-2317.
	* Schollwöck, U. (2011). The density-matrix renormalization group in the age of matrix product states. *Annals of physics*, 326(1), 96-192.
The *root* of the TT decomposition is at index *i.e.* ``A_i`` for ``i < index`` are left-orthogonal and ``A_i`` for ``i > index`` are right-orthogonal. Singular values lower than tol are discarded.
"""
function ttv_decomp(tensor::Array{T,d};index=1,tol=1e-12) where {T<:Number,d}
	# Decomposes a tensor into its tensor train with core matrices at i=index
	dims = size(tensor) #dims = [n_1,...,n_d]
	ttv_vec = Array{Array{T,3}}(undef,d)
	# ttv_ot[i]= -1 if i < index
	# ttv_ot[i] = 0 if i = index
	# ttv_ot[i] = 1 if i > index
	ttv_ot = -ones(Int64,d)
	ttv_ot[index] = 0
	if index < d
		ttv_ot[index+1:d] = ones(d-index)
	end
	rks = ones(Int64, d+1) 
	tensor_curr = tensor
	# Calculate ttv_vec[i] for i < index
	for i = 1 : (index - 1)
		# Reshape the currently left tensor
		tensor_curr = reshape(tensor_curr, Int(rks[i] * dims[i]), :)
		# Perform the singular value decomposition
		u, s, v = svd(tensor_curr)
		# Define the i-th rank
		rks[i+1] = length(s[s .>= tol])
		# Initialize ttv_vec[i]
		ttv_vec[i] = zeros(T,dims[i],rks[i],rks[i+1])
		# Fill in the ttv_vec[i]
		for x = 1 : dims[i]
			ttv_vec[i][x, :, :] = u[(rks[i]*(x-1) + 1):(rks[i]*x), :]
		end
		# Update the currently left tensor
		tensor_curr = Diagonal(s[1:rks[i+1]])*v'[1:rks[i+1],:]
	end

	# Calculate ttv_vec[i] for i > index
	if index < d
		for i = d : (-1) : (index + 1)
			# Reshape the currently left tensor
			tensor_curr = reshape(tensor_curr, :, dims[i] * rks[i+1])
			# Perform the singular value decomposition
			u, s, v = svd(tensor_curr)
			# Define the (i-1)-th rank
			rks[i]=length(s[s .>= tol])
			# Initialize ttv_vec[i]
			ttv_vec[i] = zeros(T, dims[i], rks[i], rks[i+1])
			# Fill in the ttv_vec[i]
			i_vec = zeros(Int,rks[i+1])
			for x = 1 : dims[i]
				i_vec = dims[i]*((1:rks[i+1])-ones(Int, rks[i+1])) + x*ones(Int,rks[i+1])
				ttv_vec[i][x, :, :] = v'[1:rks[i],i_vec] #(rks[i+1]*(x-1)+1):(rks[i+1]*x)
			end
			# Update the current left tensor
			tensor_curr = u[:,1:rks[i]]*Diagonal(s[1:rks[i]])
		end
	end
	# Calculate ttv_vec[i] for i = index
	# Reshape the current left tensor
	tensor_curr = reshape(tensor_curr, Int(dims[index]*rks[index]),:)
	# Initialize ttv_vec[i]
	ttv_vec[index] = zeros(T, dims[index], rks[index], rks[index+1])
	# Fill in the ttv_vec[i]
	for x = 1 : dims[index]
		ttv_vec[index][x, :, :] =
		tensor_curr[Int(rks[index]*(x-1) + 1):Int(rks[index]*x), 1:rks[index+1]]
	end

	# Define the return value as a TTvector
	return TTvector{T,d}(d,ttv_vec, dims, rks, ttv_ot)
end

"""
Returns the tensor corresponding to x_tt
"""
function ttv_to_tensor(x_tt :: TTvector{T,N}) where {T<:Number,N}
	d = length(x_tt.ttv_dims)
	r_max = maximum(x_tt.ttv_rks)
	# Initialize the to be returned tensor
	tensor = zeros(T, x_tt.ttv_dims)
	# Fill in the tensor for every t=(x_1,...,x_d)
	@simd for t in CartesianIndices(tensor)
		curr = ones(T,r_max)
		a = collect(Tuple(t))
		for i = d:-1:1
			curr[1:x_tt.ttv_rks[i]] = x_tt.ttv_vec[i][a[i],:,:]*curr[1:x_tt.ttv_rks[i+1]]
		end
		tensor[t] = curr[1]
	end
	return tensor
end

#returns the Vidal representation of a TT
function tt_to_vidal(x_tt::TTvector{T,N};tol=1e-14) where {T<:Number,N}
	d = x_tt.N
	core = Array{Array{T,3},1}(undef,d)
	Σ = Array{Array{Float64,1},1}(undef,d-1)
	y_tt = orthogonalize(x_tt)
	y_rks = copy(y_tt.ttv_rks)
	#Definition of the first core
	u,s,v = svd(reshape(y_tt.ttv_vec[1],x_tt.ttv_dims[1],y_tt.ttv_rks[2]))
	Σ[1] = s[s.>tol]
	y_rks[2] = length(Σ[1])
	core[1] = reshape(u[:,s.>tol],y_tt.ttv_dims[1],1,:)
	#Next core to SVD
	@tensor B[i2,α,β] := (Diagonal(s)*v')[α,z]*y_tt.ttv_vec[2][i2,z,β] 
	for j in 2:d-1
		u,s,v = svd(reshape(B,size(B,1)*size(B,2),:))
		Σ[j] = s[s.>tol]
		y_rks[j+1] = length(Σ[j])
		core[j] = reshape(u[:,s.>tol],x_tt.ttv_dims[j],y_rks[j],:)
		for i in 1:x_tt.ttv_dims[j]
			core[j][i,:,:] = inv(Diagonal(Σ[j-1]))*core[j][i,:,:]
		end
		@tensor B[i2,α,β] := (Diagonal(s[s.>tol])*v[:,s.>tol]')[α,z]*y_tt.ttv_vec[j+1][i2,z,β] 
	end
	@tensor core[d][i,α,β] := v'[α,z]*y_tt.ttv_vec[d][i,z,β]
	return TT_vidal{T,N}(d,core,Σ,y_tt.ttv_dims,y_rks)
end

"""
Vidal representation to tensor
"""
function vidal_to_tensor(x_v::TT_vidal{T,N}) where {T<:Number,N}
	d = x_v.N
	r_max = maximum(x_v.rks)
	tensor = zeros(T, x_v.dims)
	# Fill in the tensor for every t=(x_1,...,x_d)
	curr = ones(T,r_max)
	for t in CartesianIndices(tensor)
		curr[1] = one(T)
		a = collect(Tuple(t))
		curr[1:x_v.rks[d]] = copy(x_v.core[d][a[d],:,1]) #last core is a column vector
		for i = d-1:-1:1
			curr[1:x_v.rks[i]] = x_v.core[i][a[i],:,:]*(x_v.Σ[i].*curr[1:x_v.rks[i+1]])
		end
		tensor[t] = curr[1]
	end
	return tensor
end

"""
Returns a left-canonical TT representation
"""
function vidal_to_left_canonical(x_v::TT_vidal{T,N}) where {T<:Number,N}
	x_tt = zeros_tt(T,x_v.dims,x_v.rks,ot=vcat(ones(length(x_v.dims)), 0))
	x_tt.ttv_vec[1] = x_v.core[1]
	for i in 2:length(x_v.dims)
		for j in 1:x_v.dims[i]
			x_tt.ttv_vec[i][j,:,:] = Diagonal(x_v.Σ[i-1])*x_v.core[i][j,:,:]
		end
	end
	return x_tt
end

"""
Transforms a TToperator into a TTvector
"""
function tto_to_ttv(A::TToperator{T,N}) where {T<:Number,N}
	d = A.N
	xtt_vec = Array{Array{T,3},1}(undef,d)
	A_rks = A.tto_rks
	for i in eachindex(xtt_vec)
		xtt_vec[i] = reshape(A.tto_vec[i],A.tto_dims[i]^2,A_rks[i],A_rks[i+1])
	end
	return TTvector{T,N}(d,xtt_vec,A.tto_dims.^2,A.tto_rks,A.tto_ot)
end

"""
Transforms a TTvector (coming from a TToperator) into a TToperator
"""
function ttv_to_tto(x::TTvector{T,N}) where {T<:Number,N}
	@assert(isqrt.(x.ttv_dims).^2 == x.ttv_dims, DimensionMismatch)
	d = x.N
	Att_vec = Array{Array{T,4},1}(undef,d)
	x_rks = x.ttv_rks
	A_dims = isqrt.(x.ttv_dims)
	for i in eachindex(A_dims)
		Att_vec[i] = reshape(x.ttv_vec[i],A_dims[i],A_dims[i],x_rks[i],x_rks[i+1])
	end
	return TToperator{T,N}(d,Att_vec,A_dims,x.ttv_rks,x.ttv_ot)
end

"""
Returns the TT decomposition of a matrix using the HSVD algorithm
"""
function tto_decomp(tensor::Array{T,N}; index=1) where {T<:Number,N}
	# Decomposes a tensor operator into its tensor train
	# with core matrices at i=index
	# The tensor is given as tensor[x_1,...,x_d,y_1,...,y_d]
	d = Int(ndims(tensor)/2)
	tto_dims = size(tensor)[1:d]
	dims_sq = tto_dims.^2
	# The tensor is reorder  into tensor[x_1,y_1,...,x_d,y_d],
	# reshaped into tensor[(x_1,y_1),...,(x_d,y_d)]
	# and decomposed into its tensor train with core matrices at i= index
	index_sorted = Transpose(reshape(1:(2*d),:,2))[:]
	ttv = ttv_decomp(reshape(permutedims(tensor,index_sorted),(dims_sq[1:(end-1)]...), :); index=index)
	# Define the array of ranks [r_0=1,r_1,...,r_d]
	rks = ttv.ttv_rks
	# Initialize tto_vec
	tto_vec = Array{Array{T}}(undef,d)
	# Fill in tto_vec
	for i = 1:d
		# Initialize tto_vec[i]
		tto_vec[i] = zeros(T, tto_dims[i], tto_dims[i], rks[i], rks[i+1])
		# Fill in tto_vec[i]
		tto_vec[i] = reshape(ttv.ttv_vec[i], tto_dims[i], tto_dims[i], :, rks[i+1])
	end
	return TToperator{T,d}(d,tto_vec, tto_dims, rks, ttv.ttv_ot)
end

function tto_to_tensor(tto :: TToperator{T,N}) where {T<:Number,N}
	d = tto.N
	# Define the array of ranks [r_0=1,r_1,...,r_d]
	rks = tto.tto_rks
	r_max = maximum(rks)
	# The tensor has dimensions [n_1,...,n_d,n_1,...,n_d]
	tensor = zeros(T,(tto.tto_dims...,tto.tto_dims...))
	# Fill in the tensor for every t=(x_1,...,x_d,y_1,...,y_d)
	curr = ones(T,r_max)
	@simd for t in CartesianIndices(tensor)
		curr[1] = one(T)
		for i = d:-1:1
			curr[1:rks[i]] = tto.tto_vec[i][t[i], t[d + i], :, :]*curr[1:rks[i+1]]
		end
		tensor[t] = curr[1]
	end
	return tensor
end

function occ_to_μ(x::Vector{Int},L::Int)
	out = ones(Int64,L)
	for i in x
		out[i] = 2
	end
	out
end

function mpo_Nparticle_to_matrix(A::TToperator{T,L},N) where {T,L}
	rmax = maximum(A.tto_rks)
	temp = ones(T,rmax)
	mat = zeros(T,binomial(L,N),binomial(L,N))
	occ_list = collect(combinations(1:L,N))
	@simd for j in axes(mat,2)
		for i in axes(mat,1)
			temp[1] = one(T)
			μ_row = occ_to_μ(occ_list[i],L)
			μ_col = occ_to_μ(occ_list[j],L)
			for k in L:-1:1
				temp[1:A.tto_rks[k]] = A.tto_vec[k][μ_row[k],μ_col[k],:,:]*temp[1:A.tto_rks[k+1]]
			end
			mat[i,j] = temp[1]
		end
	end
	return mat
end

#TTO representation of the identity matrix
function id_tto(d;n_dim=2)
	return id_tto(Float64,d;n_dim=n_dim)
end

function id_tto(::Type{T},d;n_dim=2) where {T}
	dims = Tuple(n_dim*ones(Int64,d))
	A = Array{Array{T,4},1}(undef,d)
	for j in 1:d
		A[j] = zeros(T,2,2,1,1)
		A[j][:,:,1,1] = Matrix{T}(I,2,2)
	end
	return TToperator{T,d}(d,A,dims,ones(Int64,d+1),zeros(d))
end

function rand_tto(dims,rmax::Int;T=Float64)
	d = length(dims)
	tt_vec = Vector{Array{T,4}}(undef,d)
	rks = ones(Int,d+1)
	for i in eachindex(tt_vec) 
		ri = min(prod(dims[1:i-1]),prod(dims[i:d]),rmax)
		rip = min(prod(dims[1:i]),prod(dims[i+1:d]),rmax)
		rks[i+1] = rip
		tt_vec[i] = randn(T,dims[i],dims[i],ri,rip)
	end
	return TToperator{T,d}(d,tt_vec,dims,rks,zeros(Int,d))
end


function json_to_mps(x)
	dims = Tuple(convert(Vector{Int64},x[:ttv_dims]))
	rks = convert(Vector{Int64},x[:ttv_rks])
	ot = convert(Vector{Int64},x[:ttv_ot])
	vec = Vector{Array{eltype(x[:ttv_vec][1]),3}}(undef,x[:N])
	for i in eachindex(vec)
		vec[i] = reshape(convert(Vector{eltype(eltype(vec))}, (x[:ttv_vec])[i]),dims[i],rks[i],rks[i+1])
	end
	return TTvector{eltype(eltype(vec)),x[:N]}(x[:N],vec,dims,rks,ot)
end

function json_to_mpo(x)
	dims = Tuple(convert(Vector{Int64},x[:tto_dims]))
	rks = convert(Vector{Int64},x[:tto_rks])
	ot = convert(Vector{Int64},x[:tto_ot])
	vec = Vector{Array{eltype(x[:tto_vec][1]),4}}(undef,x[:N])
	for i in eachindex(vec)
		vec[i] = reshape(convert(Vector{eltype(eltype(vec))}, (x[:tto_vec])[i]),dims[i],dims[i],rks[i],rks[i+1])
	end
	return TToperator{eltype(eltype(vec)),x[:N]}(x[:N],vec,dims,rks,ot)
end