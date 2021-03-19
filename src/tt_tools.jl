using Random
using LinearAlgebra
using Base.Threads
using IterativeSolvers
using TensorOperations
import Base.isempty
import Base.eltype

"""
TT constructor of a tensor
``C[\\mu_1,…,\\mu_d] = A_1[1][\\mu_1]*⋯*A_d[d][\\mu_d] \\in \\mathbb{K}^{n_1 \\times ... \\times n_d}``
The following properties are stored
	* ttv_vec: the TT cores A_k as a list of 3-order tensors ``(A_1,...,A_d)`` where ``A_k = A_k[\\mu_k,\\alpha_{k-1},\\alpha_k]``, ``1 \\leq \\alpha_{k-1} \\leq r_{k-1}``, ``1 \\leq \\alpha_{k} \\leq r_{k}``, ``1 \\leq \\mu_k \\leq n_k``
	* ttv_dims: the dimension of the tensor along each mode
	* ttv_rks: the TT ranks ``(r_0,...,r_d)`` where ``r_0=r_d=1``
	* ttv_ot: the orthogonality of the TT where 
		* ttv_ot[i] = 1 iff ``A_i`` is right-orthogonal *i.e.* ``\\sum_{\\mu_i} A_i[\\mu_i]^T A_i[\\mu_i] = I_{r_i}``
		* ttv_ot[i] = -1 iff ``A_i`` is left-orthogonal *i.e.* ``\\sum_{\\mu_i} A_i[\\mu_i] A_i[\\mu_i]^T = I_{r_{i-1}}``
		* ttv_ot[i] = 0 if nothing is known
"""
struct ttvector{T<:Number}
	ttv_vec :: Array{Array{T,3},1}
	ttv_dims :: Array{Int64,1}
	ttv_rks :: Array{Int64,1}
	ttv_ot :: Array{Int64,1}
end

Base.eltype(::ttvector{T}) where T<:Number = T 
"""
Vidal representation of TT vector
"""
struct tt_vidal{T<:Number}
	#Vidal representation of a higher-order tensor 
	#C[μ_1,…,μ_L] = core[μ_1] * Diagonal(Σ_1) * … * Diagonal(Σ_{L-1}) * core[μ_L]
	#Cores are orthogonal and Σ are the higher-order singular values
	core :: Array{Array{T,3},1}
	Σ :: Array{Array{Float64,1},1}
	dims :: Array{Int64,1}
	rks :: Array{Int64,1}
end

"""	
TT constructor of a matrix ``M[i_1,..,i_d;j_1,..,j_d] \\in \\mathbb{K}^{n_1 \\cdots n_d \\times n_1 \\cdots n_d}`` where 
``MM[i_1,..,i_d;j_1,..,j_d] = A_1[i_1,j_1] ... A_L[i_d,j_d]``
The following properties are stored
	* tto_vec: the TT cores A_k as a list of 4-order tensors ``(A_1,...,A_d)``
	* ttv_dims: the dimension of the tensor along each mode
	* ttv_rks: the TT ranks ``(r_0,...,r_d)`` where ``r_0=r_d=1``
"""
struct ttoperator{T<:Number}
	tto_vec :: Array{Array{T,4},1}
	tto_dims :: Array{Int64,1}
	tto_rks :: Array{Int64,1}
	tto_ot :: Array{Int64,1}
end

Base.eltype(::ttoperator{T}) where T<:Number = T 

function empty_tt()
	return ttvector([],[],[],[])
end

function Base.isempty(x_tt::ttvector)
	return isempty(x_tt.ttv_vec)
end

"""
TT decomposition by the Hierarchical SVD algorithm 
	* Oseledets, I. V. (2011). Tensor-train decomposition. *SIAM Journal on Scientific Computing*, 33(5), 2295-2317.
	* Schollwöck, U. (2011). The density-matrix renormalization group in the age of matrix product states. *Annals of physics*, 326(1), 96-192.
The *root* of the TT decomposition is at index *i.e.* ``A_i`` for ``i < index`` are left-orthogonal and ``A_i`` for ``i > index`` are right-orthogonal. Singular values lower than tol are discarded.
"""
function ttv_decomp(tensor::Array{T};index=1,tol=1e-12) where T<:Number
	# Decomposes a tensor into its tensor train with core matrices at i=index
	dims = collect(size(tensor)) #dims = [n_1,...,n_d]
	n_max = maximum(dims)
	d = length(dims)
	ttv_vec = Array{Array{T}}(undef,d)
	# ttv_ot[i]= -1 if i < index
	# ttv_ot[i] = 0 if i = index
	# ttv_ot[i] = 1 if i > index
	ttv_ot = -ones(Int64,d)
	ttv_ot[index] = 0
	if index < d
		ttv_ot[index+1:d] = ones(d-index)
	end
	rks = ones(Int, d+1) 
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
		tensor_curr = Diagonal(s[1:rks[i+1]])*v[:,1:rks[i+1]]'
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
				ttv_vec[i][x, :, :] = v[i_vec, 1:rks[i]]' #(rks[i+1]*(x-1)+1):(rks[i+1]*x)
			end
			# Update the currently left tensor
			tensor_curr = u[:,1:rks[i]]*Diagonal(s[1:rks[i]])
		end
	end
	# Calculate ttv_vec[i] for i = index
	# Reshape the currently left tensor
	tensor_curr = reshape(tensor_curr, Int(dims[index]*rks[index]),:)
	# Initialize ttv_vec[i]
	ttv_vec[index] = zeros(T, dims[index], rks[index], rks[index+1])
	# Fill in the ttv_vec[i]
	for x = 1 : dims[index]
		ttv_vec[index][x, :, :] =
		tensor_curr[Int(rks[index]*(x-1) + 1):Int(rks[index]*x), 1:rks[index+1]]
	end

	# Define the return value as a ttvector
	return ttvector{T}(ttv_vec, dims, rks, ttv_ot)
end

"""
Returns the tensor corresponding to x_tt
"""
function ttv_to_tensor(x_tt :: ttvector{T}) where T<: Number
	d = length(x_tt.ttv_dims)
	r_max = maximum(x_tt.ttv_rks)
	# Initialize the to be returned tensor
	tensor = zeros(T, x_tt.ttv_dims...)
	# Fill in the tensor for every t=(x_1,...,x_d)
	for t in CartesianIndices(tensor)
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
function tt_to_vidal(x_tt::ttvector{T};tol=1e-14) where T<:Number
	d = length(x_tt.ttv_dims)
	core = Array{Array{T,3},1}(undef,d)
	Σ = Array{Array{Float64,1},1}(undef,d-1)
	y_tt = orthogonalize(x_tt,1)
	y_rks = vcat(1,y_tt.ttv_rks)
	for j in 1:d-1
		A = zeros(y_tt.ttv_dims[j],y_rks[j],y_tt.ttv_dims[j+1],y_rks[j+2])
		@tensor A[a,b,c,d] = y_tt.ttv_vec[j][a,z,b]*y_tt.ttv_vec[j+1][z,d,c]
		u,s,v = svd(reshape(A,size(A,1)*size(A,2),:))
		Σ[j] = s[s.>tol]
		y_rks[j+1] = length(Σ[j])
		core[j] = permutedims(reshape(u[:,s.>tol],y_rks[j],y_tt.ttv_dims[j],:),[1 3 2])
		y_tt.ttv_vec[j+1] = reshape(v[:,s.>tol],:,y_rks[j+2],y_tt.ttv_dims[j+1])
	end
	core[d] = y_tt.ttv_vec[d]
	return tt_vidal(core,Σ,y_tt.ttv_dims,y_rks)
end

"""
Transforms a ttoperator into a ttvector
"""
function tto_to_ttv(A::ttoperator{T}) where T<:Number
	d = length(A.tto_dims)
	xtt_vec = Array{Array{T,3},1}(undef,d)
	A_rks = A.tto_rks
	for i in 1:d
		xtt_vec[i] = reshape(A.tto_vec[i],A.tto_dims[i]^2,A_rks[i],A_rks[i+1])
	end
	return ttvector{T}(xtt_vec,A.tto_dims.^2,A.tto_rks,A.tto_ot)
end

"""
Transforms a ttvector (coming from a ttoperator) into a ttoperator
"""
function ttv_to_tto(x::ttvector{T}) where T<:Number
	d = length(x.ttv_dims)
	@assert(isqrt.(x.ttv_dims).^2 == x.ttv_dims, DimensionMismatch)
	Att_vec = Array{Array{T,4},1}(undef,d)
	x_rks = x.ttv_rks
	A_dims = zeros(Int64,d)
	for i in 1:d
		A_dims[i] = isqrt(x.ttv_dims[i])
		Att_vec[i] = reshape(x.ttv_vec[i],A_dims[i],A_dims[i],x_rks[i],x_rks[i+1])
	end
	return ttoperator{T}(Att_vec,A_dims,x.ttv_rks,x.ttv_ot)
end

"""
Returns the TT decomposition of a matrix using the HSVD algorithm
"""
function tto_decomp(tensor::Array{T}; index=1) where T<:Number
	# Decomposes a tensor operator into its tensor train
	# with core matrices at i=index
	# The tensor is given as tensor[x_1,...,x_d,y_1,...,y_d]
	d = Int(ndims(tensor)/2)
	tto_dims = collect(size(tensor))[1:d]
	n_max = maximum(tto_dims)
	dims_sq = tto_dims.^2
	# The tensor is reorder  into tensor[x_1,y_1,...,x_d,y_d],
	# reshaped into tensor[(x_1,y_1),...,(x_d,y_d)]
	# and decomposed into its tensor train with core matrices at i= index
	index_sorted = reshape(Transpose(reshape(1:(2*d),:,2)),1,:)
	ttv = ttv_decomp(reshape(permutedims(tensor,index_sorted),(dims_sq[1:(end-1)]...), :); index=index)
	# Define the array of ranks [r_0=1,r_1,...,r_d]
	rks = ttv.ttv_rks
	r_max = maximum(rks)
	# Initialize tto_vec
	tto_vec = Array{Array{T}}(undef,d)
	# Fill in tto_vec
	for i = 1:d
		# Initialize tto_vec[i]
		tto_vec[i] = zeros(T, tto_dims[i], tto_dims[i], rks[i], rks[i+1])
		# Fill in tto_vec[i]
		tto_vec[i] = reshape(ttv.ttv_vec[i], tto_dims[i], tto_dims[i], :, rks[i+1])
	end
	return ttoperator{T}(tto_vec, tto_dims, rks, ttv.ttv_ot)
end

function tto_to_tensor(tto :: ttoperator{T}) where T<:Number
	d = length(tto.tto_dims)
	# Define the array of ranks [r_0=1,r_1,...,r_d]
	rks = tto.tto_rks
	r_max = maximum(rks)
	# The tensor has dimensions [n_1,...,n_d,n_1,...,n_d]
	dims = zeros(Int, 2*d)
	dims[1:d] = tto.tto_dims
	dims[(d+1):(2*d)] = tto.tto_dims
	tensor = zeros(T,dims...)
	# Fill in the tensor for every t=(x_1,...,x_d,y_1,...,y_d)
	for t in CartesianIndices(tensor)
		curr = ones(T,r_max)
		a = collect(Tuple(t))
		for i = d:-1:1
			curr[1:rks[i]] = tto.tto_vec[i][t[i], t[d + i], :, :]*curr[1:rks[i+1]]
		end
		tensor[t] = curr[1]
	end
	return tensor
end

#TTO representation of the identity matrix
function id_tto(d;n_dim=2,T=Float64 <:Number)
	dims = n_dim*ones(Int64,d)
	A = Array{Array{T,4},1}(undef,d)
	for j in 1:d
		A[j] = zeros(2,2,1,1)
		A[j][:,:,1,1] = Matrix{T}(I,2,2)
	end
	return ttoperator{T}(A,dims,ones(Int64,d),zeros(d))
end
