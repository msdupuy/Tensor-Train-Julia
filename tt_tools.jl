using Test
using Random
using LinearAlgebra
using Base.Threads
using IterativeSolvers
using TensorOperations
import Base.isempty

#TT constructor of a tensor
#C[μ_1,…,μ_L] = C_tt.ttv_vec[1]*⋯*C_tt.ttv_vec[L]
#the normalization of the TT cores are kept in ttv_ot

struct ttvector
	# ttv_vec is an array of all matrix arrays for the tensor train format
	# ttv_vec[i] stores the matrices A_i in the following way
	# ttv_vec[i][1:x_(n_i), 1:r_(i-1), 1:r_i]
	ttv_vec :: Array{Array{Float64,3},1}

	# ttv_dims is an vector of the dimensions n_i i=1,...,d
	ttv_dims :: Array{Int64,1}

	# ttv_rks is the array of all ranks of the tensor train matrices
	# r_i i=1,...,d
	ttv_rks :: Array{Int64,1}

	# ttv_ot includes information about the orthogonality properties
	# ttv_ot[i] =  1 if the i-th matrices are rightorthogonal
	# ttv_ot[i] =  0 if there is no information or the i-th matrices aren't orthogonal
	# ttv_ot[i] = -1 if the i-th matrices are leftorthogonal
	ttv_ot :: Array{Int64,1}
end

struct tt_vidal
	#Vidal representation of a higher-order tensor 
	#C[μ_1,…,μ_L] = core[μ_1] * Diagonal(Σ_1) * … * Diagonal(Σ_{L-1}) * core[μ_L]
	#Cores are orthogonal and Σ are the higher-order singular values
	core :: Array{Array{Float64,3},1}
	Σ :: Array{Array{Float64,1},1}
	dims :: Array{Int64,1}
	rks :: Array{Int64,1}
end

struct ttoperator
	# tto_vec is an array of all matrices of the tensor train format
	# of an operator A(x1,...,xd,y1,...,yd)
	# tto_vec[i] stores the matrices A_i i=1,...,d in the following way
	# ttv_vec[i][1:x_(n_i), 1:y_(n_i), 1:r_(i-1), 1:r_i]
	tto_vec :: Array{Array{Float64,4},1}

	# tto_dims stores the dimensions n_i i=1,...,d
	tto_dims :: Array{Int64,1}

	# tto_rks is the array of all ranks of the tensor train matrices
	# r_i i=1,...,d
	tto_rks :: Array{Int64,1}

	# tto_ot is a matrix storing information about the orthoganlity properties
	# tto_ot[i] =  1 if the i-th matrices are rightorthogonal
	# tto_ot[i] =  0 if there is no information or the i-th matrices aren't orthogonal
	# tto_ot[i] = -1 if the i-th matrices are leftorthogonal
	tto_ot :: Array{Int64,1}
end

function empty_tt()
	return ttvector([],[],[],[])
end

function Base.isempty(x_tt::ttvector)
	return isempty(x_tt.ttv_vec)
end

function ttv_decomp(tensor::Array{Float64}, index;tol=1e-12)
	# Decomposes a tensor into its tensor train with core matrices at i=index
	dims = collect(size(tensor)) #dims = [n_1,...,n_d]
	n_max = maximum(dims)
	d = length(dims)
	ttv_vec = Array{Array{Float64}}(undef,d)
	# ttv_ot[i]= -1 if i < index
	# ttv_ot[i] = 0 if i = index
	# ttv_ot[i] = 1 if i > index
	ttv_ot = -ones(Int64,d)
	ttv_ot[index] = 0
	if index < d
		ttv_ot[index+1:d] = ones(d-index)
	end
	rks = ones(Int, d+2) # ttv_rks will be rks[2:d+1]
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
		ttv_vec[i] = zeros(dims[i],rks[i],rks[i+1])
		# Fill in the ttv_vec[i]
		for x = 1 : dims[i]
			ttv_vec[i][x, :, :] = u[(rks[i]*(x-1) + 1):(rks[i]*x), :]
		end
		# Update the currently left tensor
		tensor_curr = Diagonal(s[1:rks[i+1]])*Transpose(v[:,1:rks[i+1]])
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
			ttv_vec[i] = zeros(dims[i], rks[i], rks[i+1])
			# Fill in the ttv_vec[i]
			i_vec = zeros(Int,rks[i+1])
			for x = 1 : dims[i]
				i_vec = dims[i]*((1:rks[i+1])-ones(Int, rks[i+1])) + x*ones(Int,rks[i+1])
				ttv_vec[i][x, :, :] = Transpose(v[i_vec, 1:rks[i]]) #(rks[i+1]*(x-1)+1):(rks[i+1]*x)
			end
			# Update the currently left tensor
			tensor_curr = u[:,1:rks[i]]*Diagonal(s[1:rks[i]])
		end
	end
	# Calculate ttv_vec[i] for i = index
	# Reshape the currently left tensor
	tensor_curr = reshape(tensor_curr, Int(dims[index]*rks[index]),:)
	# Initialize ttv_vec[i]
	ttv_vec[index] = zeros(dims[index], rks[index], rks[index+1])
	# Fill in the ttv_vec[i]
	for x = 1 : dims[index]
		ttv_vec[index][x, :, :] =
		tensor_curr[Int(rks[index]*(x-1) + 1):Int(rks[index]*x), 1:rks[index+1]]
	end

	# Define the return value as a ttvector
	return ttvector(ttv_vec, dims, rks[2:d+1], ttv_ot)
end

function ttv_to_tensor(ttv :: ttvector)
	d = length(ttv.ttv_dims)
	# Define the array of ranks [r_0=1,r_1,...,r_d]
	rks = ones(Int,d+1)
	rks[2:(d+1)] = ttv.ttv_rks
	r_max = maximum(rks)
	# Initialize the to be returned tensor
	tensor = zeros(ttv.ttv_dims...)
	# Fill in the tensor for every t=(x_1,...,x_d)
	for t in CartesianIndices(tensor)
		curr = ones(1,r_max)
		a = collect(Tuple(t))
		for i = 1:d
			curr[1,1:rks[i+1]]=reshape(curr[1,1:rks[i]],1,:)*ttv.ttv_vec[i][a[i],1:rks[i],1:rks[i+1]]
		end
		tensor[t] = curr[1,1]
	end
	return tensor
end

#returns partial isometry Q ∈ R^{n x m}
function rand_orthogonal(n,m)
    N = max(n,m)
    q,r = qr(rand(N,N))
    return q[1:n,1:m]
end

#local ttvec rank increase function with noise ϵ_wn
function tt_up_rks_noise(tt_vec,rkm,rk,ϵ_wn)
	vec_out = zeros(Float64,size(tt_vec,1),rkm,rk)
	vec_out[:,1:size(tt_vec,2),1:size(tt_vec,3)] = tt_vec
	if !iszero(ϵ_wn)
		if rkm == size(tt_vec,2) && rk>size(tt_vec,3)
			Q = rand_orthogonal(size(tt_vec,1)*rkm,rk-size(tt_vec,3))
			vec_out[:,:,size(tt_vec,3)+1:rk] = ϵ_wn*reshape(Q,size(tt_vec,1),rkm,rk-size(tt_vec,3))
			tt_ot_i =0
		elseif rk == size(tt_vec,3) && rkm>size(tt_vec,2)
			Q = rand_orthogonal(size(tt_vec,1)*rk,rkm-size(tt_vec,2))
			vec_out[:,size(tt_vec,2)+1:rkm,:] = ϵ_wn*permutedims(reshape(Q,rk,size(tt_vec,1),rkm-size(tt_vec,2)),[2 3 1])
			tt_ot_i =0
		elseif rk>size(tt_vec,3) && rkm>size(tt_vec,2)
			if tt_ot_i == -1 #leftorthogonal
				Q = rand_orthogonal(size(tt_vec,1)*(rkm-size(tt_vec,2)),rk-size(tt_vec,3))
				vec_out[:,size(tt_vec,2)+1:rkm,size(tt_vec,3)+1:rk] = ϵ_wn*reshape(Q,size(tt_vec,1),rkm-size(tt_vec,2),rk-size(tt_vec,3))
			else #tt_ot_i =1 or 0
				Q = rand_orthogonal(rkm-size(tt_vec,2),size(tt_vec,1)*(rk-size(tt_vec,3)))
				vec_out[:,size(tt_vec,2)+1:rkm,size(tt_vec,3)+1:rk] = ϵ_wn*permutedims(reshape(Q,rkm-size(tt_vec,2),size(tt_vec,1),rk-size(tt_vec,3)),[2,1,3])
			end
		end
	end
	return vec_out
end

#returns the ttvector with ranks rks and noise ϵ_wn for the updated ranks
function tt_up_rks(x_tt::ttvector,rk_max::Int;rks=vcat(1,rk_max*ones(Int,length(x_tt.ttv_dims)-1),1),ϵ_wn=0.0)
	d = length(x_tt.ttv_dims)
	vec_out = Array{Array{Float64}}(undef,d)
	out_ot = zeros(Int64,d)
	@assert(rk_max >= maximum(x_tt.ttv_rks),"New bond dimension too low")
	n_in = 1
	n_out = prod(x_tt.ttv_dims)
	for i in 1:d
		n_in *= x_tt.ttv_dims[i]
		n_out = Int(n_out/x_tt.ttv_dims[i])
		rks[i+1] = min(rks[i+1],n_in,n_out)
		vec_out[i] = tt_up_rks_noise(x_tt.ttv_vec[i],rks[i],rks[i+1],ϵ_wn)
	end	
	return ttvector(vec_out,x_tt.ttv_dims,rks[2:end],x_tt.ttv_ot)
end

function test_tt_up_rks()
	x = randn(4,4,4,4)
	x_tt = ttv_decomp(x,2,tol=0.1)
	x = ttv_to_tensor(x_tt)
	y_tt = tt_up_rks(x_tt,20)
	@test isapprox(x,ttv_to_tensor(y_tt),atol=1e-10)
	z_tt = tt_up_rks(x_tt,20,ϵ_wn=1e-10)
	@test isapprox(x,ttv_to_tensor(z_tt),atol=1e-6)
end

#returns the orthogonalized ttvector with root i
function tt_orthogonalize(x_tt::ttvector,i::Integer)
	d = length(x_tt.ttv_dims)
	x_rks = vcat(1,x_tt.ttv_rks)
	@assert(1≤i≤d, DimensionMismatch("Impossible orthogonalization"))
	y_vec = deepcopy(x_tt.ttv_vec)
	y_ot = zeros(Int64,d)
	for j in 1:i-1
		y_ot[j]=-1
		y_vectemp = reshape(y_vec[j],x_tt.ttv_dims[j]*x_rks[j],x_rks[j+1])
		q,r = qr(y_vectemp)
		y_vec[j] = reshape(q[:,1:x_rks[j+1]],x_tt.ttv_dims[j],x_rks[j],x_rks[j+1])
		@threads for k in 1:x_tt.ttv_dims[j]
			y_vec[j+1][k,:,:] = r[1:x_rks[j+1],1:x_rks[j+1]]*y_vec[j+1][k,:,:]
		end
	end
	for j in d:-1:i+1
		y_ot[j]=1
		y_vectemp = reshape(permutedims(y_vec[j],[2,1,3]),x_rks[j],x_tt.ttv_dims[j]*x_rks[j+1])
		l,q = lq(y_vectemp)
		y_vec[j] = permutedims(reshape(q[1:x_rks[j],:],x_rks[j],x_tt.ttv_dims[j],x_rks[j+1]),[2 1 3])
		@threads for k in 1:x_tt.ttv_dims[j]
			y_vec[j-1][k,:,:] = y_vec[j-1][k,:,:]*l[1:x_rks[j],1:x_rks[j]]
		end
	end
	return ttvector(y_vec,x_tt.ttv_dims,x_tt.ttv_rks,y_ot)
end

#returns the Vidal representation of a TT
function tt_to_vidal(x_tt::ttvector;tol=1e-14)
	d = length(x_tt.ttv_dims)
	core = Array{Array{Float64,3},1}(undef,d)
	Σ = Array{Array{Float64,1},1}(undef,d-1)
	y_tt = tt_orthogonalize(x_tt,1)
	y_rks = vcat(1,y_tt.ttv_rks)
	for j in 1:d-1
		A = zeros(y_tt.ttv_dims[j],y_rks[j],y_tt.ttv_dims[j+1],y_rks[j+2])
		@tensor A[a,b,c,d] = y_tt.ttv_vec[j][a,b,z]*y_tt.ttv_vec[j+1][c,z,d]
		u,s,v = svd(reshape(A,size(A,1)*size(A,2),:))
		Σ[j] = s[s.>tol]
		y_rks[j+1] = length(Σ[j])
		core[j] = reshape(u[:,s.>tol],y_tt.ttv_dims[j],y_rks[j],:)
		y_tt.ttv_vec[j+1] = permutedims(reshape(v[:,s.>tol],y_tt.ttv_dims[j+1],y_rks[j+2],:),[1 3 2])
	end
	core[d] = y_tt.ttv_vec[d]
	return tt_vidal(core,Σ,y_tt.ttv_dims,y_rks[2:end])
end

#returns a TT representation where the Vidal SVD lower than tol are discarded
function tt_rounding(x_tt::ttvector;tol=1e-14)
	d = length(x_tt.ttv_dims)
	y_rks = vcat(1,x_tt.ttv_rks)
	y_vec = deepcopy(x_tt.ttv_vec)
	for j in 1:d-1
		A = zeros(x_tt.ttv_dims[j],y_rks[j],x_tt.ttv_dims[j+1],y_rks[j+2])
		@tensor A[a,b,c,d] = y_vec[j][a,b,z]*y_vec[j+1][c,z,d]
		u,s,v = svd(reshape(A,size(A,1)*size(A,2),:))
		Σ = s[s.>tol]
		y_rks[j+1] = length(Σ)
		y_vec[j] = reshape(u[:,s.>tol],x_tt.ttv_dims[j],y_rks[j],:)
		y_vec[j+1] = permutedims(reshape(v[:,s.>tol]*Diagonal(Σ),x_tt.ttv_dims[j+1],y_rks[j+2],:),[1 3 2])
	end
	return ttvector(y_vec,x_tt.ttv_dims,y_rks[2:end],vcat(-ones(Int64,d-1),0))
end

#returns the singular values of the reshaped tensor x[μ_1⋯μ_k;μ_{k+1}⋯μ_d]
function tt_svdvals(x_tt::ttvector;tol=1e-14)
	d = length(x_tt.ttv_dims)
	Σ = Array{Array{Float64,1},1}(undef,d-1)
	y_tt = tt_orthogonalize(x_tt,1)
	y_rks = vcat(1,y_tt.ttv_rks)
	for j in 1:d-1
		A = zeros(y_tt.ttv_dims[j],y_rks[j],y_tt.ttv_dims[j+1],y_rks[j+2])
		@tensor A[a,b,c,d] = y_tt.ttv_vec[j][a,b,z]*y_tt.ttv_vec[j+1][c,z,d]
		u,s,v = svd(reshape(A,size(A,1)*size(A,2),:))
		Σ[j] = s[s.>tol]
		y_rks[j+1] = length(Σ[j])
		y_tt.ttv_vec[j+1] = permutedims(reshape(v[:,s.>tol]*Diagonal(Σ[j]),y_tt.ttv_dims[j+1],y_rks[j+2],:),[1 3 2])
	end
	return Σ
end

function tto_decomp(tensor::Array{Float64}, index)
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
	ttv = ttv_decomp(reshape(permutedims(tensor,index_sorted),(dims_sq[1:(end-1)]...), :), index)
	# Define the array of ranks [r_0=1,r_1,...,r_d]
	rks = ones(Int, d+1)
	rks[2:(d+1)] = ttv.ttv_rks
	r_max = maximum(rks)
	# Initialize tto_vec
	tto_vec = Array{Array{Float64}}(undef,d)
	# Fill in tto_vec
	for i = 1:d
		# Initialize tto_vec[i]
		tto_vec[i] = zeros(tto_dims[i], tto_dims[i], rks[i], rks[i+1])
		# Fill in tto_vec[i]
		tto_vec[i][:, :, :, :] =
			reshape(ttv.ttv_vec[i][1:dims_sq[i], 1:rks[i], 1:rks[i+1]],
						tto_dims[i], tto_dims[i], rks[i], :)
	end
	# Define the return value as a ttoperator
	return ttoperator(tto_vec, tto_dims, rks[2:(d+1)], ttv.ttv_ot)
end

function tto_to_tensor(tto :: ttoperator)
	d = length(tto.tto_dims)
	# Define the array of ranks [r_0=1,r_1,...,r_d]
	rks = ones(Int,d+1)
	rks[2:(d+1)] = tto.tto_rks
	r_max = maximum(rks)
	# The tensor has dimensions [n_1,...,n_d,n_1,...,n_d]
	dims = zeros(Int, 2*d)
	dims[1:d] = tto.tto_dims
	dims[(d+1):(2*d)] = tto.tto_dims
	tensor = zeros(dims...)
	# Fill in the tensor for every t=(x_1,...,x_d,y_1,...,y_d)
	for t in CartesianIndices(tensor)
		curr = ones(1,r_max)
		a = collect(Tuple(t))
		for i = 1:d
			curr[1,1:rks[i+1]] = reshape(curr[1,1:rks[i]], 1, :) *
				tto.tto_vec[i][t[i], t[d + i], 1:rks[i], 1:rks[i+1]]
		end
		tensor[t] = curr[1,1]
	end
	return tensor
end


"""
basic functions for TT format
"""

function tt_add(x::ttvector,y::ttvector)
    @assert(x.ttv_dims == y.ttv_dims, "Dimensions mismatch!")
    d = length(x.ttv_dims)
    ttv_vec = Array{Array{Float64,3},1}(undef,d)
    rks = vcat([1],x.ttv_rks + y.ttv_rks)
    rks[d+1] = 1
    #initialize ttv_vec
    for k in 1:d
        ttv_vec[k] = zeros(x.ttv_dims[k],rks[k],rks[k+1])
    end
    #first core 
    ttv_vec[1][:,1,1:x.ttv_rks[1]] = x.ttv_vec[1]
    ttv_vec[1][:,1,(x.ttv_rks[1]+1):rks[2]] = y.ttv_vec[1]
    #2nd to end-1 cores
    for k in 2:(d-1)
        ttv_vec[k][:,1:x.ttv_rks[k-1],1:x.ttv_rks[k]] = x.ttv_vec[k]
        ttv_vec[k][:,(x.ttv_rks[k-1]+1):rks[k],(x.ttv_rks[k]+1):rks[k+1]] = y.ttv_vec[k]
    end
    #last core
    ttv_vec[d][:,1:x.ttv_rks[d-1],1] = x.ttv_vec[d]
    ttv_vec[d][:,(x.ttv_rks[d-1]+1):rks[d],1] = y.ttv_vec[d]
    return ttvector(ttv_vec,x.ttv_dims,rks[2:d+1],zeros(d))
end

function tto_add(x::ttoperator,y::ttoperator)
    @assert(x.tto_dims == y.tto_dims, DimensionMismatch)
    d = length(x.tto_dims)
    tto_vec = Array{Array{Float64,4},1}(undef,d)
    rks = vcat(1,x.tto_rks + y.tto_rks)
    rks[d+1] = 1
    #initialize tto_vec
    for k in 1:d
        tto_vec[k] = zeros(x.tto_dims[k],x.tto_dims[k],rks[k],rks[k+1])
    end
    #first core 
    tto_vec[1][:,:,1,1:x.tto_rks[1]] = x.tto_vec[1]
    tto_vec[1][:,:,1,(x.tto_rks[1]+1):rks[2]] = y.tto_vec[1]
    #2nd to end-1 cores
    for k in 2:(d-1)
        tto_vec[k][:,:,1:x.tto_rks[k-1],1:x.tto_rks[k]] = x.tto_vec[k]
        tto_vec[k][:,:,(x.tto_rks[k-1]+1):rks[k],(x.tto_rks[k]+1):rks[k+1]] = y.tto_vec[k]
    end
    #last core
    tto_vec[d][:,:,1:x.tto_rks[d-1],1] = x.tto_vec[d]
    tto_vec[d][:,:,(x.tto_rks[d-1]+1):rks[d],1] = y.tto_vec[d]
    return ttoperator(tto_vec,x.tto_dims,rks[2:d+1],zeros(d))
end

function test_tt_add()
    n=5
    x=randn(n,n,n,n,n)
    y=randn(n,n,n,n,n)
    x_tt = ttv_decomp(x,1)
    y_tt = ttv_decomp(y,1)
    z_tt = tt_add(x_tt,y_tt)
    @test(isapprox(ttv_to_tensor(z_tt),x+y))
end

function test_tto_add()
    n=3
    x=randn(n,n,n,n,n,n)
    y=randn(n,n,n,n,n,n)
    x_tt = tto_decomp(x,1)
    y_tt = tto_decomp(y,1)
    z_tt = tto_add(x_tt,y_tt)
    @test(isapprox(tto_to_tensor(z_tt),x+y))
end

function tto_to_ttv(A::ttoperator)
	d = length(A.tto_dims)
	xtt_vec = Array{Array{Float64,3},1}(undef,d)
	A_rks = vcat(1,A.tto_rks)
	for i in 1:d
		xtt_vec[i] = reshape(A.tto_vec[i],A.tto_dims[i]^2,A_rks[i],A_rks[i+1])
	end
	return ttvector(xtt_vec,A.tto_dims.^2,A.tto_rks,A.tto_ot)
end

function ttv_to_tto(x::ttvector)
	d = length(x.ttv_dims)
	@assert(isqrt.(x.ttv_dims).^2 == x.ttv_dims, DimensionMismatch)
	Att_vec = Array{Array{Float64,4},1}(undef,d)
	x_rks = vcat(1,x.ttv_rks)
	for i in 1:d
		Att_vec[i] = reshape(x.ttv_vec[i],isqrt(x.ttv_dims[i]),isqrt(x.ttv_dims[i]),x_rks[i],x_rks[i+1])
	end
	return ttoperator(Att_vec,isqrt.(x.ttv_dims),x.ttv_rks,x.ttv_ot)
end

"""
A : n_1 x r_0 x r_1
B : n_2 x r_1 x r_2
C : n_3 x r_2 x r_3
"""
function sv_trunc(s::Array{Float64},tol)
	if tol==0.0
		return s
	else
		d = length(s)
		i=0
		weight = 0.0
		norm2 = dot(s,s)
		while (i<d) && weight<tol*norm2
			weight+=s[d-i]^2
			i+=1
		end
		return s[1:(d-i+1)]
	end
end

function left_compression(A,B;tol=1e-12)
    dim_A = [i for i in size(A)]
    dim_B = [i for i in size(B)]

    B = permutedims(B, [2,1,3]) #B r_1 x n_2 x r_2
    U = reshape(A,:,dim_A[3])
    u,s,v = svd(U*reshape(B, dim_B[2],:),full=false) #u is the new A, dim(u) = n_1r_0 x tilde(r)_1
    s_trunc = sv_trunc(s,tol)
    dim_B[2] = length(s_trunc)
    U = reshape(u[:,1:dim_B[2]],dim_A[1],dim_A[2],dim_B[2])
    B = reshape(Diagonal(s_trunc)*v[:,1:dim_B[2]]',dim_B[2],dim_B[1],dim_B[3])
    return U, permutedims(B,[2,1,3])
end

#parallel compression of the ttvector
function tt_compression_par(X::ttvector;tol=1e-14,Imax=2)
    Y = deepcopy(X.ttv_vec) :: Array{Array{Float64,3},1}
    rks = deepcopy(X.ttv_rks) :: Array{Int64}
    d = length(X.ttv_dims)
    rks_prev = zeros(Integer,d)
    i=0
    while norm(rks-rks_prev)>0.1 && i<Imax
        i+=1
        rks_prev = deepcopy(rks) :: Array{Int64}
        if mod(i,2) == 1
            @threads for k in 1:floor(Integer,d/2)
                Y[2k-1], Y[2k] = left_compression(Y[2k-1], Y[2k], tol=tol)
                rks[2k-1] = size(Y[2k-1],3)
            end
        else
            @threads for k in 1:floor(Integer,(d-1)/2)
                Y[2k], Y[2k+1] = left_compression(Y[2k], Y[2k+1], tol=tol)
                rks[2k] = size(Y[2k],3)
            end
        end
    end
    return ttvector(Y,X.ttv_dims,rks,zeros(Integer,d))
end

function tt_compression_par(A::ttoperator;tol=1e-14,Imax=2)
	return ttv_to_tto(tt_compression_par(tto_to_ttv(A);tol=tol,Imax=Imax))
end

function test_compression_par()
    n=5
    d=3
    L = randn(n,n,n,n,n,n)
    x = randn(n,n,n)
    y = reshape(L,n^d,:)*x[:]
    L_tt = tto_decomp(reshape(L,n*ones(Int,2d)...),1)
    x_tt = ttv_decomp(x,1)
    y_tt = mult(L_tt,x_tt)
    @test(isapprox(ttv_to_tensor(tt_compression_par(y_tt))[:],y))
end

#matrix vector multiplication in TT format
function mult(A::ttoperator,v::ttvector)
    @assert(A.tto_dims==v.ttv_dims,"Dimension mismatch!")
    d = length(A.tto_dims)
    Y = Array{Array{Float64,3},1}(undef, d)
    A_rks = vcat([1],A.tto_rks) #R_0, ..., R_d
    v_rks = vcat([1],v.ttv_rks) #r_0, ..., r_d
    @threads for k in 1:d
		M = zeros(A.tto_dims[k], A_rks[k],v_rks[k], A_rks[k+1],v_rks[k+1])
		@tensor M[a,b,c,d,e] = A.tto_vec[k][a,z,b,d]*v.ttv_vec[k][z,c,e]
        Y[k] = reshape(M, A.tto_dims[k], A_rks[k]*v_rks[k], A_rks[k+1]*v_rks[k+1])
    end
    return ttvector(Y,A.tto_dims,A.tto_rks.*v.ttv_rks,zeros(Integer,d))
end

#matrix matrix multiplication in TT format
function mult(A::ttoperator,B::ttoperator)
    @assert(A.tto_dims==B.tto_dims,"Dimension mismatch!")
    d = length(A.tto_dims)
    Y = Array{Array{Float64,4},1}(undef, d)
    A_rks = vcat([1],A.tto_rks) #R_0, ..., R_d
    B_rks = vcat([1],B.tto_rks) #r_0, ..., r_d
    @threads for k in 1:d
		M = zeros(A.tto_dims[k],A.tto_dims[k],A_rks[k],B_rks[k],A_rks[k+1],B_rks[k+1])
		@tensor M[a,b,c,d,e,f] = A.tto_vec[k][a,z,c,e]*B.tto_vec[k][z,b,d,f]
        Y[k] = reshape(M, A.tto_dims[k], A.tto_dims[k], A_rks[k]*B_rks[k], A_rks[k+1]*B_rks[k+1])
    end
    return ttoperator(Y,A.tto_dims,A.tto_rks.*B.tto_rks,zeros(Integer,d))
end

function test_mult()
    n=5
    d=3
    L = randn(n,n,n,n,n,n)
    x = randn(n,n,n)
    y = reshape(L,n^d,:)*x[:]
    L_tt = tto_decomp(reshape(L,n*ones(Int,2d)...),1)
    x_tt = ttv_decomp(x,1)
    y_tt = mult(L_tt,x_tt)
    @test(isapprox(ttv_to_tensor(y_tt)[:],y))
end

function test_mult_tto()
    n=3
    d=3
    L = randn(n,n,n,n,n,n)
    S = randn(n,n,n,n,n,n)
    y = reshape(L,n^d,:)*reshape(S,n^d,:)
    L_tt = tto_decomp(L,1)
    S_tt = tto_decomp(S,1)
    y_tt = mult(L_tt,S_tt)
    @test(isapprox(reshape(tto_to_tensor(y_tt),n^d,:),y))
end

#tt_dot returns the dot product of two tt
function tt_dot(A::ttvector,B::ttvector)
    @assert(A.ttv_dims == B.ttv_dims, "Dimension mismatch")
    d = length(A.ttv_dims)
    Y = Array{Array{Float64,2},1}(undef,d)
    A_rks = vcat(1,A.ttv_rks)::Array{Int64}
    B_rks = vcat(1,B.ttv_rks)
	C = zeros(maximum(A_rks.*B_rks))
    @threads for k in 1:d
		M = zeros(A_rks[k],B_rks[k],A_rks[k+1],B_rks[k+1])
		@tensor M[a,b,c,d] = A.ttv_vec[k][z,a,c]*B.ttv_vec[k][z,b,d] #size R^A_{k-1} ×  R^B_{k-1} × R^A_{k} × R^B_{k} 
		Y[k] = reshape(M, A_rks[k]*B_rks[k], A_rks[k+1]*B_rks[k+1])
    end
    C[1:length(Y[d])] = Y[d][:]
    for k in d-1:-1:1
        C[1:size(Y[k],1)] = Y[k]*C[1:1:size(Y[k],2)]
    end
    return C[1]::Float64
end

function test_tt_dot()
    n=5
    d=3
    x = randn(n,n,n)
    y = randn(n,n,n)
    z = randn(n,n,n)
    x_tt = ttv_decomp(x,1)
    y_tt = ttv_decomp(y,1)
    z_tt = ttv_decomp(z,1)
    a_tt = tt_add(y_tt,z_tt)
    @test isapprox(x[:]'*(y+z)[:], tt_dot(x_tt,a_tt))
end

function mult_a_tt(a::Real,A::ttvector)
    i = findfirst(isequal(0),A.ttv_ot)
    X = copy(A.ttv_vec)
    X[i] = a*X[i]
    return ttvector(X,A.ttv_dims,A.ttv_rks,A.ttv_ot)
end

function mult_a_tt(a::Real,A::ttoperator)
    i = findfirst(isequal(0),A.tto_ot)
    X = copy(A.tto_vec)
    X[i] = a*X[i]
    return ttoperator(X,A.tto_dims,A.tto_rks,A.tto_ot)
end

function test_mult_real()
    n=10
    d=3
    x = randn(n,n,n)
    a = randn()
    x_tt = ttv_decomp(x,1)
    @test isapprox(a.*x, ttv_to_tensor(mult_a_tt(a,x_tt)))
end

function tt_core_compression(A,B,C;tol=1e-12)
    dim_A = [i for i in size(A)]
    dim_B = [i for i in size(B)]
    dim_C = [i for i in size(C)]

    M = reshape(B, dim_B[1]*dim_B[2],:)
    W = permutedims(C,[2,1,3])
    u,s,v = svd(M*reshape(W,dim_C[2],:))
    s_trunc = sv_trunc(s,tol)
    dim_B[3] = length(s_trunc)
    W = reshape(Diagonal(s_trunc)*v[:,1:dim_B[3]]',dim_B[3],dim_C[1],:) #C tilde(r_2) x n_3 x r_3
    dim_B[3] = length(s_trunc)
    B = permutedims(reshape(u[:,1:dim_B[3]],dim_B[1],dim_B[2],:), [2,1,3]) #B r_1 x n_2 x tilde(r_2)

    U = reshape(A,:,dim_A[3])
    u,s,v = svd(U*reshape(B, dim_B[2],:)) #u is the new A, dim(u) = n_1r_0 x tilde(r)_1
    s_trunc = sv_trunc(s,tol)
    dim_B[2] = length(s_trunc)
    U = reshape(u[:,1:dim_B[2]],dim_A[1],:,dim_B[2])
    B = reshape(Diagonal(s_trunc)*v[:,1:dim_B[2]]',:,dim_B[1],dim_B[3])
    return U,permutedims(B,[2,1,3]),permutedims(W,[2,1,3])
end

"""
U,V,W = tt_core_compression(A,B,C)
z_comp_tt = ttvector([U,V,W],[5,5,5],[5,5,1],[0,0,0])
println(norm(ttv_to_tensor(z_comp_tt).-x.-y))


TT compression
returns a compressed TT representation
"""
function tt_compression(X::ttvector,tol=1e-12)
    Y = X.ttv_vec
    rks = X.ttv_rks
    d = length(X.ttv_dims)
    for k in 2:(d-1)
        Y[k-1],Y[k],Y[k+1] = tt_core_compression(Y[k-1],Y[k],Y[k+1],tol=tol)
        rks[k-1] = size(Y[k-1],3)
    end
    rks[d-1] = size(Y[d],2)
    return ttvector(Y,X.ttv_dims,rks,zeros(Integer,d))
end

#TTO representation of the identity matrix
function id_tto(d;n_dim=2)
	dims = n_dim*ones(Int64,d)
	A = Array{Array{Float64,4},1}(undef,d)
	for j in 1:d
		A[j] = zeros(2,2,1,1)
		A[j][:,:,1,1] = Matrix{Float64}(I,2,2)
	end
	return ttoperator(A,dims,ones(Int64,d),zeros(d))
end