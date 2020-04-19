include("tt_tools.jl")
include("als.jl")
using SparseArrays
#using IterativeSolvers

struct sparsetensor_vec
    # dims is the array of dimensions of the tensor
    dims :: Array{Int64}
    # spv stores the values of the tensor in a sparse vector
    spv :: SparseVector{Float64}
end

struct sparsetensor_mat
    # dims1 is the array of dimensions of the tensor
    # used for the matrix dimension one
    dims1 :: Array{Int64}
    # dims2 is the array of dimensions of the tensor
    # used for the matrix dimension two
    dims2 :: Array{Int64}
    # spm stores the values of the tensor in a sparse matrix
    spm :: SparseMatrixCSC{Float64}
end

function spvec_to_spmat(x :: sparsetensor_vec, lindex)
    # Constructs the sparse matrix x[1:lindex, lindex+1:end]
    res_dims1 = x.dims[1:lindex]
    res_dims2 = x.dims[lindex+1:end]
    m = prod(res_dims1)
    n = prod(res_dims2)
    n_nz = nnz(x.spv)
    z = map(collect, map(a -> divrem(a,m),x.spv.nzind))
    i = zeros(Int, n_nz)
    j = zeros(Int, n_nz)
    for k = 1:n_nz
        i[k] = Int(z[k][2])
        j[k] = Int(z[k][1]) + 1
		if i[k] == 0
			i[k] = m
			j[k] += (-1)
		end
    end
    res_ma = sparse(i, j, x.spv.nzval, m, n)
    return sparsetensor_mat(res_dims1, res_dims2, res_ma)
end

function spmat_to_spvec(x :: sparsetensor_mat)
    # Transforms the sparse matrix into a sparse vector
    res_dims = zeros(length(x.dims1) + length(x.dims2))
    res_dims[1:length(x.dims1)] = x.dims1
    res_dims[(length(x.dims1)+1):end] = x.dims2
    d1 = prod(x.dims1)
    i, j, v = findnz(x.spm)
    res_i = (j-ones(length(j))).*d1 + i
    return sparsetensor_vec(res_dims, sparsevec(res_i, v, d1*prod(x.dims2)))
end

function sppermutedims(x :: sparsetensor_vec, perm)
    # permutes the dimensions of the sparse tensor

    # Prepare the transformation to cartesian coordinates
    n_prod = ones(length(perm))
    for k = 1:(length(perm))
        n_prod[1:(end-k)] = n_prod[1:(end-k)] .* x.dims[k]
    end
    # Prepare the transformation to vector coordinates
    n_prod_perm = ones(length(perm))
    for k = 1:(length(perm))
        n_prod_perm[1:(end-k)] = n_prod_perm[1:(end-k)] .* x.dims[perm[k]]
    end
    # Initialize the cartesian coordinates and their permutation
    x_katind = ones(length(x.spv.nzind), length(perm))
    x_katind_perm = ones(length(x.spv.nzind), length(perm))
    # Initialize the vector indices
    res_ind = ones(length(x.spv.nzind))

    for x_ind = 1:length(x.spv.nzind)
        #Calculate the cartesian coordinates
        x_ind_rem = x.spv.nzind[x_ind]
        for k = 1:(length(perm))
            x_katind[x_ind, length(perm) - k + 1], x_ind_rem = divrem(x_ind_rem, n_prod[k])
			if x_ind_rem == 0
				x_ind_rem = n_prod[k]
			else
				x_katind[x_ind, length(perm) - k + 1] += 1
			end
        end
        # Apply the permutation on the cartesian coordinates
        for k = 1:length(perm)
            x_katind_perm[x_ind, k] = x_katind[x_ind, perm[k]]
        end
        # Retransform the cartesian coordinates into vector indices
        for k = 1:length(perm)
            res_ind[x_ind] = res_ind[x_ind] + n_prod_perm[k]*(x_katind_perm[x_ind,length(perm) - k + 1]-1)
        end
    end
	return sparsetensor_vec(x.dims[perm], sparsevec(res_ind, x.spv.nzval))
end

# Redefine permutedims for a sparsetensor_mat
function Base.permutedims(A::sparsetensor_mat, perm, i_new)
	return permutedims(A.spm, perm, A.dims1, A.dims2, i_new)
end

# Redefine permutedims for the structs SparseVector
function Base.permutedims(A::SparseVector, perm, dims)
	return sppermutedims(sparsetensor_vec(dims, A), perm)
end

# Redefine permutedims for the structs SparseMatrixCSC
function Base.permutedims(A::SparseMatrixCSC, perm, dims1, dims2, i_new)
	return spvec_to_spmat(sppermutedims(spmat_to_spvec(sparsetensor_mat(dims1, dims2, A)), perm), i_new).spm
end

# Define ttv_decomp for sparse input
function ttv_spdecomp(spv :: sparsetensor_vec, index)
	# Decomposes a tensor into its tensor train with core matrices at i=index
	# For index < d it doesn't work yet
	dims = spv.dims #dims = [n_1,...,n_d]
	n_max = maximum(dims)
	d = length(dims)
	ttv_vec = Array{Array{Float64}}(undef,d)
	# ttv_ot[i]= -1 if i < index
	# ttv_ot[i] = 0 if i = index
	# ttv_ot[i] = 1 if i > index
	ttv_ot = -ones(d)
	ttv_ot[index] = 0
	if index < d
		ttv_ot[index+1:d] = ones(d-index)
	end
	rks = ones(Int, d+2) # ttv_rks will be rks[2:d+1]
	tensor_curr = convert(SparseMatrixCSC, reshape(spv.spv, dims[1], :))
	# Calculate ttv_vec[i] for i < index
	for i = 1 : (index - 1)
		# Reshape the currently left tensor
		tensor_curr = convert(SparseMatrixCSC, reshape(tensor_curr, Int(rks[i] * dims[i]), :))
		# Perform a qr decomposition
		qro = qr(tensor_curr)
		Q = qro.Q[invperm(qro.prow),:]
		R = qro.R[:,invperm(qro.pcol)]
		# Define the i-th rank
		rks[i+1] = minimum(size(tensor_curr))
		# Initialize ttv_vec[i]
		ttv_vec[i] = zeros(dims[i],rks[i],rks[i+1])
		# Fill in the ttv_vec[i]
		for x = 1 : dims[i]
			ttv_vec[i][x, :, :] = Q[(rks[i]*(x-1) + 1):(rks[i]*x), 1:rks[i+1]]
		end
		# Update the currently left tensor
		tensor_curr = R[1:rks[i+1],:]
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

function tto_spdecomp(spm :: sparsetensor_mat, index)
	# Decomposes a sparse tensor operator into its tensor train
	# with core matrices at i=index
	# For index < d it doesn't work yet
	# The tensor has to be given as element of R^[n_1*...*n_d,n_1*...*n_d]
	d = length(spm.dims1)
	tto_dims = spm.dims1
	n_max = maximum(tto_dims)
	dims_sq = tto_dims.^2
	# The matrix is reshaped, reordered and reshaped
	# into the matrix [(x_1,y_1),((x_2,y_2),...,(x_d,y_d))]
	# and decomposed into its tensor train with core matrices at i = index
	index_sorted = vec(reshape(Transpose(reshape(1:(2*d),:,2)),1,:))
	dims_sorted = zeros(2*d)
    dims_sorted[1:d] = tto_dims
	dims_sorted[(d+1):(2*d)] = tto_dims
	dims_sorted = dims_sorted[index_sorted]
	ttv = ttv_spdecomp(spmat_to_spvec(sparsetensor_mat([dims_sq[1]], dims_sq[2:end], permutedims(spm.spm, index_sorted, spm.dims1, spm.dims2, 2))), index)
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

function Lap(n::Integer, d::Integer)
	# Returns the tensor of the discrete Laplacian in a box [0,1]^d
	# with n equidistant discretization points in each direction
	# as an element of R^[n^d,n^d]
	I = zeros((n^d) * (2 * d + 1))
	J = zeros((n^d) * (2 * d + 1))
	V = zeros((n^d) * (2 * d + 1))

	for x = 0:n^d-1
		x_dig = digits(x, base = n, pad = d)
		I[(2*d + 1) * x + 1] = x + 1
		J[(2*d + 1) * x + 1] = x + 1
		V[(2*d + 1) * x + 1] = 2*d
		for i = 0:d-1
			if x_dig[i + 1] == 0
				I[(2*d + 1) * x + 2*i + 2] = x + 1
				J[(2*d + 1) * x + 2*i + 2] = x + n^i + 1
				V[(2*d + 1) * x + 2*i + 2] = -1
			elseif x_dig[i + 1] == n - 1
				I[(2*d + 1) * x + 2*i + 2] = x + 1
				J[(2*d + 1) * x + 2*i + 2] = x - n^i + 1
				V[(2*d + 1) * x + 2*i + 2] = -1
			else
				I[(2*d + 1) * x + 2*i + 2] = x + 1
				J[(2*d + 1) * x + 2*i + 2] = x + n^i + 1
				V[(2*d + 1) * x + 2*i + 2] = -1
				I[(2*d + 1) * x + 2*i + 3] = x + 1
				J[(2*d + 1) * x + 2*i + 3] = x - n^i + 1
				V[(2*d + 1) * x + 2*i + 3] = -1
			end
		end
	end
	I = I[I .!= 0]
	J = J[J .!= 0]
	V = V[V .!= 0]
	return sparse(I , J, V, n^d, n^d)
end

# Test
n = 25
d = 2
L = Lap(n, d)
x_o = ones(size(L,1))
x_o_tt = ttv_decomp(reshape(x_o, n * ones(Int,d)...), d)
b = L * x_o
x_s = map(round, 10 * randn(size(L,1)))
x_s_tt = ttv_decomp(reshape(x_s, n * ones(Int, d)...),d)
b_tt = ttv_decomp(reshape(b,n*ones(Int,d)...), d)
L_spm = sparsetensor_mat(n*ones(Int, d), n*ones(Int, d), L)
L_tt = tto_spdecomp(L_spm, d)
x_als_tt = als(L_tt, b_tt, x_s_tt, x_o_tt.ttv_rks)
