using Random
using LinearAlgebra
using Base.Threads
using TensorOperations
import Base.eltype
import Base.copy

"""
Tensor rings (TR), aka periodic MPS. TR are stored in 'vec' as order three tensors of size A_k ∈ R^{n_k × r_{k-1} × r_k} with r_0 = r_d.
"""
struct TRvector{T<:Number}
	N :: Int
	ttv_vec :: Array{Array{T,3},1}
	ttv_dims :: Vector{Int64}
	ttv_rks :: Array{Int64,1}
	ttv_ot :: Array{Int64,1}
end

Base.eltype(::TRvector{T}) where {T<:Number} = T 

#returns the tensor represented by 'x_tr'
function tr_to_tensor(x_tr::TRvector{T}) where T<:Number
	d = x_tr.N
	# Initialize the to be returned tensor
	tensor = zeros(T, x_tr.ttv_dims...)
	rmax = maximum(x_tr.ttv_rks)
	# Fill in the tensor for every t=(x_1,...,x_d)
	for t in CartesianIndices(tensor)
		a = collect(Tuple(t))
		curr = zeros(T,rmax,rmax)
		curr[1:x_tr.ttv_rks[d],1:x_tr.ttv_rks[d+1]] = copy(x_tr.ttv_vec[d][a[d],:,:])
		for i = d-1:-1:1
			curr[1:x_tr.ttv_rks[i],1:x_tr.ttv_rks[i+1]] = x_tr.ttv_vec[i][a[i],:,:]*curr[1:x_tr.ttv_rks[i+1],1:x_tr.ttv_rks[i+2]]
		end
		tensor[t] = tr(curr)
	end
	return tensor
end

#returns a zero TRvector
function zeros_tr(dims,rks;T=Float64)
	@assert length(rks)==length(dims)+1
	d = length(dims)
	vec = Array{Array{T,3},1}(undef,d)
	for i in eachindex(vec)
		vec[i] = zeros(dims[i],rks[i],rks[i+1])
	end
	return TRvector{T}(d,vec,dims,rks,zeros(Int,d))
end

function Base.copy(x_tr::TRvector{T}) where T<:Number
	return TRvector{T}(x_tr.N,copy(x_tr.ttv_vec),copy(x_tr.ttv_dims),copy(x_tr.ttv_rks),copy(x_tr.ttv_ot))
end

"""
Returns a random 'TRvector' of dimensions 'dims' and ranks 'rks'
"""
function rand_tr(dims,rks;T=Float64)
	@assert length(dims)+1==length(rks) "Incompatible dimensions and ranks"
	@assert rks[1] == rks[end] "Incompatible first and last ranks"
	d = length(dims)
	tt_vec = Array{Array{T,3}}(undef,d)
	for i in eachindex(tt_vec) 
		tt_vec[i] = randn(T,dims[i],rks[i],rks[i+1])
	end
	return TRvector{T}(d,tt_vec,dims,copy(rks),zeros(Int,d))
end

"""
Returns a random 'TRvector' of dimensions 'dims' and maximal rank 'rmax'
"""
function rand_tr(dims,rmax::Int;T=Float64)
	d = length(dims)
	rks = rmax*ones(Int,d+1)
	tt_vec = Array{Array{T,3}}(undef,d)
	for i in eachindex(tt_vec) 
		tt_vec[i] = randn(T,dims[i],rmax,rmax)
	end
	return TRvector{T}(d,tt_vec,dims,rks,zeros(Int,d))
end