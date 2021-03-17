import Base.+
import Base.*
import LinearAlgebra.dot

"""
Addition of two ttvector
"""
function +(x::ttvector,y::ttvector)
    @assert(x.ttv_dims == y.ttv_dims, "Dimensions mismatch!")
    d = length(x.ttv_dims)
    T = typejoin(eltype(x),eltype(y))
    ttv_vec = Array{Array{T,3},1}(undef,d)
    rks = x.ttv_rks + y.ttv_rks
    rks[1] = 1
    rks[d+1] = 1
    #initialize ttv_vec
    for k in 1:d
        ttv_vec[k] = zeros(T,rks[k],rks[k+1],x.ttv_dims[k])
    end
    #first core 
    ttv_vec[1][1,1:x.ttv_rks[2],:] = x.ttv_vec[1]
    ttv_vec[1][1,(x.ttv_rks[2]+1):rks[2],:] = y.ttv_vec[1]
    #2nd to end-1 cores
    for k in 2:(d-1)
        ttv_vec[k][1:x.ttv_rks[k],1:x.ttv_rks[k+1],:] = x.ttv_vec[k]
        ttv_vec[k][(x.ttv_rks[k]+1):rks[k],(x.ttv_rks[k+1]+1):rks[k+1],:] = y.ttv_vec[k]
    end
    #last core
    ttv_vec[d][1:x.ttv_rks[d],1,:] = x.ttv_vec[d]
    ttv_vec[d][(x.ttv_rks[d]+1):rks[d],1,:] = y.ttv_vec[d]
    return ttvector{T}(ttv_vec,x.ttv_dims,rks,zeros(d))
end

"""
Addition of two ttoperators
"""
function +(x::ttoperator,y::ttoperator)
    @assert(x.tto_dims == y.tto_dims, DimensionMismatch)
    d = length(x.tto_dims)
    T = typejoin(eltype(x),eltype(y))
    tto_vec = Array{Array{T,4},1}(undef,d)
    rks = x.tto_rks + y.tto_rks
    rks[1] = 1
    rks[d+1] = 1
    #initialize tto_vec
    for k in 1:d
        tto_vec[k] = zeros(T,rks[k],rks[k+1],x.tto_dims[k],x.tto_dims[k])
    end
    #first core 
    tto_vec[1][1,1:x.tto_rks[2],:,:] = x.tto_vec[1]
    tto_vec[1][1,(x.tto_rks[2]+1):rks[2],:,:] = y.tto_vec[1]
    #2nd to end-1 cores
    for k in 2:(d-1)
        tto_vec[k][1:x.tto_rks[k],1:x.tto_rks[k+1],:,:] = x.tto_vec[k]
        tto_vec[k][(x.tto_rks[k]+1):rks[k],(x.tto_rks[k+1]+1):rks[k+1],:,:] = y.tto_vec[k]
    end
    #last core
    tto_vec[d][1:x.tto_rks[d],1,:,:] = x.tto_vec[d]
    tto_vec[d][(x.tto_rks[d]+1):rks[d],1,:,:] = y.tto_vec[d]
    return ttoperator{T}(tto_vec,x.tto_dims,rks,zeros(d))
end


#matrix vector multiplication in TT format
function *(A::ttoperator,v::ttvector)
    @assert(A.tto_dims==v.ttv_dims, DimensionMismatch)
    d = length(A.tto_dims)
    T = typejoin(eltype(A),eltype(v))
    Y = Array{Array{T,3},1}(undef, d)
    A_rks = A.tto_rks #R_0, ..., R_d
    v_rks = v.ttv_rks #r_0, ..., r_d
    @threads for k in 1:d
		M = zeros(A_rks[k],v_rks[k], A_rks[k+1],v_rks[k+1],A.tto_dims[k])
		@tensor M[a,b,c,d,e] = A.tto_vec[k][a,c,e,z]*v.ttv_vec[k][b,d,z]
        Y[k] = reshape(M, A_rks[k]*v_rks[k], A_rks[k+1]*v_rks[k+1], A.tto_dims[k])
    end
    return ttvector{T}(Y,A.tto_dims,A.tto_rks.*v.ttv_rks,zeros(Integer,d))
end

#matrix matrix multiplication in TT format
function *(A::ttoperator,B::ttoperator)
    @assert(A.tto_dims==B.tto_dims, DimensionMismatch)
    d = length(A.tto_dims)
    T = typejoin(eltype(A),eltype(B))
    Y = Array{Array{T,4},1}(undef, d)
    A_rks = A.tto_rks #R_0, ..., R_d
    B_rks = B.tto_rks #r_0, ..., r_d
    @threads for k in 1:d
		M = zeros(A_rks[k],B_rks[k],A_rks[k+1],B_rks[k+1],A.tto_dims[k],A.tto_dims[k])
		@tensor M[a,b,c,d,e,f] = A.tto_vec[k][a,c,e,z]*B.tto_vec[k][b,d,z,f]
        Y[k] = reshape(M, A_rks[k]*B_rks[k], A_rks[k+1]*B_rks[k+1], A.tto_dims[k], A.tto_dims[k])
    end
    return ttoperator{T}(Y,A.tto_dims,A.tto_rks.*B.tto_rks,zeros(Integer,d))
end


#tt_dot returns the dot product of two tt
function dot(A::ttvector,B::ttvector)
    @assert(A.ttv_dims == B.ttv_dims, DimensionMismatch)
    d = length(A.ttv_dims)
    T = typejoin(eltype(A),eltype(B))
    Y = Array{Array{T,2},1}(undef,d)
    A_rks = A.ttv_rks
    B_rks = B.ttv_rks
	C = zeros(T,maximum(A_rks.*B_rks))
    @threads for k in 1:d
		M = zeros(A_rks[k],B_rks[k],A_rks[k+1],B_rks[k+1])
		@tensor M[a,b,c,d] = A.ttv_vec[k][a,c,z]*B.ttv_vec[k][b,d,z] #size R^A_{k-1} ×  R^B_{k-1} × R^A_{k} × R^B_{k} 
		Y[k] = reshape(M, A_rks[k]*B_rks[k], A_rks[k+1]*B_rks[k+1])
    end
    C[1:length(Y[d])] = Y[d][:]
    for k in d-1:-1:1
        C[1:size(Y[k],1)] = Y[k]*C[1:1:size(Y[k],2)]
    end
    return C[1]::T
end


function *(a::S,A::ttvector) where S<:Number
    i = findfirst(isequal(0),A.ttv_ot)
    T = typejoin(typeof(a),eltype(A))
    X = copy(A.ttv_vec)
    X[i] = a*X[i]
    return ttvector{T}(X,A.ttv_dims,A.ttv_rks,A.ttv_ot)
end

function *(a::S,A::ttoperator) where S<:Number
    i = findfirst(isequal(0),A.tto_ot)
    T = typejoin(typeof(a),eltype(A))
    X = copy(A.tto_vec)
    X[i] = a*X[i]
    return ttoperator{T}(X,A.tto_dims,A.tto_rks,A.tto_ot)
end