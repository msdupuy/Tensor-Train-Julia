using Base.Threads
using TensorOperations
import Base.+
import Base.-
import Base.*
import Base./
import LinearAlgebra.dot

"""
TTvector + constant
"""
function +(x::TTvector{T,N},y::S) where {T<:Number,S<:Number,N}
    R = typejoin(T,S)
    rks = x.rks .+ 1
    rks[1],rks[end] = 1,1
    out = zeros_tt(R,x.dims,rks)
    out.cores[1][:,:,1:x.rks[2]] = x.cores[1]
    out.cores[1][:,:,rks[2]] .= y
    for k in 2:N-1
        out.cores[k][:,1:x.rks[k],1:x.rks[k+1]] = x.cores[k]
        out.cores[k][:,rks[k],rks[k+1]] .= 1
    end
    out.cores[N][:,1:x.rks[N],1:x.rks[N+1]] = x.cores[N]
    out.cores[N][:,rks[N],rks[N+1]] .= 1
    return out
end

+(y::S,x::TTvector{T,N}) where {T<:Number,S<:Number,N} = x+y

"""
Addition of two TTvector
"""
function +(x::TTvector{T,N},y::TTvector{T,N}) where {T<:Number,N}
    @assert x.dims == y.dims "Incompatible dimensions"
    d = x.N
    cores = Array{Array{T,3},1}(undef,d)
    rks = x.rks + y.rks
    rks[1] = 1
    rks[d+1] = 1
    #initialize cores
    @threads for k in 1:d
        cores[k] = zeros(T,x.dims[k],rks[k],rks[k+1])
    end
    @inbounds begin
        #first core 
        cores[1][:,:,1:x.rks[2]] = x.cores[1]
        cores[1][:,:,(x.rks[2]+1):rks[2]] = y.cores[1]
        #2nd to end-1 cores
        @threads for k in 2:(d-1)
            cores[k][:,1:x.rks[k],1:x.rks[k+1]] = x.cores[k]
            cores[k][:,(x.rks[k]+1):rks[k],(x.rks[k+1]+1):rks[k+1]] = y.cores[k]
        end
        #last core
        cores[d][:,1:x.rks[d],1] = x.cores[d]
        cores[d][:,(x.rks[d]+1):rks[d],1] = y.cores[d]
        end
    return TTvector{T,N}(d,cores,x.dims,rks,zeros(Int64,d))
end

"""
Addition of two TToperators
"""
function +(x::TToperator{T,N},y::TToperator{T,N}) where {T<:Number,N}
    @assert x.dims == y.dims "Incompatible dimensions"
    d = x.N
    cores = Array{Array{T,4},1}(undef,d)
    rks = x.rks + y.rks
    rks[1] = 1
    rks[d+1] = 1
    #initialize cores
    @threads for k in 1:d
        cores[k] = zeros(T,x.dims[k],x.dims[k],rks[k],rks[k+1])
    end
    @inbounds begin
        #first core 
        cores[1][:,:,:,1:x.rks[1+1]] = x.cores[1]
        cores[1][:,:,:,(x.rks[2]+1):rks[2]] = y.cores[1]
        #2nd to end-1 cores
        @threads for k in 2:(d-1)
            cores[k][:,:,1:x.rks[k],1:x.rks[k+1]] = x.cores[k]
            cores[k][:,:,(x.rks[k]+1):rks[k],(x.rks[k+1]+1):rks[k+1]] = y.cores[k]
        end
        #last core
        cores[d][:,:,1:x.rks[d],1] = x.cores[d]
        cores[d][:,:,(x.rks[d]+1):rks[d],1] = y.cores[d]
    end
    return TToperator{T,N}(d,cores,x.dims,rks,zeros(Int64,d))
end


#matrix vector multiplication in TT format
function *(A::TToperator{T,N},v::TTvector{T,N}) where {T<:Number,N}
    @assert A.dims==v.dims "Incompatible dimensions"
    y = zeros_tt(T,A.dims,A.rks.*v.rks)
    @inbounds begin @simd for k in 1:v.N
        yvec_temp = reshape(y.cores[k], (y.dims[k], A.rks[k], v.rks[k], A.rks[k+1], v.rks[k+1]))
        @tensoropt((νₖ₋₁,νₖ), yvec_temp[iₖ,αₖ₋₁,νₖ₋₁,αₖ,νₖ] = A.cores[k][iₖ,jₖ,αₖ₋₁,αₖ]*v.cores[k][jₖ,νₖ₋₁,νₖ])
    end end
    return y
end

#matrix matrix multiplication in TT format
function *(A::TToperator{T,N},B::TToperator{T,N}) where {T<:Number,N}
    @assert A.dims==B.dims "Incompatible dimensions"
    d = A.N
    A_rks = A.rks #R_0, ..., R_d
    B_rks = B.rks #r_0, ..., r_d
    Y = [zeros(T,A.dims[k], A.dims[k], A_rks[k]*B_rks[k], A_rks[k+1]*B_rks[k+1]) for k in eachindex(A.dims)]
    @inbounds @simd for k in eachindex(Y)
		M_temp = reshape(Y[k], A.dims[k], A.dims[k], A_rks[k],B_rks[k], A_rks[k+1],B_rks[k+1])
        @simd for jₖ in size(M_temp,2)
            @simd for iₖ in size(M_temp,1)
                @tensor M_temp[iₖ,jₖ,αₖ₋₁,βₖ₋₁,αₖ,βₖ] = A.cores[k][iₖ,z,αₖ₋₁,αₖ]*B.cores[k][z,jₖ,βₖ₋₁,βₖ]
            end
        end
    end
    return TToperator{T,N}(d,Y,A.dims,A.rks.*B.rks,zeros(Int64,d))
end

*(A::TToperator{T,N},B...) where {T,N} = *(A,*(B...))

function *(A::Array{TTvector{T,N},1},x::Vector{T}) where {T,N}
    out = x[1]*A[1]
    for i in 2:length(A)
        out = out + x[i]*A[i]
    end
    return out
end

function *(x::TTvector{T,N},y::TTvector{T,N}) where {T<:Number,N}
    out = zeros_tt(T,x.dims,x.rks.*y.rks) 
    for k in 1:N 
        for iₖ in 1:x.dims[k]
            out.cores[k][iₖ,:,:] =  kron(x.cores[k][iₖ,:,:],y.cores[k][iₖ,:,:])
        end
    end
    return out
end

#dot returns the dot product of two TTvector
function dot(A::TTvector{T,N},B::TTvector{T,N}) where {T<:Number,N}
    @assert A.dims==B.dims "TT dimensions are not compatible"
    A_rks = A.rks
    B_rks = B.rks
    rA_max, rB_max = maximum(A_rks), maximum(B_rks)
    # Two alternating buffers to avoid copying the accumulator on every step.
    # buf holds the result of the current step; prev holds the previous step.
    # After each iteration we swap the pointers — no allocation.
    prev = zeros(T, rA_max, rB_max)
    buf  = zeros(T, rA_max, rB_max)
    prev[1,1] = one(T)
    @inbounds for k in eachindex(A.dims)
        M      = @view(buf[1:A_rks[k+1], 1:B_rks[k+1]])
        prev_v = @view(prev[1:A_rks[k],  1:B_rks[k]])
        @tensoropt M[a,b] = A.cores[k][z,α,a] * prev_v[α,β] * B.cores[k][z,β,b]
        buf, prev = prev, buf   # pointer swap — zero allocation
    end
    return prev[1,1]::T
end

"""
`dot_par(x_tt,y_tt)' returns the dot product of `x_tt` and `y_tt` in a parallelized algorithm
"""
function dot_par(A::TTvector{T,N},B::TTvector{T,N}) where {T<:Number,N}
    @assert A.dims==B.dims "TT dimensions are not compatible"
    d = length(A.dims)
    Y = Array{Array{T,2},1}(undef,d)
    A_rks = A.rks
    B_rks = B.rks
	C = zeros(T,maximum(A_rks.*B_rks))
    @threads for k in 1:d
		M = zeros(T,A_rks[k],B_rks[k],A_rks[k+1],B_rks[k+1])
		@tensor M[a,b,c,d] = A.cores[k][z,a,c]*B.cores[k][z,b,d] #size R^A_{k-1} ×  R^B_{k-1} × R^A_{k} × R^B_{k} 
		Y[k] = reshape(M, A_rks[k]*B_rks[k], A_rks[k+1]*B_rks[k+1])
    end
    @inbounds C[1:length(Y[d])] = Y[d][:]
    for k in d-1:-1:1
        @inbounds C[1:size(Y[k],1)] = Y[k]*C[1:size(Y[k],2)]
    end
    return C[1]::T
end

function *(a::S,A::TTvector{R,N}) where {S<:Number,R<:Number,N}
    T = typejoin(typeof(a),R)
    if iszero(a)
        return zeros_tt(T,A.dims,ones(Int64,A.N+1))
    else
        i = findfirst(isequal(0),A.ot)
        X = copy(A.cores)
        X[i] = a*X[i]
        return TTvector{T,N}(A.N,X,A.dims,A.rks,A.ot)
    end
end

function *(a::S,A::TToperator{R,N}) where {S<:Number,R<:Number,N}
    i = findfirst(isequal(0),A.ot)
    T = typejoin(typeof(a),R)
    X = copy(A.cores)
    X[i] = a*X[i]
    return TToperator{T,N}(A.N,X,A.dims,A.rks,A.ot)
end

function -(A::TTvector{T,N},B::TTvector{T,N}) where {T<:Number,N}
    return *(-1.0,B)+A
end

function -(A::TToperator{T,N},B::TToperator{T,N}) where {T<:Number,N}
    return *(-1.0,B)+A
end

function /(A::TTvector,a)
    return 1/a*A
end

"""
returns the matrix x y' in the TTO format
"""
function outer_product(x::TTvector{T,N},y::TTvector{T,N}) where {T<:Number,N}
    Y = [zeros(T,x.dims[k], x.dims[k], x.rks[k]*y.rks[k], x.rks[k+1]*y.rks[k+1]) for k in eachindex(x.dims)]
    @inbounds @simd for k in eachindex(Y)
		M_temp = reshape(Y[k], x.dims[k], x.dims[k], x.rks[k], y.rks[k], x.rks[k+1],y.rks[k+1])
        @simd for jₖ in size(M_temp,2)
            @simd for iₖ in size(M_temp,1)
                @tensor M_temp[iₖ,jₖ,αₖ₋₁,βₖ₋₁,αₖ,βₖ] = x.cores[k][iₖ,αₖ₋₁,αₖ]*conj(y.cores[k][jₖ,βₖ₋₁,βₖ])
            end
        end
    end
    return TToperator{T,N}(x.N,Y,x.dims,x.rks.*y.rks,zeros(Int64,x.N))
end