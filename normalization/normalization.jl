import Pkg
Pkg.activate("./")
using TensorTrains
using LinearAlgebra
using Plots
using Random
include("../src/tr_tools.jl")

"""
Norm minimization of the cores
"""

#returns the symmetric solution to the equation M^{-1} B M^{-1} = A where A and B are positive-definite matrices.
function ads(A,B)
    Bsqrt = sqrt(B)
    return Symmetric(Bsqrt*inv(sqrt(Symmetric(Bsqrt*A*Bsqrt)))*Bsqrt)
end

#computes the Frobenius norm of all the TT cores
function norm_tt(xtt::Union{TRvector{T},TTvector{T}}) where {T<:Number}
    out = 0.0
    for i in eachindex(xtt.cores)
        out += norm(xtt.cores[i])^2
    end
    return sqrt(out)
end

#returns an invertible matrix of size n × n with a condition number less than 'tol'
function rand_inv(n::Int,tol=1e6)
    out = randn(n,n)/sqrt(n)
    while cond(out)>tol
        out = randn(n,n)/sqrt(n)
    end
    return out
end

function init_X(rks;random=false)
    d = length(rks)
    X = Array{Array{Float64,2}}(undef,d)
    if random
        for i in eachindex(X)
           X[i] = rand_inv(rks[i])
        end
    else
        for i in eachindex(X)
           X[i] = Matrix{Float64}(I,rks[i],rks[i])
        end
    end
    X[1] = Matrix{Float64}(I,rks[1],rks[1])
    X[end] = Matrix{Float64}(I,rks[end],rks[end])
    return X
end

"""
Minimise the norm of ∑_i || X_i ||_F^2 where X_i are TT core representations of X
"""
function _ttcore_norm_step!(y_tt, x_tt, X, j)
    Atemp = vcat([y_tt.cores[j-1][k,:,:] for k in 1:x_tt.dims[j-1]]...)
    A = Symmetric(Atemp'*Atemp)
    Btemp = vcat([y_tt.cores[j][k,:,:]' for k in 1:x_tt.dims[j]]...)
    B = Symmetric(Btemp'*Btemp)
    X[j] = Matrix(cholesky(Symmetric(real.(ads(A,B)))).L)
    Xj_inv = inv(X[j])
    for k in 1:y_tt.dims[j-1]
        y_tt.cores[j-1][k,:,:] = y_tt.cores[j-1][k,:,:]*X[j]
    end
    for k in 1:y_tt.dims[j]
        y_tt.cores[j][k,:,:] = Xj_inv*y_tt.cores[j][k,:,:]
    end
end

#N = number of sweeps to optimize the matrices
function ttcore_norm_minimization(x_tt::TTvector{T};N=6,X=init_X(x_tt.rks),diagonalise=false) where {T<:Number}
    d = x_tt.N
    cost = zeros(N+1)
    cost[1] = norm_tt(x_tt)
    y_tt = copy(x_tt)
    for i in 1:N
        for j in 2:d
            _ttcore_norm_step!(y_tt, x_tt, X, j)
        end
        for j in d-1:-1:3
            _ttcore_norm_step!(y_tt, x_tt, X, j)
        end
        cost[i+1] = norm_tt(y_tt)
        println("New cost $(cost[i+1])")
    end
    if diagonalise
        for i in 1:d-1
            Atemp = vcat([y_tt.cores[i][k,:,:] for k in 1:x_tt.dims[i]]...)
            F = eigen(Atemp'*Atemp,sortby=-)
            for n in 1:x_tt.dims[i]
                y_tt.cores[i][n,:,:] = y_tt.cores[i][n,:,:]*F.vectors
            end
            for n in 1:x_tt.dims[i+1]
                y_tt.cores[i+1][n,:,:] = F.vectors'*y_tt.cores[i+1][n,:,:]
            end
        end
    end
    return y_tt, cost
end

function trcore_norm_minimization(x_tt::TRvector{T};N=6,X=init_X(x_tt.rks),diagonalise=false) where {T<:Number}
    d = x_tt.N
    cost = zeros(N+1)
    cost[1] = norm_tt(x_tt)
    y_tt = copy(x_tt)
    for i in 1:N
        #optimization of X₁,..,X_{d-1}   
        for j in 2:d
            Atemp = vcat([inv(X[j-1])*y_tt.cores[j-1][k,:,:] for k in 1:x_tt.dims[j-1]]...)
            A = Symmetric(Atemp'*Atemp)
            Btemp = vcat([X[j+1]'*y_tt.cores[j][k,:,:]' for k in 1:x_tt.dims[j]]...)
            B = Symmetric(Btemp'*Btemp)
            M = ads(A,B)
            X[j] = Matrix(cholesky(Symmetric(real.(M))).L)
        end
        #optimization of X_d=X_0
        Atemp = vcat([inv(X[d])*y_tt.cores[d][k,:,:] for k in 1:x_tt.dims[d]]...)
        A = Symmetric(Atemp'*Atemp)
        Btemp = vcat([X[2]'*y_tt.cores[1][k,:,:]' for k in 1:x_tt.dims[1]]...)
        B = Symmetric(Btemp'*Btemp)
        M = ads(A,B)
        X[d+1] = Matrix(cholesky(Symmetric(real.(M))).L)
        X[1] = X[d+1]
        for i in 1:d
            for j in 1:y_tt.dims[i]
                y_tt.cores[i][j,:,:] = inv(X[i])*y_tt.cores[i][j,:,:]*X[i+1]
            end
        end
        cost[i+1] = norm_tt(y_tt)
        println("New cost $(cost[i+1])")
    end
    return y_tt, cost
end

function conv_criterion(x_tt::Union{TRvector{T},TTvector{T}},i::Integer) where {T<:Number}
    d = x_tt.N
    @assert 1≤i<d
    A = vcat([x_tt.cores[i][j,:,:] for j in 1:x_tt.dims[i]]...)
    B = vcat([x_tt.cores[i+1][j,:,:]' for j in 1:x_tt.dims[i+1]]...)
    return norm(A'*A-B'*B)
end

function invariant(x_tt::Union{TRvector{T},TTvector{T}};ε=1e-6) where {T<:Number}
    d = x_tt.N
    for i in 1:d-1
        @assert isapprox(conv_criterion(x_tt,i),0.0,atol=ε)
    end
    D = Array{Array{Float64,1},1}(undef,d-1)
    for i in 1:d-1
        A = vcat([x_tt.cores[i][j,:,:] for j in 1:x_tt.dims[i]]...)
        D[i] = svdvals(A'*A)
    end
    return D
end

"""
Assume that x_tt is given in variational form
"""
function left_variational(x_tt::TTvector{T}) where {T<:Number}
    y_tt = copy(x_tt)
    Λ = Array{Array{Float64,1},1}(undef,x_tt.N)
    Λ[1] = ones(y_tt.rks[1])
    for i in 1:x_tt.N-1
        Atemp = vcat([Diagonal(Λ[i])*y_tt.cores[i][k,:,:] for k in 1:x_tt.dims[i]]...)
        F = eigen(Atemp'*Atemp,sortby=-)
        Λ[i+1] = sqrt.(F.values) #sorted decreasingly
        for n in 1:x_tt.dims[i]
            y_tt.cores[i][n,:,:] = y_tt.cores[i][n,:,:]*F.vectors
        end
        for n in 1:x_tt.dims[i+1]
            y_tt.cores[i+1][n,:,:] = F.vectors'*y_tt.cores[i+1][n,:,:]
        end
    end
    return y_tt,Λ[2:end]
end

"""
Assumes that xtt is given in a left variational form
"""
function tt_leftnormal_rounding(ytt::TTvector{T},Λ;tol=1e-4, rmax=0) where {T<:Number}
    err = 0.0
    xtt = copy(ytt)
    for i in 1:xtt.N-1
        ri = TensorTrains.cut_off_index(Λ[i],tol)
        if rmax>0
            ri = min(ri, rmax)
        end
        xtt.rks[i+1] = ri
        xtt.cores[i] = xtt.cores[i][:,:,1:ri]
        xtt.cores[i+1] = xtt.cores[i+1][:,1:ri,:]
        err += norm(@view Λ[i][ri+1:end])
    end
    return xtt,err
end

"""
Assumes that xtt is given in a diagonalised form
"""
function tt_normal_rounding(ytt::TTvector{T};tol=1e-4) where {T<:Number}
    xtt = copy(ytt)
    for i in 1:xtt.N-1
        A_temp = vcat([xtt.cores[i][j,:,:] for j in 1:xtt.dims[i]]...)
        D = eigvals(A_temp'*A_temp,sortby=-)
        ri = TensorTrains.cut_off_index(sqrt.(D),tol)
        xtt.rks[i+1] = ri
        xtt.cores[i] = xtt.cores[i][:,:,1:ri]
        xtt.cores[i+1] = xtt.cores[i+1][:,1:ri,:]
    end
    return xtt
end

function tt_rounding_print(x_tt::TTvector{T,N};tol=1e-12,rmax=max(prod(x_tt.dims[1:floor(Int,x_tt.N/2)]),prod(x_tt.dims[ceil(Int,x_tt.N/2):end]))) where {T<:Number,N}
	y_tt = orthogonalize(x_tt;i=x_tt.N)
    err2 = zero(T)
	for j in x_tt.N:-1:2
		u,s,v = svd(reshape(permutedims(y_tt.cores[j],[2 1 3]),y_tt.rks[j],:),full=false)
		k = min(TensorTrains.cut_off_index(s,tol),rmax)
        err2 += norm(s[k+1:end])^2
		y_tt.cores[j] = permutedims(reshape(v'[1:k,:],:,x_tt.dims[j],y_tt.rks[j+1]),[2 1 3])
		y_tt.cores[j-1] = reshape(reshape(y_tt.cores[j-1],y_tt.dims[j-1]*y_tt.rks[j-1],:)*u[:,1:k]*Diagonal(s[1:k]),y_tt.dims[j-1],y_tt.rks[j-1],:)
		y_tt.rks[j] = k
		y_tt.ot[j] = 1
	end
	y_tt.ot[1] = 0
	return y_tt, sqrt(err2)
end
