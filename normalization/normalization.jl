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
    for i in eachindex(xtt.ttv_vec)
        out += norm(xtt.ttv_vec[i])^2
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

#N = number of sweeps to optimize the matrices
function ttcore_norm_minimization(x_tt::TTvector{T};N=6,X=init_X(x_tt.ttv_rks)) where {T<:Number}
    d = x_tt.N
    cost = zeros(N+1)
    cost[1] = norm_tt(x_tt)
    y_tt = copy(x_tt)
    for i in 1:N
        for j in 2:d
            Atemp = vcat([y_tt.ttv_vec[j-1][k,:,:] for k in 1:x_tt.ttv_dims[j-1]]...)
            A = Symmetric(Atemp'*Atemp)
            Btemp = vcat([y_tt.ttv_vec[j][k,:,:]' for k in 1:x_tt.ttv_dims[j]]...)
            B = Symmetric(Btemp'*Btemp)
            M = ads(A,B)
            X[j] = Matrix(cholesky(Symmetric(real.(M))).L)
            for k in 1:y_tt.ttv_dims[j-1]
                y_tt.ttv_vec[j-1][k,:,:] = y_tt.ttv_vec[j-1][k,:,:]*X[j]
            end
            for k in 1:y_tt.ttv_dims[j]
                y_tt.ttv_vec[j][k,:,:] = inv(X[j])*y_tt.ttv_vec[j][k,:,:]
            end
        end
        for j in d-1:-1:3
            Atemp = vcat([y_tt.ttv_vec[j-1][k,:,:] for k in 1:x_tt.ttv_dims[j-1]]...)
            A = Symmetric(Atemp'*Atemp)
            Btemp = vcat([y_tt.ttv_vec[j][k,:,:]' for k in 1:x_tt.ttv_dims[j]]...)
            B = Symmetric(Btemp'*Btemp)
            M = ads(A,B)
            X[j] = Matrix(cholesky(Symmetric(real.(M))).L)
            for k in 1:y_tt.ttv_dims[j-1]
                y_tt.ttv_vec[j-1][k,:,:] = y_tt.ttv_vec[j-1][k,:,:]*X[j]
            end
            for k in 1:y_tt.ttv_dims[j]
                y_tt.ttv_vec[j][k,:,:] = inv(X[j])*y_tt.ttv_vec[j][k,:,:]
            end
        end
        cost[i+1] = norm_tt(y_tt)
        println("New cost $(cost[i+1])")
    end
    return y_tt, cost
end

function trcore_norm_minimization(x_tt::TRvector{T};N=6,X=init_X(x_tt.ttv_rks)) where {T<:Number}
    d = x_tt.N
    cost = zeros(N+1)
    cost[1] = norm_tt(x_tt)
    y_tt = copy(x_tt)
    for i in 1:N
        #optimization of X₁,..,X_{d-1}   
        for j in 2:d
            Atemp = vcat([inv(X[j-1])*y_tt.ttv_vec[j-1][k,:,:] for k in 1:x_tt.ttv_dims[j-1]]...)
            A = Symmetric(Atemp'*Atemp)
            Btemp = vcat([X[j+1]'*y_tt.ttv_vec[j][k,:,:]' for k in 1:x_tt.ttv_dims[j]]...)
            B = Symmetric(Btemp'*Btemp)
            M = ads(A,B)
            X[j] = Matrix(cholesky(Symmetric(real.(M))).L)
        end
        #optimization of X_d=X_0
        Atemp = vcat([inv(X[d])*y_tt.ttv_vec[d][k,:,:] for k in 1:x_tt.ttv_dims[d]]...)
        A = Symmetric(Atemp'*Atemp)
        Btemp = vcat([X[2]'*y_tt.ttv_vec[1][k,:,:]' for k in 1:x_tt.ttv_dims[1]]...)
        B = Symmetric(Btemp'*Btemp)
        M = ads(A,B)
        X[d+1] = Matrix(cholesky(Symmetric(real.(M))).L)
        X[1] = X[d+1]
        for i in 1:d
            for j in 1:y_tt.ttv_dims[i]
                y_tt.ttv_vec[i][j,:,:] = inv(X[i])*y_tt.ttv_vec[i][j,:,:]*X[i+1]
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
    A = vcat([x_tt.ttv_vec[i][j,:,:] for j in 1:x_tt.ttv_dims[i]]...)
    B = vcat([x_tt.ttv_vec[i+1][j,:,:]' for j in 1:x_tt.ttv_dims[i+1]]...)
    return norm(A'*A-B'*B)
end

function invariant(x_tt::Union{TRvector{T},TTvector{T}};ε=1e-6) where {T<:Number}
    d = x_tt.N
    for i in 1:d-1
        @assert isapprox(conv_criterion(x_tt,i),0.0,atol=ε)
    end
    D = Array{Array{Float64,1},1}(undef,d-1)
    for i in 1:d-1
        A = vcat([x_tt.ttv_vec[i][j,:,:] for j in 1:x_tt.ttv_dims[i]]...)
        D[i] = svdvals(A'*A)
    end
    return D
end

L = 10
n = 3
dims = ntuple(x->n,L)
rks = ones(Int64,L+1)
for i in 2:L
    rks[i] = min(n^(i-1),n^(L+1-i),1024)
end

#TT tests
Random.seed!(1234)
x_tt = rand_tt(dims,rks,normalise=false)
x = ttv_to_tensor(x_tt)
hsv = tt_svdvals(x_tt)
x_v = tt_to_vidal(x_tt)

Random.seed!(rand(1:100000))
ytt, cost_list = ttcore_norm_minimization(x_tt,N=60,X=init_X(x_tt.ttv_rks;random=true))
y = ttv_to_tensor(ytt)

#TR tests
Random.seed!(1234)
x_tr = rand_tr(dims,6)
xr = tr_to_tensor(x_tr)

Random.seed!(2604)
#yr, ycost = trcore_norm_minimization(x_tr,N=200,X=init_X(x_tr.ttv_rks;random=true))
#yr_tensor = tr_to_tensor(yr)