include("als.jl")
using Test
using Base.Threads

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

function test_tt_add()
    n=5
    x=randn(n,n,n,n,n)
    y=randn(n,n,n,n,n)
    x_tt = ttv_decomp(x,1)
    y_tt = ttv_decomp(y,1)
    z_tt = tt_add(x_tt,y_tt)
    @test(isapprox(ttv_to_tensor(z_tt),x+y))
end


"""
A : n_1 x r_0 x r_1
B : n_2 x r_1 x r_2
C : n_3 x r_2 x r_3
"""
function sv_trunc(s::Array{Float64},tol)
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

function right_compression(B,C;tol=1e-12)
    dim_B = [i for i in size(B)]
    dim_C = [i for i in size(C)]

    M = reshape(B, dim_B[1]*dim_B[2],:) #r_0 n_1 x r_1
    W = permutedims(C,[2,1,3]) #r_1 x n_2 r_2
    u,s,v = svd(M*reshape(W,dim_C[2],:))
    s_trunc = sv_trunc(s,tol)
    dim_B[3] = length(s_trunc)
    W = reshape(Diagonal(s_trunc)*v[:,1:dim_B[3]]',dim_B[3],dim_C[1],:) #C tilde(r_2) x n_3 x r_3
    dim_B[3] = length(s_trunc)
    B = reshape(u[:,1:dim_B[3]],dim_B[1],dim_B[2],:) #B n_2 x r_1 x tilde(r_2)
    return B,permutedims(W,[2,1,3])
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

function tt_compression_par(X::ttvector;tol=1e-14,Imax=2)
    Y = X.ttv_vec
    rks = X.ttv_rks :: Array{Int64}
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

#not efficient
function tt_compression_par!(X::ttvector;tol=1e-14,Imax=2)
    d = length(X.ttv_dims)
    rks_prev = zeros(Integer,d)
    i=0
    while norm(X.ttv_rks-rks_prev)>0.1 && i<Imax
        i+=1
        rks_prev = deepcopy(X.ttv_rks) :: Array{Int64}
        if mod(i,2) == 1
            @threads for k in 1:floor(Integer,d/2)
                X.ttv_vec[2k-1], X.ttv_vec[2k] = left_compression(X.ttv_vec[2k-1], X.ttv_vec[2k], tol=tol)
                X.ttv_rks[2k-1] = size(X.ttv_vec[2k-1],3)
            end
        else
            @threads for k in 1:floor(Integer,(d-1)/2)
                X.ttv_vec[2k], X.ttv_vec[2k+1] = left_compression(X.ttv_vec[2k], X.ttv_vec[2k+1], tol=tol)
                X.ttv_rks[2k] = size(X.ttv_vec[2k],3)
            end
        end
    end
    nothing
end

function test_tt_compression_par2()
    n=5
    d=3
    L = randn(n,n,n,n,n,n)
    x = randn(n,n,n)
    y = reshape(L,n^d,:)*x[:]
    L_tt = tto_decomp(reshape(L,n*ones(Int,2d)...),1)
    x_tt = ttv_decomp(x,1)
    y_tt = mult(L_tt,x_tt)
    tt_compression_par!(y_tt)
    @test(isapprox(ttv_to_tensor(y_tt)[:],y))
end

function mult(A::ttoperator,v::ttvector)
    @assert(A.tto_dims==v.ttv_dims,"Dimension mismatch!")
    d = length(A.tto_dims)
    Y = Array{Array{Float64}}(undef, d)
    A_rks = vcat([1],A.tto_rks) #R_0, ..., R_d
    v_rks = vcat([1],v.ttv_rks) #r_0, ..., r_d
    @threads for k in 1:d
        M = reshape(permutedims(A.tto_vec[k],[1,3,4,2]),:,A.tto_dims[k]) #A_k of size n_k R_{k-1} R_k x n_k
        x = reshape(v.ttv_vec[k],v.ttv_dims[k],v_rks[k]*v_rks[k+1]) #v_k of size n_k x r_{k-1} r_k
        Y[k] = reshape(M*x, A.tto_dims[k], A_rks[k], A_rks[k+1], v_rks[k], v_rks[k+1])
        Y[k] = permutedims(Y[k], [1,2,4,3,5])
        Y[k] = reshape(Y[k], A.tto_dims[k], A_rks[k]*v_rks[k], A_rks[k+1]*v_rks[k+1])
    end
    return ttvector(Y,A.tto_dims,A.tto_rks.*v.ttv_rks,zeros(Integer,d))
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

#tt_dot returns the dot product of two tt
function tt_dot(A::ttvector,B::ttvector)
    @assert(A.ttv_dims == B.ttv_dims, "Dimension mismatch")
    d = length(A.ttv_dims)
    Y = Array{Array{Float64},1}(undef,d)
    A_rks = vcat([1],A.ttv_rks)::Array{Int64}
    B_rks = vcat([1],B.ttv_rks)
    @threads for k in 1:d
        M = reshape(permutedims(A.ttv_vec[k],[2,3,1]),A_rks[k]*A_rks[k+1],A.ttv_dims[k]) #A_k of size R_{k-1} R_k x n_k
        x = reshape(B.ttv_vec[k],B.ttv_dims[k],B_rks[k]*B_rks[k+1]) #v_k of size n_k x r_{k-1} r_k
        Y[k] = reshape(M*x, A_rks[k], A_rks[k+1], B_rks[k], B_rks[k+1])
        Y[k] = permutedims(Y[k], [1,3,2,4])
        Y[k] = reshape(Y[k],A_rks[k]*B_rks[k], A_rks[k+1]*B_rks[k+1])
    end
    C = Y[1]
    for k in 2:d
        C = C*Y[k]
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
    i = rand(findall(isequal(0),A.ttv_ot))
    X = copy(A.ttv_vec)
    X[i] = a*X[i]
    return ttvector(X,A.ttv_dims,A.ttv_rks,A.ttv_ot)
end

function test_mult_real()
    n=10
    d=3
    x = randn(n,n,n)
    a = randn()
    x_tt = ttv_decomp(x,1)
    @test isapprox(a.*x, ttv_to_tensor(mult_a_tt(a,x_tt)))
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



