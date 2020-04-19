include("tt_tools.jl")

"""
TT version of the restarted GMRES algorithm
"""

#takes a hessenberg matrix and returns q,H resp. orthogonal and upper triangular such that q^T H_out = H_in
function qr_hessenberg(H)
    m = size(H,2)
    q = Matrix{Float64}(I,m+1,m+1)
    T = H
    for i in 1:m
        R = Matrix{Float64}(I,m+1,m+1)
        s = T[i+1,i]/sqrt(T[i,i]^2+T[i+1,i]^2)
        c = T[i,i]/sqrt(T[i,i]^2+T[i+1,i]^2)
        R[i:(i+1),i:(i+1)] = [c s; -s c]
        q = R*q
        T = R*T
    end
    return q,T
end

function tt_gmres(A::ttoperator,b::ttvector,x0::ttvector;Imax=500,tol=1e-8,m=30,hist=false,γ_list=Float64[])
    V = Array{ttvector}(undef,m)
    W = Array{ttvector}(undef,m)
    H = zeros(m+1,m)
    r0 = tt_compression_par(tt_add(b,mult_a_tt(-1.0,tt_compression_par(mult(A,x0))))) 
    β = sqrt(tt_dot(r0,r0))
    V[1] = mult_a_tt(1/β,r0)
    W[1] = tt_compression_par(mult(A,V[1]))
    H[1,1] = tt_dot(W[1],V[1])
    W[1] = tt_compression_par(tt_add(W[1],mult_a_tt(-H[1,1],V[1])))
    H[2,1] = sqrt(tt_dot(W[1],W[1]))
    q,r = qr_hessenberg(H[1:2,1])
    γ = abs(β*q[2,1]) #γ = \| Ax_j-b \|_2
    if hist
        γ_list = vcat(γ_list,γ)
    end
    j = 1
    if Imax <=0 || isapprox(H[j+1,j],0.,atol=tol) || isapprox(γ,0,atol=tol)
        if hist
            return x0,γ_list,H[2,1]
        else
            return x0,[γ],H[2,1]
        end
    else
        while j <= min(m-1,Imax) && !isapprox(H[j+1,j],0.,atol=tol) && !isapprox(γ,0,atol=tol)
            V[j+1] = mult_a_tt(1/H[j+1,j],W[j])
            j+=1 
            W[j] = mult(A,V[j])
            W[j] = tt_compression_par(W[j])
            for i in 1:j
                H[i,j] = tt_dot(W[j],V[i])
                W[j] = tt_add(W[j],mult_a_tt(-H[i,j],V[i]))
                W[j] = tt_compression_par(W[j])                        
            end
            H[j+1,j] = sqrt(tt_dot(W[j],W[j]))
            q,r = qr_hessenberg(@view H[1:(j+1),1:j])
            γ = abs(β*q[j+1,1])
            if hist
                γ_list = vcat(γ_list,γ)
            end
        end
        z = r[1:j,1:j]\q[1:j,1]
        for i in 1:j
            x0 = tt_add(x0,mult_a_tt(β*z[i],V[i]))
        end
        x0 = tt_compression_par(x0)
        return tt_gmres(A,b,x0,tol=tol,Imax=Imax-j,m=m,hist=hist,γ_list=γ_list)
    end
end

function tt_cg(A::ttoperator,b::ttvector,x0::ttvector;Imax=500,tol=1e-8)
    p = tt_compression_par(tt_add(b,mult_a_tt(-1.0,tt_compression_par(mult(A,x0)))))
    r = p
    j=1
    res= zeros(Imax)
    res[1] = sqrt(tt_dot(p,p))
    while j < Imax && res[j]>tol
        Ap = mult(A,p)
        Ap = tt_compression_par(Ap)
        a = res[j]^2/tt_dot(p,Ap)
        x0 = tt_add(x0,mult_a_tt(a,p))
        x0 = tt_compression_par(x0)
        r = tt_add(r,mult_a_tt(-a,Ap))
        r = tt_compression_par(r)
        res[j+1] = sqrt(tt_dot(r,r))
        p = tt_add(r,mult_a_tt(res[j+1]^2/res[j]^2,p))
        p = tt_compression_par(p)
        j+=1
    end
    return x, res[1:j]
end

"""
A_k : n_k x n_k x R_{k-1} x R_k
X_k : n_k x r^X_{k-1} x r^X_k
B_k : n_k x r^B_{k-1} x r^B_k
Ql : R_{k-1} r^X_{k-1} x r^B_{k-1}
Qr : R_k r^X_k x r^B_k
"""
function init_core(A_k,dim_X,B_k,Ql,Qr)
    X_k = zeros(dim_X...)
    #B right shape
    B = reshape(B_k,:,size(B_k,3)) # n_k r^B_{k-1} x r^B_k
    B = reshape(B*(Qr'),size(B_k,1),size(B_k,2),size(A_k,4),size(X_k,3)) #n_k x r^B_{k-1} x R_k x r^X_k
    B = permutedims(B,[2,1,3,4]) #r^B_{k-1} x n_k x R_k x r^X_k
    B = Ql*reshape(B,size(B,1),:) #R_{k-1} r^X_{k-1} x n_k R_k r^X_k
    B = permutedims(reshape(B,size(A_k,3),size(X_k,2),size(A_k,1),size(A_k,4),size(X_k,3)),[3,1,4,2,5]) #n_k x R_{k-1} x R_k x r^X_{k-1} x r^X_k
    B = reshape(B,size(A_k,1)*size(A_k,3)*size(A_k,4),dim_X[2]*dim_X[3]) #n_k R_{k-1} R_k x r^X_{k-1} r^X_k
    A = reshape(permutedims(A_k,[1,3,4,2]),size(A_k,1)*size(A_k,3)*size(A_k,4),size(A_k,2))
    X_k = reshape(A\B,dim_X[1],dim_X[2],dim_X[3])
    return X_k
end

#returns partial isometry Q ∈ R^{n x m}
function rand_orthogonal(n,m)
    N = max(n,m)
    q,r = qr(rand(N,N))
    return q[1:n,1:m]
end

function init(A::ttoperator,b::ttvector,opt_rks)
    @assert(A.tto_dims == b.ttv_dims,DimensionMismatch)
    d = length(A.tto_dims)
    opt_rks = vcat([1],opt_rks)
    Q_list = Array{Array{Float64},1}(undef,d+1)
    ttvec = Array{Array{Float64},1}(undef,d)
    Q_list[1] = [1]
    Q_list[d+1] = [1]
    for k in 1:(d-1)
        Q_list[k+1] = rand_orthogonal(A.tto_rks[k]*opt_rks[k+1],b.ttv_rks[k])
    end
    for k in 1:d
        ttvec[k] = init_core(A.tto_vec[k],[A.tto_dims[k],opt_rks[k],opt_rks[k+1]],b.ttv_vec[k],Q_list[k],Q_list[k+1])
    end
    return ttvector(ttvec,A.tto_dims,opt_rks[2:(d+1)],ones(Int64,d))
end