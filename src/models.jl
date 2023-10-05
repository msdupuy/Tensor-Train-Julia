using Combinatorics

"""
one_body_to_matrix and two_body_to_matrix for testing purposes
returns the 2^L×2^L matrices of a_k^†a_l and a_k^† a_l^† a_m a_n 
"""
function one_body_to_matrix(k,l,L)
    A = zeros(Float64,2^L,2^L)
    for i in 0:2^L-1
        i_bin = digits(i,base=2,pad=L)
        for j in 0:2^L-1
            j_bin = digits(j,base=2,pad=L) #corresponding index as d-tuples
            if i_bin[k]==1 && j_bin[l]==1 && j-2^(l-1) == i-2^(k-1)
                A[i+1,j+1] = (-1)^(sum(i_bin[1:(k-1)])+sum(j_bin[1:(l-1)]))
            end
        end
    end
    return A
end

function two_body_to_matrix(k,l,m,n,L)
    A = zeros(Float64,2^L,2^L)
    for i in 0:2^L-1
        i_bin = digits(i,base=2,pad=L)
        for j in 0:2^L-1
            j_bin = digits(j,base=2,pad=L) #corresponding index as d-tuples
            if i_bin[k]==1 && i_bin[l]==1 && j_bin[m]==1 && j_bin[n]==1 && j-2^(m-1)-2^(n-1) == i-2^(k-1)-2^(l-1)
                A[i+1,j+1] = (-1)^(sum(i_bin[1:(k-1)])+sum(i_bin[1:(l-1)])+sum(j_bin[1:(m-1)])+sum(j_bin[1:(n-1)]))
            end
        end
    end
    return A
end

#naive implementation

"""
h: two-dimensional array of the connected sites (kinetic+external potential)
V: four-dimensional array of the connected sites (interaction)
"""
function sc_to_mat(h,V)
    L = size(h,1)
    A = zeros(2^L,2^L)
    for i in 1:L
        for j in i:L
            for k in j:L
                for l in k:L
                    if !isapprox(h[i,j],0.0,atol=1e-10)
                        H = one_body_to_matrix(i,j,L)
                        A += h[i,j]*(H+H')
                    end
                    if !isapprox(V[i,j,k,l],0.0,atol=1e-10)
                        H = two_body_to_matrix(i,j,k,l,L)
                        A += V[i,j,k,l]*(H+H')
                    end
                end
            end
        end
    end
    return A
end

#odd index = spin up, even index = spin down
#returns TT of a half-filled state
function half_filling(N)
    tt_vec = Array{Array{Float64,3},1}(undef,2N)
    for i in 1:N
        tt_vec[2i-1] = zeros(2,1,1)
        tt_vec[2i-1][1,1,1] = 1.0
        tt_vec[2i] = zeros(2,1,1)
        tt_vec[2i][2,1,1] = 1.0
    end
    return TTvector{Float64,2N}(2N,tt_vec,Tuple(2*ones(Int,2N)),ones(Int,2N+1),zeros(2N))
end

#odd index = spin up 
#returns TT of a Slater determinant
function slater(n,d;σ=1:n)
    x = zeros_tt(ntuple(x->2,d),ones(Int64,d+1))
    for i in σ
        x.ttv_vec[i][2,1,1] = 1.0
    end
    for i in setdiff(1:d,σ)
        x.ttv_vec[i][1,1,1] = 1.0
    end
    return x
end

#auxiliary functions for a_p^†a_q
mpo_core_id() = mpo_core_id(Float64)

function mpo_core_id(::Type{T}) where T
    out = zeros(T,2,2,1,1)
    out[1,1,1,1] = 1.0
    out[2,2,1,1] = 1.0
    return out
end

mpo_core_ferm_sign() = mpo_core_ferm_sign(Float64)
function mpo_core_ferm_sign(::Type{T}) where T
    out = zeros(T,2,2,1,1)
    out[1,1,1,1] = 1.0
    out[2,2,1,1] = -1.0
    return out
end

mpo_core_creation() = mpo_core_creation(Float64)
function mpo_core_creation(::Type{T}) where T
    out = zeros(T,2,2,1,1)
    out[2,1,1,1] = 1.0
    return out
end

mpo_core_annihilation() = mpo_core_annihilation(Float64)
function mpo_core_annihilation(::Type{T}) where T
    out = zeros(T,2,2,1,1)
    out[1,2,1,1] = 1.0
    return out
end

"""
MPO of the creation operator a^†_p (default = fermionic creation)
"""
function tto_creation(p,dims::NTuple{N,Int64};fermion=true) where N
    return tto_creation(Float64,p,dims;fermion=fermion)
end

function tto_creation(::Type{T},p,dims::NTuple{N,Int64};fermion=true) where {T,N}
    A = zeros_tto(T,dims,ones(Int64,N+1))
    for i in 1:p-1
        fermion ? (A.tto_vec[i] = mpo_core_ferm_sign(T)) : (A.tto_vec[i] = mpo_core_id(T))
    end
    A.tto_vec[p] = mpo_core_creation(T)
    for i in p+1:N
        A.tto_vec[i] = mpo_core_id(T)
    end
    return A
end

"""
MPO of the annihilation operator a_q (default = fermionic annihilation)
"""
tto_annihilation(q,dims::NTuple{N,Int64};fermion=true) where N = tto_annihilation(Float64,q,dims;fermion=fermion)
function tto_annihilation(::Type{T},q,dims::NTuple{N,Int64};fermion=true) where {T,N}
    A = zeros_tto(T,dims,ones(Int,N+1))
    for i in 1:q-1
        fermion ? (A.tto_vec[i] = mpo_core_ferm_sign(T)) : (A.tto_vec[i] = mpo_core_id(T))
    end
    A.tto_vec[q] = mpo_core_annihilation(T)
    for i in q+1:N
        A.tto_vec[i] = mpo_core_id(T)
    end
    return A
end

"""
returns bosonic or fermionic MPO of a_p^†a_q
"""
one_body_mpo(p::Integer,q::Integer,dims::NTuple{N,Int64};fermion=true) where N = one_body_mpo(Float64,p::Integer,q::Integer,dims;fermion=fermion)
function one_body_mpo(T,p::Integer,q::Integer,dims;fermion=true)
    return tto_creation(T,p,dims;fermion=fermion)*tto_annihilation(T,q,dims;fermion=fermion)
end

"""
returns bosonic or fermionic MPO of a_k^† a^†_l a_m a_n 
"""

two_body_mpo(k,l,m,n,dims::NTuple{N,Int64};fermion=true) where N = two_body_mpo(Float64,k,l,m,n,dims;fermion=fermion)
function two_body_mpo(::Type{T},k,l,m,n,dims::NTuple{N,Int64};fermion=true) where {T,N}
    return tto_creation(T,k,dims;fermion=fermion)*tto_creation(T,l,dims;fermion=fermion)*tto_annihilation(T,m,dims;fermion=fermion)*tto_annihilation(T,n,dims;fermion=fermion)
end


"""
returns an MPO version of 
H = Σ_{ij} h_ij a_i† a_j + Σ_{ijkl} V_{ijkl} a_i†a_j†a_ka_l 
h and V have to be Symmetric 
"""
#assuming diagonal terms are divided by 2 in the h and V matrix
function hV_to_mpo(h::Array{T,2},V,dims::NTuple{N,Int64};tol=1e-8::Float64,n_rnd=20::Int) where {T,N}
    L = size(h,1)
    @assert issymmetric(h)
#    @assert isapprox(V,permutedims(V,(2,1,4,3)))
    A = zeros_tto(T,dims,ones(Int64,L+1))
    i_rnd = 1
    #Precomputation of creation and annihilation operators
    tto_crea = [tto_creation(i,dims) for i in 1:L]
    tto_anni = [tto_annihilation(i,dims) for i in 1:L]
    for i in findall(x->!isapprox(x,0.0,atol=1e-12),h)
        if i[1]<i[2]
            H = tto_crea[i[1]]*tto_anni[i[2]] + tto_crea[i[2]]*tto_anni[i[1]]
            A = A+h[i]*H
        elseif i[1]==i[2]
            H = tto_crea[i[1]]*tto_anni[i[2]]
            A = A+h[i]*H
        end
        if i_rnd > n_rnd 
            A = tt_rounding(A,tol=tol)
            i_rnd = 1
        else
            i_rnd+=1
        end
    end
    for i in findall(x->!isapprox(x,0.0,atol=1e-12),V)
        if (i[1],i[2])<(i[4],i[3]) 
            H = (tto_crea[i[1]]*tto_crea[i[2]]*tto_anni[i[3]]*tto_anni[i[4]])+(tto_crea[i[4]]*tto_crea[i[3]]*tto_anni[i[2]]*tto_anni[i[1]])
            A = A+V[i]*H
        elseif (i[1],i[2])==(i[4],i[3])
            H = tto_crea[i[1]]*tto_crea[i[2]]*tto_anni[i[3]]*tto_anni[i[4]]
            A = A+V[i]*H
        end
        if i_rnd > n_rnd 
            A = tt_rounding(A,tol=tol)
            i_rnd = 1
        else
            i_rnd+=1
        end
    end
    A = tt_rounding(A,tol=tol)
    return A
end

"""
for normal ordered Hamiltonians
"""
function normal_ordering(p,q,r,s,n)
    if s≤n  
        if r≤n 
            return r,s,p,q,1.0
        else 
            return s,p,q,r,-1.0
        end
    else 
        if r≤n 
            return r,p,q,s,1.0
        end
    end
    if p≤n 
        return q,p,r,s,-1.0
    end 
    return p,q,r,s,1.0
end

function hV_no_to_mpo(h::Array{T,2},V,dims::NTuple{N,Int64};n=ceil(Int64,L/2),tol=1e-8::Float64,n_rnd=20::Int) where {T,N}
    L = size(h,1)
    @assert issymmetric(h)
#    @assert isapprox(V,permutedims(V,(2,1,4,3)))
    A = zeros_tto(T,dims,ones(Int64,L+1))
    i_rnd = 1
    #Precomputation of creation and annihilation operators
    tto_crea = [tto_creation(i,dims) for i in 1:L]
    tto_anni = [tto_annihilation(i,dims) for i in 1:L]
    for i in findall(x->!isapprox(x,0.0,atol=1e-12),h)
        if i[1]<i[2]
            H = tto_crea[i[1]]*tto_anni[i[2]] + tto_crea[i[2]]*tto_anni[i[1]]
            A = A+h[i]*H
        elseif i[1]==i[2]
            H = tto_crea[i[1]]*tto_anni[i[2]]
            A = A+h[i]*H
        end
        if i_rnd > n_rnd 
            A = tt_rounding(A,tol=tol)
            i_rnd = 1
        else
            i_rnd+=1
        end
    end
    for i in findall(x->!isapprox(x,0.0,atol=1e-12),V)
        p,q,r,s,ϕ =  normal_ordering(i[1],i[2],i[3],i[4],n)
        H = tto_crea[p]*tto_crea[q]*tto_anni[r]*tto_anni[s]
        A = A+ϕ*V[i]*H
        if i_rnd > n_rnd 
            A = tt_rounding(A,tol=tol)
            i_rnd = 1
        else
            i_rnd+=1
        end
    end
    A = tt_rounding(A,tol=tol)
    return A
end

"""
Examples of standard Hamiltonians
"""

#returns the h,V for the hubbard model (assuming natural labelling of sites)
function hubbard_1D(L;t=1,U=1, pbc=false)
    h = zeros(2L,2L)
    V = zeros(2L,2L,2L,2L)
    for i in 1:L-1
        h[2i,2i+2]=-t
        h[2i-1,2i+1]=-t
        V[2i-1,2i,2i-1,2i]=-U #because of the anticommutation rules
        V[2i,2i-1,2i,2i-1]=-U #because of the anticommutation rules
    end
    V[2L-1,2L,2L-1,2L]=-U
    V[2L,2L-1,2L,2L-1]=-U
    if pbc 
        h[2L-1,1]=-t
        h[2L,2]=-t
    end
    return Symmetric(h),V
end

#switch sites j and k in a one-dimensional spin chain model
function site_switch(j::Integer,k::Integer,L::Integer)
    out = collect(1:2L)
    out[2j-1],out[2j] = 2k-1,2k
    out[2k-1],out[2k] = 2j-1,2j
    return out
end

#2D Hubbard with cylindrical boundary conditions along L
# TO MODIFY
function hubbard_2D(w,L;t=1,U=1,w_pbc = false,L_pbc=true)
    h = zeros(2w*L,2w*L)
    V = zeros(2w*L,2w*L,2w*L,2w*L)
    for i in 0:w-2
        for j in 0:L-2
            V[2(j*w+i)+1,2(j*w+i)+2,2(j*w+i)+1,2(j*w+i)+2] = -U
            V[2(j*w+i)+2,2(j*w+i)+1,2(j*w+i)+2,2(j*w+i)+1] = -U
            h[2(j*w+i)+1,2((j+1)*w+i)+1] = -t #right connection
            h[2(j*w+i)+1,2(j*w+i+1)+1] = -t #bottom connection
            h[2(j*w+i)+2,2((j+1)*w+i)+2] = -t #right connection
            h[2(j*w+i)+2,2(j*w+i+1)+2] = -t #bottom connection
        end
        #j=L-1
        V[2((L-1)*w+i)+1,2((L-1)*w+i)+2,2((L-1)*w+i)+1,2((L-1)*w+i)+2] = -U
        V[2((L-1)*w+i)+2,2((L-1)*w+i)+1,2((L-1)*w+i)+2,2((L-1)*w+i)+1] = -U
        h[2((L-1)*w+i)+1,2((L-1)*w+i+1)+1] = -t #bottom connection
        h[2((L-1)*w+i)+2,2((L-1)*w+i+1)+2] = -t #bottom connection
        if L_pbc && L>2
            h[2(L*w+i-w)+1,2i+1] =-t #right connection for j=L-1
            h[2(L*w+i-w)+2,2i+2] =-t #right connection for j=L-1
        end
    end
    #i = w-1
    for j in 0:L-2
        V[2(j*w+w-1)+1,2(j*w+w-1)+2,2(j*w+w-1)+1,2(j*w+w-1)+2] = -U
        V[2(j*w+w-1)+2,2(j*w+w-1)+1,2(j*w+w-1)+2,2(j*w+w-1)+1] = -U
        h[2(j*w+w-1)+1,2((j+1)*w+w-1)+1] = -t #right connection
        h[2(j*w+w-1)+2,2((j+1)*w+w-1)+2] = -t #right connection
        if w_pbc && w>2
            h[2(j*w+w-1)+1,2j*w+1] = -t #top connection
            h[2(j*w+w-1)+2,2j*w+2] = -t #top connection
        end
    end
    #i = w-1, j=L-1
    V[2L*w-1,2L*w,2L*w-1,2L*w] = -U
    V[2L*w,2L*w-1,2L*w,2L*w-1] = -U
    if L_pbc && L>2
        h[2(L*w-1)+1,2(w-1)+1] = -t #right connection
        h[2(L*w-1)+2,2(w-1)+2] = -t #right connection
    end
    if w_pbc && w>2
        h[2(L*w-1)+1,2(L-1)*w+1] = -t #right connection
        h[2(L*w-1)+2,2(L-1)*w+2] = -t #right connection
    end
    return Symmetric(h),V
end

"""
PPP Hamiltonian for cyclic polyene C_NH_N (ref. G. Fano, F. Ortolani † and L. Ziosi, The density matrix renormalization group method. Application to the PPP model of a cyclic polyene chain, J. Chem. Phys. 1998)
H = β ∑_{<μ,ν>,σ} a^†_μσ a^†_νσ + c.c. + 0.5 ∑_μ,ν γ_μν (n_μ-1)(n_ν-1)

β = -2.5eV, γ_μν = 1/(γ0^-1+d_μν), γ0 = 10.84eV
d_μν = b sin(π/N*(μ-ν [N]))/sin(π/N), b = 1.4 A

Ground-state energy = -12.72 eV (N=6)
"""
function PPP_C_NH_N(N;β=-2.5/27.2113845,b=1.4*1.8897259886,γ0=10.84/27.2113845,order=collect(1:2N),tol=1e-8)
    @assert isperm(order) "Ordering given is not a permutation."
    dims = ntuple(x->2,2N)
    h = zeros(2N,2N)
    γ = sum(1/(1/γ0+b*sin(k/N*pi)/sin(pi/N)) for k in 1:N)
    for i in 1:N-1
        h[2i-1,2i+1],h[2i+1,2i-1] = β,β
        h[2i,2i+2],h[2i+2,2i] = β,β
    end
    h[2N,2],h[2,2N] = β,β
    h[2N-1,1],h[1,2N-1] = β,β
    H_tto = hV_to_mpo(Symmetric(h)[order,order],zeros(2N,2N,2N,2N),dims)
    σ = invperm(order)
    id = -1.0*id_tto(2N)
    for i in 1:N
        Hi = (one_body_mpo(σ[2i],σ[2i],H_tto.tto_dims)+one_body_mpo(σ[2i-1],σ[2i-1],H_tto.tto_dims))+id
        for j in 1:N
            Hj = (one_body_mpo(σ[2j],σ[2j],H_tto.tto_dims)+one_body_mpo(σ[2j-1],σ[2j-1],H_tto.tto_dims)) + id 
            Htemp = Hi*Hj
            γij = 1/(1/γ0+b*sin(pi/N*mod(i-j,N))/sin(pi/N))
            H_tto = H_tto + 0.5*γij*Htemp
        end
        H_tto = tt_rounding(H_tto,tol=tol)
    end
    return H_tto
end

"""
Returns the particle number operator ̂N of L sites
"""
function part_num(dims::NTuple{N,Int64}) where N
    A = zeros_tto(dims,ones(Int64,N+1))
    for i in 1:N
        A = A + one_body_mpo(i,i,dims)
    end
    return tt_rounding(A)
end

"""
1e and 2e integrals coefficients to h,V coefficients of the second quantized Hamiltonian in the spin orbital basis
"""
function one_e_two_e_integrals_to_hV(int1e,int2e)
    L = size(int1e,1)
    h = zeros(2L,2L)
    V = zeros(2L,2L,2L,2L)
    for i in 1:L
        for j in 1:L
            h[2i-1,2j-1] = int1e[i,j] #spin up integrals 
            h[2i,2j] = int1e[i,j]
            for k in 1:L
                for l in 1:L
                    V[2i-1,2j-1,2l-1,2k-1] = int2e[i,k,j,l]
                    V[2i,2j-1,2l-1,2k] = int2e[i,k,j,l]
                    V[2i-1,2j,2l,2k-1] = int2e[i,k,j,l]
                    V[2i,2j,2l,2k] = int2e[i,k,j,l]
                end
            end
        end
    end
    return h,0.5*V
end