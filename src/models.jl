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
    return ttvector(tt_vec,2*ones(Int,2N),ones(Int,2N),zeros(2N))
end

#auxiliary functions for a_p^†a_q
function mpo_core_id()
    out = zeros(2,2,1,1)
    out[1,1,1,1] = 1.0
    out[2,2,1,1] = 1.0
    return out
end

function mpo_core_ferm_sign()
    out = zeros(2,2,1,1)
    out[1,1,1,1] = 1.0
    out[2,2,1,1] = -1.0
    return out
end

function mpo_core_creation()
    out = zeros(2,2,1,1)
    out[2,1,1,1] = 1.0
    return out
end

function mpo_core_annihilation()
    out = zeros(2,2,1,1)
    out[1,2,1,1] = 1.0
    return out
end

#returns bosonic or fermionic MPO of a_p^†a_q
function one_body_mpo(p,q,L;fermion=true)
    H = Array{Array{Float64,4},1}(undef,L)
    if p == q
        for i in 1:L
            H[i] = mpo_core_id()
        end
        H[p][1,1,1,1] = 0.0
    else
        H[p] = mpo_core_creation()
        H[q] = mpo_core_annihilation()
        for i in 1:min(p,q)-1
            H[i] = mpo_core_id()
        end
        for i in max(p,q)+1:L
            H[i] = mpo_core_id()
        end
        for i in min(p,q)+1:max(p,q)-1
            if fermion
                H[i] = mpo_core_ferm_sign()
            else
                H[i] = mpo_core_id()
            end
        end
    end 
    return ttoperator(H,2*ones(Int64,L),ones(Int64,L),zeros(Int64,L))
end


#returns bosonic or fermionic MPO of a_k^† a^†_l a_m a_n 
#assume k<l,m<n
function two_body_mpo(k,l,m,n,L)
    H = Array{Array{Float64,4},1}(undef,L) 
    if l == m
        A = one_body_mpo(k,m,L)
        B = one_body_mpo(l,n,L)
        C = one_body_mpo(k,n,L)
        return tto_add(mult_a_tt(-1.0,mult(A,B)),C)
    else
        A = one_body_mpo(k,m,L)
        B = one_body_mpo(l,n,L)
        return mult_a_tt(-1.0,mult(A,B))
    end
end


"""
returns an MPO version of 
H = Σ_{ij} h_ij (a_i† a_j + c.c.) + Σ_{ijkl} V_{ijkl} (a_i†a_j†a_ka_l + c.c.)
"""
#assuming diagonal terms are divided by 2 in the h and V matrix
function hV_to_mpo(h,V;tol=1e-10)
    L = size(h,1)
    i_list = findall(!iszero,h)
    if length(i_list)>0
        i = i_list[1]
        A = mult_a_tt(h[i],tto_add(one_body_mpo(i[1],i[2],L),one_body_mpo(i[2],i[1],L)))
        for i in i_list[2:end]
            H = one_body_mpo(i[1],i[2],L)
            G = one_body_mpo(i[2],i[1],L)
            A = tto_add(A,mult_a_tt(h[i],tto_add(G,H)))
        end
        A = tt_rounding(A,tol=tol)
    end
    for i in findall(!iszero,V)
        H = two_body_mpo(i[1],i[2],i[3],i[4],L)
        G = two_body_mpo(i[4],i[3],i[2],i[1],L)
        A = tto_add(A,mult_a_tt(V[i],tto_add(H,G)))
    end
    return tt_rounding(A,tol=tol)
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
    end
    V[2L-1,2L,2L-1,2L]=-U
    if pbc 
        h[2L-1,1]=-t
        h[2L,2]=-t
    end
    return h,V
end

#switch sites j and k in a one-dimensional spin chain model
function site_switch(j::Integer,k::Integer,L::Integer)
    out = collect(1:2L)
    out[2j-1],out[2j] = 2k-1,2k
    out[2k-1],out[2k] = 2j-1,2j
    return out
end

#2D Hubbard with cylindrical boundary conditions along L
function hubbard_2D(w,L;t=1,U=1,w_pbc = false,L_pbc=true)
    h = zeros(2w*L,2w*L)
    V = zeros(2w*L,2w*L,2w*L,2w*L)
    for i in 0:w-2
        for j in 0:L-2
            V[2(j*w+i)+1,2(j*w+i)+2,2(j*w+i)+1,2(j*w+i)+2] = -U
            h[2(j*w+i)+1,2((j+1)*w+i)+1] = -t #right connection
            h[2(j*w+i)+1,2(j*w+i+1)+1] = -t #bottom connection
            h[2(j*w+i)+2,2((j+1)*w+i)+2] = -t #right connection
            h[2(j*w+i)+2,2(j*w+i+1)+2] = -t #bottom connection
        end
        #j=L-1
        V[2((L-1)*w+i)+1,2((L-1)*w+i)+2,2((L-1)*w+i)+1,2((L-1)*w+i)+2] = -U
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
        h[2(j*w+w-1)+1,2((j+1)*w+w-1)+1] = -t #right connection
        h[2(j*w+w-1)+2,2((j+1)*w+w-1)+2] = -t #right connection
        if w_pbc && w>2
            h[2(j*w+w-1)+1,2j*w+1] = -t #top connection
            h[2(j*w+w-1)+2,2j*w+2] = -t #top connection
        end
    end
    #i = w-1, j=L-1
    V[2L*w-1,2L*w,2L*w-1,2L*w] = -U
    if L_pbc && L>2
        h[2(L*w-1)+1,2(w-1)+1] = -t #right connection
        h[2(L*w-1)+2,2(w-1)+2] = -t #right connection
    end
    if w_pbc && w>2
        h[2(L*w-1)+1,2(L-1)*w+1] = -t #right connection
        h[2(L*w-1)+2,2(L-1)*w+2] = -t #right connection
    end
    return h,V
end

"""
PPP Hamiltonian for cyclic polyene C_NH_N (ref. G. Fano, F. Ortolani † and L. Ziosi, The density matrix renormalization group method. Application to the PPP model of a cyclic polyene chain, J. Chem. Phys. 1998)
H = β ∑_{<μ,ν>,σ} a^†_μσ a^†_νσ + c.c. + 0.5 ∑_μ,ν (n_μ-1)(n_ν-1)

β = -2.5eV, γ_μν = 1/(γ0^-1+d_μν), γ0 = 10.84eV
d_μν = b sin(π/N*(μ-ν [N]))/sin(π/N), b = 1.4 A

Ground-state energy = -12.72 eV (N=6)
"""
function PPP_C_NH_N(N;β=-2.5/27.2113845,b=1.4*1.8897259886,γ0=10.84/27.2113845,order=collect(1:2N),tol=1e-10)
    @assert(isperm(order),"Ordering given is not a permutation.")
    h = zeros(2N,2N)
    γ = sum(1/(1/γ0+b*sin(k/N*pi)/sin(pi/N)) for k in 1:N)
    for i in 1:N-1
        h[2i-1,2i+1] = β
        h[2i,2i+2] = β
    end
    h[2N,2] = β
    h[2N-1,1] = β
    H_tto = hV_to_mpo(h[order,order],zeros(2N,2N,2N,2N))
    σ = invperm(order)
    for i in 1:N
        Hi = tto_add(tto_add(one_body_mpo(σ[2i],σ[2i],2N),one_body_mpo(σ[2i-1],σ[2i-1],2N)),mult_a_tt(-1.0,id_tto(2N)))
        for j in 1:N
            Hj = tto_add(tto_add(one_body_mpo(σ[2j],σ[2j],2N),one_body_mpo(σ[2j-1],σ[2j-1],2N)),mult_a_tt(-1.0,id_tto(2N)))
            Htemp = mult(Hi,Hj)
            γij = 1/(1/γ0+b*sin(pi/N*mod(i-j,N))/sin(pi/N))
            H_tto = tto_add(H_tto,mult_a_tt(0.5*γij,Htemp))
        end
        H_tto = tt_rounding(H_tto,tol=tol)
    end
    return H_tto
end
