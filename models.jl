include("tt_tools.jl")
include("sptensors.jl")
include("als.jl")
using Combinatorics

"""
returns an MPO version of 
H = Σ_{ij} h_ij a_i† a_j + Σ_{ijkl} V_{ijkl} a_i†a_j†a_ka_l

TODO: a_i†a_j†a_ka_l in MPO, tto_add function and hubbard_1d/2d? 
test with aklt model
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

function undigits(x;base=2)
    return sum([x[k]*base^(k-1) for k in 1:length(x)])
end

function sp_one_body_to_matrix(k,l,L,N)
    Is = spzeros(2^L)
    Js = spzeros(2^L)
    Vs = spzeros(2^L)
    j = 0 #counter
    for occ_list in combinations(1:L,N)
        if (l in occ_list) && ((k==l) || !(k in occ_list))
            j+=1
            x = [m in occ_list for m in 1:L]
            Js[j] = undigits(x)+1
            Vs[j] = (-1)^sum(x[1:l-1])
            x[l] = 0
            x[k] = 1
            Is[j] = undigits(x)+1
            Vs[j] *= (-1)^sum(x[1:k-1])
        end
    end
    return sparse(Is[1:j],Js[1:j],Vs[1:j],2^L,2^L)
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

#assume k<l, m<n
function sp_two_body_to_matrix(k,l,m,n,L,N)
    Is = spzeros(2^L)
    Js = spzeros(2^L)
    Vs = spzeros(2^L)
    j = 0 #counter
    for occ_list in combinations(1:L,N)
        if (m in occ_list) && (n in occ_list) && ((k==m) || (k==n) || !(k in occ_list)) && ((l==m) || (l==n) || !(l in occ_list))
            j+=1
            x = [r in occ_list for r in 1:L]
            Js[j] = undigits(x)+1
            Vs[j] = (-1)^sum(x[1:n-1])
            x[n] = 0
            Vs[j] *= (-1)^sum(x[1:m-1])
            x[m] = 0
            Vs[j] *= (-1)^sum(x[1:l-1])
            x[l] = 1
            Vs[j] *= (-1)^sum(x[1:k-1])
            x[k] = 1
            Is[j] = undigits(x)+1
        end
    end
    return sparse(Is[1:j],Js[1:j],Vs[1:j],2^L,2^L)
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

#h and V sparse
function sp_sc_to_mat(h,V,N)
    L = size(h,1)
    A = spzeros(2^L,2^L)
    for i in findall(!iszero,h)
        H = sp_one_body_to_matrix(i[1],i[2],L,N)
        A += h[i]*(H+H')
    end
    for i in findall(!iszero,V)
        H = sp_two_body_to_matrix(i[1],i[2],i[3],i[4],L,N)
        A += V[i]*(H+H')
    end
    return dropzeros!(A)
end

#returns the h,V for the hubbard model
#odd index = spin up, even index = spin down

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

#returns MPO of a_p^†a_q
function one_body_mpo(p,q,L)
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
            H[i] = mpo_core_ferm_sign()
        end
    end 
    return ttoperator(H,2*ones(Int64,L),ones(Int64,L),zeros(Int64,L))
end

function test_1body_mpo()
    L=6
    p=rand(1:L)
    q=rand(1:L)
    H = one_body_to_matrix(p,q,L)
    Hmpo = one_body_mpo(p,q,L)
    H2 = tto_to_tensor(Hmpo)
    @test isapprox(norm(H-reshape(H2,2^L,2^L)),0.0,atol=1e-10)
end

#returns MPO of a_k^† a^†_l a_m a_n 
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

function test_2body_mpo()
    L=6
    k,l,m,n=1,2,3,4 
    H = two_body_to_matrix(k,l,m,n,L)
    Hmpo = two_body_mpo(k,l,m,n,L)
    H2 = tto_to_tensor(Hmpo)
    @test isapprox(norm(H-reshape(H2,2^L,2^L)),0.0,atol=1e-10)
end

#assuming natural labelling of sites
function hubbard_1D(L;t=1,U=1)
    h = zeros(2L,2L)
    V = zeros(2L,2L,2L,2L)
    for i in 1:L-1
        h[2i,2i+2]=-t
        h[2i-1,2i+1]=-t
        V[2i-1,2i,2i-1,2i]=-U #because of the anticommutation rules
    end
    h[2L-1,1]=-t
    h[2L,2]=-t
    V[2L-1,2L,2L-1,2L]=-U
    return h,V
end

#assuming diagonal terms are divided by 2 in the h and V matrix
function hV_to_mpo(h,V)
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
    end
    for i in findall(!iszero,V)
        H = two_body_mpo(i[1],i[2],i[3],i[4],L)
        G = two_body_mpo(i[4],i[3],i[2],i[1],L)
        A = tto_add(A,mult_a_tt(V[i],tto_add(H,G)))
    end
    return A
end

function ham_to_mpo(h,V,N)
    L = size(h,1)
    H = sp_sc_to_mat(h,V,N)
    dim = 2*ones(Int,L)
    return tto_spdecomp(sparsetensor_mat(dim,dim,H),L)
end