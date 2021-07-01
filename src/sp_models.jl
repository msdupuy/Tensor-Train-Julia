using Base: Integer
include("tt_tools.jl")

"""
sparse implementation of one_body and two_body matrices
"""

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

function ham_to_mpo(h,V,N)
    L = size(h,1)
    H = sp_sc_to_mat(h,V,N)
    dim = 2*ones(Int,L)
    return tto_spdecomp(sparsetensor_mat(dim,dim,H),L)
end
