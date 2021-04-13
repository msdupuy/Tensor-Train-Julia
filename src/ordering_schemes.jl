using StatsBase
using Random
using Combinatorics

"""
ordering schemes for QC-DMRG or 2D statistical models
"""

#returns the list of one-orbital reduced density matrix
function one_rdm(x_tt::ttvector{T}) where T<:Number
    d = length(x_tt.ttv_dims)
    @assert(2*ones(Int,d)==x_tt.ttv_dims)
    γ = zeros(T,d,2,2)
    for i in 1:d
        y_tt = one_body_mpo(i,i,d;T=T)*x_tt
        γ[i,2,2] = dot(x_tt,y_tt)
        γ[i,1,1] = 1-γ[i,2,2]
    end
    return γ
end


#returns the list of two-orbitals reduced density matrix
function two_rdm(x_tt::ttvector{S};fermion=true) where S<:Number
    d = length(x_tt.ttv_dims)
    @assert(2*ones(Int,d)==x_tt.ttv_dims)
    γ = zeros(S,d,d,2,2,2,2) #(i,j;i,j) occupancy
    for i in 1:d-1
        for j in i+1:d
            γ[i,j,2,2,2,2] = -dot(x_tt,two_body_mpo(i,j,i,j,d,T=S)*x_tt)
            γ[i,j,1,2,1,2] = -γ[i,j,2,2,2,2] + dot(x_tt,one_body_mpo(j,j,d,T=S)*x_tt)
            γ[i,j,2,1,2,1] = -γ[i,j,2,2,2,2] + dot(x_tt,one_body_mpo(i,i,d,T=S)*x_tt)
            γ[i,j,1,1,1,1] = 1.0 -γ[i,j,2,2,2,2] -γ[i,j,2,1,2,1] -γ[i,j,1,2,1,2] 
            γ[i,j,2,1,1,2] = dot(x_tt,one_body_mpo(i,j,d;fermion=fermion,T=S)*x_tt)
            γ[i,j,1,2,2,1] = γ[i,j,2,1,1,2]
        end
    end
    return γ
end


"""
returns the entropy of a matrix M
if a==1 : von Neumann entropy, else Renyi entropy
"""
function entropy(a,M)
   x = eigvals(M)
   x = x[x.>1e-15]
   if a==Inf
      return -log(maximum(x))
   elseif isapprox(a,1.)
      return -sum(x.*log.(x))
   else
      return 1/(1-a)*log.(sum(x.^a))
   end
end

#γ_1 = one orbital RDM, γ_2 = two orbital RDM
function mutual_information(γ1::Array{T,3},γ2::Array{T,6};a=1) where T<:Number #a=1 von neumann entropy
    d = size(γ1,1)
    IM = zeros(T,d,d)
    s1 = [entropy(a,γ1[i,:,:]) for i in 1:d]
    for i in 1:d-1
        for j in i+1:d
            IM[i,j] = s1[i]+s1[j]-entropy(a,reshape(γ2[i,j,:,:,:,:],4,4))
        end
    end
    return Hermitian(IM)
end

#returns the fiedler order of a mutual information matrix IM
function fiedler(IM)
   L = size(IM)[1]
   Lap = Diagonal([sum(IM[i,:]) for i=1:L]) - IM
   F = eigen(Lap)
   @assert isapprox(F.values[1],0.,atol=1e-14)
   return sortperm(F.vectors[:,2]) #to get 2nd eigenvector
end

"""
Returns the Fiedler order of the state `x_tt`, assumed to be normalized.
"""
function fiedler_order(x_tt::ttvector;a=1) #a=1 : von Neumann entropy
    γ1 = one_rdm(x_tt)
    γ2 = two_rdm(x_tt)
    IM = mutual_information(γ1,γ2;a=a)
    return fiedler(IM)
end

#returns the one particle reduced density matrix of a state encoded in the TT x_tt
function one_prdm(x_tt::ttvector{T}) where T<:Number
    d = length(x_tt.ttv_dims)
    γ = zeros(T,d,d)
    for i in 1:d
        γ[i,i] = dot(x_tt,one_body_mpo(i,i,d;T=T)*x_tt)
        for j in i+1:d
            γ[i,j] = dot(x_tt,one_body_mpo(i,j,d;T=T)*x_tt)
        end
    end
    return Hermitian(γ)
end

function cost(x;tol=1e-10)
    return -sum(log10.((x.+tol).^2.0 .*((1+tol).-x.^2.0)))
end

#best weighted prefactor order
#Warning: V needs to be given in a row-"occupancy" way i.e. V[i,:] represents the coefficients of the natural orbital ψ_i in the basis of the ϕ_j, 1 ≤ j ≤ L

function bwpo_order(V,N,L;
    pivot = round(Int,L/2),nb_l = pivot,
    nb_r = L-pivot, order=collect(1:L),
    CAS=collect(1:N),imax=1000,rand_or_full=500, tol =1e-8, temp=1e-4)
    if imax <= 0 || min(nb_l,nb_r) == 0
        return order[pivot-nb_l+1:pivot+nb_r]
    else
        x_N = order
        cost_max = cost(ones(min(pivot,L-pivot)),tol=tol)*size(CAS,2)
        prefactor = sum([cost(svdvals(V[CAS[:,i],x_N[1:pivot]]),tol=tol) for i in size(CAS,2)]) 
        iter = 0
        if binomial(nb_r+nb_l,min(nb_l,nb_r)) > rand_or_full
            while iter < imax && temp*prefactor/(imax*cost_max) < rand()
                #nouveau voisin
                x_temp = vcat(order[1:(pivot-nb_l)],shuffle(order[pivot-nb_l+1:pivot+nb_r]),order[pivot+nb_r+1:L])
                new_prefactor = sum([cost(svdvals(V[CAS[:,i],x_temp[1:pivot]]),tol=tol) for i in size(CAS,2)])
                if new_prefactor > prefactor
                    x_N = x_temp
                    prefactor = new_prefactor
                end
                iter = iter+1
            end
        else #do exhaustive search in the space of the combinations
            combs_list = collect(combinations(order[pivot-nb_l+1:pivot+nb_r],nb_l))
            for σ in combs_list
                σ_c = setdiff(order[pivot-nb_l+1:pivot+nb_r],σ)
                x_temp = vcat(order[1:(pivot-nb_l)],σ,σ_c,order[pivot+nb_r+1:L])
                new_prefactor = sum([cost(svdvals(V[CAS[:,i],x_temp[1:pivot]]),tol=tol) for i in size(CAS,2)])
                if new_prefactor > prefactor
                    x_N = x_temp
                    prefactor = new_prefactor
                end
            end
        end
        pivotL = pivot-nb_l+1+round(Int,(nb_l-1)/2)
        order_l = bwpo_order(V,N,L; pivot=pivotL, nb_l=round(Int,(nb_l-1)/2)+1, nb_r=nb_l-1-round(Int,(nb_l-1)/2), order=x_N, CAS=CAS,imax=imax-iter,rand_or_full=rand_or_full, tol =tol, temp=temp)
        pivotR = pivot + round(Int,nb_r/2)
        order_r = bwpo_order(V,N,L; pivot=pivotR, nb_l=round(Int,nb_r/2), nb_r=nb_r-round(Int,nb_r/2), order=x_N, CAS=CAS,imax=imax-iter,rand_or_full=rand_or_full, tol =tol, temp=temp)
        return vcat(order_l,order_r)::Array{Int,1}
    end
end

function bwpo_order(ψ_tt::ttvector;order = collect(1:length(ψ_tt.ttv_dims)),tol=1e-8,imax=2000,rand_or_full=500,temp=1e-4)
    γ = one_prdm(ψ_tt)
	N = round(Int,tr(γ))
    F = eigen(γ)
    V = reverse(F.vectors,dims=2)[:,1:N]'
    return bwpo_order(V,N,length(ψ_tt.ttv_dims),tol=tol,imax=imax,rand_or_full=rand_or_full,temp=temp)
end

function bwpo_order(γ::AbstractArray{Float64,2};order = collect(1:size(γ,1)),CAS=collect(1:round(Int,tr(γ))),tol=1e-8,imax=2000,rand_or_full=500,temp=1e-4)
	N = round(Int,tr(γ))
    F = eigen(γ)
    V = reverse(F.vectors,dims=2)[:,:]'
    return bwpo_order(V,N,size(γ,1);CAS=CAS,tol=tol,imax=imax,rand_or_full=rand_or_full,temp=temp)
end

"""
generate CAS from a list of occupied orbitals and a list of virtual orbitals
"""
function CAS_generator(N::Int,n_occ::Array{Int,1},n_virt::Array{Int,1})
    @assert isempty(intersect(n_occ,n_virt))
    CAS = collect(1:N)
    n_frozen = setdiff(1:N,n_occ)
    for occ in combinations(n_occ,2)
        for virt in combinations(n_virt,2)
            CAS = hcat(CAS,vcat(n_frozen, setdiff(n_occ,occ),virt ))
        end
    end
    return CAS
end