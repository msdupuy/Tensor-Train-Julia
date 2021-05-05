using StatsBase
using Random
using Combinatorics

"""
ordering schemes for QC-DMRG or 2D statistical models
"""

"""
'one_rdm' returns the list of one-orbital reduced density matrices assuming that 'x_tt' is a pure state for some particle number
"""
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

"""
'two_rdm' returns the list of two-orbitals reduced density matrices assuming that 'x_tt' is a pure state for some particle number
"""
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
Returns the orbital reduced density matrix ρ_{i:j}, i<j.
"""
function N_rdm(x_tt::ttvector{T},i::Integer,j::Integer) where T<:Number
    @assert(i<j≤length(x_tt.ttv_dims))
    y_tt = orthogonalize(x_tt,i=i)
    ρ = zeros(T,x_tt.ttv_dims[i:j]...,x_tt.ttv_dims[i:j]...)
    index = CartesianIndices(Tuple([1:k for k in x_tt.ttv_dims[i:j]]))
    for J in index
        M = copy(y_tt.ttv_vec[i][J[1],:,:])
        for k in i+1:j
            M = M*y_tt.ttv_vec[k][J[k-i+1],:,:]
        end
        @threads for K in index
            N = copy(y_tt.ttv_vec[i][K[1],:,:])
            for k in i+1:j
                N = N*y_tt.ttv_vec[k][K[k-i+1],:,:]
            end
            ρ[J,K] = tr(M*N')
        end
    end
    return ρ
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

function bwpo_aux(x_N,V,CAS,prefactor,pivot,nb_l,nb_r,tol;σ=randperm(nb_r+nb_l))
    x_temp = vcat(x_N[1:(pivot-nb_l)],x_N[pivot-nb_l+1:pivot+nb_r][σ])
    new_prefactor = sum([cost(svdvals(V[CAS[:,i],x_temp[1:pivot]]),tol=tol) for i in size(CAS,2)])
    if new_prefactor > prefactor
        x_N = vcat(x_temp,x_N[pivot+nb_r+1:end])
        prefactor = new_prefactor
    end
    return x_N,prefactor
end
"""
Returns the best weighted prefactor order σ, which minimizes det(V[:,σ(1:L/2)]*V[:,σ(1:L/2)]')det(V[:,σ(L/2+1:L)]*V[:,σ(L/2+1:L)]')
Warning: V needs to be given in a row-"occupancy" way i.e. V[i,:] represents the coefficients of the natural orbital ψ_i in the basis of the ϕ_j, 1 ≤ j ≤ L
"""
function bwpo_order(V,N,L;
    pivot = round(Int,L/2),nb_l = pivot,
    nb_r = L-pivot, order=collect(1:L),
    CAS=collect(1:N),imax=1000,rand_or_full=500, tol =1e-8, temp=1e-4, k=4)
    if imax <= 0 
        return order[pivot-nb_l+1:pivot+nb_r]
    elseif nb_l<=0
        return order[pivot+1:pivot+k]
    elseif nb_r<=0
        return order[pivot-k+1:pivot]
    else
        x_N = order
        cost_max = cost(ones(min(pivot,L-pivot)),tol=tol)*size(CAS,2)
        prefactor = sum([cost(svdvals(V[CAS[:,i],x_N[1:pivot]]),tol=tol) for i in size(CAS,2)]) 
        iter = 0
        if binomial(nb_r+nb_l,min(nb_l,nb_r)) > rand_or_full #check the number of different combinations
            while iter < imax && temp*prefactor/(imax*cost_max) < rand()
                #selection of the new combination
                x_N,prefactor = bwpo_aux(x_N,V,CAS,prefactor,pivot,nb_l,nb_r,tol)
                iter = iter+1
            end
        else #do exhaustive search in the space of the combinations
            combs_list = collect(combinations(1:(nb_l+nb_r),nb_l))
            for σ in combs_list
                σ_c = vcat(σ,setdiff(collect(1:nb_l+nb_r),σ))
                x_N,prefactor = bwpo_aux(x_N,V,CAS,prefactor,pivot,nb_l,nb_r,tol;σ=σ_c)
            end
        end
        pivotL = max(pivot-k,0)
        order_l = bwpo_order(V,N,L; pivot=pivotL, nb_l=nb_l-pivot+pivotL, nb_r=pivot-pivotL, order=x_N, CAS=CAS,imax=imax-iter,rand_or_full=rand_or_full, tol =tol, temp=temp,k=pivot-pivotL)
        pivotR = min(pivot + k,L)
        order_r = bwpo_order(V,N,L; pivot=pivotR, nb_l=pivotR-pivot, nb_r=nb_r-pivotR+pivot, order=x_N, CAS=CAS,imax=imax-iter,rand_or_full=rand_or_full, tol =tol, temp=temp,k=pivotR-pivot)
        return vcat(order_l,order_r)::Array{Int,1}
    end
end

function bwpo_order(ψ_tt::ttvector;order = collect(1:length(ψ_tt.ttv_dims)),tol=1e-8,imax=2000,rand_or_full=500,temp=1e-4)
    γ = one_prdm(ψ_tt)
	N = round(Int,tr(γ))
    F = eigen(γ)
    V = reverse(F.vectors,dims=2)[:,1:N]'
    return bwpo_order(V,N,length(ψ_tt.ttv_dims),order=order,tol=tol,imax=imax,rand_or_full=rand_or_full,temp=temp)
end

function bwpo_order(γ::AbstractArray{Float64,2};order = collect(1:size(γ,1)),CAS=collect(1:round(Int,tr(γ))),tol=1e-8,imax=2000,rand_or_full=500,temp=1e-4)
	N = round(Int,tr(γ))
    F = eigen(γ)
    V = reverse(F.vectors,dims=2)[:,:]'
    return bwpo_order(V,N,size(γ,1);order=order,CAS=CAS,tol=tol,imax=imax,rand_or_full=rand_or_full,temp=temp)
end

function bwpo_aux_sites(x_N,V,CAS,prefactor,pivot,nb_l,nb_r,tol;σ=randperm(nb_r+nb_l))
    σ_mid = zeros(Int,2length(σ))
    for i in eachindex(σ)
        σ_mid[2i-1] = 2(pivot-nb_l)+2σ[i]-1
        σ_mid[2i] = 2(pivot-nb_l)+2σ[i]
    end
    x_temp = vcat(x_N[1:2*(pivot-nb_l)],x_N[σ_mid],x_N[2(pivot+nb_r)+1:end])
    new_prefactor = sum([cost(svdvals(V[CAS[:,i],x_temp[1:2pivot]]),tol=tol) for i in size(CAS,2)])
    if new_prefactor > prefactor
        x_N = x_temp
        prefactor = new_prefactor
    end
    return x_N,prefactor
end

#L = number of sites
function bwpo_order_sites(V,N,L;
    pivot = round(Int,L/2), nb_l = pivot,
    nb_r = L-pivot, order=collect(1:2L),
    CAS=collect(1:N),imax=1000,rand_or_full=500, tol =1e-8, temp=1e-6,k=2)
    if imax <= 0 
        return order[2(pivot -nb_l)+1:2(pivot+nb_r)]
    elseif nb_l<=0
        return order[2pivot+1:2pivot+2k]
    elseif nb_r<=0
        return order[2pivot-2k+1:2pivot]    
    else
        x_N = order
        cost_max = cost(ones(min(2pivot,2L-2pivot)),tol=tol)*size(CAS,2)
        prefactor = sum([cost(svdvals(V[CAS[:,i],x_N[1:2pivot]]),tol=tol) for i in size(CAS,2)]) 
        iter = 0
        if binomial(nb_r+nb_l,min(nb_l,nb_r)) > rand_or_full
            while iter < imax && temp*prefactor/(imax*cost_max) < rand()
                #nouveau voisin
                x_N, prefactor = bwpo_aux_sites(x_N,V,CAS,prefactor,pivot,nb_l,nb_r,tol)
                iter = iter+1
            end
        else #do exhaustive search in the space of the combinations
            combs_list = collect(combinations(1:nb_l+nb_r,nb_l))
            for σ in combs_list
                σ_c = vcat(σ, setdiff(1:nb_l+nb_r,σ))
                x_N, prefactor = bwpo_aux_sites(x_N,V,CAS,prefactor,pivot,nb_l,nb_r,tol;σ=σ_c)
            end
        end
        pivotL = max(pivot-k,0)
        order_l = bwpo_order_sites(V,N,L; pivot=pivotL, nb_l=nb_l-pivot+pivotL, nb_r=pivot-pivotL, order=x_N, CAS=CAS,imax=imax-iter,rand_or_full=rand_or_full, tol =tol, temp=temp,k=pivot-pivotL)
        pivotR = min(pivot + k,L)
        order_r = bwpo_order_sites(V,N,L; pivot=pivotR, nb_l=pivotR-pivot, nb_r=nb_r-pivotR+pivot, order=x_N, CAS=CAS,imax=imax-iter,rand_or_full=rand_or_full, tol =tol, temp=temp,k=pivotR-pivot)
        return vcat(order_l,order_r)::Array{Int,1}
    end
end

function bwpo_order_sites(ψ_tt::ttvector;order = collect(1:length(ψ_tt.ttv_dims)),tol=1e-8,imax=2000,rand_or_full=500,temp=1e-4)
    γ = one_prdm(ψ_tt)
	N = round(Int,tr(γ))
    F = eigen(γ)
    V = reverse(F.vectors,dims=2)[:,1:N]'
    return bwpo_order_sites(V,N,round(Int,length(ψ_tt.ttv_dims)/2),order=order,tol=tol,imax=imax,rand_or_full=rand_or_full,temp=temp)
end

function bwpo_order_sites(γ::AbstractArray{Float64,2};order = collect(1:size(γ,1)),CAS=collect(1:round(Int,tr(γ))),tol=1e-8,imax=2000,rand_or_full=500,temp=1e-4)
	N = round(Int,tr(γ))
    F = eigen(γ)
    V = reverse(F.vectors,dims=2)[:,:]'
    return bwpo_order_sites(V,N,round(Int,size(γ,1)/2);order=order,CAS=CAS,tol=tol,imax=imax,rand_or_full=rand_or_full,temp=temp)
end

"""
Returns the approximate BWPO order using a greedy algorithm
Warning : not efficient
"""
function greedy_bwpo(V,N,L;k=4,CAS=collect(1:N),tol=1e-8)
    order = collect(1:L)
    for i in 1:floor(Int,L/k)
        for j in (i-1)*k+1:i*k
            prefactor = sum([cost(svdvals(V[CAS[:,i],order[1:i*k]]),tol=tol) for i in size(CAS,2)])
            for σ in collect(combinations(order[(i-1)*k+1:L],k))
                test_order = vcat(order[1:(i-1)*k],σ)
                test_prefactor = sum([cost(svdvals(V[CAS[:,i],test_order]),tol=tol) for i in size(CAS,2)])
                if test_prefactor < prefactor
                    order = vcat(test_order,setdiff(order[(i-1)*k+1:L],σ))
                    prefactor = test_prefactor
                end
            end
        end
    end
    return order
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