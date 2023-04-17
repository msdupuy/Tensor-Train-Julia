using Base.Threads
using LinearAlgebra

#local ttvec rank increase function with noise ϵ_wn
function tt_up_rks_noise(tt_vec,tt_ot_i,rkm,rk,ϵ_wn)
	vec_out = zeros(eltype(tt_vec),size(tt_vec,1),rkm,rk)
	vec_out[:,1:size(tt_vec,2),1:size(tt_vec,3)] = tt_vec
	if !iszero(ϵ_wn)
		if rkm == size(tt_vec,2) && rk>size(tt_vec,3)
			Q = rand_orthogonal(size(tt_vec,1)*rkm,rk-size(tt_vec,3))
			vec_out[:,:,size(tt_vec,3)+1:rk] = ϵ_wn*reshape(Q,size(tt_vec,1),rkm,rk-size(tt_vec,3))
			tt_ot_i =0
		elseif rk == size(tt_vec,3) && rkm>size(tt_vec,2)
			Q = rand_orthogonal(rkm-size(tt_vec,2),size(tt_vec,1)*rk)
			vec_out[:,size(tt_vec,2)+1:rkm,:] = ϵ_wn*reshape(Q,size(tt_vec,1),rkm-size(tt_vec,2),rk)
			tt_ot_i =0
		elseif rk>size(tt_vec,3) && rkm>size(tt_vec,2)
			Q = rand_orthogonal((rkm-size(tt_vec,2))*size(tt_vec,1),(rk-size(tt_vec,3)))
			vec_out[:,size(tt_vec,2)+1:rkm,size(tt_vec,3)+1:rk] = ϵ_wn*reshape(Q,size(tt_vec,1),rkm-size(tt_vec,2),rk-size(tt_vec,3))
		end
	end
	return vec_out
end

"""
returns the TTvector with ranks rks and noise ϵ_wn for the updated ranks
"""
function tt_up_rks(x_tt::TTvector{T},rk_max::Int;rks=vcat(1,rk_max*ones(Int,length(x_tt.ttv_dims)-1),1),ϵ_wn=0.0) where {T<:Number}
	d = x_tt.N
	vec_out = Array{Array{T}}(undef,d)
	out_ot = zeros(Int64,d)
	@assert(rk_max > maximum(x_tt.ttv_rks),"New bond dimension too low")
	n_in = 1
	n_out = prod(x_tt.ttv_dims)
	for i in 1:d
		n_in *= x_tt.ttv_dims[i]
		n_out = Int(n_out/x_tt.ttv_dims[i])
		rks[i+1] = min(rks[i+1],n_in,n_out)
		vec_out[i] = tt_up_rks_noise(x_tt.ttv_vec[i],x_tt.ttv_ot[i],rks[i],rks[i+1],ϵ_wn)
	end	
	return TTvector{T}(d,vec_out,x_tt.ttv_dims,rks,x_tt.ttv_ot)
end

"""
returns the orthogonalized TTvector with root i
"""
function orthogonalize(x_tt::TTvector{T};i=1::Int) where {T<:Number}
	d = x_tt.N
	@assert(1≤i≤d, DimensionMismatch("Impossible orthogonalization"))
	y_tt = copy(x_tt)
	y_tt.ttv_ot[i]=0
	for j in 1:i-1
		y_tt.ttv_ot[j]=1
		y_vectemp = reshape(y_tt.ttv_vec[j],y_tt.ttv_dims[j]*y_tt.ttv_rks[j],y_tt.ttv_rks[j+1])
		F = qr(y_vectemp)
		y_tt.ttv_vec[j] = reshape(Matrix(F.Q)[:,1:y_tt.ttv_rks[j+1]],x_tt.ttv_dims[j],y_tt.ttv_rks[j],y_tt.ttv_rks[j+1])
		@threads for k in 1:x_tt.ttv_dims[j]
			y_tt.ttv_vec[j+1][k,:,:] = F.R[1:y_tt.ttv_rks[j+1],1:y_tt.ttv_rks[j+1]]*y_tt.ttv_vec[j+1][k,:,:]
		end
	end
	for j in d:-1:i+1
		y_tt.ttv_ot[j]=-1
		y_vectemp = reshape(permutedims(y_tt.ttv_vec[j],[2,1,3]),y_tt.ttv_rks[j],y_tt.ttv_dims[j]*y_tt.ttv_rks[j+1])
		F = lq(y_vectemp)
		y_tt.ttv_rks[j] = size(Matrix(F.Q),1)
		y_tt.ttv_vec[j] = permutedims(reshape(Matrix(F.Q)[1:y_tt.ttv_rks[j],:],y_tt.ttv_rks[j],x_tt.ttv_dims[j],y_tt.ttv_rks[j+1]),[2 1 3])
		y_tt_temp = copy(y_tt.ttv_vec[j-1])
		y_tt.ttv_vec[j-1] = zeros(T,x_tt.ttv_dims[j],y_tt.ttv_rks[j-1],y_tt.ttv_rks[j])
		@threads for k in 1:x_tt.ttv_dims[j]
			y_tt.ttv_vec[j-1][k,:,:] = y_tt_temp[k,:,:]*F.L[:,1:y_tt.ttv_rks[j]]
		end
	end
	return y_tt
end

function cut_off_index(s::Array{T}, tol::Float64; degen_tol=1e-10) where {T<:Number}
	k = sum(s.>norm(s)*tol)
	while k<length(s) && isapprox(s[k],s[k+1];rtol=degen_tol, atol=degen_tol)
		k = k+1
	end
	return k
end

"""
returns a TT representation where the singular values lower than tol are discarded
"""
function tt_rounding(x_tt::TTvector{T};tol=1e-12) where {T<:Number}
	d = x_tt.N
	y_rks = copy(x_tt.ttv_rks)
	y_vec = copy(x_tt.ttv_vec)
	for j in 1:d-1
		A = zeros(T,x_tt.ttv_dims[j],y_rks[j],x_tt.ttv_dims[j+1],y_rks[j+2])
		@tensor A[a,b,c,d] = y_vec[j][a,b,z]*y_vec[j+1][c,z,d]
		u,s,v = svd(reshape(A,size(A,1)*size(A,2),:),full=false)
		k = cut_off_index(s,tol)
		Σ = s[1:k]
		y_rks[j+1] = length(Σ)
		y_vec[j] = reshape(u[:,1:k],x_tt.ttv_dims[j],y_rks[j],:)
		y_vec[j+1] = permutedims(reshape(Diagonal(Σ)*v'[1:k,:],:,x_tt.ttv_dims[j+1],y_rks[j+2]),[2 1 3])
	end
	for j in d:-1:2
		A = zeros(T,x_tt.ttv_dims[j-1],y_rks[j-1],x_tt.ttv_dims[j],y_rks[j+1])
		@tensor A[a,b,c,d] = y_vec[j-1][a,b,z]*y_vec[j][c,z,d]
		u,s,v = svd(reshape(A,size(A,1)*size(A,2),:),full=false)
		k = cut_off_index(s,tol)
		Σ = s[1:k]
		y_rks[j] = length(Σ)
		y_vec[j] = permutedims(reshape(v'[1:k,:],:,x_tt.ttv_dims[j],y_rks[j+1]),[2 1 3])
		y_vec[j-1] = reshape(u[:,1:k]*Diagonal(Σ),x_tt.ttv_dims[j-1],y_rks[j-1],:)
	end
	return TTvector{T}(d,y_vec,x_tt.ttv_dims,y_rks,vcat(0,ones(Int64,d-1)))
end

"""
returns the rounding of the TT operator
"""
function tt_rounding(A_tto::TToperator;tol=1e-12)
	return ttv_to_tto(tt_rounding(tto_to_ttv(A_tto);tol=tol))
end

"""
returns the singular values of the reshaped tensor x[μ_1⋯μ_k;μ_{k+1}⋯μ_d] for all 1≤ k ≤ d
"""
function tt_svdvals(x_tt::TTvector{T};tol=1e-14) where {T<:Number}
	d = x_tt.N
	Σ = Array{Array{Float64,1},1}(undef,d-1)
	y_tt = orthogonalize(x_tt)
	y_rks = y_tt.ttv_rks
	for j in 1:d-1
		A = zeros(y_tt.ttv_dims[j],y_rks[j],y_tt.ttv_dims[j+1],y_rks[j+2])
		@tensor A[a,b,c,d] = y_tt.ttv_vec[j][a,b,z]*y_tt.ttv_vec[j+1][c,z,d]
		u,s,v = svd(reshape(A,size(A,1)*size(A,2),:),alg=LinearAlgebra.QRIteration())
		Σ[j] = s[s.>tol]
		y_rks[j+1] = length(Σ[j])
		y_tt.ttv_vec[j+1] = permutedims(reshape(Diagonal(Σ[j])*v'[s.>tol,:],:,y_tt.ttv_dims[j+1],y_rks[j+2]),[2 1 3])
	end
	return Σ
end

function sv_trunc(s::Array{Float64},tol)
	if tol==0.0
		return s
	else
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

"""
parallel compression of the TTvector
TODO refactoring
"""
function tt_compression_par(X::TTvector;tol=1e-14,Imax=2)
    Y = deepcopy(X.ttv_vec) :: Array{Array{Float64,3},1}
    rks = deepcopy(X.ttv_rks) :: Array{Int64}
	d = x_tt.N
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
    return TTvector(d,Y,X.ttv_dims,rks,zeros(Integer,d))
end

function tt_compression_par(A::TToperator;tol=1e-14,Imax=2)
	return ttv_to_tto(tt_compression_par(tto_to_ttv(A);tol=tol,Imax=Imax))
end
