using IterativeSolvers
using LinearMaps

"""
Implementation based on the presentation in 
Holtz, Sebastian, Thorsten Rohwedder, and Reinhold Schneider. "The alternating linear scheme for tensor optimization in the tensor train format." SIAM Journal on Scientific Computing 34.2 (2012): A683-A713.
"""

function updateH_mals!(x_vec, A_vec, Hi::AbstractArray{T,5}, Him::AbstractArray{T,5}) where T<:Number
	@tensor(Him[a,i,α,l,β] = x_vec[j,α,ϕ]*A_vec[i,l,a,z]*Hi[z,j,ϕ,k,χ]*x_vec[k,β,χ]) #size(rAim,ri,ni,ri,ni)
	nothing
end

function updateH_bim!(xtt_vec, btt_vec, Hbi, Hbim)
	Hbtemp = @view(Hbim[1:size(xtt_vec,2),:,:])
	@tensoropt((a,y),N_b[a,b] := xtt_vec[z,a,y]*view(Hbi[1:size(xtt_vec,3),:,:],:,:,:)[y,z,b]) #size(ri,rbi)
	@tensor Hbtemp[a,b,c] = N_b[a,z]*btt_vec[b,c,z] #size(ri,ni,rbim)
	nothing
end

function init_H_mals(x_tt::ttvector{T},A::ttoperator{T},rmax::Int) where T<:Number
	d = length(x_tt.ttv_dims)
	H = Array{Array{T}}(undef, d)
	H[d-1] = reshape(permutedims(A.tto_vec[d],[3,1,2,4]), :, x_tt.ttv_dims[d], 1, x_tt.ttv_dims[d], 1) #size(R^A_d, n_d, r_{d+1}, n_d, r_{d+1})
	for i = (d-1) : -1 : 2
		# Update H[i-1]
		rmax_i = min(rmax,prod(x_tt.ttv_dims[1:i]),prod(x_tt.ttv_dims[i+1:end]))
		H[i-1] = zeros(T,A.tto_rks[i],x_tt.ttv_dims[i],rmax_i,x_tt.ttv_dims[i],rmax_i)
		Hi = @view(H[i][:,:,1:x_tt.ttv_rks[i+2],:,1:x_tt.ttv_rks[i+2]])
		Him = @view(H[i-1][:,:,1:x_tt.ttv_rks[i+1],:,1:x_tt.ttv_rks[i+1]])
		updateH_mals!(x_tt.ttv_vec[i+1], A.tto_vec[i], Hi, Him)
	end
	return H
end

#function updateGip!(xtt_vec,Atto_vec,Gi,Gip)
#	Gtemp = @view(Gip[:,1:size(xtt_vec,3),:,1:size(xtt_vec,3),:])
#	@tensoropt((a,z,c), M1[a,b,c,d] := xtt_vec[y,z,a]*view(Gi[:,1:size(xtt_vec,2),:,1:size(xtt_vec,2),:],:,:,:,:,:)[y,z,b,c,d]) #size(ri,ni,rim,rAi)
#	@tensoropt((a,b,z), M2[a,b,c] := xtt_vec[y,z,a]*M1[b,y,z,c]) #size(ri,ri,rAi)
#	@tensoropt((b,d), Gtemp[a,b,c,d,e] = M2[d,b,z]*Atto_vec[a,c,z,e]) #size(nip,ri,nip,ri,rAip)
#	nothing
#end
#
#function updateG_bip!(xtt_vec,btt_vec,G_bi,G_bip)
#	Gbtemp = @view(G_bip[:,:,1:size(xtt_vec,3)])
#	@tensor begin
#		M_b[a,b] := view(G_bi[:,:,1:size(xtt_vec,2)],:,:,:)[a,y,z]*xtt_vec[y,z,b] #size(rbi,ri)
#		Gbtemp[a,b,c] = btt_vec[b,z,a]*M_b[z,c] #size(rbip,nip,ri)
#	end
#end

function left_core_move_mals(xtt::ttvector{T},i::Integer,V::Array{T,4},tol::Float64,rmax::Integer) where T<:Number
	# Perform the truncated svd
	u_V, s_V, v_V, = svd(reshape(V, prod(size(V)[1:2]), :))
	# Determine the truncated rank
	s_trunc = sv_trunc(s_V,tol)
	# Update the ranks to the truncated one
	xtt.ttv_rks[i+1] = min(length(s_trunc),rmax)
	println("Rank: $(xtt.ttv_rks[i]),	Max rank=$rmax")
	println("Discarded weight: $((norm(s_V)-norm(s_V[1:xtt.ttv_rks[i]]))/norm(s_V))")

	# xtt.ttv_vec[i+1] = truncated Transpose(v_V)
	xtt.ttv_vec[i+1] = permutedims(reshape(v_V[:, 1:xtt.ttv_rks[i+1]],size(V,3),size(V,4),xtt.ttv_rks[i+1]),[1,3,2])
	xtt.ttv_ot[i+1] = 1

	# xtt.ttv_vec[i] = truncated u_V * Diagonal(s_V)
	xtt.ttv_vec[i] = reshape(u_V[:, 1:xtt.ttv_rks[i+1]] * Diagonal(s_trunc[1:xtt.ttv_rks[i+1]]),size(V,1),size(V,2),:)
	xtt.ttv_ot[i] = 0
	return xtt
end

function right_core_move_mals(xtt::ttvector,i::Integer,V::Array{Float64,4},tol::Float64,rmax::Integer)
	# Perform the truncated svd
	u_V, s_V, v_V, = svd(reshape(V, prod(size(V)[1:2]), :))
	# Determine the truncated rank
	s_trunc = sv_trunc(s_V,tol)
	# Update the ranks to the truncated one
	xtt.ttv_rks[i+1] = min(length(s_trunc),rmax)
	println("Rank: $(xtt.ttv_rks[i]),	Max rank=$rmax")
	println("Discarded weight: $((norm(s_V)-norm(s_V[1:xtt.ttv_rks[i]]))/norm(s_V))")

	# xtt.ttv_vec[i] = truncated u_V
	xtt.ttv_vec[i] = reshape(u_V[:, 1:xtt.ttv_rks[i+1]], size(V,1), size(V,2), xtt.ttv_rks[i+1])
	xtt.ttv_ot[i] = -1

	# xtt.ttv_vec[i+1] = truncated Diagonal(s_V) * Transpose(v_V)
	xtt.ttv_vec[i+1] = permutedims(reshape(Diagonal(s_trunc[1:xtt.ttv_rks[i+1]]) * v_V[:, 1:xtt.ttv_rks[i+1]]',xtt.ttv_rks[i+1],size(V,3),size(V,4)), [2 1 3])
	xtt.ttv_ot[i+1] = 0
	return xtt
end

function K_full_mals(Gi::AbstractArray{T,5},Hi::AbstractArray{T,5}) where T<:Number
	K = zeros(T,size(Gi,1),size(Gi,2),size(Hi,2),size(Hi,3),size(Gi,1),size(Gi,2),size(Hi,2),size(Hi,3))
	@tensor K[a,b,c,d,e,f,g,h] = Gi[a,b,e,f,z]*Hi[z,c,d,g,h]
	return K
end

function Ksolve_mals(Gi, Hi, G_bi, H_bi, rim, rip)
	@tensor begin
		K[a,b,c,d,e,f,g,h] := Gi[:,1:rim,:,1:rim,:][f,e,b,a,z]*Hi[1:rip,1:rip,:,:,:][d,h,g,c,z] #size(rim,ni,nip,rip,rim,ni,nip,rip)
		Pb[a,b,c,d] := H_bi[1:rip,:,:][d,c,z]*G_bi[:,:,1:rim][z,b,a] #size(rim,ni,nip,rip)
	end
	V = reshape(K,prod(size(K)[1:4]),:) \ Pb[:]
	return reshape(V,size(K)[1:4]...)
end

function K_eigmin_mals(Gi::Array{T,5},Hi::Array{T,5},ttv_vec_i::Array{T,3},ttv_vec_ip::Array{T,3};it_solver=false,itslv_thresh=1024::Int64,maxiter=maxiter::Int64,tol=tol::Float64) where T<:Number
	K_dims = [size(ttv_vec_i,1),size(ttv_vec_i,2),size(ttv_vec_ip,1),size(ttv_vec_ip,3)]
	Gtemp = view(Gi[:,1:K_dims[2],:,1:K_dims[2],:],:,:,:,:,:)
	Htemp = view(Hi[:,:,1:K_dims[4],:,1:K_dims[4]],:,:,:,:,:)
	if it_solver || prod(K_dims) > itslv_thresh
		H = zeros(Float64,prod(K_dims))
		function K_matfree(V;K_dims=K_dims::Array{Int64,1},H=H)
			Hrshp = reshape(H,K_dims...)
			@tensoropt((f,h), Hrshp[a,b,c,d] = Gtemp[a,b,e,f,z]*Htemp[z,c,d,g,h]*reshape(V,K_dims...)[e,f,g,h])
			return H
		end
		X0 = zeros(Float64,prod(K_dims))
		X0_temp = reshape(X0,K_dims...)
		@tensor X0_temp[a,b,c,d] = ttv_vec_i[a,b,z]*ttv_vec_ip[c,z,d]
		r = lobpcg(LinearMap(K_matfree,prod(K_dims);ishermitian = true),false,X0,1;maxiter=maxiter,tol=tol)
		return r.λ[1]::Float64, reshape(r.X[:,1],K_dims...)::Array{Float64,4}
	else
		K = K_full_mals(Gtemp,Htemp)
		F = eigen(reshape(K,prod(K_dims),:))
		return real(F.values[1])::Float64,real.(reshape(F.vectors[:,1],K_dims...))::Array{Float64,4}
	end	
end

function mals_linsolv(A :: ttoperator, b :: ttvector, tt_start :: ttvector; tol=1e-12::Float64,rmax=round(Int,sqrt(prod(tt_start.ttv_dims))))
	# mals finds the minimum of the operator J(x)=1/2*<Ax,x> - <x,b>
	# input:
	# 	A: the tensor operator in its tensor train format
	#   b: the tensor in its tensor train format
	#	tt_start: start value in its tensor train format
	#	opt_rks: rank vector considered to be optimal enough
	#	tol: tolerated inaccuracy
	# output:
	#	tt_opt: stationary point of J up to tolerated inaccuracy
	# 			in its tensor train format

	# Initialize the to be returned tensor in its tensor train format
	tt_opt = orthogonalize(tt_start)
	dims = tt_start.ttv_dims
	d = length(dims)
	# Define the array of ranks of tt_opt [r_0=1,r_1,...,r_d]
	rks = tt_start.ttv_rks
	# Define the array of ranks of A [R_0=1,R_1,...,R_d]
	A_rks = A.tto_rks
	# Define the array of ranks of b [R^b_0=1,R^b_1,...,R^b_d]
	b_rks = b.ttv_rks

	# Initialize the arrays of G and H
	G = Array{Array{Float64}}(undef, d)
	H = Array{Array{Float64}}(undef, d-1)
	# Initialize the arrays of G_b and H_b
	G_b = Array{Array{Float64}}(undef, d)
	H_b = Array{Array{Float64}}(undef, d-1)
	# Initialize G[1], G_b[1], H[d] and H_b[d]
	for i in 1:d
		rmax_i = min(rmax,prod(dims[1:i-1]),prod(dims[i:end]))
		G[i] = zeros(dims[i],rmax_i,dims[i],rmax_i,A_rks[i+1])
		G_b[i] = zeros(b_rks[i+1],dims[i],rmax_i)
	end
	for i in 1:d-1
		rmax_i = min(rmax,prod(dims[1:i+1]),prod(dims[i+2:end]))
		H[i] = zeros(rmax_i,rmax_i,dims[i+1],dims[i+1],A_rks[i+1])
		H_b[i] = zeros(rmax_i,dims[i+1],b_rks[i])
	end
	G[1][:,1:1,:,1:1,:] = reshape(A.tto_vec[1][:,:,1,:], dims[1],1,dims[1], 1, :)
	G_b[1][:,:,1:1] = permutedims(reshape(b.ttv_vec[1][:,1,1:b_rks[2]], dims[1], 1, :), [3 1 2])
	H[d-1][1:1,1:1,:,:,:] = reshape(A.tto_vec[d], 1, 1, dims[d], dims[d], :) # k'_d,k_d,x_d,y_d,j_(d-1)
	H_b[d-1][1:1,:,:] = reshape(b.ttv_vec[d], 1, dims[d], :) # k_d, x_d, j^b_(d-1)

	#while 1==1 #TODO make it work for real
		for i = (d-1) : -1 : 2
			# Update H[i-1], H_b[i-1]
			updateHim!(tt_opt.ttv_vec[i+1], A.tto_vec[i], H[i], H[i-1])
			updateH_bim!(tt_opt.ttv_vec[i+1], b.ttv_vec[i], H_b[i],H_b[i-1])
		end

		# First half sweep
		for i = 1:(d-1)
			# If i is the index of the core matrices do the optimization
			if tt_opt.ttv_ot[i] == 0
				# Define V as solution of K*x=P2b in x
				V = Ksolve_mals(G[i],H[i],G_b[i],H_b[i],size(tt_opt.ttv_vec[i],2),size(tt_opt.ttv_vec[i+1],3))
				tt_opt = right_core_move_mals(tt_opt,i,V,tol,rmax)
			end
			# Update G[i+1],G_b[i+1]
			updateGip!(tt_opt.ttv_vec[i],A.tto_vec[i+1],G[i],G[i+1])
			updateG_bip!(tt_opt.ttv_vec[i],b.ttv_vec[i+1],G_b[i],G_b[i+1])
		end

		# Second half sweep
		for i = d-1:(-1):1
			# Define V as solution of K*x=P2b in x
			V = Ksolve_mals(G[i],H[i],G_b[i],H_b[i],size(tt_opt.ttv_vec[i],2),size(tt_opt.ttv_vec[i+1],3))
			tt_opt = left_core_move_mals(tt_opt,i,V,tol,rmax)
			# Update H[i-1], H_b[i-1]
			if i > 1
				updateHim!(tt_opt.ttv_vec[i+1], A.tto_vec[i], H[i], H[i-1])
				updateH_bim!(tt_opt.ttv_vec[i+1], b.ttv_vec[i], H_b[i],H_b[i-1])
			end
		end
	#end
	return tt_opt
end

function mals_eigsolv(A :: ttoperator{T}, tt_start :: ttvector{T}; tol=1e-12::Float64,sweep_schedule=[2]::Array{Int64,1},rmax_schedule=[round(Int,sqrt(prod(tt_start.ttv_dims)))]::Array{Int64,1},it_solver=false::Bool,linsolv_maxiter=200::Int64,linsolv_tol=max(sqrt(tol),1e-8)::Float64) where T<:Number
	# mals_eig finds the minimum of the operator J(x)=<Ax,x>/<x,x>
	# input:
	# 	A: the tensor operator in its tensor train format
	#	tt_start: start value in its tensor train format
	#	opt_rks: rank vector considered to be optimal enough
	#	tol: tolerated inaccuracy
	# output:
	#	tt_opt: stationary point of J up to tolerated inaccuracy
	# 			in its tensor train format

	@assert(length(rmax_schedule)==length(sweep_schedule),"Sweep schedule error")	
	# Initialize the to be returned tensor in its tensor train format
	tt_opt = orthogonalize(tt_start)
	dims = tt_start.ttv_dims
	d = length(dims)
	# Define the array of ranks of tt_opt [r_0=1,r_1,...,r_d]
	rks = tt_start.ttv_rks
	# Define the array of ranks of A [R_0=1,R_1,...,R_d]
	A_rks = A.tto_rks
	E = Float64[]
	r_hist = Int64[]
	# Initialize the arrays of G
	G = Array{Array{T}}(undef, d)
	rmax = maximum(rmax_schedule)
	# Initialize G[i]
	for i in 1:d
		rmax_i = min(rmax,prod(dims[1:i-1]),prod(dims[i:end]))
		G[i] = zeros(dims[i],rmax_i,dims[i],rmax_i,A_rks[i+1])
	end
	G[1][:,1:1,:,1:1,:] = reshape(A.tto_vec[1][:,:,1,:], dims[1],1,dims[1], 1, :)

	H = init_H_mals(tt_opt,A,rmax)

	nsweeps = 0 #sweeps counter
	i_schedule = 1
	while i_schedule <= length(sweep_schedule) 
		nsweeps+=1
		println("Macro-iteration $nsweeps; bond dimension $(rmax_schedule[i_schedule])")

		if nsweeps == sweep_schedule[i_schedule]
			i_schedule+=1
			if i_schedule > length(sweep_schedule)
				return E,tt_opt, r_hist
			end
		end

		# First half sweep
		for i = 1:(d-1)
			println("Forward sweep: core optimization $i out of $(d-1)")
			# If i is the index of the core matrices do the optimization
			if tt_opt.ttv_ot[i] == 0
				# Define V as solution of K*x=P2b in x
				λ,V = K_eigmin_mals(G[i],H[i],tt_opt.ttv_vec[i],tt_opt.ttv_vec[i+1];it_solver=it_solver,maxiter=linsolv_maxiter,tol=linsolv_tol)
				println("Eigenvalue: $λ")
				E = vcat(E,λ)
				tt_opt = right_core_move_mals(tt_opt,i,V,tol,rmax_schedule[i_schedule])
				r_hist = vcat(r_hist,maximum(tt_opt.ttv_rks))
			end
			# Update G[i+1],G_b[i+1]
			Gi = @view(G[i][:,1:tt_opt.ttv_rks[i],:,1:tt_opt.ttv_rks[i],:])
			Gip = @view(G[i+1][:,1:tt_opt.ttv_rks[i+1],:,1:tt_opt.ttv_rks[i+1],:])
			update_G!(tt_opt.ttv_vec[i],A.tto_vec[i+1],Gi,Gip)
		end

		# Second half sweep
		for i = d-1:(-1):1
			println("Backward sweep: core optimization $(d-i) out of $(d-1)")
			# Define V as solution of K*x=P2b in x
			λ,V = K_eigmin_mals(G[i],H[i],tt_opt.ttv_vec[i],tt_opt.ttv_vec[i+1];it_solver=it_solver,maxiter=linsolv_maxiter,tol=linsolv_tol)
			println("Eigenvalue: $λ")
			E = vcat(E,λ)
			tt_opt = left_core_move_mals(tt_opt,i,V,tol,rmax_schedule[i_schedule])
			r_hist = vcat(r_hist,maximum(tt_opt.ttv_rks))
			# Update H[i-1]
			if i > 1
				Hi = @view(H[i][:,:,1:tt_opt.ttv_rks[i+2],:,1:tt_opt.ttv_rks[i+2]])
				Him = @view(H[i-1][:,:,1:tt_opt.ttv_rks[i+1],:,1:tt_opt.ttv_rks[i+1]])
				updateH_mals!(tt_opt.ttv_vec[i+1], A.tto_vec[i], Hi, Him)
			end
		end
	end
	return E,tt_opt, r_hist
end

