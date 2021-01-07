include("tt_tools.jl")
include("als.jl")

using Plots
using IterativeSolvers

"""
MALS auxiliary functions
"""

function updateHim!(xtt_vec, Atto, Hi, Him)
	Htemp = @view(Him[1:size(xtt_vec,2),1:size(xtt_vec,2),:,:,:])
	@tensoropt((a,b,z),N1[a,b,c,d] := xtt_vec[y,a,z]*view(Hi[1:size(xtt_vec,3),1:size(xtt_vec,3),:,:,:],:,:,:,:,:)[b,z,y,c,d]) #size(ri,rip,nip,rAi)
	@tensoropt((a,b,z),N2[a,b,c] := xtt_vec[y,a,z]*N1[b,z,y,c])#size(ri,ri,rAi)
	@tensoropt((a,b),Htemp[a,b,c,d,e] = N2[a,b,z]*Atto[c,d,e,z]) #size(ri,ri,ni,ni,rAim)
	nothing
end

function updateH_bim!(xtt_vec, btt_vec, Hbi, Hbim)
	Hbtemp = @view(Hbim[1:size(xtt_vec,2),:,:])
	@tensoropt((a,y),N_b[a,b] := xtt_vec[z,a,y]*view(Hbi[1:size(xtt_vec,3),:,:],:,:,:)[y,z,b]) #size(ri,rbi)
	@tensor Hbtemp[a,b,c] = N_b[a,z]*btt_vec[b,c,z] #size(ri,ni,rbim)
	nothing
end

function updateGip!(xtt_vec,Atto_vec,Gi,Gip)
	Gtemp = @view(Gip[:,1:size(xtt_vec,3),:,1:size(xtt_vec,3),:])
	@tensoropt((a,z,c), M1[a,b,c,d] := xtt_vec[y,z,a]*view(Gi[:,1:size(xtt_vec,2),:,1:size(xtt_vec,2),:],:,:,:,:,:)[y,z,b,c,d]) #size(ri,ni,rim,rAi)
	@tensoropt((a,b,z), M2[a,b,c] := xtt_vec[y,z,a]*M1[b,y,z,c]) #size(ri,ri,rAi)
	@tensoropt((b,d), Gtemp[a,b,c,d,e] = M2[d,b,z]*Atto_vec[a,c,z,e]) #size(nip,ri,nip,ri,rAip)
	nothing
end

function updateG_bip!(xtt_vec,btt_vec,G_bi,G_bip)
	Gbtemp = @view(G_bip[:,:,1:size(xtt_vec,3)])
	@tensor begin
		M_b[a,b] := view(G_bi[:,:,1:size(xtt_vec,2)],:,:,:)[a,y,z]*xtt_vec[y,z,b] #size(rbi,ri)
		Gbtemp[a,b,c] = btt_vec[b,z,a]*M_b[z,c] #size(rbip,nip,ri)
	end
end

function left_core_move_mals(xtt::ttvector,i::Integer,V::Array{Float64,4},tol::Float64,rmax::Integer)
	# Perform the truncated svd
	u_V, s_V, v_V, = svd(reshape(V, prod(size(V)[1:2]), :))
	# Determine the truncated rank
	s_trunc = sv_trunc(s_V,tol)
	# Update the ranks to the truncated one
	xtt.ttv_rks[i] = min(length(s_trunc),rmax)
	println("Rank: $(xtt.ttv_rks[i]),	Max rank=$rmax")
	println("Discarded weight: $((norm(s_V)-norm(s_V[1:xtt.ttv_rks[i]]))/norm(s_V))")

	# xtt.ttv_vec[i+1] = truncated Transpose(v_V)
	xtt.ttv_vec[i+1] = permutedims(reshape(v_V[:, 1:xtt.ttv_rks[i]],size(V,3),size(V,4),xtt.ttv_rks[i]),[1,3,2])
	xtt.ttv_ot[i+1] = 1

	# xtt.ttv_vec[i] = truncated u_V * Diagonal(s_V)
	xtt.ttv_vec[i] = permutedims(reshape(u_V[:, 1:xtt.ttv_rks[i]] * Diagonal(s_trunc[1:xtt.ttv_rks[i]]),size(V,1),size(V,2),:),[2,1,3])
#		permutedims(reshape(u_V[:, 1:ri_trunc] * Diagonal(s_trunc),
#							rim, ni, :), [2 1 3])
	xtt.ttv_ot[i] = 0
	return xtt
end

function right_core_move_mals(xtt::ttvector,i::Integer,V::Array{Float64,4},tol::Float64,rmax::Integer)
	# Perform the truncated svd
	u_V, s_V, v_V, = svd(reshape(V, prod(size(V)[1:2]), :))
	# Determine the truncated rank
	s_trunc = sv_trunc(s_V,tol)
	# Update the ranks to the truncated one
	xtt.ttv_rks[i] = min(length(s_trunc),rmax)
	println("Rank: $(xtt.ttv_rks[i]),	Max rank=$rmax")
	println("Discarded weight: $((norm(s_V)-norm(s_V[1:xtt.ttv_rks[i]]))/norm(s_V))")

	# xtt.ttv_vec[i] = truncated u_V
	xtt.ttv_vec[i] = permutedims(reshape(u_V[:, 1:xtt.ttv_rks[i]], size(V,1), size(V,2), xtt.ttv_rks[i]), [2 1 3])
	xtt.ttv_ot[i] = -1

	# xtt.ttv_vec[i+1] = truncated Diagonal(s_V) * Transpose(v_V)
	xtt.ttv_vec[i+1] = permutedims(reshape(Diagonal(s_trunc[1:xtt.ttv_rks[i]]) *
							Transpose(v_V[:, 1:xtt.ttv_rks[i]]),xtt.ttv_rks[i],size(V,3),size(V,4)), [2 1 3])
	xtt.ttv_ot[i+1] = 0
	return xtt
end

function Ksolve_mals(Gi, Hi, G_bi, H_bi, rim, rip)
	@tensor begin
		K[a,b,c,d,e,f,g,h] := Gi[:,1:rim,:,1:rim,:][f,e,b,a,z]*Hi[1:rip,1:rip,:,:,:][d,h,g,c,z] #size(rim,ni,nip,rip,rim,ni,nip,rip)
		Pb[a,b,c,d] := H_bi[1:rip,:,:][d,c,z]*G_bi[:,:,1:rim][z,b,a] #size(rim,ni,nip,rip)
	end
	V = reshape(K,prod(size(K)[1:4]),:) \ Pb[:]
	return reshape(V,size(K)[1:4]...)
end

function K_eigmin_mals(Gi::Array{Float64,5},Hi::Array{Float64,5},ttv_vec_i::Array{Float64,3},ttv_vec_ip::Array{Float64,3};it_solver=false,itslv_thresh=1024::Int64,maxiter=maxiter::Int64,tol=tol::Float64)
	K_dims = [size(ttv_vec_i,2),size(ttv_vec_i,1),size(ttv_vec_ip,1),size(ttv_vec_ip,3)]
	Gtemp = view(Gi[:,1:K_dims[1],:,1:K_dims[1],:],:,:,:,:,:)
	Htemp = view(Hi[1:K_dims[4],1:K_dims[4],:,:,:],:,:,:,:,:)
	if it_solver || prod(K_dims) > itslv_thresh
		H = zeros(Float64,K_dims...)
		function K_matfree(V;K_dims=K_dims::Array{Int64,1},H=H)
			@tensoropt((a,d,e,h), H[a,b,c,d] = Gtemp[f,e,b,a,z]*Htemp[d,h,g,c,z]*reshape(V,K_dims...)[e,f,g,h])
			return H[:]
		end
		X0 = zeros(Float64,K_dims...)
		@tensoropt((a,z,d), X0[a,b,c,d] := ttv_vec_i[b,a,z]*ttv_vec_ip[c,z,d])
		r = lobpcg(LinearMap(K_matfree,prod(K_dims);issymmetric = true),false,X0[:],1;maxiter=maxiter,tol=tol)
		return r.λ[1]::Float64, reshape(r.X[:,1],K_dims...)::Array{Float64,4}
	else
		K = zeros(Float64,K_dims...,K_dims...)
		@tensoropt((a,d,e,h), K[a,b,c,d,e,f,g,h] = Gtemp[f,e,b,a,z]*Htemp[d,h,g,c,z]) #size(rim,ni,nip,rip,rim,ni,nip,rip)
		F = eigen(reshape(K,prod(K_dims),:))
		return real(F.values[1])::Float64,real.(reshape(F.vectors[:,1],K_dims...))::Array{Float64,4}
	end	
end

function mals(A :: ttoperator, b :: ttvector, tt_start :: ttvector; tol=1e-12::Float64,rmax=round(Int,sqrt(prod(tt_start.ttv_dims))))
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
	tt_opt = deepcopy(tt_start)
	dims = tt_start.ttv_dims
	d = length(dims)
	# Define the array of ranks of tt_opt [r_0=1,r_1,...,r_d]
	rks = vcat(1,tt_start.ttv_rks)
	# Define the array of ranks of A [R_0=1,R_1,...,R_d]
	A_rks = vcat(1,A.tto_rks)
	# Define the array of ranks of b [R^b_0=1,R^b_1,...,R^b_d]
	b_rks = vcat(1,b.ttv_rks)

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

function mals_eig(A :: ttoperator, tt_start :: ttvector; tol=1e-12::Float64,sweep_schedule=[2]::Array{Int64,1},rmax_schedule=[round(Int,sqrt(prod(tt_start.ttv_dims)))]::Array{Int64,1},it_solver=false::Bool,linsolv_maxiter=200::Int64,linsolv_tol=max(sqrt(tol),1e-8)::Float64)
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
	tt_opt = deepcopy(tt_start)
	dims = tt_start.ttv_dims
	d = length(dims)
	# Define the array of ranks of tt_opt [r_0=1,r_1,...,r_d]
	rks = vcat(1,tt_start.ttv_rks)
	# Define the array of ranks of A [R_0=1,R_1,...,R_d]
	A_rks = vcat(1,A.tto_rks)
	E = Float64[]
	r_hist = Int64[]
	# Initialize the arrays of G and H
	G = Array{Array{Float64}}(undef, d)
	H = Array{Array{Float64}}(undef, d-1)
	rmax = maximum(rmax_schedule)
	# Initialize G[1], G_b[1], H[d] and H_b[d]
	for i in 1:d
		rmax_i = min(rmax,prod(dims[1:i-1]),prod(dims[i:end]))
		G[i] = zeros(dims[i],rmax_i,dims[i],rmax_i,A_rks[i+1])
	end
	for i in 1:d-1
		rmax_i = min(rmax,prod(dims[1:i+1]),prod(dims[i+2:end]))
		H[i] = zeros(rmax_i,rmax_i,dims[i+1],dims[i+1],A_rks[i+1])
	end
	G[1][:,1:1,:,1:1,:] = reshape(A.tto_vec[1][:,:,1,:], dims[1],1,dims[1], 1, :)
	H[d-1][1:1,1:1,:,:,:] = reshape(A.tto_vec[d], 1, 1, dims[d], dims[d], :) # k'_d,k_d,x_d,y_d,j_(d-1)

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
		for i = (d-1) : -1 : 2
			# Update H[i-1]
			updateHim!(tt_opt.ttv_vec[i+1], A.tto_vec[i], H[i], H[i-1])
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
			updateGip!(tt_opt.ttv_vec[i],A.tto_vec[i+1],G[i],G[i+1])
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
				updateHim!(tt_opt.ttv_vec[i+1], A.tto_vec[i], H[i], H[i-1])
			end
		end
	end
end

