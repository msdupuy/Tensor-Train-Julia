include("tt_tools.jl")

#TODO: eigs version and include IterativeSolvers option

function init_H_and_Hb(x_tt::ttvector,A_tto::ttoperator;b_tt::ttvector=empty_tt())
	d = length(x_tt.ttv_dims)
	H = Array{Array{Float64}}(undef, d)
	H_b = Array{Array{Float64}}(undef, d) 
	H[d] = ones(1,1,1)
	if !isempty(b_tt)
		H_b[d] = ones(1,1)
	end
	for i = d : -1 : 2
		@tensor begin
			N1[a,b,c,d] := x_tt.ttv_vec[i][a,b,z]*H[i][z,c,d]
			N2[a,b,c,d] := N1[y,a,b,z]*A_tto.tto_vec[i][y,c,d,z]
			H[i-1][a,b,c] := x_tt.ttv_vec[i][z,b,y]*N2[a,y,z,c] #size (rim, rim, rAim)
		end
		if !isempty(b_tt)
			@tensor begin
				N_b[a,b,c] := x_tt.ttv_vec[i][a,b,z]*H_b[i][z,c] #size (ni,rim,rbi)
				H_b[i-1][a,b] := N_b[y,a,z]*b_tt.ttv_vec[i][y,b,z]  # k_(i-1), j^b_(i-1)
			end
		end
	end
	return H,H_b
end

function left_core_move(x_tt::ttvector,V,i::Int,x_rks,x_dims)
	rim2,rim,ri = x_rks[i-1],x_rks[i],x_rks[i+1]
	nim,ni = x_dims[i-1],x_dims[i]
	rim_new = min(ri*ni, rim)

	# Prepare core movements
	rim_new = min(ri*ni, rim)
	QV = zeros(ni*ri, ni*ri)
	RV = zeros(rim, rim)
	x_tt.ttv_rks[i-1] = rim_new
	QV[1:ni*ri, 1:ni*ri], RV[1:rim_new, 1:rim] =
			qr(reshape(permutedims(V[1:ni, 1:rim, 1:ri], [1 3 2]), ni*ri, :))

	# Apply core movement 3.2
	x_tt.ttv_vec[i][1:ni, 1:rim_new, 1:ri] =
			permutedims(reshape(QV[1:ni*ri, 1:rim_new], ni, ri, :),[1 3 2])
	x_tt.ttv_vec[i][1:ni, (rim_new+1):rim, 1:ri] = zeros(ni, rim-rim_new, ri)
	x_tt.ttv_ot[i] = 1

	# Apply core movement 3.2
	@tensor x_tt.ttv_vec[i-1][a,b,c] := x_tt.ttv_vec[i-1][a,b,z]*RV[1:rim_new, 1:rim][c,z] #size (nim,rim2,rim_new)
	x_tt.ttv_ot[i-1] = 0
	return x_tt,rim_new
end

function right_core_move(x_tt::ttvector,V,i::Int,x_rks,x_dims)
	rim,ri,rip = x_rks[i],x_rks[i+1],x_rks[i+2]
	ni,nip = x_dims[i],x_dims[i+1]
	ri_new = min(rim*ni, ri)

	QV = zeros(ni*rim, ni*rim)
	RV = zeros(ri, ri)
	x_tt.ttv_rks[i] = ri_new
	QV[1:ni*rim, 1:ni*rim], RV[1:ri_new, 1:ri] =
		qr(reshape(V[1:ni, 1:rim, 1:ri], ni*rim, :))

	# Apply core movement 3.1
	x_tt.ttv_vec[i][1:ni, 1:rim, 1:ri_new] = reshape(QV[1:ni*rim, 1:ri_new], ni, rim, :)
	x_tt.ttv_vec[i][1:ni, 1:rim, (ri_new+1):ri] = zeros(ni,rim,ri-ri_new)
	x_tt.ttv_ot[i] = -1

	# Apply core movement 3.2
	@tensor x_tt.ttv_vec[i+1][a,b,c] := RV[1:ri, 1:ri][b,z]*x_tt.ttv_vec[i+1][a,z,c] #size (nip,ri,rip)
	x_tt.ttv_ot[i+1] = 0
	return x_tt,ri_new
end

function update_G_Gb(x_tt::ttvector,A_tto::ttoperator,i,Gi;G_bi=[],b_tt::ttvector=empty_tt())
	@tensor begin
		M1[a,b,c,d] := x_tt.ttv_vec[i][y,z,a]*Gi[y,z,b,c,d] #size (ri,ni,rim,rAi)
		M2[a,b,c] := x_tt.ttv_vec[i][y,z,a]*M1[b,y,z,c] #size (ri,ri,rAi)
		G[a,b,c,d,e] := M2[d,b,z]*A_tto.tto_vec[i+1][a,c,z,e]
	end
	if G_bi != []
		@tensor begin
			M_b[a,b] := G_bi[a,y,z]*x_tt.ttv_vec[i][y,z,b]
			G_b[a,b,c] := b_tt.ttv_vec[i+1][b,z,a]*M_b[z,c] # j_(i+1), x_(i+1), k_i
		end
		return G,G_b
	else
		return G,Float64[]
	end
end

function update_H_Hb(x_tt::ttvector,A_tto::ttoperator,i,Hi;H_bi=[],b_tt::ttvector=empty_tt())
	@tensor begin
		N1[a,b,c,d] := x_tt.ttv_vec[i][a,b,z]*Hi[z,c,d] #size (ni,rim,ri,rAi)
		N2[a,b,c,d] := N1[y,a,b,z]*A_tto.tto_vec[i][y,c,d,z] #size (rim,ri,ni,rAim)
		H[a,b,c] := x_tt.ttv_vec[i][z,b,y]*N2[a,y,z,c] #k_i,k'_i,l_i
	end
	if H_bi != []
		@tensor begin
			N_b[a,b,c] := x_tt.ttv_vec[i][a,b,z]*H_bi[z,c] #size(ni,rim,rbi)
			H_b[a,b] := N_b[y,a,z]*b_tt.ttv_vec[i][y,b,z] #size(rbim, rim)
		end
		return H,H_b
	else
		return H,Float64[]
	end	
end

#function sweep_schedule(nsweep,r_seq;tol=1e-6) #
#	d = length(nsweep)
#	swp_out = zeros(Int64,n_int_sweep*d,2)
#	for i in 1:d
#		swp_out[2i-1,:] = [nsweep[i] r_seq[i]]
#		swp_out[2i,:] = [nsweep[i] r_seq[i]]
#	end
#	return swp_out
#end

function Ksolve(Gi,G_bi,Hi,H_bi)
	@tensor begin
		K[a,b,c,d,e,f] := Gi[d,e,a,b,z]*Hi[f,c,z] #size (ni,rim,ri,ni,rim,ri)
		Pb[a,b,c] := G_bi[z,a,b]*H_bi[c,z] #size (ni,rim,ri)
	end
	V = md_div(K,Pb,[1 2 3 4 5 6],[1 2 3],3,3,[1 2 3])
	return V
end

function K_eigmin(Gi,Hi,ttv_vec;it_solver=false,itslv_thresh=2500)
	@tensor K[a,b,c,d,e,f] := Gi[d,e,a,b,z]*Hi[f,c,z] #size (ni,rim,ri,ni,rim,ri)
	println(size(ttv_vec))
	println(size(K)[1:3])	
	if it_solver || prod(size(K)[1:3]) > itslv_thresh
		r = lobpcg(reshape(K,prod(size(K)[1:3]),:),false,ttv_vec[:],1)
		return r.λ[1], reshape(r.X[:,1],size(K)[1:3]...)
	else
		F = eigen(reshape(K,prod(size(K)[1:3]),:))
		return real(F.values[1]),real.(reshape(F.vectors[:,1],size(K)[1:3]...))
	end	
end

function K_eiggenmin(Gi,Hi,Ki,Li,ttv_vec;it_solver=false,itslv_thresh=2500)
	@tensor begin
		K[a,b,c,d,e,f] := Gi[d,e,a,b,z]*Hi[f,c,z] #size (ni,rim,ri,ni,rim,ri)	
		S[a,b,c,d,e,f] := Ki[d,e,a,b,z]*Li[f,c,z] #size (ni,rim,ri,ni,rim,ri)	
	end
	if it_solver || prod(size(K)[1:3]) > itslv_thresh
		r = lobpcg(reshape(K,prod(size(K)[1:3]),:),reshape(S,prod(size(S)[1:3]),:),false,ttv_vec[:],1)
		return r.λ[1], reshape(r.X[:,1],size(K)[1:3]...)
	else
		F = eigen(reshape(K,prod(size(K)[1:3]),:),reshape(S,prod(size(K)[1:3]),:),)
		return real(F.values[1]),real.(reshape(F.vectors[:,1],size(K)[1:3]...))
	end
end

#sweep scheduler: Array of Int, Int, Float: sweep, rks, E_change in last sweep 
function als(A :: ttoperator, b :: ttvector, tt_start :: ttvector ;sweep_count=2,it_solver=false,r_itsolver=5000)
	# als finds the minimum of the operator J:1/2*<Ax,Ax> - <x,b>
	# input:
	# 	A: the tensor operator in its tensor train format
	#   b: the tensor in its tensor train format
	#	tt_start: start value in its tensor train format
	#	opt_rks: rank vector considered to be optimal enough
	# output:
	#	tt_opt: stationary point of J up to tolerated rank opt_rks
	# 			in its tensor train format

	# Initialize the to be returned tensor in its tensor train format
	tt_opt = deepcopy(tt_start)
	dims = tt_start.ttv_dims
	d = length(dims)
	# Define the array of ranks of tt_opt [r_0=1,r_1,...,r_d]
	rks = vcat([1], tt_start.ttv_rks)

	# Initialize the arrays of G and G_b
	G = Array{Array{Float64}}(undef, d)
	G_b = Array{Array{Float64}}(undef, d)

	# Initialize G[1], G_b[1], H[d] and H_b[d]
	G[1] = zeros(dims[1], 1, dims[1], 1, A.tto_rks[1])
	G[1] = reshape(A.tto_vec[1][:,:,1,:], dims[1],1,dims[1], 1, :)
	G_b[1] = zeros(b.ttv_rks[1], dims[1], 1)
	G_b[1] = permutedims(reshape(b.ttv_vec[1][:,1,1:b.ttv_rks[1]], dims[1], 1, :), [3 1 2])

	#Initialize H and H_b
	H,H_b = init_H_and_Hb(tt_opt,A,b_tt=b)

	nsweeps = 0 #sweeps counter

	while nsweeps < sweep_count
		nsweeps+=1
		# First half sweep
		for i = 1:(d-1)
			println("Forward sweep: core optimization $i out of $d")
			# If i is the index of the core matrices do the optimization
			if tt_opt.ttv_ot[i] == 0
				# Define V as solution of K*x=Pb in x
				V = Ksolve(G[i],G_b[i],H[i],H_b[i])
				tt_opt, rks[i+1] = right_core_move(tt_opt,V,i,rks,dims)
			end
			#update G,G_b
			G[i+1],G_b[i+1] = update_G_Gb(tt_opt,A,i,G[i];G_bi=G_b[i],b_tt=b)
		end

		if nsweeps == sweep_count
			return tt_opt
		else
			nsweeps+=1
			# Second half sweep
			for i = d:(-1):2
				println("Backward sweep: core optimization $i out of $d")
				# Define V as solution of K*x=Pb in x
				V = Ksolve(G[i],G_b[i],H[i],H_b[i])
				tt_opt,rks[i] = left_core_move(tt_opt,V,i,rks,dims)
				H[i-1],H_b[i-1] = update_H_Hb(tt_opt,A,i,H[i];H_bi=H_b[i],b_tt=b)
			end
		end
	end
	return tt_opt
end

"""
Warning probably only works for left-orthogonal starting tensor
"""
function als_eig(A :: ttoperator, tt_start :: ttvector ; sweep_schedule=[2],rmax_schedule=[maximum(tt_start.ttv_rks)],tol=1e-10,it_solver=false,itslv_thresh=2500)
	# als finds the minimum of the operator J:1/2*<Ax,Ax> - <x,b>
	# input:
	# 	A: the tensor operator in its tensor train format
	#	tt_start: start value in its tensor train format
	#	opt_rks: rank vector considered to be optimal enough
	# output:
	#	tt_opt: stationary point of J up to tolerated rank opt_rks
	# 			in its tensor train format
	@assert(length(rmax_schedule)==length(sweep_schedule),"Sweep schedule error")	

	# Initialize the to be returned tensor in its tensor train format
	E = 0.0 #output eigenvalue
	tt_opt = deepcopy(tt_start)
	dims = tt_start.ttv_dims
	d = length(dims)
	# Define the array of ranks of tt_opt [r_0=1,r_1,...,r_d]
	rks = vcat([1], tt_start.ttv_rks)

	# Initialize the array G
	G = Array{Array{Float64}}(undef, d)
	G[1] = zeros(dims[1], 1, dims[1], 1, A.tto_rks[1])
	G[1] = reshape(A.tto_vec[1][:,:,1,:], dims[1],1,dims[1], 1, :)

	#Initialize H and H_b
	H,H_b = init_H_and_Hb(tt_opt,A)

	nsweeps = 0 #sweeps counter
	i_schedule = 1
	while i_schedule <= length(sweep_schedule) 
		nsweeps+=1
		if nsweeps == sweep_schedule[i_schedule]
			i_schedule+=1
			if i_schedule > length(sweep_schedule)
				return E,tt_opt
			else
				tt_opt = tt_up_rks(tt_opt,rmax_schedule[i_schedule])
				for i in 1:d-1
					Htemp = zeros(tt_opt.ttv_rks[i],tt_opt.ttv_rks[i],A.tto_rks[i])
					Htemp[1:size(H[i],1),1:size(H[i],2),1:size(H[i],3)] = H[i] 
					H[i] = Htemp
				end
			end
		end
		# First half sweep
		for i = 1:(d-1)
			println("Forward sweep: core optimization $i out of $d")

			# If i is the index of the core matrices do the optimization
			if tt_opt.ttv_ot[i] == 0
				# Define V as solution of K*x=Pb in x
				E,V = K_eigmin(G[i],H[i],tt_opt.ttv_vec[i];it_solver=it_solver,itslv_thresh=itslv_thresh)
				println("Eigenvalue: $E")
				tt_opt, rks[i+1] = right_core_move(tt_opt,V,i,vcat(1,tt_opt.ttv_rks),dims)
			end

			#update G,G_b
			G[i+1],G_b = update_G_Gb(tt_opt,A,i,G[i])
		end

		# Second half sweep
		for i = d:(-1):2
			println("Backward sweep: core optimization $i out of $d")
			# Define V as solution of K*x=Pb in x
			E,V = K_eigmin(G[i],H[i],tt_opt.ttv_vec[i];it_solver=it_solver,itslv_thresh=itslv_thresh)
			println("Eigenvalue: $E")

			tt_opt,rks[i] = left_core_move(tt_opt,V,i,vcat(1,tt_opt.ttv_rks),dims)

			H[i-1],H_bi = update_H_Hb(tt_opt,A,i,H[i])
		end
	end
end

"""
returns the smallest eigenpair Ax = Sx
"""

function als_gen_eig(A :: ttoperator, S::ttoperator, tt_start :: ttvector ; sweep_count=2,it_solver=false,itslv_thresh=2500)
	# als finds the minimum of the operator J:1/2*<Ax,Ax> - <x,b>
	# input:
	# 	A: the tensor operator in its tensor train format
	#	tt_start: start value in its tensor train format
	#	opt_rks: rank vector considered to be optimal enough
	# output:
	#	tt_opt: stationary point of J up to tolerated rank opt_rks
	# 			in its tensor train format

	# Initialize the to be returned tensor in its tensor train format
	E = 0.0 #output eigenvalue
	tt_opt = deepcopy(tt_start)
	dims = tt_start.ttv_dims
	d = length(dims)
	# Define the array of ranks of tt_opt [r_0=1,r_1,...,r_d]
	rks = vcat([1], tt_start.ttv_rks)

	# Initialize the arrays of G and K
	G = Array{Array{Float64}}(undef, d)
	K = Array{Array{Float64}}(undef, d) 

	# Initialize G[1]
	G[1] = zeros(dims[1], 1, dims[1], 1, A.tto_rks[1])
	G[1] = reshape(A.tto_vec[1][:,:,1,:], dims[1],1,dims[1], 1, :)
	K[1] = zeros(dims[1], 1, dims[1], 1, S.tto_rks[2])
	K[1] = reshape(S.tto_vec[1][:,:,1,:], dims[1],1,dims[1], 1, :)

	#Initialize H and H_b
	H,H_b = init_H_and_Hb(tt_opt,A)
	L,L_b = init_H_and_Hb(tt_opt,S)

	nsweeps = 0 #sweeps counter
	while nsweeps < sweep_count
		nsweeps+=1

		# First half sweep
		for i = 1:(d-1)
			println("Forward sweep: core optimization $i out of $d")

			# If i is the index of the core matrices do the optimization
			if tt_opt.ttv_ot[i] == 0
				# Define V as solution of K*x=Pb in x
				E,V = K_eiggenmin(G[i],H[i],K[i],L[i],tt_opt.ttv_vec[i];it_solver=it_solver,itslv_thresh=itslv_thresh)
				println("Eigenvalue: $E")
				tt_opt, rks[i+1] = right_core_move(tt_opt,V,i,rks,dims)
			end

			#update G and K
			G[i+1],G_b = update_G_Gb(tt_opt,A,i,G[i])
			K[i+1],K_b = update_G_Gb(tt_opt,S,i,K[i])
		end

		if nsweeps == sweep_count
			return tt_opt
		else
			nsweeps+=1
			# Second half sweep
			for i = d:(-1):2
				println("Backward sweep: core optimization $i out of $d")
				# Define V as solution of K*x=Pb in x
				E,V = K_eiggenmin(G[i],H[i],K[i],L[i],tt_opt.ttv_vec[i];it_solver=it_solver,itslv_thresh=itslv_thresh)
				println("Eigenvalue: $E")

				tt_opt,rks[i] = left_core_move(tt_opt,V,i,rks,dims)

				H[i-1],H_b = update_H_Hb(tt_opt,A,i,H[i])
				L[i-1],L_b = update_H_Hb(tt_opt,S,i,L[i])
			end
		end
	end
	return E,tt_opt
end
