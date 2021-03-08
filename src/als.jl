using LinearMaps
using TensorOperations

"""
TODO: include IterativeSolvers option for als

Implementation based on the presentation in 
Holtz, Sebastian, Thorsten Rohwedder, and Reinhold Schneider. "The alternating linear scheme for tensor optimization in the tensor train format." SIAM Journal on Scientific Computing 34.2 (2012): A683-A713.
"""

function init_H(x_tt::ttvector,A_tto::ttoperator)
	d = length(x_tt.ttv_dims)
	H = Array{Array{Float64}}(undef, d)
	H[d] = ones(1,1,1)
	for i = d : -1 : 2
		@tensoropt((b,c,z), N1[a,b,c,d] := x_tt.ttv_vec[i][a,b,z]*H[i][z,c,d])
		@tensoropt((a,b), N2[a,b,c,d] := N1[y,a,b,z]*A_tto.tto_vec[i][y,c,d,z])
		@tensoropt((a,b,y), H[i-1][a,b,c] := x_tt.ttv_vec[i][z,b,y]*N2[a,y,z,c]) #size (rim, rim, rAim)
	end
	return H
end

function init_Hb(x_tt::ttvector,b_tt::ttvector)
	d = length(x_tt.ttv_dims)
	H_b = Array{Array{Float64}}(undef, d) 
	H_b[d] = ones(1,1)
	for i = d : -1 : 2
		@tensor begin
			N_b[a,b,c] := x_tt.ttv_vec[i][a,b,z]*H_b[i][z,c] #size (ni,rim,rbi)
			H_b[i-1][a,b] := N_b[y,a,z]*b_tt.ttv_vec[i][y,b,z]  # k_(i-1), j^b_(i-1)
		end
	end
	return H_b
end

function left_core_move(x_tt::ttvector,V::Array{Float64,3},i::Int,x_rks)
	rim,ri = x_rks[i],x_rks[i+1]
	ni = x_tt.ttv_dims[i]

	# Prepare core movements
	QV, RV = qr(reshape(permutedims(V, [1 3 2]), ni*ri, :)) #QV: ni*ri x ni*ri; RV ni*ri x rim

	# Apply core movement 3.1
	x_tt.ttv_vec[i] = permutedims(reshape(QV[:, 1:rim], ni, ri, :),[1 3 2])
	x_tt.ttv_ot[i] = 1

	# Apply core movement 3.2
	@tensoropt((b,c,z) , Xim[a,b,c] := x_tt.ttv_vec[i-1][a,b,z]*RV[1:rim,:][c,z]) #size (nim,rim2,rim_new)
	x_tt.ttv_vec[i-1] = Xim
	x_tt.ttv_ot[i-1] = 0
	return x_tt,rim
end

function right_core_move(x_tt::ttvector,V::Array{Float64,3},i::Int,x_rks)
	rim,ri = x_rks[i],x_rks[i+1]
	ni = x_tt.ttv_dims[i]
	QV, RV = qr(reshape(V, ni*rim, :)) #QV: ni*rim x ni*rim; RV ni*rim x ri

	# Apply core movement 3.1
	x_tt.ttv_vec[i] = reshape(QV[:, 1:ri], ni, rim, :)
	x_tt.ttv_ot[i] = -1

	# Apply core movement 3.2
	@tensoropt((b,c,z), Xip[a,b,c] := RV[1:ri,:][b,z]*x_tt.ttv_vec[i+1][a,z,c]) #size (nip,ri,rip)
	x_tt.ttv_vec[i+1] = Xip
	x_tt.ttv_ot[i+1] = 0
	return x_tt,ri
end


function update_G!(x_tt::ttvector,A_tto::ttoperator,i,Gi::Array{Float64,5},Gip::Array{Float64,5})
	@tensoropt((a,c,z), M1[a,b,c,d] := x_tt.ttv_vec[i][y,z,a]*Gi[y,z,b,c,d]) #size (ri,ni,rim,rAi)
	@tensoropt((a,b,z), M2[a,b,c] := x_tt.ttv_vec[i][y,z,a]*M1[b,y,z,c]) #size (ri,ri,rAi)
	@tensoropt((b,d), Gip[a,b,c,d,e] = M2[d,b,z]*A_tto.tto_vec[i+1][a,c,z,e])
	nothing
end

function update_Gb!(x_tt::ttvector,i,G_bi::Array{Float64,3},G_bip::Array{Float64,3},b_tt::ttvector)
	@tensor begin
		M_b[a,b] := G_bi[a,y,z]*x_tt.ttv_vec[i][y,z,b]
		G_bip[a,b,c] = b_tt.ttv_vec[i+1][b,z,a]*M_b[z,c] # j_(i+1), x_(i+1), k_i
	end
end

function update_H!(x_tt::ttvector,A_tto::ttoperator,i,Hi::Array{Float64,3},Him::Array{Float64,3})
	@tensoropt((b,c,z), N1[a,b,c,d] := x_tt.ttv_vec[i][a,b,z]*Hi[z,c,d]) #size (ni,rim,ri,rAi)
	@tensoropt((a,b), N2[a,b,c,d] := N1[y,a,b,z]*A_tto.tto_vec[i][y,c,d,z]) #size (rim,ri,ni,rAim)
	@tensoropt((a,b,y), Him[a,b,c] = x_tt.ttv_vec[i][z,b,y]*N2[a,y,z,c]) #k_i,k'_i,l_i
	nothing
end

function update_Hb!(x_tt::ttvector,i,H_bi::Array{Float64,2},H_bim::Array{Float64,2},b_tt::ttvector)
	@tensor begin
		N_b[a,b,c] := x_tt.ttv_vec[i][a,b,z]*H_bi[z,c] #size(ni,rim,rbi)
		H_bim[a,b] = N_b[y,a,z]*b_tt.ttv_vec[i][y,b,z] #size(rbim, rim)
	end
end

#full assemble of matrix K
function K_full(Gi::Array{Float64,5},Hi::Array{Float64,3},K_dims::Array{Int})
	K = zeros(Float64,prod(K_dims),prod(K_dims))
	Krshp = reshape(K,K_dims...,K_dims...)
	@tensoropt((b,c,e,f), Krshp[a,b,c,d,e,f] = Gi[d,e,a,b,z]*Hi[f,c,z]) #size (ni,rim,ri,ni,rim,ri)
	return K
end

function Ksolve(Gi::Array{Float64,5},G_bi::Array{Float64,3},Hi::Array{Float64,3},H_bi::Array{Float64,2})
	K_dims = [size(Gi,1),size(Gi,2),size(Hi,1)]
	K = K_full(Gi,Hi,K_dims)
	@tensoropt((b,c), Pb[a,b,c] := G_bi[z,a,b]*H_bi[c,z]) #size (ni,rim,ri)
	return reshape(K\Pb[:],K_dims...)
end

function K_eigmin(Gi::Array{Float64,5},Hi::Array{Float64,3},ttv_vec::Array{Float64,3};it_solver=false,itslv_thresh=1024::Int64,maxiter=maxiter::Int64,tol=tol::Float64)
	K_dims = [size(Gi,1),size(Gi,2),size(Hi,1)]
	if it_solver || prod(K_dims) > itslv_thresh
		H = zeros(Float64,prod(K_dims))
		function K_matfree(V;Gi=Gi::Array{Float64,5},Hi=Hi::Array{Float64,3},K_dims=K_dims,H=H)
			Hrshp = reshape(H,K_dims...)
			@tensoropt((b,c,e,f), Hrshp[a,b,c] = Gi[d,e,a,b,z]*Hi[f,c,z]*reshape(V,K_dims...)[d,e,f])
			return H
		end
		r = lobpcg(LinearMap(K_matfree,prod(K_dims);issymmetric = true),false,ttv_vec[:],1;maxiter=maxiter,tol=tol)
		return r.λ[1], reshape(r.X[:,1],K_dims...)
	else
		K = K_full(Gi,Hi,K_dims)
		F = eigen(K)
		return real(F.values[1]),real.(reshape(F.vectors[:,1],K_dims...))
	end	
end

function K_eiggenmin(Gi,Hi,Ki,Li,ttv_vec;it_solver=false,itslv_thresh=2500)
	@tensor begin
		K[a,b,c,d,e,f] := Gi[d,e,a,b,z]*Hi[f,c,z] #size (ni,rim,ri,ni,rim,ri)	
		S[a,b,c,d,e,f] := Ki[d,e,a,b,z]*Li[f,c,z] #size (ni,rim,ri,ni,rim,ri)	
	end
	if it_solver || prod(size(K)[1:3]) > itslv_thresh
		r = lobpcg(reshape(K,prod(size(K)[1:3]),:),reshape(S,prod(size(S)[1:3]),:),false,ttv_vec[:],1;maxiter=500,tol=1e-8)
		return r.λ[1], reshape(r.X[:,1],size(K)[1:3]...)
	else
		F = eigen(reshape(K,prod(size(K)[1:3]),:),reshape(S,prod(size(K)[1:3]),:),)
		return real(F.values[1]),real.(reshape(F.vectors[:,1],size(K)[1:3]...))
	end
end

#sweep scheduler: Array of Int, Int, Float: sweep, rks, E_change in last sweep 
function als_linsolv(A :: ttoperator, b :: ttvector, tt_start :: ttvector ;sweep_count=2,it_solver=false,r_itsolver=5000)
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
	rks = vcat(1, tt_start.ttv_rks)

	# Initialize the arrays of G and G_b
	G = Array{Array{Float64}}(undef, d)
	G_b = Array{Array{Float64}}(undef, d)

	# Initialize G[1], G_b[1], H[d] and H_b[d]
	for i in 1:d
		G[i] = zeros(dims[i],rks[i],dims[i],rks[i],A.tto_rks[i])
		G_b[i] = zeros(b.ttv_rks[i],dims[i],rks[i])
	end
	G[1] = reshape(A.tto_vec[1][:,:,1,:], dims[1],1,dims[1], 1, :)
	G_b[1] = permutedims(reshape(b.ttv_vec[1], dims[1], 1, :), [3 1 2])

	#Initialize H and H_b
	H = init_H(tt_opt,A)
	H_b = init_Hb(tt_opt,b)

	nsweeps = 0 #sweeps counter

	while nsweeps < sweep_count
		nsweeps+=1
		# First half sweep
		for i = 1:(d-1)
			println("Forward sweep: core optimization $i out of $d")
			# Define V as solution of K*x=Pb in x
			V = Ksolve(G[i],G_b[i],H[i],H_b[i])
			tt_opt, rks[i+1] = right_core_move(tt_opt,V,i,rks)
			println(norm(tt_opt.ttv_vec[i+1]))
			#update G,G_b
			update_G!(tt_opt,A,i,G[i],G[i+1])
			update_Gb!(tt_opt,i,G_b[i],G_b[i+1],b)
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
				tt_opt,rks[i] = left_core_move(tt_opt,V,i,rks)
				println(norm(tt_opt.ttv_vec[i-1]))
				update_H!(tt_opt,A,i,H[i],H[i-1])
				update_Hb!(tt_opt,i,H_b[i],H_b[i-1],b)
			end
		end
	end
	return tt_opt
end

"""
Warning probably only works for left-orthogonal starting tensor
Returns the lowest eigenvalue of A by minimizing the Rayleigh quotient
"""
function als_eigsolv(A :: ttoperator,
	 tt_start :: ttvector ; #TT initial guess
	 sweep_schedule=[2]::Array{Int64,1}, #Number of sweeps for each bond dimension in rmax_schedule
	 rmax_schedule=[maximum(tt_start.ttv_rks)]::Array{Int64,1}, #bond dimension at each sweep
	 noise_schedule=zeros(length(rmax_schedule))::Array{Float64,1}, #noise at each bond dimension increase
	 it_solver=false::Bool, #linear solver for the microstep
	 itslv_thresh=1024::Int64, #switch from full to iterative
	 maxiter=200::Int64, #maximum of iterations for the iterative solver
	 linsolv_tol=1e-8) #tolerance of the iterative linear solver
	@assert(length(rmax_schedule)==length(sweep_schedule)==length(noise_schedule),"Sweep schedule error")	

	# Initialize the to be returned tensor in its tensor train format
	tt_opt = deepcopy(tt_start)
	tt_opt = tt_orthogonalize(tt_opt,1)
	dims = tt_start.ttv_dims
	d = length(dims)
	E = zeros(Float64,2d*(sweep_schedule[end]+1)) #output eigenvalue
	# Define the array of ranks of tt_opt [r_0=1,r_1,...,r_d]
	rks = vcat(1, tt_start.ttv_rks)

	# Initialize the array G
	G = Array{Array{Float64}}(undef, d)
	for i in 1:d
		G[i] = zeros(dims[i],rks[i],dims[i],rks[i],A.tto_rks[i])
	end
	G[1] = reshape(A.tto_vec[1][:,:,1,:], dims[1],1,dims[1], 1, :)

	#Initialize H and H_b
	H = init_H(tt_opt,A)

	nsweeps = 0 #sweeps counter
	i_schedule,i_μit = 1,0
	while i_schedule <= length(sweep_schedule) 
		nsweeps+=1
		println("Macro-iteration $nsweeps; bond dimension $(rmax_schedule[i_schedule])")
		if nsweeps == sweep_schedule[i_schedule]
			i_schedule+=1
			if i_schedule > length(sweep_schedule)
				return E[1:i_μit],tt_opt
			else
				tt_opt = tt_up_rks(tt_opt,rmax_schedule[i_schedule];ϵ_wn=noise_schedule[i_schedule])
				tt_opt = tt_orthogonalize(tt_opt,1)
				H = init_H(tt_opt,A)
				for i in 1:d-1
					Gtemp = zeros(dims[i+1],tt_opt.ttv_rks[i],dims[i+1],tt_opt.ttv_rks[i],A.tto_rks[i+1])
					Gtemp[1:size(G[i+1],1),1:size(G[i+1],2),1:size(G[i+1],3),1:size(G[i+1],4),1:size(G[i+1],5)] = G[i+1]
					G[i+1] = Gtemp
				end
			end
		end
		# First half sweep
		for i = 1:(d-1)
			println("Forward sweep: core optimization $i out of $(d-1)")
			# Define V as solution of K*x=Pb in x
			i_μit += 1
			E[i_μit],V = K_eigmin(G[i],H[i],tt_opt.ttv_vec[i];it_solver=it_solver,itslv_thresh=itslv_thresh,maxiter=maxiter,tol=linsolv_tol)
			println("Eigenvalue: $(E[i_μit])")
			tt_opt, rks[i+1] = right_core_move(tt_opt,V,i,vcat(1,tt_opt.ttv_rks))
			#update G
			update_G!(tt_opt,A,i,G[i],G[i+1])
		end

		# Second half sweep
		for i = d:(-1):2
			println("Backward sweep: core optimization $(d+1-i) out of $(d-1)")
			# Define V as solution of K*x=Pb in x
			i_μit += 1
			E[i_μit],V = K_eigmin(G[i],H[i],tt_opt.ttv_vec[i];it_solver=it_solver,itslv_thresh=itslv_thresh,maxiter=maxiter,tol=linsolv_tol)
			println("Eigenvalue: $(E[i_μit])")
			tt_opt,rks[i] = left_core_move(tt_opt,V,i,vcat(1,tt_opt.ttv_rks))
			update_H!(tt_opt,A,i,H[i],H[i-1])
		end
	end
	return E[1:i_μit],tt_opt
end

"""
returns the smallest eigenpair Ax = Sx
"""

function als_gen_eigsolv(A :: ttoperator, S::ttoperator, tt_start :: ttvector ; sweep_schedule=[2],rmax_schedule=[maximum(tt_start.ttv_rks)],tol=1e-10,it_solver=false,itslv_thresh=2500)
	# als finds the minimum of the operator J:1/2*<Ax,Ax> - <x,b>
	# input:
	# 	A: the tensor operator in its tensor train format
	#	tt_start: start value in its tensor train format
	#	opt_rks: rank vector considered to be optimal enough
	# output:
	#	tt_opt: stationary point of J up to tolerated rank opt_rks
	# 			in its tensor train format

	# Initialize the to be returned tensor in its tensor train format
	tt_opt = deepcopy(tt_start)
	dims = tt_start.ttv_dims
	d = length(dims)
	E = zeros(Float64,d*sweep_schedule[end]) #output eigenvalue
	# Define the array of ranks of tt_opt [r_0=1,r_1,...,r_d]
	rks = vcat([1], tt_start.ttv_rks)

	# Initialize the arrays of G and K
	G = Array{Array{Float64}}(undef, d)
	K = Array{Array{Float64}}(undef, d) 

	# Initialize G[1]
	for i in 1:d
		G[i] = zeros(dims[i],rks[i],dims[i],rks[i],A.tto_rks[i])
		K[i] = zeros(dims[i],rks[i],dims[i],rks[i],S.tto_rks[i])
	end
	G[1] = reshape(A.tto_vec[1][:,:,1,:], dims[1],1,dims[1], 1, :)
	K[1] = reshape(S.tto_vec[1][:,:,1,:], dims[1],1,dims[1], 1, :)

	#Initialize H and H_b
	H = init_H(tt_opt,A)
	L = init_H(tt_opt,S)

	nsweeps = 0 #sweeps counter
	i_schedule,i_μit = 1,0
	while i_schedule <= length(sweep_schedule) 
		nsweeps+=1
		if nsweeps == sweep_schedule[i_schedule]
			i_schedule+=1
			if i_schedule > length(sweep_schedule)
				return E[1:i_μit],tt_opt
			else
				tt_opt = tt_up_rks(tt_opt,rmax_schedule[i_schedule])
				for i in 1:d-1
					Htemp = zeros(tt_opt.ttv_rks[i],tt_opt.ttv_rks[i],A.tto_rks[i])
					Ltemp = zeros(tt_opt.ttv_rks[i],tt_opt.ttv_rks[i],S.tto_rks[i])
					Htemp[1:size(H[i],1),1:size(H[i],2),1:size(H[i],3)] = H[i] 
					Ltemp[1:size(L[i],1),1:size(L[i],2),1:size(L[i],3)] = L[i] 
					H[i] = Htemp
					L[i] = Ltemp
				end
			end
		end

		# First half sweep
		for i = 1:(d-1)
			println("Forward sweep: core optimization $i out of $d")

			# If i is the index of the core matrices do the optimization
			if tt_opt.ttv_ot[i] == 0
				# Define V as solution of K*x=Pb in x
				i_μit += 1
				E[i_μit],V = K_eiggenmin(G[i],H[i],K[i],L[i],tt_opt.ttv_vec[i];it_solver=it_solver,itslv_thresh=itslv_thresh)
				println("Eigenvalue: $(E[i_μit])")
				tt_opt, rks[i+1] = right_core_move(tt_opt,V,i,rks)
			end

			#update G and K
			update_G!(tt_opt,A,i,G[i],G[i+1])
			update_G!(tt_opt,S,i,K[i],K[i+1])
		end

		# Second half sweep
		for i = d:(-1):2
			println("Backward sweep: core optimization $i out of $d")
			# Define V as solution of K*x=Pb in x
			i_μit += 1
			E[i_μit],V = K_eiggenmin(G[i],H[i],K[i],L[i],tt_opt.ttv_vec[i];it_solver=it_solver,itslv_thresh=itslv_thresh)
			println("Eigenvalue: $(E[i_μit])")
			tt_opt,rks[i] = left_core_move(tt_opt,V,i,rks)
			update_H!(tt_opt,A,i,H[i],H[i-1])
			update_H!(tt_opt,S,i,L[i],L[i-1])
		end
	end
end