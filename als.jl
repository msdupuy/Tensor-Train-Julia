include("tt_tools.jl")

#TODO: eigs version and include IterativeSolvers option

function init_H_and_Hb(x_tt::ttvector,b_tt::ttvector,A_tto::ttoperator)
	x_dims = x_tt.ttv_dims
	d = length(x_tt.ttv_dims)
	A_rks = vcat([1],A_tto.tto_rks)
	x_rks = vcat([1], x_tt.ttv_rks)
	b_rks = vcat([1], b_tt.ttv_rks)
	H = Array{Array{Float64}}(undef, d)
	H_b = Array{Array{Float64}}(undef, d) 

	H[d] = ones(1,1,1)
	H_b[d] = ones(1,1)
	for i = d : -1 : 2
		N1 = md_mult(x_tt.ttv_vec[i],H[i],[1,2,3],[1,2,3],2,1,[1 2 3 4])
		N2 = md_mult(N1,A_tto.tto_vec[i],[2 3 1 4],[1 4 2 3],2,2,[1,2,3,4])
		# Initialize H[i-1]
		H[i-1] = md_mult(x_tt.ttv_vec[i],N2,[2 3 1],[2 3 1 4],1,2,[2 1 3]) #size (rim, rim, rAim)

		N_b = reshape(reshape(x_tt.ttv_vec[i], x_dims[i]*x_rks[i], :) * H_b[i], x_dims[i], x_rks[i], :) #size (ni,rim,rbi)
		# Fill in H_b[i-1]
		H_b[i-1] = md_mult(N_b,b_tt.ttv_vec[i],[2 1 3],[1 3 2],1,2,[1 2]) # k_(i-1), j^b_(i-1)
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
	x_tt.ttv_vec[i-1][1:nim, 1:rim2, 1:rim_new] = md_mult(x_tt.ttv_vec[i-1][1:nim, 1:rim2, 1:rim],RV[1:rim_new, 1:rim],[1 2 3],[2 1],2,1,[1 2 3])
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
	x_tt.ttv_vec[i+1][1:nip, 1:ri, 1:rip] = md_mult(RV[1:ri, 1:ri],x_tt.ttv_vec[i+1][1:nip, 1:ri, 1:rip],[1 2],[2 1 3],1,1,[2 1 3])
	x_tt.ttv_ot[i+1] = 0

	return x_tt,ri_new
end

function update_G_Gb(x_tt::ttvector,x_dims,x_rks,b_tt::ttvector,b_rks,A_tto::ttoperator,A_rks,i,Gi,G_bi)
	rim,ri,rip = x_rks[i],x_rks[i+1],x_rks[i+2]
	rbi,rbip = b_rks[i+1],b_rks[i+2]
	ni,nip = x_dims[i],x_dims[i+1]
	rAip = A_rks[i+2]
	M1 = md_mult(x_tt.ttv_vec[i][1:ni, 1:rim, 1:ri],Gi,[3 1 2],[1 2 3 4 5],1,2,[1 2 3 4]) #size (ri,ni,rim,rAi)
	M2 = md_mult(x_tt.ttv_vec[i],M1,[3 1 2],[2 3 1 4],1,2,[1 2 3]) #size (ri,ri,rAi)

	# Initialize G[i+1]
	G = zeros(nip, ri, nip, ri, rAip) # x_(i+1), k_i, y_(i+1), k'_i, j_(i+1)
	# Fill in G[i+1]
	G = md_mult(M2,A_tto.tto_vec[i+1],[1 2 3],[3 1 2 4],2,1,[3 2 4 1 5])
	M_b = reshape(G_bi, rbi, :) * reshape(x_tt.ttv_vec[i], ni*rim, :)
	# Initialize G_b[i+1]
	G_b = zeros(rbip, nip, ri) # j_(i+1), x_(i+1), k_i
	# Fill in G_b[i+1]
	G_b = md_mult(b_tt.ttv_vec[i+1],M_b,[3 1 2],[1 2],2,1,[1 2 3])
	return G,G_b
end

function update_H_Hb(x_tt::ttvector,x_dims,x_rks,b_tt::ttvector,b_rks,A_tto::ttoperator,A_rks,i,Hi,H_bi)
	rim,ri = x_rks[i],x_rks[i+1]
	rbim,rbi = b_rks[i],b_rks[i+1]
	ni = x_dims[i]
	rAim,rAi = A_rks[i],A_rks[i+1]

	N1 = md_mult(x_tt.ttv_vec[i],Hi,[1 2 3],[1 2 3],2,1,[1 2 3 4]) #size (ni,rim,ri,rAi)
	N2 = md_mult(N1,A_tto.tto_vec[i],[2 3 1 4],[1 4 2 3],2,2,[1 2 3 4]) #size (rim,ri,ni,rAim)

	# Reinitialize H[i-1]
	H = zeros(rim, rim, rAim) #k_i,k'_i,l_i
	# Fill in H[i-1]
	H = md_mult(x_tt.ttv_vec[i],N2[1:rim, 1:ri, 1:ni, 1:rAim],[2 3 1],[2 3 1 4],1,2,[2 1 3])

	N_b = md_mult(x_tt.ttv_vec[i],H_bi,[1 2 3],[1 2],2,1,[1 2 3]) #size(ni,rim,rbi)
	# Reinitialize H_b[i-1]
	H_b = zeros(rbim, rim) #k_i-1, j^b_i-1
	# Fill in H_b[i-1]
	H_b = md_mult(N_b,b_tt.ttv_vec[i],[2 1 3],[1 3 2],1,2,[1 2])
	return H,H_b
end

function Ksolve(Gi,G_bi,Hi,H_bi,x_dims,x_rks,A_rks,i)
	rim,ri = x_rks[i],x_rks[i+1]
	rAi = A_rks[i+1]
	ni = x_dims[i]
	K = md_mult(Gi[1:ni, 1:rim, 1:ni, 1:rim, 1:rAi],Hi[1:ri, 1:ri, 1:rAi],[3 4 1 2 5],[3 2 1],4,1,[1 2 5 3 4 6]) #size (ni,rim,ri,ni,rim,ri)
	Pb = md_mult(G_bi,H_bi,[2 3 1],[2 1],2,1,[1 2 3]) #size (ni,rim,ri)
	V = md_div(K,Pb,[1 2 3 4 5 6],[1 2 3],3,3,[1 2 3])
	return V
end

function als(A :: ttoperator, b :: ttvector, tt_start :: ttvector, opt_rks :: Array{Int64};sweep_count=2,it_solver=false,r_itsolver=5000)
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
	# Define the array of ranks of A [R_0=1,R_1,...,R_d]
	A_rks = vcat([1],A.tto_rks)
	# Define the array of ranks of b [R^b_0=1,R^b_1,...,R^b_d]
	b_rks = vcat([1],b.ttv_rks)

	# Initialize the arrays of G and G_b
	G = Array{Array{Float64}}(undef, d)
	G_b = Array{Array{Float64}}(undef, d)

	# Initialize G[1], G_b[1], H[d] and H_b[d]
	G[1] = zeros(dims[1], 1, dims[1], 1, A_rks[2])
	G[1] = reshape(A.tto_vec[1][:,:,1,:], dims[1],1,dims[1], 1, :)
	G_b[1] = zeros(b_rks[2], dims[1], 1)
	G_b[1] = permutedims(reshape(b.ttv_vec[1][:,1,1:b_rks[2]], dims[1], 1, :), [3 1 2])

	#Initialize H and H_b
	H,H_b = init_H_and_Hb(tt_opt,b,A)

	nsweeps = 0 #sweeps counter

	while nsweeps < sweep_count
		nsweeps+=1

		# First half sweep
		for i = 1:(d-1)
			println("Forward sweep: core optimization $i out of $d")

			# If i is the index of the core matrices do the optimization
			if tt_opt.ttv_ot[i] == 0
				# Define V as solution of K*x=Pb in x
				V = Ksolve(G[i],G_b[i],H[i],H_b[i],dims,rks,A_rks,i)
				tt_opt, rks[i+1] = right_core_move(tt_opt,V,i,rks,dims)
			end

			#update G,G_b
			G[i+1],G_b[i+1] = update_G_Gb(tt_opt,dims,rks,b,b_rks,A,A_rks,i,G[i],G_b[i])
		end

		if nsweeps == sweep_count
			return tt_opt
		else
			nsweeps+=1
			# Second half sweep
			for i = d:(-1):2
				println("Backward sweep: core optimization $i out of $d")
				# Define V as solution of K*x=Pb in x
				V = Ksolve(G[i],G_b[i],H[i],H_b[i],dims,rks,A_rks,i)

				tt_opt,rks[i] = left_core_move(tt_opt,V,i,rks,dims)

				H[i-1],H_b[i-1] = update_H_Hb(tt_opt,dims,rks,b,b_rks,A,A_rks,i,H[i],H_b[i])
			end
		end
	end
	return tt_opt
end
