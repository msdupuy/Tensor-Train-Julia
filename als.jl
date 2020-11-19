include("tt_tools.jl")

#TODO: eigs version and include IterativeSolvers option

function als(A :: ttoperator, b :: ttvector, tt_start :: ttvector, opt_rks :: Array{Int64};N_halfsweep=1,it_solver=false,r_itsolver=5000)
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
	n_max = maximum(dims)
	# Define the array of ranks of tt_opt [r_0=1,r_1,...,r_d]
	rks = vcat([1], tt_start.ttv_rks)
	r_max = maximum(tt_start.ttv_rks)
	# Define the array of ranks of A [R_0=1,R_1,...,R_d]
	A_rks = vcat([1],A.tto_rks)
	rA_max = maximum(A_rks)
	# Define the array of ranks of b [R^b_0=1,R^b_1,...,R^b_d]
	b_rks = vcat([1],b.ttv_rks)
	rb_max = maximum(b_rks)

	# Initialize the arrays of G and H
	G = Array{Array{Float64}}(undef, d)
	H = Array{Array{Float64}}(undef, d)
	# Initialize the tensors M1, M2, N1 and N2
	M1 = zeros(r_max, n_max, r_max, rA_max) # k_i, y_i, k_(i-1), j_i
	M2 = zeros(r_max, r_max, rA_max) # k'_i, k_i, j_i
	N1 = zeros(n_max, r_max, r_max, rA_max) # x_i, k_(i-1), k'_i, j_i
	N2 = zeros(r_max, r_max, n_max, rA_max) # k_(i-1), k'_i, y_i, j_(i-1)
	# Initialize the arrays of G_b and H_b
	G_b = Array{Array{Float64}}(undef, d)
	H_b = Array{Array{Float64}}(undef, d)
	# Initialize the tensors M_b and N_b
	M_b = zeros(rb_max, r_max) # l_i, k_i
	N_b = zeros(n_max, r_max, rb_max) # x_i,k_(i-1),j^b_i
	# Initialize the matrices K, Pb, V, QV and RV
	K = zeros(n_max, r_max, r_max, n_max, r_max, r_max)
	Pb = zeros(n_max, r_max, r_max)
	V = zeros(n_max, r_max, r_max)
	QV = zeros(r_max*n_max, r_max*n_max)
	RV = zeros(r_max, r_max)

	# Initialize G[1], G_b[1], H[d] and H_b[d]
	G[1] = zeros(dims[1], 1, dims[1], 1, A_rks[2])
	G[1] = reshape(A.tto_vec[1][:,:,1,:], dims[1],1,dims[1], 1, :)
	G_b[1] = zeros(b_rks[2], dims[1], 1)
	G_b[1] = permutedims(reshape(b.ttv_vec[1][:,1,1:b_rks[2]], dims[1], 1, :), [3 1 2])
	H[d] = ones(1,1,1)
	H_b[d] = ones(1,1)

	#while 1==1 #TODO make it work for real
		# Fill in  H[1],...,H[d-1] and H_b[1],...,H_b[d-1]
		for i = d : -1 : 2
			ni = dims[i] # n_i
			nim = dims[i-1] # n_(i-1)
			ri = rks[i+1] # r_i
			rim = rks[i] # r_(i-1)
			rAi = A_rks[i+1] # R_i
			rAim = A_rks[i] # R_(i-1)
			rbi = b_rks[i+1] # R^b_i
			rbim = b_rks[i] # R^b_(i-1)

			N1[1:ni, 1:rim, 1:ri, 1:rAi] =
				reshape(reshape(tt_opt.ttv_vec[i][:,:,:], ni*rim, :) *
						reshape(H[i][:, :, :], ri, :), ni, rim, ri, :)
			N2[1:rim, 1:ri, 1:ni, 1:rAim] =
				reshape(reshape(permutedims(N1[1:ni, 1:rim, 1:ri, 1:rAi],[2 3 1 4]), rim*ri, :) *
						reshape(permutedims(A.tto_vec[i][:,:,:,:],[1 4 2 3]),ni*rAi, :),
					rim, ri, ni, :)
			# Initialize H[i-1]
			H[i-1] = zeros(rim, rim, rAim) #k_(i-1),k'_(i-1),l_(i-1)
			# Fill in H[i-1]
			H[i-1] = permutedims(reshape(reshape(permutedims(tt_opt.ttv_vec[i][:,:,:],[2 3 1]), rim,:) *
										reshape(permutedims(N2[1:rim, 1:ri, 1:ni, 1:rAim],[2 3 1 4]), ri*ni,:),
								rim, rim, :), [2 1 3])

			N_b[1:ni, 1:rim, 1:rbi] =
				reshape(reshape(tt_opt.ttv_vec[i][:,:,:],ni*rim, :) * H_b[i][:,:],
						ni, rim, :)
			# Initialize H_b[i-1]
			H_b[i-1] = zeros(rbim, rim) # k_(i-1), j^b_(i-1)
			# Fill in H_b[i-1]
			H_b[i-1] = reshape(permutedims(N_b[1:ni, 1:rim, 1:rbi], [2 1 3]), rim, :) *
						reshape(permutedims(b.ttv_vec[i][:,:,:],[1 3 2]), ni * rbi, :)
		end

		# First half sweap
		for i = 1:(d-1)
			ni = dims[i] # n_i
			nip = dims[i+1] # n_(i+1)
			ri = rks[i+1] # r_i
			rip = rks[i+2] # r_(i+1)
			rim = rks[i] # r_(i-1)
			rAi = A_rks[i+1] # R_i
			rAip = A_rks[i+2] # R_(i+1)
			rbi = b_rks[i+1] # R^b_i
			rbip = b_rks[i+2] # R^b_(i+1)

			# If i is the index of the core matrices do the optimization
			if tt_opt.ttv_ot[i] == 0
				# Define V as solution of K*x=Pb in x
				K[1:ni, 1:rim, 1:ri, 1:ni, 1:rim, 1:ri] =
				permutedims(reshape(reshape(permutedims(G[i][1:ni, 1:rim, 1:ni, 1:rim, 1:rAi],
												[3 4 1 2 5]), ni*rim*ni*rim, :) *
									reshape(permutedims(H[i][1:ri, 1:ri, 1:rAi], [3 2 1]), rAi, :),
					ni, rim, ni, rim, ri, :), [1 2 5 3 4 6])
				Pb[1:ni, 1:rim, 1:ri] =
				reshape(reshape(permutedims(G_b[i][:,:,:], [2 3 1]), ni*rim, :) *
						Transpose(H_b[i][:,:]), ni, rim, :)
				V[1:ni, 1:rim, 1:ri]=
					reshape(reshape(K[1:ni, 1:rim, 1:ri, 1:ni, 1:rim, 1:ri], ni*rim*ri, :) \ reshape(Pb[1:ni, 1:rim, 1:ri],:,1), ni, rim, :)

				# Prepare core movements
				ri_new = min(rim*ni, ri)
				QV = zeros(ni*rim, ni*rim)
				RV = zeros(ri, ri)
				tt_opt.ttv_rks[i] = ri_new
				rks[i+1] = ri_new
				QV[1:ni*rim, 1:ni*rim], RV[1:ri_new, 1:ri] =
					qr(reshape(V[1:ni, 1:rim, 1:ri], ni*rim, :))

				# Apply core movement 3.1
				tt_opt.ttv_vec[i][1:ni, 1:rim, 1:ri_new] =
					reshape(QV[1:ni*rim, 1:ri_new], ni, rim, :)
				tt_opt.ttv_vec[i][1:ni, 1:rim, (ri_new+1):ri] = zeros(ni,rim,ri-ri_new)
				tt_opt.ttv_ot[i] = -1

				# Apply core movement 3.2
				tt_opt.ttv_vec[i+1][1:nip, 1:ri, 1:rip] =
					permutedims(reshape(RV[1:ri, 1:ri] *
										reshape(permutedims(tt_opt.ttv_vec[i+1][1:nip, 1:ri, 1:rip], [2 1 3]), ri, :),
						ri, nip, :), [2 1 3])
				tt_opt.ttv_ot[i+1] = 0
			end

			M1[1:ri, 1:ni, 1:rim, 1:rAi] =
				reshape(reshape(permutedims(tt_opt.ttv_vec[i][1:ni, 1:rim, 1:ri], [3 1 2]), ri, :) *
						reshape(G[i][:,:,:,:,:], ni*rim, :), ri, ni, rim, :)
			M2[1:ri, 1:ri, 1:rAi] =
				reshape(reshape(permutedims(tt_opt.ttv_vec[i][:,:,:], [3 1 2]), ri,:) *
						reshape(permutedims(M1[1:ri, 1:ni, 1:rim, 1:rAi], [2 3 1 4]), ni*rim, :),
						ri, ri, :)
			# Initialize G[i+1]
			G[i+1] = zeros(nip, ri, nip, ri, rAip) # x_(i+1), k_i, y_(i+1), k'_i, j_(i+1)
			# Fill in G[i+1]
			G[i+1][:,:,:,:,:] =
				permutedims(reshape(reshape(M2[1:ri,1:ri,1:rAi], ri*ri,:) *
						reshape(permutedims(A.tto_vec[i+1][:,:,:,:], [3 1 2 4]), rAi,:),
						ri, ri, nip, nip,:), [3 2 4 1 5])

			M_b[1:rbi, 1:ri] = reshape(G_b[i][:,:,:], rbi, :) *
								reshape(tt_opt.ttv_vec[i][:,:,:], ni*rim, :)
			# Initialize G_b[i+1]
			G_b[i+1] = zeros(rbip, nip, ri) # j_(i+1), x_(i+1), k_i
			# Fill in G_b[i+1]
			G_b[i+1][:,:,:] =
				reshape(reshape(permutedims(b.ttv_vec[i+1][:,:,:], [3 1 2]), :, rbi) *
				M_b[1:rbi, 1:ri], rbip, nip, :)
		end

		# If the first half sweap was enough return tt_opt
		if (tt_opt.ttv_rks == opt_rks) & (tt_start.ttv_ot[1] == 0)
			return tt_opt
		end

		# Second half sweap
		for i = d:(-1):2
			ni = dims[i] # n_i
			nim = dims[i-1] # n_(i-1)
			ri = rks[i+1] # r_i
			rim = rks[i] # r_(i-1)
			rim2 = rks[i-1] # r_(i-2)
			rAi = A_rks[i+1] # R_i
			rAim = A_rks[i] # R_(i-1)
			rbi = b_rks[i+1] # R^b_i
			rbim = b_rks[i] # R^b_(i-1)

			# Define V as solution of K*x=Pb in x
			K[1:ni, 1:rim, 1:ri, 1:ni, 1:rim, 1:ri] =
				permutedims(reshape(reshape(permutedims(G[i][1:ni, 1:rim, 1:ni, 1:rim, 1:rAi],
												[3 4 1 2 5]), ni*rim*ni*rim, :) *
										reshape(permutedims(H[i][1:ri, 1:ri, 1:rAi], [3 2 1]), rAi, :),
									ni, rim, ni, rim, ri, :), [1 2 5 3 4 6])
			Pb[1:ni, 1:rim, 1:ri] =
				reshape(reshape(permutedims(G_b[i][:,:,:], [2 3 1]), ni*rim, :) *
						Transpose(H_b[i][:,:]), ni, rim, :)
			V[1:ni, 1:rim, 1:ri]=
				reshape(reshape(K[1:ni, 1:rim, 1:ri, 1:ni, 1:rim, 1:ri], ni*rim*ri, :) \
						reshape(Pb[1:ni, 1:rim, 1:ri],:,1), ni, rim, :)

			# Prepare core movements
			rim_new = min(ri*ni, rim)
			QV = zeros(ni*ri, ni*ri)
			RV = zeros(rim, rim)
			tt_opt.ttv_rks[i-1] = rim_new
			rks[i] = rim_new
			QV[1:ni*ri, 1:ni*ri], RV[1:rim_new, 1:rim] =
					qr(reshape(permutedims(V[1:ni, 1:rim, 1:ri], [1 3 2]), ni*ri, :))

			# Apply core movement 3.2
			tt_opt.ttv_vec[i][1:ni, 1:rim_new, 1:ri] =
					permutedims(reshape(QV[1:ni*ri, 1:rim_new], ni, ri, :),[1 3 2])
			tt_opt.ttv_vec[i][1:ni, (rim_new+1):rim, 1:ri] = zeros(ni, rim-rim_new, ri)
			tt_opt.ttv_ot[i] = 1

			# Apply core movement 3.2
			tt_opt.ttv_vec[i-1][1:nim, 1:rim2, 1:rim_new] =
					reshape(reshape(tt_opt.ttv_vec[i-1][1:nim, 1:rim2, 1:rim], nim*rim2, :) *
							Transpose(RV[1:rim_new, 1:rim]),
						nim, rim2, :)
				tt_opt.ttv_ot[i-1] = 0

			N1[1:ni, 1:rim, 1:ri, 1:rAi] =
				reshape(reshape(tt_opt.ttv_vec[i][:,:,:], ni*rim, :) *
						reshape(H[i][:, :, :], ri, :), ni, rim, ri, :)
			N2[1:rim, 1:ri, 1:ni, 1:rAim] =
				reshape(reshape(permutedims(N1[1:ni, 1:rim, 1:ri, 1:rAi],[2 3 1 4]), rim*ri, :) *
						reshape(permutedims(A.tto_vec[i][:,:,:,:],[1 4 2 3]),ni*rAi, :),
					rim, ri, ni, :)
			# Reinitialize H[i-1]
			H[i-1] = zeros(rim, rim, rAim) #k_i,k'_i,l_i
			# Fill in H[i-1]
			H[i-1] = permutedims(reshape(reshape(permutedims(tt_opt.ttv_vec[i][:,:,:],[2 3 1]), rim,:) *
										reshape(permutedims(N2[1:rim, 1:ri, 1:ni, 1:rAim],[2 3 1 4]), ri*ni,:),
								rim, rim, :), [2 1 3])

			N_b[1:ni, 1:rim, 1:rbi] =
			reshape(reshape(tt_opt.ttv_vec[i][:,:,:],ni*rim, :) * H_b[i][:,:],
					ni, rim, :)
			# Reinitialize H_b[i-1]
			H_b[i-1] = zeros(rbim, rim) #k_i-1, j^b_i-1
			# Fill in H_b[i-1]
			H_b[i-1] = reshape(permutedims(N_b[1:ni, 1:rim, 1:rbi], [2 1 3]), rim, :) *
						reshape(permutedims(b.ttv_vec[i][:,:,:],[1 3 2]), ni * rbi, :)
		end
	#end
	return tt_opt
end
