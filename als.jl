using LinearAlgebra

struct ttvector
	# ttv_vec is an array of all matrix arrays for the tensor train format
	# ttv_vec[i] stores the matrices A_i in the following way
	# ttv_vec[i][1:x_(n_i), 1:r_(i-1), 1:r_i]
	ttv_vec :: Array{Array{Float64,3},1}

	# ttv_dims is an vector of the dimensions n_i i=1,...,d
	ttv_dims :: Array{Int64}

	# ttv_rks is the array of all ranks of the tensor train matrices
	# r_i i=1,...,d
	ttv_rks :: Array{Int64}

	# ttv_ot includes information about the orthogonality properties
	# ttv_ot[i] =  1 if the i-th matrices are rightorthonormal
	# ttv_ot[i] =  0 if there is no information or the i-th matrices aren't orthogonal
	# ttv_ot[i] = -1 if the i-th matrices are leftorthonormal
	ttv_ot :: Array{Int64}
end

struct ttoperator
	# tto_vec is an array of all matrices of the tensor train format
	# of an operator A(x1,...,xd,y1,...,yd)
	# tto_vec[i] stores the matrices A_i i=1,...,d in the following way
	# ttv_vec[i][1:x_(n_i), 1:y_(n_i), 1:r_(i-1), 1:r_i]
	tto_vec :: Array{Array{Float64,4},1}

	# tto_dims stores the dimensions n_i i=1,...,d
	tto_dims :: Array{Int64}

	# tto_rks is the array of all ranks of the tensor train matrices
	# r_i i=1,...,d
	tto_rks :: Array{Int64}

	# tto_ot is a matrix storing information about the orthoganlity properties
	# tto_ot[i] =  1 if the i-th matrices are rightorthogonal
	# tto_ot[i] =  0 if there is no information or the i-th matrices aren't orthogonal
	# tto_ot[i] = -1 if the i-th matrices are leftorthogonal
	tto_ot :: Array{Int64}
end

function ttv_decomp(tensor, index;tol=1e-12)
	# Decomposes a tensor into its tensor train with core matrices at i=index
	dims = collect(size(tensor)) #dims = [n_1,...,n_d]
	n_max = maximum(dims)
	d = length(dims)
	ttv_vec = Array{Array{Float64}}(undef,d)
	# ttv_ot[i]= -1 if i < index
	# ttv_ot[i] = 0 if i = index
	# ttv_ot[i] = 1 if i > index
	ttv_ot = -ones(Int64,d)
	ttv_ot[index] = 0
	if index < d
		ttv_ot[index+1:d] = ones(d-index)
	end
	rks = ones(Int, d+2) # ttv_rks will be rks[2:d+1]
	tensor_curr = tensor
	# Calculate ttv_vec[i] for i < index
	for i = 1 : (index - 1)
		# Reshape the currently left tensor
		tensor_curr = reshape(tensor_curr, Int(rks[i] * dims[i]), :)
		# Perform the singular value decomposition
		u, s, v = svd(tensor_curr)
		# Define the i-th rank
		rks[i+1] = length(s[s .>= tol])
		# Initialize ttv_vec[i]
		ttv_vec[i] = zeros(dims[i],rks[i],rks[i+1])
		# Fill in the ttv_vec[i]
		for x = 1 : dims[i]
			ttv_vec[i][x, :, :] = u[(rks[i]*(x-1) + 1):(rks[i]*x), :]
		end
		# Update the currently left tensor
		tensor_curr = Diagonal(s[1:rks[i+1]])*Transpose(v[:,1:rks[i+1]])
	end

	# Calculate ttv_vec[i] for i > index
	if index < d
		for i = d : (-1) : (index + 1)
			# Reshape the currently left tensor
			tensor_curr = reshape(tensor_curr, :, dims[i] * rks[i+1])
			# Perform the singular value decomposition
			u, s, v = svd(tensor_curr)
			# Define the (i-1)-th rank
			rks[i]=length(s[s .>= tol])
			# Initialize ttv_vec[i]
			ttv_vec[i] = zeros(dims[i], rks[i], rks[i+1])
			# Fill in the ttv_vec[i]
			i_vec = zeros(Int,rks[i+1])
			for x = 1 : dims[i]
				i_vec = dims[i]*((1:rks[i+1])-ones(Int, rks[i+1])) + x*ones(Int,rks[i+1])
				ttv_vec[i][x, :, :] = Transpose(v[i_vec, 1:rks[i]]) #(rks[i+1]*(x-1)+1):(rks[i+1]*x)
			end
			# Update the currently left tensor
			tensor_curr = u[:,1:rks[i]]*Diagonal(s[1:rks[i]])
		end
	end
	# Calculate ttv_vec[i] for i = index
	# Reshape the currently left tensor
	tensor_curr = reshape(tensor_curr, Int(dims[index]*rks[index]),:)
	# Initialize ttv_vec[i]
	ttv_vec[index] = zeros(dims[index], rks[index], rks[index+1])
	# Fill in the ttv_vec[i]
	for x = 1 : dims[index]
		ttv_vec[index][x, :, :] =
		tensor_curr[Int(rks[index]*(x-1) + 1):Int(rks[index]*x), 1:rks[index+1]]
	end

	# Define the return value as a ttvector
	return ttvector(ttv_vec, dims, rks[2:d+1], ttv_ot)
end

function ttv_to_tensor(ttv :: ttvector)
	d = length(ttv.ttv_dims)
	# Define the array of ranks [r_0=1,r_1,...,r_d]
	rks = ones(Int,d+1)
	rks[2:(d+1)] = ttv.ttv_rks
	r_max = maximum(rks)
	# Initialize the to be returned tensor
	tensor = zeros(ttv.ttv_dims...)
	# Fill in the tensor for every t=(x_1,...,x_d)
	for t in CartesianIndices(tensor)
		curr = ones(1,r_max)
		a = collect(Tuple(t))
		for i = 1:d
			curr[1,1:rks[i+1]]=reshape(curr[1,1:rks[i]],1,:)*ttv.ttv_vec[i][a[i],1:rks[i],1:rks[i+1]]
		end
		tensor[t] = curr[1,1]
	end
	return tensor
end

function tto_decomp(tensor, index)
	# Decomposes a tensor operator into its tensor train
	# with core matrices at i=index
	# The tensor is given as tensor[x_1,...,x_d,y_1,...,y_d]
	d = Int(ndims(tensor)/2)
	tto_dims = collect(size(tensor))[1:d]
	n_max = maximum(tto_dims)
	dims_sq = tto_dims.^2
	# The tensor is reorder  into tensor[x_1,y_1,...,x_d,y_d],
	# reshaped into tensor[(x_1,y_1),...,(x_d,y_d)]
	# and decomposed into its tensor train with core matrices at i= index
	index_sorted = reshape(Transpose(reshape(1:(2*d),:,2)),1,:)
	ttv = ttv_decomp(reshape(permutedims(tensor,index_sorted),(dims_sq[1:(end-1)]...), :), index)
	# Define the array of ranks [r_0=1,r_1,...,r_d]
	rks = ones(Int, d+1)
	rks[2:(d+1)] = ttv.ttv_rks
	r_max = maximum(rks)
	# Initialize tto_vec
	tto_vec = Array{Array{Float64}}(undef,d)
	# Fill in tto_vec
	for i = 1:d
		# Initialize tto_vec[i]
		tto_vec[i] = zeros(tto_dims[i], tto_dims[i], rks[i], rks[i+1])
		# Fill in tto_vec[i]
		tto_vec[i][:, :, :, :] =
			reshape(ttv.ttv_vec[i][1:dims_sq[i], 1:rks[i], 1:rks[i+1]],
						tto_dims[i], tto_dims[i], rks[i], :)
	end
	# Define the return value as a ttoperator
	return ttoperator(tto_vec, tto_dims, rks[2:(d+1)], ttv.ttv_ot)
end

function tto_to_tensor(tto :: ttoperator)
	d = length(tto.tto_dims)
	# Define the array of ranks [r_0=1,r_1,...,r_d]
	rks = ones(Int,d+1)
	rks[2:(d+1)] = tto.tto_rks
	r_max = maximum(rks)
	# The tensor has dimensions [n_1,...,n_d,n_1,...,n_d]
	dims = zeros(Int, 2*d)
	dims[1:d] = tto.tto_dims
	dims[(d+1):(2*d)] = tto.tto_dims
	tensor = zeros(dims...)
	# Fill in the tensor for every t=(x_1,...,x_d,y_1,...,y_d)
	for t in CartesianIndices(tensor)
		curr = ones(1,r_max)
		a = collect(Tuple(t))
		for i = 1:d
			curr[1,1:rks[i+1]] = reshape(curr[1,1:rks[i]], 1, :) *
				tto.tto_vec[i][t[i], t[d + i], 1:rks[i], 1:rks[i+1]]
		end
		tensor[t] = curr[1,1]
	end
	return tensor
end

function als(A :: ttoperator, b :: ttvector, tt_start :: ttvector, opt_rks :: Array{Int64})
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
	rks = ones(Int, d+1)
	rks[2:(d+1)] = tt_start.ttv_rks
	r_max = maximum(tt_start.ttv_rks)
	# Define the array of ranks of A [R_0=1,R_1,...,R_d]
	A_rks = ones(Int, d+1)
	A_rks[2:(d+1)] = A.tto_rks
	rA_max = maximum(A_rks)
	# Define the array of ranks of b [R^b_0=1,R^b_1,...,R^b_d]
	b_rks = ones(Int, d+1)
	b_rks[2:(d+1)] = b.ttv_rks
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
				V = map(x -> round(x, digits=10),V)
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
			V = map(x -> round(x, digits=10),V)
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

# Define some random tensor T
T = map(round, rand(2,3,4).*10)

function tdr(A, index)
	# Decomposes the tensor A with core matrices at i = index,
	# reconstructs the tensor and returns both tensors
	AT = ttv_decomp(A, index)
	B = ttv_to_tensor(AT)
	return A, B
end

# Define some random tensor operator O
O = map(round, rand(2,3,4,5,2,3,4,5).*10)

function odr(O, index)
	# Decomposes the tensor operator O with core matrices at i = index,
	# reconstructs the tensor operator and returns both tensor operators
	OT = tto_decomp(O, index)
	P = tto_to_tensor(OT)
	return O, P
end

# First test for the function als()
# Define some random tensor operator A and its tensor train format
A = map(round, rand(2,3,4,2,3,4).*10)
A_tt = tto_decomp(A, 1)
# Define some random tensor x and its tensor train format
x = map(round,rand(2,3,4).*10)
x_tt = ttv_decomp(x, 3)
# Define b=A*x and its tensor train format
b = reshape(reshape(permutedims(A,[4 5 6 1 2 3]), 24, :)*reshape(x,24,:),2,3,:)
b_tt = ttv_decomp(b, 1)
# Define the start tensor as some random tensor and its tensor train format
x_start = map(round,rand(2,3,4).*10)
x_start_tt = ttv_decomp(x_start, 1)

o_tt=als(A_tt, b_tt, x_start_tt, x_tt.ttv_rks)

# Second test for the function als()
# Define some random tensor operator A1 and its tensor train format
A1 = map(round, rand(2,3,4,2,3,4).*10)
A1_tt = tto_decomp(A1, 1)
# Define some random tensor x1 and its tensor train format
x1 = map(round,rand(2,3,4).*10)
x1_tt = ttv_decomp(x1, 3)
# Define b1=A1*x1 and its tensor train format
b1 = reshape(reshape(permutedims(A1,[4 5 6 1 2 3]), 24, :)*reshape(x1,24,:),2,3,:)
b1_tt = ttv_decomp(b1, 1)
# Define the start tensor as some random tensor and its tensor train format
x1_start = map(round,rand(2,3,4).*10)
x1_start_tt = ttv_decomp(x1_start, 3)

o1_tt=als(A1_tt, b1_tt, x1_start_tt, x1_tt.ttv_rks)
