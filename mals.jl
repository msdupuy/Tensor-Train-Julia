include("tt_tools.jl")
include("als.jl")

using Plots
using IterativeSolvers

"""
MALS auxiliary functions
"""

function updateHim!(xtt_vec, Atto, Hi, Him)
	@tensor begin
		N1[a,b,c,d] := xtt_vec[y,a,z]*Hi[b,z,y,c,d] #size(ri,rip,nip,rAi)
		N2[a,b,c] := xtt_vec[y,a,z]*N1[b,z,y,c] #size(ri,ri,rAi)
		Him[a,b,c,d,e] = N2[a,b,z]*Atto[c,d,e,z] #size(ri,ri,ni,ni,rAim)
	end
end

function updateH_bim!(xtt_vec, btt_vec, Hbi, Hbim)
	@tensor begin
		N_b[a,b] := xtt_vec[z,a,y]*Hbi[y,z,b] #size(ri,rbi)
		Hbim[a,b,c] = N_b[a,z]*btt_vec[b,c,z] #size(ri,ni,rbim)
	end
end


function updateGip!(i, ni, nip, rim, ri, rAi, rAip, M1, M2, G, tt_opt, A)
	M1[1:ri, 1:ni, 1:rim, 1:rAi] =
		reshape(reshape(permutedims(tt_opt.ttv_vec[i][1:ni, 1:rim, 1:ri], [3 1 2]), ri, :) *
				reshape(G[i][ 1:ni, 1:rim, 1:ni, 1:rim, 1:rAi], ni*rim, :), ri, ni, rim, :)
	M2[1:ri, 1:ri, 1:rAi] =
		reshape(reshape(permutedims(tt_opt.ttv_vec[i][1:ni, 1:rim, 1:ri], [3 1 2]), ri,:) *
				reshape(permutedims(M1[1:ri, 1:ni, 1:rim, 1:rAi], [2 3 1 4]), ni*rim, :),
				ri, ri, :)
	# Initialize G[i+1]
	G[i+1] = zeros(nip, ri, nip, ri, rAip) # x_(i+1), k_i, y_(i+1), k'_i, j_(i+1)
	# Fill in G[i+1]
	G[i+1][:,:,:,:,:] =
		permutedims(reshape(reshape(M2[1:ri,1:ri,1:rAi], ri*ri,:) *
				reshape(permutedims(A.tto_vec[i+1][1:nip, 1:nip, 1:rAi, 1:rAip], [3 1 2 4]), rAi,:),
				ri, ri, nip, nip,:), [3 2 4 1 5])
end

function updateG_bip!(i, ni, nip, rim, ri, rbi, rbip, M_b, G_b, tt_opt, b)
	M_b[1:rbi, 1:ri] = reshape(G_b[i][1:rbi, 1:ni, 1:rim], rbi, :) *
						reshape(tt_opt.ttv_vec[i][1:ni, 1:rim, 1:ri], ni*rim, :)
	# Initialize G_b[i+1]
	G_b[i+1] = zeros(rbip, nip, ri) # j_(i+1), x_(i+1), k_i
	# Fill in G_b[i+1]
	G_b[i+1][1:rbip, 1:nip, 1:ri] =
		reshape(reshape(permutedims(b.ttv_vec[i+1][1:nip, 1:rbi, 1:rbip], [3 1 2]), :, rbi) *
		M_b[1:rbi, 1:ri], rbip, nip, :)
end

function calcopt!(i, ni, nip, rim, rip, rAi, rbi, K, Pb, V, G, H, G_b, H_b)
	K[1:rim, 1:ni, 1:nip, 1:rip, 1:rim, 1:ni, 1:nip, 1:rip] =
		permutedims(reshape(reshape(G[i][1:ni, 1:rim, 1:ni, 1:rim, 1:rAi], ni*rim*ni*rim, :) *
							reshape(permutedims(H[i][1:rip, 1:rip, 1:nip, 1:nip, 1:rAi], [5 1 2 3 4]), rAi, :),
							ni, rim, ni, rim, rip, rip, nip, :), [4 3 8 5 2 1 7 6])
	Pb[1:rim, 1:ni, 1:nip, 1:rip] =
		permutedims(reshape(reshape(H_b[i][1:rip, 1:nip, 1:rbi], rip*nip, :) *
							reshape(G_b[i][1:rbi, 1:ni, 1:rim], rbi, :),
							rip, nip, ni, rim), [4 3 2 1])
	V[1:rim, 1:ni, 1:nip, 1:rip] =
		reshape(reshape(K[1:rim, 1:ni, 1:nip, 1:rip, 1:rim, 1:ni, 1:nip, 1:rip], rim*ni*nip*rip, :) \
				reshape(Pb[1:rim, 1:ni, 1:nip, 1:rip], :, 1), rim, ni, nip, :)
end

function mals(A :: ttoperator, b :: ttvector, tt_start :: ttvector, eps::Float64)
	# mals finds the minimum of the operator J(x)=1/2*<Ax,x> - <x,b>
	# input:
	# 	A: the tensor operator in its tensor train format
	#   b: the tensor in its tensor train format
	#	tt_start: start value in its tensor train format
	#	opt_rks: rank vector considered to be optimal enough
	#	eps: tolerated inaccuracy
	# output:
	#	tt_opt: stationary point of J up to tolerated inaccuracy
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
	H = Array{Array{Float64}}(undef, d-1)
	# Initialize the tensors M1, M2, N1 and N2
	M1 = zeros(r_max, n_max, r_max, rA_max) # k_i, y_i, k_(i-1), j_i
	M2 = zeros(r_max, r_max, rA_max) # k'_i, k_i, j_i
	# Initialize the arrays of G_b and H_b
	G_b = Array{Array{Float64}}(undef, d)
	H_b = Array{Array{Float64}}(undef, d-1)
	# Initialize the tensors M_b and N_b
	M_b = zeros(rb_max, r_max) # k_i, l_i
	# Initialize the matrices K, Pb and V
	K = zeros(r_max, n_max, n_max, r_max, r_max, n_max, n_max, r_max)
	Pb = zeros(r_max, n_max, n_max, r_max)
	V = zeros(r_max, n_max, n_max, r_max)
	# Initialize G[1], G_b[1], H[d] and H_b[d]
	for i in 1:d
		G[i] = zeros(dims[i],rks[i],dims[i],rks[i],A_rks[i+1])
		G_b[i] = zeros(b_rks[i+1],dims[i],rks[i])
	end
	for i in 1:d-1
		H[i] = zeros(rks[i+2],rks[i+2],dims[i+1],dims[i+1],A_rks[i+1])
		H_b[i] = zeros(rks[i+2],dims[i+1],b_rks[i])
	end
	G[1] = reshape(A.tto_vec[1][:,:,1,:], dims[1],1,dims[1], 1, :)
	G_b[1] = permutedims(reshape(b.ttv_vec[1][:,1,1:b_rks[2]], dims[1], 1, :), [3 1 2])
	H[d-1] = reshape(A.tto_vec[d], 1, 1, dims[d], dims[d], :) # k'_d,k_d,x_d,y_d,j_(d-1)
	H_b[d-1] = reshape(b.ttv_vec[d], 1, dims[d], :) # k_d, x_d, j^b_(d-1)

	#while 1==1 #TODO make it work for real
		# Fill in  H[1],...,H[d-2] and H_b[1],...,H_b[d-2]
		for i = (d-1) : -1 : 2
			ni = dims[i] # n_i
			nip = dims[i+1] # n_(i+1)
			ri = rks[i+1] # r_i
			rip = rks[i+2] # r_(i+1)
			rAi = A_rks[i+1] # R_i
			rAim = A_rks[i] # R_(i-1)
			rbi = b_rks[i+1] # R^b_i
			rbim = b_rks[i] # R^b_(i-1)

			# Update H[i-1]
			updateHim!(tt_opt.ttv_vec[i+1], A.tto_vec[i], H[i], H[i-1])
			# Update H_b[i-1]
			updateH_bim!(tt_opt.ttv_vec[i+1], b.ttv_vec[i], H_b[i],H_b[i-1])
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
				# Define V as solution of K*x=P2b in x
				calcopt!(i, ni, nip, rim, rip, rAi, rbi, K, Pb, V, G, H, G_b, H_b)
				# Perform the truncated svd
				V = map(x -> round(x, digits=10),V)
				u_V, s_V, v_V, = svd(reshape(V[1:rim, 1:ni, 1:nip, 1:rip], rim*ni, :))
				# Determine the truncated rank
				s_V_sq = s_V.^2
				bd = (eps^2)*sum(s_V_sq)
				ri_trunc = 1
				while (sum(s_V_sq[ri_trunc:ri]) >= bd) & (ri_trunc < ri)
					ri_trunc+=1
				end
				# Update the ranks to the truncated one
				ri = ri_trunc
				tt_opt.ttv_rks[i] = ri_trunc
				rks[i+1] = ri_trunc

				# tt_opt.ttv_vec[i] = truncated u_V
				tt_opt.ttv_vec[i][1:ni, 1:rim, 1:ri_trunc] =
					permutedims(reshape(u_V[1:(rim*ni), 1:ri_trunc], rim, ni, :), [2 1 3])
				tt_opt.ttv_vec[i][1:ni, 1:rim, (ri_trunc+1):ri] = zeros(ni,rim,ri-ri_trunc)
				tt_opt.ttv_ot[i] = -1

				# tt_opt.ttv_vec[i+1] = truncated Diagonal(s_V) * Transpose(v_V)
				tt_opt.ttv_vec[i+1][1:nip, 1:ri_trunc, 1:rip] =
					permutedims(reshape(Diagonal(s_V[1:ri_trunc]) *
										Transpose(v_V[1:nip*rip, 1:ri_trunc]),
										ri_trunc, nip, :), [2 1 3])
				tt_opt.ttv_ot[i+1] = 0
			end

			# Update G[i+1]
			updateGip!(i, ni, nip, rim, ri, rAi, rAip, M1, M2, G, tt_opt, A)
			# Update G_b[i+1]
			updateG_bip!(i, ni, nip, rim, ri, rbi, rbip, M_b, G_b, tt_opt, b)
		end

		# Second half sweap
		for i = d-1:(-1):1
			ni = dims[i] # n_i
			nip = dims[i+1] # n_(i+1)
			ri = rks[i+1] # r_i
			rim = rks[i] # r_(i-1)
			rip = rks[i+2] #r_(i+1)
			rAi = A_rks[i+1] # R_i
			rAim = A_rks[i] # R_(i-1)
			rbi = b_rks[i+1] # R^b_i
			rbim = b_rks[i] # R^b_(i-1)

			# Define V as solution of K*x=P2b in x
			calcopt!(i, ni, nip, rim, rip, rAi, rbi, K, Pb, V, G, H, G_b, H_b)

			# Perform the truncated svd
			V = map(x -> round(x, digits=10),V)
			u_V, s_V, v_V =svd(reshape(V[1:rim, 1:ni, 1:nip, 1:rip], rim*ni, :))
			# Determine the truncated rank
			s_V_sq = s_V.^2
			s = (eps^2)*sum(s_V_sq)
			ri_trunc = 1
			while (sum(s_V_sq[ri_trunc:ri]) >= s) & (ri_trunc < ri)
				ri_trunc+=1
			end
			# Update the ranks to the truncated one
			ri = ri_trunc
			tt_opt.ttv_rks[i] = ri_trunc
			rks[i+1] = ri_trunc

			# tt_opt.ttv_vec[i+1] = truncated Transpose(v_V)
			tt_opt.ttv_vec[i+1][1:nip, 1:ri_trunc, 1:rip] =
				permutedims(reshape(permutedims(v_V[:, 1:ri_trunc], [2 1]), ri_trunc, nip, :), [2 1 3])
			tt_opt.ttv_vec[i+1][1:nip, (ri_trunc+1):ri, 1:rip] = zeros(nip, ri-ri_trunc, rip)
			tt_opt.ttv_ot[i+1] = 1

			# tt_opt.ttv_vec[i] = truncated u_V * Diagonal(s_V)
			tt_opt.ttv_vec[i][1:ni, 1:rim, 1:ri_trunc] =
				permutedims(reshape(u_V[:, 1:ri_trunc] * Diagonal(s_V[1:ri_trunc]),
									rim, ni, :), [2 1 3])
			tt_opt.ttv_ot[i] = 0

			# Update H[i-1], H_b[i-1]
			if i > 1
				updateHim!(tt_opt.ttv_vec[i+1], A.tto_vec[i], H[i], H[i-1])
				updateH_bim!(tt_opt.ttv_vec[i+1], b.ttv_vec[i], H_b[i],H_b[i-1])
			end
		end
	#end
	return tt_opt
end

function interact_with_user(prompt)
	println(stdout, prompt)
	return chomp(readline())
end

function iterate_MALS(auto :: Bool, step_max :: Int64, A :: ttoperator, b :: ttvector, x_start :: ttvector, eps :: Float64)
	# Perform the MALS full step iteratively
	# Input:
	# 	auto: if auto == false after every full step the user can quit
	#	step_max: number of full steps performed at most
	# 	A: tensor operator in its tensor train format
	#   b: tensor in its tensor train format
	#	x_start: start value in its tensor train format
	#	eps: tolerated inaccuracy
	# output:
	#	x_tts: array of solutions of the performed MALS full steps
	# 	errs_rel: array of relative errors of the performed MALS full steps

	# Determine optimal solution with the ALS
	opttt_als = als(A,b,x_start,x_start.ttv_rks)
	opt_als = ttv_to_tensor(opttt_als)
	# Initialize x_tts and fbns
	x_tts = Array{ttvector}(undef, step_max + 1)
	x_tts[1] = x_start
	errs_rel = Array{Array{Float64}}(undef, (step_max + 1))
	errs_rel[1] = zeros(1)
	errs_rel[1][1] = norm(ttv_to_tensor(x_start) - opt_als) / norm(opt_als)

	for step_curr = 1:step_max
		x_tts[step_curr + 1], errs_rel[step_curr + 1] =
			mals(A, b, x_tts[step_curr], opt_als, eps)
		# Return if the calculated solution is good enough
		if (errs_rel[step_curr + 1][1] < eps) | (errs_rel[step_curr + 1][2] < eps)
			infotxt = string("Successfully terminated after step number: ", step_curr)
			println(stdout, infotxt)
			return x_tts[1:(step_curr + 1)], errs_rel[1:(step_curr + 1)]
		end
		# Check whether to stop
		if step_curr < step_max
			if !auto
				# Ask whether to go on
				infotxt = string("Number of currently performed steps: ", step_curr)
				println(stdout, infotxt)
				infotxt = string("The current relative error is: ", errs_rel[step_curr+1])
				println(stdout, infotxt)
				infotxt = "Press <q> and <Enter> to quit or <Enter> to continue: "
				input = interact_with_user(infotxt)
				# Quit and return the calculated steps if user wants to quit
				if input=="q"
					return x_tts[1:(step_curr + 1)], errs_rel[1:(step_curr + 1)]
				end
			end
		end
	end
	println(stdout, "All steps have been calculated and returned.")
	return x_tts, errs_rel
end

function plot_two_sweap_data(r)
	nmb_swp = length(r) - 1
	r_plot = zeros(2 * nmb_swp + 1)
	r_plot[1] = r[1][1]
	for i = 1:nmb_swp
		r_plot[2 * i] = r[i + 1][1]
		r_plot[2 * i + 1] = r[i + 1][2]
	end
	display(scatter(0:0.5:nmb_swp, r_plot, shape = [:circle, :star5]))
end


## First test for the function iterate_MALS()
## Define some random tensor operator A2 and its tensor train format
#A2 = map(round, rand(2,3,3,2,2,3,3,2).*10)
#A2_tt = tto_decomp(A2, 1)
## Define some random tensor x2 and its tensor train format
#x2 = map(round,rand(2,3,3,2).*10)
#x2_tt = ttv_decomp(x2, 4)
## Define b2=A2*x2 and its tensor train format
#b2 = reshape(reshape(permutedims(A2,[5 6 7 8 1 2 3 4]), 36, :)*reshape(x2,36,:),2,3,3,:)
#b2_tt = ttv_decomp(b2, 1)
## Define the start tensor as some random tensor and its tensor train format
#x2_start = map(round,rand(2,3,3,2).*10)
#x2_start_tt = ttv_decomp(x2_start, 4)
#
#o2_tts, err_rel2s = iterate_MALS(false, 10, A2_tt, b2_tt, x2_start_tt, 10^(-4))
#o2s = map(ttv_to_tensor, o2_tts)
#
#plot_two_sweap_data(err_rel2s)
