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

function updateGip!(xtt_vec,Atto_vec,Gi,Gip)
	@tensor begin
		M1[a,b,c,d] := xtt_vec[y,z,a]*Gi[y,z,b,c,d] #size(ri,ni,rim,rAi)
		M2[a,b,c] := xtt_vec[y,z,a]*M1[b,y,z,c] #size(ri,ri,rAi)
		Gip[a,b,c,d,e] = M2[d,b,z]*Atto_vec[a,c,z,e] #size(nip,ri,nip,ri,rAip)
	end
end

function updateG_bip!(xtt_vec,btt_vec,G_bi,G_bip)
	@tensor begin
		M_b[a,b] := G_bi[a,y,z]*xtt_vec[y,z,b] #size(rbi,ri)
		G_bip[a,b,c] = btt_vec[b,z,a]*M_b[z,c] #size(rbip,nip,ri)
	end
end

function left_core_move_mals(xtt::ttvector,i::Integer,V,tol::Float64,rmax::Integer)
	# Perform the truncated svd
	u_V, s_V, v_V, = svd(reshape(V, prod(size(V)[1:2]), :))
	# Determine the truncated rank
	s_trunc = sv_trunc(s_V,tol)
	# Update the ranks to the truncated one
	xtt.ttv_rks[i] = max(length(s_trunc),rmax)

	# xtt.ttv_vec[i+1] = truncated Transpose(v_V)
	xtt.ttv_vec[i+1] = permutedims(reshape(v_V[:, 1:xtt.ttv_rks[i]],size(V,3),size(V,4),:),[1,3,2])
	xtt.ttv_ot[i+1] = 1

	# xtt.ttv_vec[i] = truncated u_V * Diagonal(s_V)
	xtt.ttv_vec[i] = permutedims(reshape(u_V[:, 1:xtt.ttv_rks[i]] * Diagonal(s_trunc),size(V,1),size(V,2),:),[2,1,3])
#		permutedims(reshape(u_V[:, 1:ri_trunc] * Diagonal(s_trunc),
#							rim, ni, :), [2 1 3])
	xtt.ttv_ot[i] = 0
	return xtt
end

function right_core_move_mals(xtt::ttvector,i::Integer,V,tol::Float64,rmax::Integer)
	# Perform the truncated svd
	u_V, s_V, v_V, = svd(reshape(V, prod(size(V)[1:2]), :))
	# Determine the truncated rank
	s_trunc = sv_trunc(s_V,tol)
	# Update the ranks to the truncated one
	xtt.ttv_rks[i] = max(length(s_trunc),rmax)

	# xtt.ttv_vec[i] = truncated u_V
	xtt.ttv_vec[i] = permutedims(reshape(u_V[:, 1:xtt.ttv_rks], size(V,1), size(V,2), :), [2 1 3])
	xtt.ttv_ot[i] = -1

	# xtt.ttv_vec[i+1] = truncated Diagonal(s_V) * Transpose(v_V)
	xtt.ttv_vec[i+1] = permutedims(reshape(Diagonal(s_trunc) *
							Transpose(v_V[:, 1:ri_trunc]),xtt.ttv_rks[i],size(V,3),size(V,4)), [2 1 3])
	xtt.ttv_ot[i+1] = 0
	return xtt
end

function Ksolve_mals(Gi, Hi, G_bi, H_bi)
	@tensor begin
		K[a,b,c,d,e,f,g,h] := Gi[f,e,b,a,z]*Hi[d,h,g,c,z] #size(rim,ni,nip,rip,rim,ni,nip,rip)
		Pb[a,b,c,d] := H_bi[d,c,z]*G_bi[z,b,a] #size(rim,ni,nip,rip)
	end
	V = reshape(K,prod(size(K)[1:4]),:) \ Pb[:]
	return reshape(V,size(K)[1:4]...)
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
		for i = (d-1) : -1 : 2
			# Update H[i-1], H_b[i-1]
			updateHim!(tt_opt.ttv_vec[i+1], A.tto_vec[i], H[i], H[i-1])
			updateH_bim!(tt_opt.ttv_vec[i+1], b.ttv_vec[i], H_b[i],H_b[i-1])
		end

		# First half sweap
		for i = 1:(d-1)
			# If i is the index of the core matrices do the optimization
			if tt_opt.ttv_ot[i] == 0
				# Define V as solution of K*x=P2b in x
				V = Ksolve_mals(G[i],H[i],G_b[i],H_b[i])
				tt_opt = right_core_move_mals(tt_opt,i,V,tol,rmax)
			end
			# Update G[i+1],G_b[i+1]
			updateGip!(tt_opt.ttv_vec[i],A.tto_vec[i+1],G[i],G[i+1])
			updateG_bip!(tt_opt.ttv_vec[i],b.ttv_vec[i+1],G_b[i],G_b[i+1])
		end

		# Second half sweap
		for i = d-1:(-1):1
			# Define V as solution of K*x=P2b in x
			V = Ksolve_mals(G[i],H[i],G_b[i],H_b[i])
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
