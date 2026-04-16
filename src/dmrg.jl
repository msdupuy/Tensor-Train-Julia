using LinearMaps
using TensorOperations
using KrylovKit

struct DMRGScheduler
	sweep_schedule
	rmax_schedule
	N #N site DMRG
	tol::Float64 # SVD truncation parameter
	it_solver::Bool
	linsolv_maxiter::Int64 #maximum of iterations for the iterative solver
	linsolv_tol::Float64 #tolerance of the iterative linear solver
	itslv_thresh::Int #switch from full to iterative
end

struct DMRGverbose
	schedule::DMRGScheduler
	TTvs::Vector{TTvector}
	matvec_count::Vector{Int64}
	ttsvd_weights::Vector{Float64}
end

# Pre-allocated scratch buffers for update_H! and update_G! (shared, never called simultaneously).
# Sized to the worst-case dimensions so no allocation occurs inside the hot sweep loops.
struct DMRGScratch{T<:Number}
	# Buffers for update_H! / update_G! (single-site environment update)
	xp    :: Matrix{T}  # (r_max,          n_max * r_max)
	HGp   :: Matrix{T}  # (r_max,          rA_max * r_max)
	T1    :: Matrix{T}  # (n_max * r_max,  rA_max * r_max)
	T1p   :: Matrix{T}  # (n_max * rA_max, r_max * r_max)
	Ap    :: Matrix{T}  # (n_max * rA_max, n_max * rA_max)
	T2    :: Matrix{T}  # (r_max * r_max,  n_max * rA_max)
	T2p   :: Matrix{T}  # (rA_max * r_max, n_max * r_max)
	xp2   :: Matrix{T}  # (n_max * r_max,  r_max)
	out   :: Matrix{T}  # (rA_max * r_max, r_max)
	# Buffers for K_matfree (N-site micro-step matvec in Ksolve! / K_eigmin)
	Km_T1  :: Matrix{T}  # (rA_max * r_max,   n_max^N * r_max)
	Km_T1p :: Matrix{T}  # (r_max^2,          rA_max * n_max^N)
	Km_T2  :: Matrix{T}  # (r_max^2,          n_max^N * rA_max)
	Km_T2p :: Matrix{T}  # (r_max * n_max^N,  r_max * rA_max)
	Km_T3  :: Matrix{T}  # (r_max * n_max^N,  r_max)
end

function DMRGScratch(::Type{T}, n_max::Int, r_max::Int, rA_max::Int, N::Int) where {T<:Number}
	nN_max = n_max^N
	DMRGScratch{T}(
		zeros(T, r_max,           n_max   * r_max),
		zeros(T, r_max,           rA_max  * r_max),
		zeros(T, n_max  * r_max,  rA_max  * r_max),
		zeros(T, n_max  * rA_max, r_max   * r_max),
		zeros(T, n_max  * rA_max, n_max   * rA_max),
		zeros(T, r_max  * r_max,  n_max   * rA_max),
		zeros(T, rA_max * r_max,  n_max   * r_max),
		zeros(T, n_max  * r_max,  r_max),
		zeros(T, rA_max * r_max,  r_max),
		zeros(T, rA_max * r_max,  nN_max  * r_max),
		zeros(T, r_max  * r_max,  rA_max  * nN_max),
		zeros(T, r_max  * r_max,  nN_max  * rA_max),
		zeros(T, r_max  * nN_max, r_max   * rA_max),
		zeros(T, r_max  * nN_max, r_max),
	)
end

function dmrg_schedule(sweeps_per_rank, rmin, rmax, n_rks;N=2,tol=1e-12,linsolv_maxiter=200,linsolv_tol=1e-8,it_solver=true,itslv_thresh=256)
	sweep_schedule = sweeps_per_rank*[k for k in 1:n_rks]
	rmax_schedule = round.(Int,logrange(rmin,rmax,length=n_rks))
	DMRGScheduler(sweep_schedule,rmax_schedule,N,tol,it_solver,linsolv_maxiter,linsolv_tol,itslv_thresh)
end

function dmrg_schedule_default(;N=2,rmax=64,nsweeps=2,it_solver=true)
	tol=1e-12
	N=N
	sweep_schedule=[nsweeps+1] #Number of sweeps for each bond dimension in rmax_schedule
	rmax_schedule=[rmax] #maximum rank in sweep_schedule
	it_solver=it_solver
	linsolv_maxiter=200 #maximum of iterations for the iterative solver
	linsolv_tol=max(sqrt(tol),1e-8) #tolerance of the iterative linear solver
	itslv_thresh=256 #switch from full to iterative
	DMRGScheduler(sweep_schedule,rmax_schedule,N,tol,it_solver,linsolv_maxiter,linsolv_tol,itslv_thresh)
end

"""
Implementation based on the presentation in 
Holtz, Sebastian, Thorsten Rohwedder, and Reinhold Schneider. "The alternating linear scheme for tensor optimization in the tensor train format." SIAM Journal on Scientific Computing 34.2 (2012): A683-A713.
"""

function init_H(x_tt::TTvector{T},A_tto::TToperator{T},N::Int,rmax) where {T<:Number}
	d = x_tt.N
	H = Array{Array{T,3},1}(undef, d+1-N)
	H[d+1-N] = ones(T,1,1,1)
	rks = r_and_d_to_rks(vcat(1,rmax*ones(Int,d-1),1),x_tt.dims;rmax=rmax)
	for i = d+1-N:-1:2
		H[i-1] = zeros(T,A_tto.rks[i+N-1],rks[i+N-1],rks[i+N-1])
		Hi_view = @view(H[i][:,1:x_tt.rks[i+N],1:x_tt.rks[i+N]])
		Him = @view(H[i-1][:,1:x_tt.rks[i+N-1],1:x_tt.rks[i+N-1]])
		x_vec = x_tt.cores[i+N-1]
		A_vec = A_tto.cores[i+N-1]
		update_H!(x_vec,A_vec,Hi_view,Him)
	end
	return H
end

function update_H!(x_vec::Array{T,3},A_vec::Array{T,4},Hi::AbstractArray{T,3},Him::AbstractArray{T,3}) where T<:Number
	@tensor Him[a,α,β] = ((conj.(x_vec)[j,α,αₖ]*Hi[z,αₖ,βₖ])*A_vec[j,k,a,z])*x_vec[k,β,βₖ] #size (rAim, rim, rim)
	nothing
end

function update_G!(x_vec::Array{T,3},A_vec::Array{T,4},Gi::AbstractArray{T,3},Gip::AbstractArray{T,3}) where T<:Number
	@tensoropt((ϕ,χ), Gip[a,α,β] = conj.(x_vec)[j,ϕ,α]*Gi[z,ϕ,χ]*x_vec[k,χ,β]*A_vec[j,k,z,a]) #size (rAi, ri, ri)
	nothing
end

# Zero-allocation version of update_H! using pre-allocated scratch buffers.
# Him[a,α,β] = Σ_{j,k,z,αₖ,βₖ} conj(x[j,α,αₖ]) * Hi[z,αₖ,βₖ] * A[j,k,a,z] * x[k,β,βₖ]
# x:(n,rxl,rxr), Hi:(rAr,rxr,rxr), A:(n,n,rAl,rAr), Him:(rAl,rxl,rxl)
function update_H!(x_vec::Array{T,3},A_vec::Array{T,4},Hi::AbstractArray{T,3},Him::AbstractArray{T,3},scratch::DMRGScratch{T}) where T<:Number
	n, rxl, rxr = size(x_vec)
	rAr = size(Hi, 1)
	rAl = size(A_vec, 3)

	xp  = @view scratch.xp[1:rxr,    1:n*rxl]
	HGp = @view scratch.HGp[1:rxr,   1:rAr*rxr]
	T1  = @view scratch.T1[1:n*rxl,  1:rAr*rxr]
	T1p = @view scratch.T1p[1:n*rAr, 1:rxl*rxr]
	Ap  = @view scratch.Ap[1:n*rAr,  1:n*rAl]
	T2  = @view scratch.T2[1:rxl*rxr,1:n*rAl]
	T2p = @view scratch.T2p[1:rAl*rxl,1:n*rxr]
	xp2 = @view scratch.xp2[1:n*rxr, 1:rxl]
	out = @view scratch.out[1:rAl*rxl,1:rxl]

	# Step 1: xp[αₖ, j+n*(α-1)] = x[j,α,αₖ]   HGp[αₖ, z+rAr*(βₖ-1)] = Hi[z,αₖ,βₖ]
	#         T1 = adjoint(xp) * HGp  →  T1_4d:(n,rxl,rAr,rxr)
	@inbounds for α in 1:rxl, j in 1:n, αₖ in 1:rxr
		xp[αₖ, j + n*(α-1)] = x_vec[j, α, αₖ]
	end
	@inbounds for βₖ in 1:rxr, z in 1:rAr, αₖ in 1:rxr
		HGp[αₖ, z + rAr*(βₖ-1)] = Hi[z, αₖ, βₖ]
	end
	mul!(T1, xp', HGp)

	# Step 2: T1p = T1_4d perm(1,3,2,4):(n,rAr,rxl,rxr)   Ap = A perm(1,4,2,3):(n,rAr,n,rAl)
	#         T2 = T1p' * Ap  →  T2_4d:(rxl,rxr,n,rAl)
	@inbounds for βₖ in 1:rxr, α in 1:rxl, z in 1:rAr, j in 1:n
		T1p[j + n*(z-1), α + rxl*(βₖ-1)] = T1[j + n*(α-1), z + rAr*(βₖ-1)]
	end
	@inbounds for a in 1:rAl, k in 1:n, z in 1:rAr, j in 1:n
		Ap[j + n*(z-1), k + n*(a-1)] = A_vec[j, k, a, z]
	end
	mul!(T2, T1p', Ap)

	# Step 3: T2p = T2_4d perm(4,1,3,2):(rAl,rxl,n,rxr)   xp2[k+n*(βₖ-1),β] = x[k,β,βₖ]
	#         out = T2p * xp2  →  out_3d:(rAl,rxl,rxl) = Him
	@inbounds for a in 1:rAl, α in 1:rxl, βₖ in 1:rxr, k in 1:n
		T2p[a + rAl*(α-1), k + n*(βₖ-1)] = T2[α + rxl*(βₖ-1), k + n*(a-1)]
	end
	@inbounds for β in 1:rxl, βₖ in 1:rxr, k in 1:n
		xp2[k + n*(βₖ-1), β] = x_vec[k, β, βₖ]
	end
	mul!(out, T2p, xp2)
	Him .= reshape(out, rAl, rxl, rxl)
	nothing
end

# Zero-allocation version of update_G! using pre-allocated scratch buffers.
# Gip[a,α,β] = Σ_{j,k,z,ϕ,χ} conj(x[j,ϕ,α]) * Gi[z,ϕ,χ] * A[j,k,z,a] * x[k,χ,β]
# x:(n,rxl,rxr), Gi:(rAl,rxl,rxl), A:(n,n,rAl,rAr), Gip:(rAr,rxr,rxr)
function update_G!(x_vec::Array{T,3},A_vec::Array{T,4},Gi::AbstractArray{T,3},Gip::AbstractArray{T,3},scratch::DMRGScratch{T}) where T<:Number
	n, rxl, rxr = size(x_vec)
	rAl = size(Gi, 1)
	rAr = size(A_vec, 4)

	xp  = @view scratch.xp[1:rxl,    1:n*rxr]
	HGp = @view scratch.HGp[1:rxl,   1:rAl*rxl]
	T1  = @view scratch.T1[1:n*rxr,  1:rAl*rxl]
	T1p = @view scratch.T1p[1:n*rAl, 1:rxr*rxl]
	Ap  = @view scratch.Ap[1:n*rAl,  1:n*rAr]
	T2  = @view scratch.T2[1:rxr*rxl,1:n*rAr]
	T2p = @view scratch.T2p[1:rAr*rxr,1:n*rxl]
	out = @view scratch.out[1:rAr*rxr,1:rxr]

	# Step 1: xp[ϕ, j+n*(α-1)] = x[j,ϕ,α]   HGp[ϕ, z+rAl*(χ-1)] = Gi[z,ϕ,χ]
	#         T1 = adjoint(xp) * HGp  →  T1_4d:(n,rxr,rAl,rxl)
	@inbounds for α in 1:rxr, j in 1:n, ϕ in 1:rxl
		xp[ϕ, j + n*(α-1)] = x_vec[j, ϕ, α]
	end
	@inbounds for χ in 1:rxl, z in 1:rAl, ϕ in 1:rxl
		HGp[ϕ, z + rAl*(χ-1)] = Gi[z, ϕ, χ]
	end
	mul!(T1, xp', HGp)

	# Step 2: T1p = T1_4d perm(1,3,2,4):(n,rAl,rxr,rxl)   Ap = A perm(1,3,2,4):(n,rAl,n,rAr)
	#         T2 = T1p' * Ap  →  T2_4d:(rxr,rxl,n,rAr)
	@inbounds for χ in 1:rxl, α in 1:rxr, z in 1:rAl, j in 1:n
		T1p[j + n*(z-1), α + rxr*(χ-1)] = T1[j + n*(α-1), z + rAl*(χ-1)]
	end
	@inbounds for a in 1:rAr, k in 1:n, z in 1:rAl, j in 1:n
		Ap[j + n*(z-1), k + n*(a-1)] = A_vec[j, k, z, a]
	end
	mul!(T2, T1p', Ap)

	# Step 3: T2p = T2_4d perm(4,1,3,2):(rAr,rxr,n,rxl)   xp2 = reshape(x, n*rxl, rxr) [free]
	#         out = T2p * xp2  →  out_3d:(rAr,rxr,rxr) = Gip
	@inbounds for a in 1:rAr, α in 1:rxr, χ in 1:rxl, k in 1:n
		T2p[a + rAr*(α-1), k + n*(χ-1)] = T2[α + rxr*(χ-1), k + n*(a-1)]
	end
	mul!(out, T2p, reshape(x_vec, n*rxl, rxr))
	Gip .= reshape(out, rAr, rxr, rxr)
	nothing
end

#returns the contracted tensor A_i[\\mu_i] ⋯ A_j[\\mu_j] ∈ R^{R^A_{i-1} × n_i × n_i × ⋯ × n_j × n_j ×  R^A_j}
function Amid(A_tto::TToperator{T},i::Int,j::Int) where {T<:Number}
	A = permutedims(A_tto.cores[i],(3,1,2,4))
	for k in i+1:j
		C = reshape(A,A_tto.rks[i],prod(A_tto.dims[i:k-1]),:,A_tto.rks[k])
		@tensor Atemp[αk,Ik,ik,Jk,jk,βk] := A_tto.cores[k][ik,jk,ξk,βk]*C[αk,Ik,Jk,ξk]
		A = reshape(Atemp,A_tto.rks[i],prod(A_tto.dims[i:k]),:,A_tto.rks[k+1])
	end
	return A #size R^A_{i-1} × (n_i⋯n_j) × (n_i⋯n_j) × R^A_j
end

#full assemble of matrix K
function K_full(Gi::AbstractArray{T,3},Hi::AbstractArray{T,3},Amid_tensor::AbstractArray{T,4}) where {T<:Number}
	K_dims = (size(Gi,2),size(Amid_tensor,2),size(Hi,2))
	K = zeros(T,K_dims...,K_dims...)
	@tensoropt((a,c,d,f), K[a,b,c,d,e,f] = Gi[y,a,d]*Hi[z,c,f]*Amid_tensor[y,b,e,z]) #size (r^X_{i-1},n_i⋯n_j,r^X_j)
	return Hermitian(reshape(K,prod(K_dims),prod(K_dims)))
end

# Builds the K matrix-free linear map for the iterative micro-step solver/eigensolver.
# K[a,b,c,d,e,f] = Gi[y,a,d]*Amid[y,b,e,z]*Hi[z,c,f] (symmetrized)
function _make_K_linmap(Gi_view::AbstractArray{T,3}, Hi_view::AbstractArray{T,3}, Amid_tensor::AbstractArray{T,4}, scratch::DMRGScratch{T}) where {T<:Number}
	rxl = size(Gi_view, 2)
	nN  = size(Amid_tensor, 2)
	rxr = size(Hi_view, 2)
	rAl = size(Gi_view, 1)
	rAr = size(Hi_view, 1)
	Gi = Base.iscontiguous(Gi_view) ? Gi_view : Array(Gi_view)
	Hi = Base.iscontiguous(Hi_view) ? Hi_view : Array(Hi_view)
	Gi_m    = reshape(Gi, rAl*rxl, rxl)
	Gi_m2   = reshape(permutedims(Gi, (1,3,2)), rAl*rxl, rxl)
	Amid_p  = reshape(permutedims(Amid_tensor, (1,3,2,4)), rAl*nN, nN*rAr)
	Amid_p2 = reshape(Amid_tensor, rAl*nN, nN*rAr)
	Hi_p    = reshape(permutedims(Hi, (3,1,2)), rxr*rAr, rxr)
	Hi_p2   = reshape(permutedims(Hi, (2,1,3)), rxr*rAr, rxr)
	T1_mat  = @view scratch.Km_T1[1:rAl*rxl,  1:nN*rxr]
	T1p_mat = @view scratch.Km_T1p[1:rxl*rxr, 1:rAl*nN]
	T2      = @view scratch.Km_T2[1:rxl*rxr,  1:nN*rAr]
	T2p_mat = @view scratch.Km_T2p[1:rxl*nN,  1:rxr*rAr]
	T3      = @view scratch.Km_T3[1:rxl*nN,   1:rxr]
	T1      = reshape(T1_mat, rAl, rxl, nN, rxr)
	T1_perm = reshape(T1p_mat, rxl, rxr, rAl, nN)
	T2_4d   = reshape(T2, rxl, rxr, nN, rAr)  # aliases T2; stays valid after mul!(T2,...)
	T2_perm = reshape(T2p_mat, rxl, nN, rxr, rAr)
	function K_matfree(Vout, V)
		V_mat = reshape(V, rxl, nN*rxr)
		# Term 1: Gi[y,a,d]*V[d,e,f]*Amid[y,b,e,z]*Hi[z,c,f]
		mul!(T1_mat, Gi_m, V_mat)
		@inbounds for e in 1:nN, y in 1:rAl, f in 1:rxr, a in 1:rxl
			T1_perm[a,f,y,e] = T1[y,a,e,f]
		end
		mul!(T2, T1p_mat, Amid_p)
		@inbounds for z in 1:rAr, b in 1:nN, f in 1:rxr, a in 1:rxl
			T2_perm[a,b,f,z] = T2_4d[a,f,b,z]
		end
		mul!(T3, T2p_mat, Hi_p)
		copyto!(Vout, T3)
		# Term 2: Gi[y,d,a]*V[d,e,f]*Amid[y,e,b,z]*Hi[z,f,c]
		mul!(T1_mat, Gi_m2, V_mat)
		@inbounds for e in 1:nN, y in 1:rAl, f in 1:rxr, a in 1:rxl
			T1_perm[a,f,y,e] = T1[y,a,e,f]
		end
		mul!(T2, T1p_mat, Amid_p2)
		@inbounds for z in 1:rAr, b in 1:nN, f in 1:rxr, a in 1:rxl
			T2_perm[a,b,f,z] = T2_4d[a,f,b,z]
		end
		mul!(T3, T2p_mat, Hi_p2)
		@inbounds for i in eachindex(Vout)
			Vout[i] = 0.5*(Vout[i] + T3[i])
		end
		return nothing
	end
	return LinearMap{T}(K_matfree, rxl*nN*rxr; issymmetric=true, ismutating=true)
end

function init_Hb(x_tt::TTvector{T},b_tt::TTvector{T},N::Integer,rmax) where {T<:Number}
	d = x_tt.N
	H_b = Array{Array{T,2},1}(undef, d+1-N) 
	H_b[d+1-N] = ones(T,1,1)
	rks = r_and_d_to_rks(vcat(1,rmax*ones(Int,d-1),1),x_tt.dims;rmax=rmax)
	for i = d+1-N:-1:2
		H_b[i-1] = zeros(T,rks[i+N-1],b_tt.rks[i+N-1])
		b_vec = b_tt.cores[i+N-1]
		x_vec = x_tt.cores[i+N-1]
		Hbi = @view(H_b[i][1:x_tt.rks[i+N],:])
		update_Hb!(x_vec,b_vec,Hbi,H_b[i-1]) #size(r^X_{i-1},r^b_{i-1}) 
	end
	return H_b
end

function update_Hb!(x_vec::Array{T,3},b_vec::Array{T,3},H_bi::AbstractArray{T,2},H_bim::AbstractArray{T,2}) where T<:Number
	H_bimview = @view(H_bim[1:size(x_vec,2),1:size(b_vec,2)])
	@tensor H_bimview[α,β] = (H_bi[ϕ,χ]*b_vec[i,β,χ])*conj.(x_vec)[i,α,ϕ]
	nothing
end

function update_Gb!(x_vec::Array{T,3},b_vec::Array{T,3},G_bi::AbstractArray{T,2},G_bip::AbstractArray{T,2}) where T<:Number
	@tensor G_bip[α,β] = (G_bi[ϕ,χ]*b_vec[i,χ,β])*conj.(x_vec)[i,ϕ,α]
	nothing
end

function b_mid(b_tt::TTvector{T},i::Integer,j::Integer) where {T<:Number}
	b_out = permutedims(b_tt.cores[i],(2,1,3))
	for k in i+1:j
		@tensor btemp[αk,ik,jk,βk] := b_out[αk,ik,ξk]*b_tt.cores[k][jk,ξk,βk]
		b_out = reshape(btemp,b_tt.rks[i],:,b_tt.rks[k+1]) #size r^b_{i-1} × (n_i⋯n_k) × r^b_k
	end
	return b_out
end

function Ksolve!(Gi_view::AbstractArray{T,3},G_bi::AbstractArray{T,2},Hi_view::AbstractArray{T,3},H_bi::AbstractArray{T,2},Amid_tensor::AbstractArray{T,4},Bmid::AbstractArray{T,3},Pb::AbstractArray{T,3},V0::AbstractArray{T,3},Vapp::AbstractArray{T,3},scratch::DMRGScratch{T};it_solver=false,maxiter=200,tol=1e-6,itslv_thresh=256) where T<:Number
	K_dims = (size(Gi_view,2),size(Amid_tensor,2),size(Hi_view,2))
	@tensor (Pb[α1,i,α2] = (G_bi[α1,β1]*Bmid[β1,i,β2])*H_bi[α2,β2]) #size (r^X_{i-1},n_i⋯n_j,r^X_j)
	if it_solver && prod(K_dims) > itslv_thresh
		Kmap = _make_K_linmap(Gi_view, Hi_view, Amid_tensor, scratch)
		Vapp[:], top = linsolve(Kmap, vec(Pb), vec(V0); isposdef=true, tol=tol, maxiter=maxiter, verbosity=0)
		top.converged > 0 && return nothing
	end
	Vapp[:] = K_full(Gi_view,Hi_view,Amid_tensor)\Pb[:]
	return nothing
end

function right_core_move!(x_tt::TTvector{T},V,V_move,i::Int,N,tol::Float64,r_max::Integer;verbose,dmrg_info) where {T<:Number}
	# Perform the truncated svd
	u_V, s_V, v_V = svd(reshape(V,x_tt.rks[i]*x_tt.dims[i],:))
	# Update the ranks to the truncated one
	x_tt.rks[i+1] = max(min(cut_off_index(s_V,tol),r_max),x_tt.rks[i+1])
	println("Rank: $(x_tt.rks[i+1]),	Max rank=$r_max")
	svd_truncation = (norm(s_V)-norm(s_V[1:x_tt.rks[i+1]]))/norm(s_V)
	println("Discarded weight: $(svd_truncation)")
	if verbose
		push!(dmrg_info.ttsvd_weights, svd_truncation)
	end

	x_tt.cores[i] = permutedims(reshape(u_V[:,1:x_tt.rks[i+1]],x_tt.rks[i],x_tt.dims[i],:),(2,1,3))
	x_tt.ot[i] = 1
	x_tt.ot[i+1]=0
	V_moveview = @view(V_move[1:x_tt.rks[i+1],1:prod(x_tt.dims[i+1:i+N-1]),1:size(V,3)])
	@tensor V_moveview[αk,ik,βk] = reshape(v_V'[1:x_tt.rks[i+1],:],x_tt.rks[i+1],:,size(V,3))[αk,ik,βk]
	for ak in axes(V_moveview,1)
		V_moveview[ak,:,:] = V_moveview[ak,:,:]*(s_V[ak])
	end
	return nothing
end

function left_core_move!(x_tt::TTvector{T},V,V_move,j::Int,N,tol::Float64,r_max::Integer;verbose,dmrg_info) where {T<:Number}
	# Perform the truncated svd
	u_V, s_V, v_V = svd(reshape(V,:, x_tt.dims[j]*x_tt.rks[j+1]))
	# Update the ranks to the truncated one
	x_tt.rks[j] = max(min(cut_off_index(s_V,tol),r_max),x_tt.rks[j])
	println("Rank: $(x_tt.rks[j]),	Max rank=$r_max")
	svd_truncation = (norm(s_V)-norm(s_V[1:x_tt.rks[j]]))/norm(s_V)
	println("Discarded weight: $(svd_truncation)")
	if verbose
		push!(dmrg_info.ttsvd_weights, svd_truncation)
	end

	x_tt.cores[j] = permutedims(reshape(v_V'[1:x_tt.rks[j],:],x_tt.rks[j],:,x_tt.rks[j+1]),(2,1,3))
	x_tt.ot[j] = -1
	x_tt.ot[j-1]=0
	V_moveview = @view(V_move[1:size(V,1),1:prod(x_tt.dims[j+1-N:j-1]),1:x_tt.rks[j]])
	@tensor V_moveview[αk,ik,βk] = reshape(u_V[:,1:x_tt.rks[j]],size(V,1),:,x_tt.rks[j])[αk,ik,βk]
	for bk in axes(V_moveview,3)
		V_moveview[:,:,bk] = V_moveview[:,:,bk]*s_V[bk]
	end
	return nothing
end


function K_eigmin(Gi_view::AbstractArray{T,3},Hi_view::AbstractArray{T,3},V0::AbstractArray{T,3},Amid_tensor::AbstractArray{T,4},V,scratch::DMRGScratch{T};it_solver=false::Bool,itslv_thresh=256::Int64,maxiter=200::Int64,tol=1e-6::Float64) where T<:Number
	K_dims = size(V0)
	if it_solver && prod(K_dims) > itslv_thresh
		Kmap = _make_K_linmap(Gi_view, Hi_view, Amid_tensor, scratch)
		r = eigsolve(Kmap, copy(vec(V0)), 1, :SR; issymmetric=true, tol=tol, maxiter=maxiter)
		if r[3].converged >= 1
			V[:] = r[2][1][:]
			return real(r[1][1])
		end
	end
	K = K_full(Gi_view,Hi_view,Amid_tensor)
	F = eigen(K,1:1)
	V[:] = reshape(F.vectors[:,1],K_dims)
	return F.values[1]
end

function init_dmrg(A::TToperator{T,d},tt_opt::TTvector{T,d},rks,N::Integer) where {T<:Number,d}
	rmax = maximum(rks)
	G = Array{Array{T,3},1}(undef, d+1-N)
	Amid_list = Array{Array{T,4},1}(undef, d+1-N)
	for i in 1:d+1-N
		G[i] = zeros(T,A.rks[i],rks[i],rks[i])
		Amid_list[i] = Amid(A,i,i+N-1)
	end
	G[1] = ones(T,size(G[1]))
	H = init_H(tt_opt,A,N,rmax)

	V0 = zeros(T,rmax,maximum(tt_opt.dims)^N,rmax)
	V0_view = @view(V0[1:tt_opt.rks[1],1:prod(tt_opt.dims[1:N]),1:tt_opt.rks[1+N]])
	V0_view = b_mid(tt_opt,1,N)
	V = zeros(T,rmax,maximum(tt_opt.dims)^N,rmax)
	V_move = zeros(T,rmax,maximum(tt_opt.dims),rmax)
	V_temp = zeros(T,rmax,maximum(tt_opt.dims),maximum(tt_opt.dims),rmax)
	n_max = maximum(tt_opt.dims)
	rA_max = maximum(A.rks)
	scratch = DMRGScratch(T, n_max, rmax, rA_max, N)
	return G,Amid_list,H,V0,V,V_move,V_temp,V0_view,scratch
end

function init_dmrg_b(b::TTvector{T,d},tt_opt::TTvector{T,d},rks,N) where {T<:Number,d}
	rmax = maximum(rks)
	G_b = zeros.(T,rks[1:d+1-N],b.rks[1:d+1-N]) #Array{Array{T,2},1}(undef, d+1-N)
	bmid_list = Array{Array{T,3},1}(undef, d+1-N)
	for i in 1:d+1-N
		bmid_list[i] = b_mid(b,i,i+N-1)
	end
	G_b[1] = ones(T,size(G_b[1]))
	Pb_temp = zeros(T,rmax,maximum(tt_opt.dims)^N,rmax)
	H_b = init_Hb(tt_opt,b,N,rmax)
	return G_b,bmid_list,H_b,Pb_temp
end

function update_G_H_V(Gi,Hi,V,tt_dims,tt_rks,i,N)
	Gi_view = @view(Gi[:,1:tt_rks[i],1:tt_rks[i]])
	Hi_view = @view(Hi[:,1:tt_rks[i+N],1:tt_rks[i+N]])
	V_view = @view(V[1:tt_rks[i],1:prod(tt_dims[i:i+N-1]),1:tt_rks[i+N]])
	return Gi_view,Hi_view,V_view
end

function update_G_H_V_b(Gbi,Hbi,Pb_temp,tt_dims,tt_rks,i,N)
	G_bi_view = @view(Gbi[1:tt_rks[i],:])
	H_bi_view = @view(Hbi[1:tt_rks[i+N],:])
	Pb_view = @view(Pb_temp[1:tt_rks[i],1:prod(tt_dims[i:i+N-1]),1:tt_rks[i+N]])
	return G_bi_view,H_bi_view,Pb_view
end

function update_right(tt_opt,V0,V_view,V_move,V_temp,i,N,tol,rmax,Ai,Gi_view,Gip,scratch;verbose,dmrg_info,move=true)
	if move
		#update tt_opt
		right_core_move!(tt_opt,V_view,V_move,i,N,tol,rmax;verbose,dmrg_info)
	end

	V_moveview = @view(V_move[1:tt_opt.rks[i+1],1:prod(tt_opt.dims[i+1:i+N-1]),1:tt_opt.rks[i+N]])
	V_tempview = @view(V_temp[1:size(V_moveview,1),1:size(V_moveview,2),1:tt_opt.dims[i+N],1:tt_opt.rks[i+1+N]])
	@tensor V_tempview[αk,J,ik,γk] = V_moveview[αk,J,βk]*tt_opt.cores[i+N][ik,βk,γk]
	V0_view = @view(V0[1:tt_opt.rks[i+1],1:prod(tt_opt.dims[i+1:i+N-1]),1:tt_opt.rks[i+N]])
	V0_view = reshape(V_tempview,size(V_tempview,1),:,size(V_tempview,4))

	#update G[i+1]
	Gip_view = @view(Gip[:,1:tt_opt.rks[i+1],1:tt_opt.rks[i+1]])
	update_G!(tt_opt.cores[i],Ai,Gi_view,Gip_view,scratch)
	return V0_view
end

function update_left(tt_opt,V0,V_view,V_move,V_temp,i,N,tol,rmax,Aip,Hi_view,Him,scratch;verbose,dmrg_info,move=true)
	if move
		left_core_move!(tt_opt,V_view,V_move,i+N-1,N,tol,rmax;verbose,dmrg_info)
	end

	#update the initialization
	V_moveview = @view(V_move[1:tt_opt.rks[i],1:prod(tt_opt.dims[i:i+N-2]),1:tt_opt.rks[i+N-1]])
	V_tempview = @view(V_temp[1:tt_opt.rks[i-1],1:size(V_moveview,2),1:tt_opt.dims[i-1],1:size(V_moveview,3)])
	@tensor V_tempview[αk,J,ik,γk] =  V_moveview[βk,J,γk]*tt_opt.cores[i-1][ik,αk,βk]
	V0_view = @view(V0[1:tt_opt.rks[i],1:prod(tt_opt.dims[i:i+N-2]),1:tt_opt.rks[i+N-1]])
	V0_view = reshape(V_tempview,size(V_tempview,1),:,size(V_tempview,4))

	#update H[i-1]
	Him_view = @view(Him[:,1:tt_opt.rks[i+N-1],1:tt_opt.rks[i+N-1]])
	update_H!(tt_opt.cores[i+N-1],Aip,Hi_view,Him_view,scratch)
	return V0_view
end

function extract_schedule_parameters(schedule)
    N = schedule.N
    tol = schedule.tol
    sweep_schedule = schedule.sweep_schedule
    rmax_schedule = schedule.rmax_schedule
    it_solver = schedule.it_solver
    linsolv_maxiter = schedule.linsolv_maxiter
    linsolv_tol = schedule.linsolv_tol
    itslv_thresh = schedule.itslv_thresh

    # Return the parameters 
    return N, tol, sweep_schedule, rmax_schedule, it_solver, linsolv_maxiter, linsolv_tol, itslv_thresh
end

function ttv_after_dmrg_microstep!(tt_opt,rmax,V,schedule;j=tt_opt.N,verbose,dmrg_info,left_to_right=true)
	N = schedule.N
	tol = schedule.tol
	d = tt_opt.N
	rks = tt_opt.rks 
	dims = tt_opt.dims
	if N==2
		V_reshape = reshape(V,rks[j-1]*dims[j-1],rks[j+1]*dims[j])
		u_V, s_V, v_V = svd(V_reshape)
		tt_opt.rks[j] = min(cut_off_index(s_V,tol),rmax)
		println("Rank: $(tt_opt.rks[j]),	Max rank=$rmax")
		svd_truncation = (norm(s_V)-norm(s_V[1:tt_opt.rks[j]]))/norm(s_V)
		println("Discarded weight: $(svd_truncation)")
		if verbose
			push!(dmrg_info.ttsvd_weights, svd_truncation)
		end
		if left_to_right
			tt_opt.cores[j-1] = permutedims(reshape(u_V[:,1:tt_opt.rks[j]]*Diagonal(s_V[1:tt_opt.rks[j]]), tt_opt.rks[j-1], dims[j],tt_opt.rks[j]),(2,1,3))
			tt_opt.ot[j-1] = 0
			tt_opt.cores[j] = permutedims(reshape(v_V'[1:tt_opt.rks[j],:],tt_opt.rks[j],dims[j],tt_opt.rks[j+1]),(2,1,3))
			tt_opt.ot[j] = -1
		else
			tt_opt.cores[j-1] = permutedims(reshape(u_V[:,1:tt_opt.rks[j]], tt_opt.rks[j-1], dims[j],tt_opt.rks[j]),(2,1,3))
			tt_opt.ot[j-1] = 1
			tt_opt.cores[j] = permutedims(reshape(Diagonal(s_V[1:tt_opt.rks[j]])*v_V'[1:tt_opt.rks[j],:],tt_opt.rks[j],dims[j],tt_opt.rks[j+1]),(2,1,3))
			tt_opt.ot[j] = 0
		end
	else #N=1
		if left_to_right
			l,q = lq(reshape(V,tt_opt.rks[j],:))
			tt_opt.cores[j-1] = reshape(reshape(tt_opt.cores[j-1],tt_opt.dims[j-1]*tt_opt.rks[j-1],:)*l, tt_opt.dims[j-1],tt_opt.rks[j-1],:)
			tt_opt.ot[j-1] = 0
			tt_opt.cores[j] = permutedims(reshape(Matrix(q),tt_opt.rks[j],tt_opt.dims[j],tt_opt.rks[j+1]),(2,1,3))
			tt_opt.ot[j] = -1
		else 
			q,r = qr(reshape(V,tt_opt.rks[j]*tt_opt.dims[j],:))
			@tensor vec_temp[i,α,β] := tt_opt.cores[j+1][i,ξ,β]*r[α,ξ]
			tt_opt.cores[j+1] = vec_temp
			tt_opt.ot[j+1] = 0
			tt_opt.cores[j] = permutedims(reshape(Matrix(q),tt_opt.rks[j],tt_opt.dims[j],tt_opt.rks[j+1]),(2,1,3))
			tt_opt.ot[j] = 1
		end
	end
	return nothing 
end

# Forward turnaround: SVD the super-core, update H[d-N], return new V0_view.
# Called when the forward sweep reaches the last site (i == d-N+1).
# linsolv additionally calls update_Hb! on the same core afterward.
function _dmrg_turnaround_fwd!(tt_opt::TTvector{T}, A::TToperator{T}, H, V_view, Hi_view, schedule, rmax, scratch::DMRGScratch{T}; verbose, dmrg_info) where {T<:Number}
	d, N = tt_opt.N, schedule.N
	ttv_after_dmrg_microstep!(tt_opt, rmax, V_view, schedule; verbose, dmrg_info)
	verbose && push!(dmrg_info.TTvs, copy(tt_opt))
	Him_view = @view(H[d-N][:,1:tt_opt.rks[d],1:tt_opt.rks[d]])
	update_H!(tt_opt.cores[d], A.cores[d], Hi_view, Him_view, scratch)
	return b_mid(tt_opt, d-N, d-1)
end

# Backward turnaround: SVD the super-core, update G[2], return new V0_view.
# Called when the backward sweep reaches the first site (i == 1).
# linsolv additionally calls update_Gb! on the same core afterward.
function _dmrg_turnaround_bwd!(tt_opt::TTvector{T}, A::TToperator{T}, G, V_view, schedule, rmax, scratch::DMRGScratch{T}; verbose, dmrg_info) where {T<:Number}
	N = schedule.N
	ttv_after_dmrg_microstep!(tt_opt, rmax, V_view, schedule; verbose, dmrg_info, left_to_right=false, j=N)
	verbose && push!(dmrg_info.TTvs, copy(tt_opt))
	Gi_view_1 = @view(G[1][1:1,1:1,1:1])
	Gip_view  = @view(G[2][:,1:tt_opt.rks[2],1:tt_opt.rks[2]])
	update_G!(tt_opt.cores[1], A.cores[1], Gi_view_1, Gip_view, scratch)
	return b_mid(tt_opt, 2, N+1)
end

"""
Solve Ax=b using the ALS algorithm where A is given as `TToperator` and `b`, `tt_start` are `TTvector`.
The ranks of the solution is the same as `tt_start`.
`sweep_count` is the number of total sweeps in the ALS.
"""
function dmrg_linsolv(A :: TToperator{T}, b :: TTvector{T}, tt_start :: TTvector{T};schedule = dmrg_schedule_default(),verbose=true) where {T<:Number}
	# als finds the minimum of the operator J:1/2*<Ax,Ax> - <x,b>
	# input:
	# 	A: the tensor operator in its tensor train format
	#   b: the tensor in its tensor train format
	#	tt_start: start value in its tensor train format
	#	tt_opt: stationary point of J up to tolerated rank opt_rks
	# 			in its tensor train format
	
	# Unpack schedule parameters
	N, tol, sweep_schedule, rmax_schedule, it_solver, linsolv_maxiter, linsolv_tol, itslv_thresh = extract_schedule_parameters(schedule)

	d = b.N
	# Verbose
	dmrg_info = DMRGverbose(schedule,TTvector{T,d}[],Int64[], Float64[])

	# Initialize the to be returned tensor in its tensor train format
	rmax = maximum(rmax_schedule)
	if N==1
		tt_start = tt_up_rks(tt_start,rmax)
	end
	tt_opt = orthogonalize(tt_start)
	dims = tt_start.dims
	rmax = maximum(rmax_schedule)
	rks = r_and_d_to_rks(vcat(1,rmax*ones(Int,d-1),1),dims;rmax=rmax)

	#Initialize DMRG 
	G,Amid_list,H,V0,V,V_move,V_temp,V0_view,scratch = init_dmrg(A,tt_opt,rks,N)
	G_b,bmid_list,H_b,Pb_temp = init_dmrg_b(b,tt_opt,rks,N)

	nsweeps = 0 #sweeps counter
	i_schedule = 1

	#1st step of the sweep
	Gi_view,Hi_view,V_view = update_G_H_V(G[1],H[1],V,tt_opt.dims,tt_opt.rks,1,N)
	G_bi_view, H_bi_view, Pb_view = update_G_H_V_b(G_b[1],H_b[1],Pb_temp,tt_opt.dims,tt_opt.rks,1,N)
	Ksolve!(Gi_view,G_bi_view,Hi_view,H_bi_view,Amid_list[1],bmid_list[1],Pb_view,V0_view, V_view,scratch;it_solver=it_solver,maxiter=linsolv_maxiter,tol=linsolv_tol,itslv_thresh=itslv_thresh)
	V0_view = update_right(tt_opt,V0,V_view,V_move,V_temp,1,N,tol,rmax_schedule[i_schedule],A.cores[1],Gi_view,G[2],scratch;verbose,dmrg_info)

	G_bip = @view(G_b[2][1:tt_opt.rks[2],:])
	update_Gb!(tt_opt.cores[1],b.cores[1],G_bi_view,G_bip)

	while i_schedule <= length(sweep_schedule) 
		nsweeps+=1
		println("---------------------------")
		if nsweeps == sweep_schedule[i_schedule]
			i_schedule+=1
			if i_schedule > length(sweep_schedule)
				return tt_opt, dmrg_info
			end
		end
		println("Macro-iteration $nsweeps; bond dimension $(rmax_schedule[i_schedule])")
		# First half sweep
		for i = 2:d-N+1
			println("Forward sweep: core optimization $i out of $(d+1-N)")
			Gi_view,Hi_view,V_view = update_G_H_V(G[i],H[i],V,tt_opt.dims,tt_opt.rks,i,N)
			G_bi_view, H_bi_view, Pb_view = update_G_H_V_b(G_b[i],H_b[i],Pb_temp,tt_opt.dims,tt_opt.rks,i,N)
			# Define V as solution of K*x=Pb in x
			Ksolve!(Gi_view,G_bi_view,Hi_view,H_bi_view,Amid_list[i],bmid_list[i],Pb_view,V0_view, V_view,scratch;it_solver=it_solver,maxiter=linsolv_maxiter,tol=linsolv_tol,itslv_thresh=itslv_thresh)

			if i<d-N+1
				#Update TT core i and the next initialization
				V0_view = update_right(tt_opt,V0,V_view,V_move,V_temp,i,N,tol,rmax_schedule[i_schedule],A.cores[i],Gi_view,G[i+1],scratch;verbose,dmrg_info)

				G_bip = @view(G_b[i+1][1:tt_opt.rks[i+1],:])
				update_Gb!(tt_opt.cores[i],b.cores[i],G_bi_view,G_bip)
			else #i==d-N+1
				V0_view = _dmrg_turnaround_fwd!(tt_opt, A, H, V_view, Hi_view, schedule, rmax_schedule[i_schedule], scratch; verbose, dmrg_info)
				update_Hb!(tt_opt.cores[d], b.cores[d], H_bi_view, H_b[d-N])
			end
		println("---------------------------")
		end

		# Second half sweep
		for i = (d-N):(-1):2
			println("Backward sweep: core optimization $i out of $(d+1-N)")
			Gi_view,Hi_view,V_view = update_G_H_V(G[i],H[i],V,tt_opt.dims,tt_opt.rks,i,N)
			G_bi_view, H_bi_view, Pb_view = update_G_H_V_b(G_b[i],H_b[i],Pb_temp,tt_opt.dims,tt_opt.rks,i,N)
			Ksolve!(Gi_view,G_bi_view,Hi_view,H_bi_view,Amid_list[i],bmid_list[i],Pb_view,V0_view,V_view,scratch;it_solver=it_solver,maxiter=linsolv_maxiter,tol=linsolv_tol,itslv_thresh=itslv_thresh)
			V0_view = update_left(tt_opt,V0,V_view,V_move,V_temp,i,N,tol,rmax_schedule[i_schedule],A.cores[i+N-1],Hi_view,H[i-1],scratch;verbose,dmrg_info)
			update_Hb!(tt_opt.cores[i+N-1],b.cores[i+N-1],H_bi_view,H_b[i-1])
		println("---------------------------")
		end
		# Last backward step: turnaround at i=1
		println("Backward sweep: core optimization 1 out of $(d+1-N)")
		Gi_view,Hi_view,V_view = update_G_H_V(G[1],H[1],V,tt_opt.dims,tt_opt.rks,1,N)
		G_bi_view, H_bi_view, Pb_view = update_G_H_V_b(G_b[1],H_b[1],Pb_temp,tt_opt.dims,tt_opt.rks,1,N)
		Ksolve!(Gi_view,G_bi_view,Hi_view,H_bi_view,Amid_list[1],bmid_list[1],Pb_view,V0_view,V_view,scratch;it_solver=it_solver,maxiter=linsolv_maxiter,tol=linsolv_tol,itslv_thresh=itslv_thresh)
		V0_view = _dmrg_turnaround_bwd!(tt_opt, A, G, V_view, schedule, rmax_schedule[i_schedule], scratch; verbose, dmrg_info)
		G_bip = @view(G_b[2][1:tt_opt.rks[2],:])
		update_Gb!(tt_opt.cores[1], b.cores[1], G_bi_view, G_bip)
		println("---------------------------")
	end
	return tt_opt, dmrg_info
end

"""
Returns the lowest eigenvalue of A by minimizing the Rayleigh quotient in the ALS algorithm.

The ranks can be increased in the course of the ALS: if `sweep_schedule[k] ≤ i <sweep_schedule[k+1]` is the current number of sweeps then the ranks is given by `rmax_schedule[k]`.
"""
function dmrg_eigsolv(A :: TToperator{T},
	tt_start :: TTvector{T} ;schedule = dmrg_schedule_default(),verbose=true)  where {T<:Number} 

	# Unpack schedule parameters
	N, tol, sweep_schedule, rmax_schedule, it_solver, linsolv_maxiter, linsolv_tol, itslv_thresh = extract_schedule_parameters(schedule)

	d = tt_start.N
	# Verbose
	dmrg_info = DMRGverbose(schedule,TTvector{T,d}[],Int64[], Float64[])
	# Initialize the to be returned tensor in its tensor train format
	tt_opt = orthogonalize(tt_start)
	dims = tt_start.dims
	rmax = maximum(rmax_schedule)
	rks = r_and_d_to_rks(vcat(1,rmax*ones(Int,d-1),1),dims;rmax=rmax)
	# Initialize the output objects
	E = Float64[]
	r_hist = Int64[]

	#Initialize DMRG 
	G,Amid_list,H,V0,V,V_move,V_temp,V0_view,scratch = init_dmrg(A,tt_opt,rks,N)

	nsweeps = 0 #sweeps counter
	i_schedule = 1

	#1st step of the sweep
	println("Macro-iteration 1; bond dimension $(rmax_schedule[1])")
	println("Forward sweep: core optimization 1 out of $(d-N+1)")
	Gi_view,Hi_view,V_view = update_G_H_V(G[1],H[1],V,tt_opt.dims,tt_opt.rks,1,N)
	λ = K_eigmin(G[1],H[1],V0_view ,Amid_list[1], V_view,scratch ;it_solver,maxiter=linsolv_maxiter,tol=linsolv_tol,itslv_thresh)
	println("Eigenvalue: $λ")
	push!(E,λ)
	push!(r_hist,maximum(tt_opt.rks))
	V0_view = update_right(tt_opt,V0,V_view,V_move,V_temp,1,N,tol,rmax_schedule[i_schedule],A.cores[1],Gi_view,G[2],scratch;verbose,dmrg_info)
	println("--------------------------------------------")

	while i_schedule <= length(sweep_schedule) 
		nsweeps+=1

		if nsweeps == sweep_schedule[i_schedule]
			i_schedule+=1
			if i_schedule > length(sweep_schedule)
				return E::Array{Float64,1}, tt_opt::TTvector{T}, dmrg_info
			end
		end
		println("Macro-iteration $nsweeps; bond dimension $(rmax_schedule[i_schedule])")
		# First half sweep
		for i = 2:(d-N+1)
			println("Forward sweep: core optimization $i out of $(d-N+1)")
			# Define V as solution of K V= λ V for smallest λ
			Gi_view,Hi_view,V_view = update_G_H_V(G[i],H[i],V,tt_opt.dims,tt_opt.rks,i,N)
			λ = K_eigmin(Gi_view,Hi_view,V0_view, Amid_list[i],V_view,scratch; it_solver, maxiter=linsolv_maxiter,tol=linsolv_tol,itslv_thresh)
			println("Eigenvalue: $λ")
			push!(E,λ)
			#Update TT core i and the next initialization
			if i<d-N+1
				V0_view = update_right(tt_opt,V0,V_view,V_move,V_temp,i,N,tol,rmax_schedule[i_schedule],A.cores[i],Gi_view,G[i+1],scratch;verbose,dmrg_info)
				push!(r_hist,maximum(tt_opt.rks))
			else #i==d-N+1
				V0_view = _dmrg_turnaround_fwd!(tt_opt, A, H, V_view, Hi_view, schedule, rmax_schedule[i_schedule], scratch; verbose, dmrg_info)
			end
			println("--------------------------------------------")
		end

		# Second half sweep
		for i = (d-N):(-1):2
			println("Backward sweep: core optimization $(i) out of $(d-N)")
			# Define V as solution of K*x=P2b in x
			Gi_view,Hi_view,V_view = update_G_H_V(G[i],H[i],V,tt_opt.dims,tt_opt.rks,i,N)
			λ = K_eigmin(Gi_view,Hi_view,V0_view,Amid_list[i],V_view,scratch ;it_solver=it_solver,maxiter=linsolv_maxiter,tol=linsolv_tol,itslv_thresh=itslv_thresh)
			println("Eigenvalue: $λ")
			push!(E,λ)
			#update the initialization
			V0_view = update_left(tt_opt,V0,V_view,V_move,V_temp,i,N,tol,rmax_schedule[i_schedule],A.cores[i+N-1],Hi_view,H[i-1],scratch;verbose,dmrg_info)
			println("--------------------------------------------")
			push!(r_hist,maximum(tt_opt.rks))
		end
		#last step to complete the sweep
		println("Backward sweep: core optimization $(1) out of $(d-N)")
		Gi_view,Hi_view,V_view = update_G_H_V(G[1],H[1],V,tt_opt.dims,tt_opt.rks,1,N)
		λ = K_eigmin(Gi_view,Hi_view,V0_view,Amid_list[1],V_view,scratch;it_solver,maxiter=linsolv_maxiter,tol=linsolv_tol,itslv_thresh)
		println("Eigenvalue: $λ")
		push!(E,λ)
		push!(r_hist,maximum(tt_opt.rks))
		V0_view = _dmrg_turnaround_bwd!(tt_opt, A, G, V_view, schedule, rmax_schedule[i_schedule], scratch; verbose, dmrg_info)
		println("--------------------------------------------")
	end
	return E::Array{Float64,1}, tt_opt::TTvector{T},dmrg_info 
end

"""
returns the smallest eigenpair Ax = Sx
NOT WORKING
"""
function dmrg_gen_eigsolv(A :: TToperator{T}, S::TToperator{T}, tt_start :: TTvector{T} ; sweep_schedule=[2],rmax_schedule=[maximum(tt_start.rks)],tol=1e-10,it_solver=false,itslv_thresh=2500) where {T<:Number}
	d = tt_start.N
	# Initialize the to be returned tensor in its tensor train format
	tt_opt = orthogonalize(tt_start)
	dims = tt_start.dims
	E = zeros(Float64,d*sweep_schedule[end]) #output eigenvalue
	# Define the array of ranks of tt_opt [r_0=1,r_1,...,r_d]
	rks = tt_start.rks

	# Initialize the arrays of G and K
	G = Array{Array{T}}(undef, d)
	K = Array{Array{T}}(undef, d) 

	# Initialize G[1]
	for i in 1:d
		G[i] = zeros(dims[i],rks[i],dims[i],rks[i],A.rks[i+1])
		K[i] = zeros(dims[i],rks[i],dims[i],rks[i],S.rks[i+1])
	end
	G[1] = reshape(A.cores[1][:,:,1,:], dims[1],1,dims[1], 1, :)
	K[1] = reshape(S.cores[1][:,:,1,:], dims[1],1,dims[1], 1, :)

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
					Htemp = zeros(tt_opt.rks[i],tt_opt.rks[i],A.rks[i])
					Ltemp = zeros(tt_opt.rks[i],tt_opt.rks[i],S.rks[i])
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
			if tt_opt.ot[i] == 0
				# Define V as solution of K*x=Pb in x
				i_μit += 1
				E[i_μit],V = K_eiggenmin(G[i],H[i],K[i],L[i],tt_opt.cores[i];it_solver=it_solver,itslv_thresh=itslv_thresh)
				println("Eigenvalue: $(E[i_μit])")
				tt_opt = right_core_move(tt_opt,V,i,N,rks;verbose,dmrg_info)
			end

			#update G and K
			update_G!(tt_opt.cores[i],A.cores[i+1],G[i],G[i+1])
			update_G!(tt_opt.cores[i],S.cores[i+1],K[i],K[i+1])
		end

		# Second half sweep
		for i = d:(-1):2
			println("Backward sweep: core optimization $i out of $d")
			# Define V as solution of K*x=Pb in x
			i_μit += 1
			E[i_μit],V = K_eiggenmin(G[i],H[i],K[i],L[i],tt_opt.cores[i];it_solver=it_solver,itslv_thresh=itslv_thresh)
			println("Eigenvalue: $(E[i_μit])")
			tt_opt = left_core_move(tt_opt,V,i,rks)
			update_H!(tt_opt.cores[i],A.cores[i],H[i],H[i-1])
			update_H!(tt_opt.cores[i],S.cores[i],L[i],L[i-1])
		end
	end
end
