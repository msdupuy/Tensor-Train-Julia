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
	rks = r_and_d_to_rks(vcat(1,rmax*ones(Int,d-1),1),x_tt.ttv_dims;rmax=rmax)
	for i = d+1-N:-1:2
		H[i-1] = zeros(T,A_tto.tto_rks[i+N-1],rks[i+N-1],rks[i+N-1])
		Hi_view = @view(H[i][:,1:x_tt.ttv_rks[i+N],1:x_tt.ttv_rks[i+N]])
		Him = @view(H[i-1][:,1:x_tt.ttv_rks[i+N-1],1:x_tt.ttv_rks[i+N-1]])
		x_vec = x_tt.ttv_vec[i+N-1]
		A_vec = A_tto.tto_vec[i+N-1]
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

#returns the contracted tensor A_i[\\mu_i] ⋯ A_j[\\mu_j] ∈ R^{R^A_{i-1} × n_i × n_i × ⋯ × n_j × n_j ×  R^A_j}
function Amid(A_tto::TToperator{T},i::Int,j::Int) where {T<:Number}
	A = permutedims(A_tto.tto_vec[i],(3,1,2,4))
	for k in i+1:j
		C = reshape(A,A_tto.tto_rks[i],prod(A_tto.tto_dims[i:k-1]),:,A_tto.tto_rks[k])
		@tensor Atemp[αk,Ik,ik,Jk,jk,βk] := A_tto.tto_vec[k][ik,jk,ξk,βk]*C[αk,Ik,Jk,ξk]
		A = reshape(Atemp,A_tto.tto_rks[i],prod(A_tto.tto_dims[i:k]),:,A_tto.tto_rks[k+1])
	end
	return A #size R^A_{i-1} × (n_i⋯n_j) × (n_i⋯n_j) × R^A_j
end

#full assemble of matrix K
function K_full(Gi::AbstractArray{T,3},Hi::AbstractArray{T,3},Amid_tensor::AbstractArray{T,4}) where {T<:Number,d}
	K_dims = (size(Gi,2),size(Amid_tensor,2),size(Hi,2))
	K = zeros(T,K_dims...,K_dims...)
	@tensoropt((a,c,d,f), K[a,b,c,d,e,f] = Gi[y,a,d]*Hi[z,c,f]*Amid_tensor[y,b,e,z]) #size (r^X_{i-1},n_i⋯n_j,r^X_j)
	return Hermitian(reshape(K,prod(K_dims),prod(K_dims)))
end

function init_Hb(x_tt::TTvector{T},b_tt::TTvector{T},N::Integer,rmax) where {T<:Number}
	d = x_tt.N
	H_b = Array{Array{T,2},1}(undef, d+1-N) 
	H_b[d+1-N] = ones(T,1,1)
	rks = r_and_d_to_rks(vcat(1,rmax*ones(Int,d-1),1),x_tt.ttv_dims;rmax=rmax)
	for i = d+1-N:-1:2
		H_b[i-1] = zeros(T,rks[i+N-1],b_tt.ttv_rks[i+N-1])
		b_vec = b_tt.ttv_vec[i+N-1]
		x_vec = x_tt.ttv_vec[i+N-1]
		Hbi = @view(H_b[i][1:x_tt.ttv_rks[i+N],:])
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
	b_out = permutedims(b_tt.ttv_vec[i],(2,1,3))
	for k in i+1:j
		@tensor btemp[αk,ik,jk,βk] := b_out[αk,ik,ξk]*b_tt.ttv_vec[k][jk,ξk,βk]
		b_out = reshape(btemp,b_tt.ttv_rks[i],:,b_tt.ttv_rks[k+1]) #size r^b_{i-1} × (n_i⋯n_k) × r^b_k
	end
	return b_out
end

function Ksolve!(Gi_view::AbstractArray{T,3},G_bi::AbstractArray{T,2},Hi_view::AbstractArray{T,3},H_bi::AbstractArray{T,2},Amid_tensor::AbstractArray{T,4},Bmid::AbstractArray{T,3},Pb,V0::AbstractArray{T,3},Vapp::AbstractArray{T,3};it_solver=false,maxiter=200,tol=1e-6,itslv_thresh=256) where T<:Number
	K_dims = (size(Gi_view,2),size(Amid_tensor,2),size(Hi_view,2))
	@tensor (Pb[α1,i,α2] = (G_bi[α1,β1]*Bmid[β1,i,β2])*H_bi[α2,β2]) #size (r^X_{i-1},n_i⋯n_j,r^X_j)

	if it_solver && prod(K_dims) > itslv_thresh	
		function K_matfree(Vout,V)
			Hrshp = reshape(Vout,K_dims)
			@tensor( Hrshp[a,b,c] = 0.5*((Gi_view[y,a,d]*reshape(V,K_dims)[d,e,f])*Amid_tensor[y,b,e,z])*Hi_view[z,c,f] + 0.5*((Gi_view[y,d,a]*reshape(V,K_dims)[d,e,f])*Amid_tensor[y,e,b,z])*Hi_view[z,f,c])
			return nothing
		end
		Vapp[:],top = linsolve(LinearMap{T}(K_matfree,Integer(prod(K_dims));ismutating=true), Pb[:], V0[:];issymmetric=true,isposdef=true,tol=tol,verbosity=2)
		println("Number of matvec: $(top.numops)")
		return nothing
	else
		K = K_full(Gi_view,Hi_view,Amid_tensor)
		Vapp[:] = K\Pb[:]
		return nothing
	end
end

function right_core_move!(x_tt::TTvector{T},V,V_move,i::Int,N,tol::Float64,r_max::Integer;verbose,dmrg_info) where {T<:Number}
	# Perform the truncated svd
	u_V, s_V, v_V = svd(reshape(V,x_tt.ttv_rks[i]*x_tt.ttv_dims[i],:))
	# Update the ranks to the truncated one
	x_tt.ttv_rks[i+1] = max(min(cut_off_index(s_V,tol),r_max),x_tt.ttv_rks[i+1])
	println("Rank: $(x_tt.ttv_rks[i+1]),	Max rank=$r_max")
	svd_truncation = (norm(s_V)-norm(s_V[1:x_tt.ttv_rks[i+1]]))/norm(s_V)
	println("Discarded weight: $(svd_truncation)")
	if verbose
		push!(dmrg_info.ttsvd_weights, svd_truncation)
	end

	x_tt.ttv_vec[i] = permutedims(reshape(u_V[:,1:x_tt.ttv_rks[i+1]],x_tt.ttv_rks[i],x_tt.ttv_dims[i],:),(2,1,3))
	x_tt.ttv_ot[i] = 1
	x_tt.ttv_ot[i+1]=0
	V_moveview = @view(V_move[1:x_tt.ttv_rks[i+1],1:prod(x_tt.ttv_dims[i+1:i+N-1]),1:size(V,3)])
	@tensor V_moveview[αk,ik,βk] = reshape(v_V'[1:x_tt.ttv_rks[i+1],:],x_tt.ttv_rks[i+1],:,size(V,3))[αk,ik,βk]
	for ak in axes(V_moveview,1)
		V_moveview[ak,:,:] = V_moveview[ak,:,:]*(s_V[ak])
	end
	return nothing
end

function left_core_move!(x_tt::TTvector{T},V,V_move,j::Int,N,tol::Float64,r_max::Integer;verbose,dmrg_info) where {T<:Number}
	# Perform the truncated svd
	u_V, s_V, v_V = svd(reshape(V,:, x_tt.ttv_dims[j]*x_tt.ttv_rks[j+1]))
	# Update the ranks to the truncated one
	x_tt.ttv_rks[j] = max(min(cut_off_index(s_V,tol),r_max),x_tt.ttv_rks[j])
	println("Rank: $(x_tt.ttv_rks[j]),	Max rank=$r_max")
	svd_truncation = (norm(s_V)-norm(s_V[1:x_tt.ttv_rks[j]]))/norm(s_V)
	println("Discarded weight: $(svd_truncation)")
	if verbose
		push!(dmrg_info.ttsvd_weights, svd_truncation)
	end

	x_tt.ttv_vec[j] = permutedims(reshape(v_V'[1:x_tt.ttv_rks[j],:],x_tt.ttv_rks[j],:,x_tt.ttv_rks[j+1]),(2,1,3))
	x_tt.ttv_ot[j] = -1
	x_tt.ttv_ot[j-1]=0
	V_moveview = @view(V_move[1:size(V,1),1:prod(x_tt.ttv_dims[j+1-N:j-1]),1:x_tt.ttv_rks[j]])
	@tensor V_moveview[αk,ik,βk] = reshape(u_V[:,1:x_tt.ttv_rks[j]],size(V,1),:,x_tt.ttv_rks[j])[αk,ik,βk]
	for bk in axes(V_moveview,3)
		V_moveview[:,:,bk] = V_moveview[:,:,bk]*s_V[bk]
	end
	return nothing
end


function K_eigmin(Gi_view::AbstractArray{T,3},Hi_view::AbstractArray{T,3},V0::AbstractArray{T,3},Amid_tensor::AbstractArray{T,4},V;it_solver=false::Bool,itslv_thresh=256::Int64,maxiter=200::Int64,tol=1e-6::Float64) where T<:Number
	K_dims = size(V0)
	λ = zero(T)
	if it_solver && prod(K_dims) > itslv_thresh
		function K_matfree(Vout,V)
			Hrshp = reshape(Vout,K_dims)
			@tensor( Hrshp[a,b,c] = 0.5*((Gi_view[y,a,d]*reshape(V,K_dims)[d,e,f])*Amid_tensor[y,b,e,z])*Hi_view[z,c,f] + 0.5*((Gi_view[y,d,a]*reshape(V,K_dims)[d,e,f])*Amid_tensor[y,e,b,z])*Hi_view[z,f,c])
			return nothing
		end
		r = eigsolve(LinearMap{T}(K_matfree,prod(K_dims);issymmetric = true,ismutating=true),copy(V0[:]),1,:SR,issymmetric=true,tol=tol,maxiter=maxiter)
		println("Number of matvec: $(r[3].numops)")
		V[:] = r[2][1][:]
		λ = real(r[1][1])
	else
		K = K_full(Gi_view,Hi_view,Amid_tensor)
		F = eigen(K,1:1)
		for i in eachindex(V)
			V[i] = reshape(F.vectors[:,1],K_dims)[i]
		end
		λ = F.values[1]
	end	
	return λ
end

function init_dmrg(A::TToperator{T,d},tt_opt::TTvector{T,d},rks,N::Integer) where {T<:Number,d}
	rmax = maximum(rks)
	G = Array{Array{T,3},1}(undef, d+1-N)
	Amid_list = Array{Array{T,4},1}(undef, d+1-N)
	for i in 1:d+1-N
		G[i] = zeros(T,A.tto_rks[i],rks[i],rks[i])
		Amid_list[i] = Amid(A,i,i+N-1)
	end
	G[1] = ones(T,size(G[1]))
	H = init_H(tt_opt,A,N,rmax)

	V0 = zeros(T,rmax,maximum(tt_opt.ttv_dims)^N,rmax)
	V0_view = @view(V0[1:tt_opt.ttv_rks[1],1:prod(tt_opt.ttv_dims[1:N]),1:tt_opt.ttv_rks[1+N]])
	V0_view = b_mid(tt_opt,1,N)
	V = zeros(T,rmax,maximum(tt_opt.ttv_dims)^N,rmax)
	V_move = zeros(T,rmax,maximum(tt_opt.ttv_dims),rmax)
	V_temp = zeros(T,rmax,maximum(tt_opt.ttv_dims),maximum(tt_opt.ttv_dims),rmax)
	return G,Amid_list,H,V0,V,V_move,V_temp,V0_view
end

function init_dmrg_b(b::TTvector{T,d},tt_opt::TTvector{T,d},rks,N) where {T<:Number,d}
	rmax = maximum(rks)
	G_b = zeros.(T,rks[1:d+1-N],b.ttv_rks[1:d+1-N]) #Array{Array{T,2},1}(undef, d+1-N)
	bmid_list = Array{Array{T,3},1}(undef, d+1-N)
	for i in 1:d+1-N
		bmid_list[i] = b_mid(b,i,i+N-1)
	end
	G_b[1] = ones(T,size(G_b[1]))
	Pb_temp = zeros(T,rmax,maximum(tt_opt.ttv_dims)^N,rmax)
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

function update_right(tt_opt,V0,V_view,V_move,V_temp,i,N,tol,rmax,Ai,Gi_view,Gip;verbose,dmrg_info,move=true)
	if move
		#update tt_opt
		right_core_move!(tt_opt,V_view,V_move,i,N,tol,rmax;verbose,dmrg_info)
	end

	V_moveview = @view(V_move[1:tt_opt.ttv_rks[i+1],1:prod(tt_opt.ttv_dims[i+1:i+N-1]),1:tt_opt.ttv_rks[i+N]])
	V_tempview = @view(V_temp[1:size(V_moveview,1),1:size(V_moveview,2),1:tt_opt.ttv_dims[i+N],1:tt_opt.ttv_rks[i+1+N]])
	@tensor V_tempview[αk,J,ik,γk] = V_moveview[αk,J,βk]*tt_opt.ttv_vec[i+N][ik,βk,γk]
	V0_view = @view(V0[1:tt_opt.ttv_rks[i+1],1:prod(tt_opt.ttv_dims[i+1:i+N-1]),1:tt_opt.ttv_rks[i+N]])
	V0_view = reshape(V_tempview,size(V_tempview,1),:,size(V_tempview,4))

	#update G[i+1]
	Gip_view = @view(Gip[:,1:tt_opt.ttv_rks[i+1],1:tt_opt.ttv_rks[i+1]])
	update_G!(tt_opt.ttv_vec[i],Ai,Gi_view,Gip_view)
	return V0_view
end

function update_left(tt_opt,V0,V_view,V_move,V_temp,i,N,tol,rmax,Aip,Hi_view,Him;verbose,dmrg_info,move=true)
	if move
		left_core_move!(tt_opt,V_view,V_move,i+N-1,N,tol,rmax;verbose,dmrg_info)
	end

	#update the initialization
	V_moveview = @view(V_move[1:tt_opt.ttv_rks[i],1:prod(tt_opt.ttv_dims[i:i+N-2]),1:tt_opt.ttv_rks[i+N-1]])
	V_tempview = @view(V_temp[1:tt_opt.ttv_rks[i-1],1:size(V_moveview,2),1:tt_opt.ttv_dims[i-1],1:size(V_moveview,3)])
	@tensor V_tempview[αk,J,ik,γk] =  V_moveview[βk,J,γk]*tt_opt.ttv_vec[i-1][ik,αk,βk]
	V0_view = @view(V0[1:tt_opt.ttv_rks[i],1:prod(tt_opt.ttv_dims[i:i+N-2]),1:tt_opt.ttv_rks[i+N-1]])
	V0_view = reshape(V_tempview,size(V_tempview,1),:,size(V_tempview,4))

	#update H[i-1]
	Him_view = @view(Him[:,1:tt_opt.ttv_rks[i+N-1],1:tt_opt.ttv_rks[i+N-1]])
	update_H!(tt_opt.ttv_vec[i+N-1],Aip,Hi_view,Him_view)
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
	rks = tt_opt.ttv_rks 
	dims = tt_opt.ttv_dims
	if N==2
		V_reshape = reshape(V,rks[j-1]*dims[j-1],rks[j+1]*dims[j])
		u_V, s_V, v_V = svd(V_reshape)
		tt_opt.ttv_rks[j] = min(cut_off_index(s_V,tol),rmax)
		println("Rank: $(tt_opt.ttv_rks[j]),	Max rank=$rmax")
		svd_truncation = (norm(s_V)-norm(s_V[1:tt_opt.ttv_rks[j]]))/norm(s_V)
		println("Discarded weight: $(svd_truncation)")
		if verbose
			push!(dmrg_info.ttsvd_weights, svd_truncation)
		end
		if left_to_right
			tt_opt.ttv_vec[j-1] = permutedims(reshape(u_V[:,1:tt_opt.ttv_rks[j]]*Diagonal(s_V[1:tt_opt.ttv_rks[j]]), tt_opt.ttv_rks[j-1], dims[j],tt_opt.ttv_rks[j]),(2,1,3))
			tt_opt.ttv_ot[j-1] = 0
			tt_opt.ttv_vec[j] = permutedims(reshape(v_V'[1:tt_opt.ttv_rks[j],:],tt_opt.ttv_rks[j],dims[j],tt_opt.ttv_rks[j+1]),(2,1,3))
			tt_opt.ttv_ot[j] = -1
		else
			tt_opt.ttv_vec[j-1] = permutedims(reshape(u_V[:,1:tt_opt.ttv_rks[j]], tt_opt.ttv_rks[j-1], dims[j],tt_opt.ttv_rks[j]),(2,1,3))
			tt_opt.ttv_ot[j-1] = 1
			tt_opt.ttv_vec[j] = permutedims(reshape(Diagonal(s_V[1:tt_opt.ttv_rks[j]])*v_V'[1:tt_opt.ttv_rks[j],:],tt_opt.ttv_rks[j],dims[j],tt_opt.ttv_rks[j+1]),(2,1,3))
			tt_opt.ttv_ot[j] = 0
		end
	else #N=1
		if left_to_right
			l,q = lq(reshape(V,tt_opt.ttv_rks[j],:))
			tt_opt.ttv_vec[j-1] = reshape(reshape(tt_opt.ttv_vec[j-1],tt_opt.ttv_dims[j-1]*tt_opt.ttv_rks[j-1],:)*l, tt_opt.ttv_dims[j-1],tt_opt.ttv_rks[j-1],:)
			tt_opt.ttv_ot[j-1] = 0
			tt_opt.ttv_vec[j] = permutedims(reshape(Matrix(q),tt_opt.ttv_rks[j],tt_opt.ttv_dims[j],tt_opt.ttv_rks[j+1]),(2,1,3))
			tt_opt.ttv_ot[j] = -1
		else 
			q,r = qr(reshape(V,tt_opt.ttv_rks[j]*tt_opt.ttv_dims[j],:))
			@tensor vec_temp[i,α,β] := tt_opt.ttv_vec[j+1][i,ξ,β]*r[α,ξ]
			tt_opt.ttv_vec[j+1] = vec_temp
			tt_opt.ttv_ot[j+1] = 0
			tt_opt.ttv_vec[j] = permutedims(reshape(Matrix(q),tt_opt.ttv_rks[j],tt_opt.ttv_dims[j],tt_opt.ttv_rks[j+1]),(2,1,3))
			tt_opt.ttv_ot[j] = 1
		end
	end
	return nothing 
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
	dims = tt_start.ttv_dims
	rmax = maximum(rmax_schedule)
	rks = r_and_d_to_rks(vcat(1,rmax*ones(Int,d-1),1),dims;rmax=rmax)

	#Initialize DMRG 
	G,Amid_list,H,V0,V,V_move,V_temp,V0_view = init_dmrg(A,tt_opt,rks,N)
	G_b,bmid_list,H_b,Pb_temp = init_dmrg_b(b,tt_opt,rks,N)

	nsweeps = 0 #sweeps counter
	i_schedule = 1

	#1st step of the sweep
	Gi_view,Hi_view,V_view = update_G_H_V(G[1],H[1],V,tt_opt.ttv_dims,tt_opt.ttv_rks,1,N)
	G_bi_view, H_bi_view, Pb_view = update_G_H_V_b(G_b[1],H_b[1],Pb_temp,tt_opt.ttv_dims,tt_opt.ttv_rks,1,N)
	Ksolve!(Gi_view,G_bi_view,Hi_view,H_bi_view,Amid_list[1],bmid_list[1],Pb_view,V0_view, V_view;it_solver=it_solver,maxiter=linsolv_maxiter,tol=linsolv_tol,itslv_thresh=itslv_thresh)
	V0_view = update_right(tt_opt,V0,V_view,V_move,V_temp,1,N,tol,rmax_schedule[i_schedule],A.tto_vec[1],Gi_view,G[2];verbose,dmrg_info)

	G_bip = @view(G_b[2][1:tt_opt.ttv_rks[2],:])
	update_Gb!(tt_opt.ttv_vec[1],b.ttv_vec[1],G_bi_view,G_bip)

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
			Gi_view,Hi_view,V_view = update_G_H_V(G[i],H[i],V,tt_opt.ttv_dims,tt_opt.ttv_rks,i,N)
			G_bi_view, H_bi_view, Pb_view = update_G_H_V_b(G_b[i],H_b[i],Pb_temp,tt_opt.ttv_dims,tt_opt.ttv_rks,i,N)
			# Define V as solution of K*x=Pb in x
			Ksolve!(Gi_view,G_bi_view,Hi_view,H_bi_view,Amid_list[i],bmid_list[i],Pb_view,V0_view, V_view;it_solver=it_solver,maxiter=linsolv_maxiter,tol=linsolv_tol,itslv_thresh=itslv_thresh)

			if i<d-N+1
				#Update TT core i and the next initialization
				V0_view = update_right(tt_opt,V0,V_view,V_move,V_temp,i,N,tol,rmax_schedule[i_schedule],A.tto_vec[i],Gi_view,G[i+1];verbose,dmrg_info)

				G_bip = @view(G_b[i+1][1:tt_opt.ttv_rks[i+1],:])
				update_Gb!(tt_opt.ttv_vec[i],b.ttv_vec[i],G_bi_view,G_bip)
			else #i==d-N+1
				ttv_after_dmrg_microstep!(tt_opt,rmax_schedule[i_schedule],V_view,schedule;verbose,dmrg_info)
				if verbose
					push!(dmrg_info.TTvs,copy(tt_opt))
				end
				#update H[d-N],H_b[d-N]
				Him_view = @view(H[d-N][:,1:tt_opt.ttv_rks[d],1:tt_opt.ttv_rks[d]])
				update_H!(tt_opt.ttv_vec[d],A.tto_vec[d],Hi_view,Him_view)
				update_Hb!(tt_opt.ttv_vec[d],b.ttv_vec[d],H_bi_view,H_b[d-N])

				#update V0_view 
				V0_view = @view(V0[1:tt_opt.ttv_rks[d-N],1:prod(tt_opt.ttv_dims[d-N:d-1]),1:tt_opt.ttv_rks[d]])
				V0_view = b_mid(tt_opt,d-N,d-1)
			end
		println("---------------------------")
		end

		# Second half sweep
		for i = (d-N):(-1):1
			println("Backward sweep: core optimization $i out of $(d+1-N)")
			# Define V as solution of K*x=Pb in x
			Gi_view,Hi_view,V_view = update_G_H_V(G[i],H[i],V,tt_opt.ttv_dims,tt_opt.ttv_rks,i,N)
			G_bi_view, H_bi_view, Pb_view = update_G_H_V_b(G_b[i],H_b[i],Pb_temp,tt_opt.ttv_dims,tt_opt.ttv_rks,i,N)
			# Define V as solution of K*x=Pb in x
			Ksolve!(Gi_view,G_bi_view,Hi_view,H_bi_view,Amid_list[i],bmid_list[i],Pb_view,V0_view, V_view;it_solver=it_solver,maxiter=linsolv_maxiter,tol=linsolv_tol,itslv_thresh=itslv_thresh)

			if i>1 
				V0_view = update_left(tt_opt,V0,V_view,V_move,V_temp,i,N,tol,rmax_schedule[i_schedule],A.tto_vec[i+N-1],Hi_view,H[i-1];verbose,dmrg_info)
				update_Hb!(tt_opt.ttv_vec[i+N-1],b.ttv_vec[i+N-1],H_bi_view,H_b[i-1])
			else 
				ttv_after_dmrg_microstep!(tt_opt,rmax_schedule[i_schedule],V_view,schedule;verbose,dmrg_info,left_to_right=false,j=N)
				if verbose
					push!(dmrg_info.TTvs,copy(tt_opt))
				end

				#update G[2],G_b[2]
				Gi_view = @view(G[1][1:1,1:1,1:1])
				Gip_view = @view(G[2][:,1:tt_opt.ttv_rks[2],1:tt_opt.ttv_rks[2]])
				update_G!(tt_opt.ttv_vec[1],A.tto_vec[1],Gi_view,Gip_view)

				G_bip = @view(G_b[2][1:tt_opt.ttv_rks[2],:])
				update_Gb!(tt_opt.ttv_vec[1],b.ttv_vec[1],G_bi_view,G_bip)

				#update V0_view 
				V0_view = @view(V0[1:tt_opt.ttv_rks[2],1:prod(tt_opt.ttv_dims[2:N+1]),1:tt_opt.ttv_rks[N+2]])
				V0_view = b_mid(tt_opt,2,N+1)
			end
		println("---------------------------")
		end
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
	dims = tt_start.ttv_dims
	rmax = maximum(rmax_schedule)
	rks = r_and_d_to_rks(vcat(1,rmax*ones(Int,d-1),1),dims;rmax=rmax)
	# Initialize the output objects
	E = Float64[]
	r_hist = Int64[]

	#Initialize DMRG 
	G,Amid_list,H,V0,V,V_move,V_temp,V0_view = init_dmrg(A,tt_opt,rks,N)

	nsweeps = 0 #sweeps counter
	i_schedule = 1

	#1st step of the sweep
	println("Macro-iteration 1; bond dimension $(rmax_schedule[1])")
	println("Forward sweep: core optimization 1 out of $(d-N+1)")
	Gi_view,Hi_view,V_view = update_G_H_V(G[1],H[1],V,tt_opt.ttv_dims,tt_opt.ttv_rks,1,N)
	λ = K_eigmin(G[1],H[1],V0_view ,Amid_list[1], V_view ;it_solver,maxiter=linsolv_maxiter,tol=linsolv_tol,itslv_thresh)
	println("Eigenvalue: $λ")
	push!(E,λ)
	push!(r_hist,maximum(tt_opt.ttv_rks))
	V0_view = update_right(tt_opt,V0,V_view,V_move,V_temp,1,N,tol,rmax_schedule[i_schedule],A.tto_vec[1],Gi_view,G[2];verbose,dmrg_info)
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
			Gi_view,Hi_view,V_view = update_G_H_V(G[i],H[i],V,tt_opt.ttv_dims,tt_opt.ttv_rks,i,N)
			λ = K_eigmin(Gi_view,Hi_view,V0_view, Amid_list[i],V_view; it_solver, maxiter=linsolv_maxiter,tol=linsolv_tol,itslv_thresh)
			println("Eigenvalue: $λ")
			push!(E,λ)
			#Update TT core i and the next initialization
			if i<d-N+1
				V0_view = update_right(tt_opt,V0,V_view,V_move,V_temp,i,N,tol,rmax_schedule[i_schedule],A.tto_vec[i],Gi_view,G[i+1];verbose,dmrg_info)
				push!(r_hist,maximum(tt_opt.ttv_rks))
			else #i==d-N+1
				ttv_after_dmrg_microstep!(tt_opt,rmax_schedule[i_schedule],V_view,schedule;verbose,dmrg_info)
				if verbose
					push!(dmrg_info.TTvs,copy(tt_opt))
				end
				#update H[d-N],H_b[d-N]
				Him_view = @view(H[d-N][:,1:tt_opt.ttv_rks[d],1:tt_opt.ttv_rks[d]])
				update_H!(tt_opt.ttv_vec[d],A.tto_vec[d],Hi_view,Him_view)
			end
			println("--------------------------------------------")
		end

		#update V0_view 
		V0_view = @view(V0[1:tt_opt.ttv_rks[d-N],1:prod(tt_opt.ttv_dims[d-N:d-1]),1:tt_opt.ttv_rks[d]])
		V0_view = b_mid(tt_opt,d-N,d-1)

		# Second half sweep
		for i = (d-N):(-1):2
			println("Backward sweep: core optimization $(i) out of $(d-N)")
			# Define V as solution of K*x=P2b in x
			Gi_view,Hi_view,V_view = update_G_H_V(G[i],H[i],V,tt_opt.ttv_dims,tt_opt.ttv_rks,i,N)
			λ = K_eigmin(Gi_view,Hi_view,V0_view,Amid_list[i],V_view ;it_solver=it_solver,maxiter=linsolv_maxiter,tol=linsolv_tol,itslv_thresh=itslv_thresh)
			println("Eigenvalue: $λ")
			push!(E,λ)
			#update the initialization
			V0_view = update_left(tt_opt,V0,V_view,V_move,V_temp,i,N,tol,rmax_schedule[i_schedule],A.tto_vec[i+N-1],Hi_view,H[i-1];verbose,dmrg_info)
			println("--------------------------------------------")
			push!(r_hist,maximum(tt_opt.ttv_rks))
		end
		#last step to complete the sweep
		println("Backward sweep: core optimization $(1) out of $(d-N)")
		Gi_view,Hi_view,V_view = update_G_H_V(G[1],H[1],V,tt_opt.ttv_dims,tt_opt.ttv_rks,1,N)
		λ = K_eigmin(G[1],H[1],V0_view ,Amid_list[1], V_view ;it_solver,maxiter=linsolv_maxiter,tol=linsolv_tol,itslv_thresh)
		println("Eigenvalue: $λ")
		push!(E,λ)
		ttv_after_dmrg_microstep!(tt_opt,rmax_schedule[i_schedule],V_view,schedule;verbose,dmrg_info,left_to_right=false,j=N)
		if verbose
			push!(dmrg_info.TTvs,copy(tt_opt))
		end
		push!(r_hist,maximum(tt_opt.ttv_rks))
		println("--------------------------------------------")

		#update G[2]
		Gi_view = @view(G[1][1:1,1:1,1:1])
		Gip_view = @view(G[2][:,1:tt_opt.ttv_rks[2],1:tt_opt.ttv_rks[2]])
		update_G!(tt_opt.ttv_vec[1],A.tto_vec[1],Gi_view,Gip_view)

		#update V0_view 
		V0_view = @view(V0[1:tt_opt.ttv_rks[2],1:prod(tt_opt.ttv_dims[2:N+1]),1:tt_opt.ttv_rks[N+2]])
		V0_view = b_mid(tt_opt,2,N+1)
	end
	return E::Array{Float64,1}, tt_opt::TTvector{T},dmrg_info 
end

"""
returns the smallest eigenpair Ax = Sx
NOT WORKING
"""
function dmrg_gen_eigsolv(A :: TToperator{T}, S::TToperator{T}, tt_start :: TTvector{T} ; sweep_schedule=[2],rmax_schedule=[maximum(tt_start.ttv_rks)],tol=1e-10,it_solver=false,itslv_thresh=2500) where {T<:Number}
	d = tt_start.N
	# Initialize the to be returned tensor in its tensor train format
	tt_opt = orthogonalize(tt_start)
	dims = tt_start.ttv_dims
	E = zeros(Float64,d*sweep_schedule[end]) #output eigenvalue
	# Define the array of ranks of tt_opt [r_0=1,r_1,...,r_d]
	rks = tt_start.ttv_rks

	# Initialize the arrays of G and K
	G = Array{Array{T}}(undef, d)
	K = Array{Array{T}}(undef, d) 

	# Initialize G[1]
	for i in 1:d
		G[i] = zeros(dims[i],rks[i],dims[i],rks[i],A.tto_rks[i+1])
		K[i] = zeros(dims[i],rks[i],dims[i],rks[i],S.tto_rks[i+1])
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
				tt_opt = right_core_move(tt_opt,V,i,N,rks;verbose,dmrg_info)
			end

			#update G and K
			update_G!(tt_opt.ttv_vec[i],A.tto_vec[i+1],G[i],G[i+1])
			update_G!(tt_opt.ttv_vec[i],S.tto_vec[i+1],K[i],K[i+1])
		end

		# Second half sweep
		for i = d:(-1):2
			println("Backward sweep: core optimization $i out of $d")
			# Define V as solution of K*x=Pb in x
			i_μit += 1
			E[i_μit],V = K_eiggenmin(G[i],H[i],K[i],L[i],tt_opt.ttv_vec[i];it_solver=it_solver,itslv_thresh=itslv_thresh)
			println("Eigenvalue: $(E[i_μit])")
			tt_opt = left_core_move(tt_opt,V,i,rks)
			update_H!(tt_opt.ttv_vec[i],A.tto_vec[i],H[i],H[i-1])
			update_H!(tt_opt.ttv_vec[i],S.tto_vec[i],L[i],L[i-1])
		end
	end
end
