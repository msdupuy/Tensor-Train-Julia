using TensorOperations
using LinearAlgebra

"""
TT rounding algorithm in https://doi.org/10.1137/21M1451191
Partial contraction defined in Definition 3.1
"""
function partial_contraction(A::TTvector{T,N},B::TTvector{T,N};reverse=true) where {T,N}
  @assert A.ttv_dims==B.ttv_dims "TT dimensions are not compatible"
  L = length(A.ttv_dims)
  W = zeros.(T,A.ttv_rks,B.ttv_rks)
  if reverse
    W[L+1] = ones(T,1,1)
    @inbounds for k in L:-1:1
      @tensoropt((a,b,α,β), W[k][a,b] = A.ttv_vec[k][z,a,α]*B.ttv_vec[k][z,b,β]*W[k+1][α,β]) #size R^A_{k} × R^B_{k} 
    end
  else
    W[1] = ones(T,1,1)
    @inbounds for k in 1:L
      @tensoropt((a,b,α,β), W[k+1][a,b] = A.ttv_vec[k][z,α,a]*B.ttv_vec[k][z,β,b]*W[k][α,β]) #size R^A_{k} × R^B_{k} 
    end
  end
  return W 
end

"""
TT rounding algorithm in https://doi.org/10.1137/21M1451191
Algorithm 3.2 "Randomize then Orthogonalize"
"""
function ttrand_rounding(y_tt::TTvector{T,N};rks=y_tt.ttv_rks,rmax=prod(y_tt.ttv_dims),orthogonal=true,ℓ=round(Int,0.5*maximum(rks))) where {T,N}
  rks = r_and_d_to_rks(rks.+ℓ,y_tt.ttv_dims;rmax=rmax+ℓ) #rks with oversampling
  x_tt = zeros_tt(T,y_tt.ttv_dims,rks)
  ℜ_tt = rand_tt(T,y_tt.ttv_dims,rks;normalise=true,orthogonal=orthogonal)
  W = partial_contraction(y_tt,ℜ_tt)
  A_temp = zeros(T,maximum(rks),maximum(y_tt.ttv_rks))
  Y_temp = zeros(T,maximum(y_tt.ttv_dims),maximum(rks),maximum(y_tt.ttv_rks))
  Y_temp[1:y_tt.ttv_dims[1],1:1,1:y_tt.ttv_rks[2]] = copy(y_tt.ttv_vec[1])
  Z_temp = zeros(T,maximum(y_tt.ttv_dims),maximum(rks),maximum(rks))
  @inbounds begin
    for k in 1:N-1
      @tensoropt((βₖ,αₖ₋₁,αₖ), Z_temp[1:y_tt.ttv_dims[k],1:rks[k],1:rks[k+1]][iₖ,αₖ₋₁,αₖ] = @view(Y_temp[1:y_tt.ttv_dims[k],1:rks[k],1:y_tt.ttv_rks[k+1]])[iₖ,αₖ₋₁,βₖ]*W[k+1][βₖ,αₖ]) # nₖ × ℓₖ₋₁ × ℓₖ
      @views Qₖ_temp,Rₖ = qr(reshape(Z_temp[1:y_tt.ttv_dims[k],1:rks[k],1:rks[k+1]],x_tt.ttv_dims[k]*rks[k],:))
      x_tt.ttv_vec[k] = reshape(Matrix(Qₖ_temp),y_tt.ttv_dims[k],rks[k],:)
      @views A_temp[1:rks[k+1],1:y_tt.ttv_rks[k+1]] = Matrix(Qₖ_temp)'*reshape(Y_temp[1:x_tt.ttv_dims[k],1:rks[k],1:y_tt.ttv_rks[k+1]],x_tt.ttv_dims[k]*rks[k],:) # × Rˣₖ
      @tensoropt((βₖ,αₖ₊₁), Y_temp[1:y_tt.ttv_dims[k+1],1:rks[k+1],1:y_tt.ttv_rks[k+2]][iₖ₊₁,αₖ,αₖ₊₁] = @view(A_temp[1:rks[k+1],1:y_tt.ttv_rks[k+1]])[αₖ,βₖ]*y_tt.ttv_vec[k+1][iₖ₊₁,βₖ,αₖ₊₁])
    end
    x_tt.ttv_vec[N] = Y_temp[1:y_tt.ttv_dims[N],1:rks[N],1:1]
  end
  return x_tt
end

function dot_randrounding(A::TToperator,x::TTvector)
  y = A*x
  y = ttrand_rounding(y)
  return tt_rounding(y;tol=1e-8)
end

"""
STTA: http://arxiv.org/abs/2208.02600
"""

function stta_sketch(x::TTvector{T,N},L::TTvector{T,N},R::TTvector{T,N}) where {T,N}
  Ψ = zeros.(x.ttv_dims,L.ttv_rks[1:end-1],R.ttv_rks[2:end])
  Ω = zeros.(L.ttv_rks[2:end-1],R.ttv_rks[2:end-1])
  left_contractions = partial_contraction(x,L;reverse=false)
  right_contractions = partial_contraction(x,R;reverse=true)
  for k in eachindex(Ω)
    @tensor (Ω[k][a,b] = left_contractions[k+1][z,a]*right_contractions[k+1][z,b])
  end
  for k in eachindex(Ψ)
    @tensor (Ψ[k][i,α,β] = (x.ttv_vec[k][i,y,z]*left_contractions[k][y,α])*right_contractions[k+1][z,β])
  end
  return Ω,Ψ
end

"""
returns a stable solution to A\\B
"""
function stable_inverse(A;ε=1e-12)
  u,s,v = svd(A)
  return v[:,s.>maximum(s)*ε]*Diagonal(1 ./s[s.>maximum(s)*ε])*u[:,s.>maximum(s)*ε]'
end

function stta(y_tt::TTvector{T,N};rks=vcat(1,round.(Int,1.5*y_tt.ttv_rks[2:end-1]),1),rmax=maximum(rks),ℓ=round(Int,maximum(rks))) where {T,N}
  r_rks = r_and_d_to_rks(rks.+ℓ,y_tt.ttv_dims;rmax=rmax+ℓ)
  l_rks = r_and_d_to_rks(round.(Int,1.5*(rks.+ℓ)),y_tt.ttv_dims;rmax=round(Int,1.5*(rmax+ℓ)))
  L = rand_tt(y_tt.ttv_dims,l_rks,normalise=true,orthogonal=true,right=false)
  R = rand_tt(y_tt.ttv_dims,r_rks,normalise=true,orthogonal=true)
  Ω,Ψ = stta_sketch(y_tt,L,R)
  rks = ones(N+1)
  for k in 1:N-1
    if size(Ω[k],1)<size(Ω[k],2) #rR>rL
      Ψ_temp = reshape(Ψ[k],:,size(Ψ[k],3))*stable_inverse(Ω[k])
      Ψ[k]= reshape(Ψ_temp,size(Ψ[k],1),size(Ψ[k],2),:)
      rks[k+1] = size(Ψ[k],3)
    else 
      Ψ_temp = stable_inverse(Ω[k])*reshape(permutedims(Ψ[k+1],(2,1,3)),size(Ψ[k+1],2),:)
      Ψ[k+1]= permutedims(reshape(Ψ_temp,:,size(Ψ[k+1],1),size(Ψ[k+1],3)),(2,1,3))
      rks[k+1] = size(Ψ[k+1],2)
    end
  end
  return TTvector{T,N}(N,Ψ,y_tt.ttv_dims,rks,zeros(N))
end

"""
Wrong algorithm
"""
function tt_hmt(y_tt::TTvector{T,N};rks=y_tt.ttv_rks,rmax=maximum(rks),ℓ=round(Int,maximum(rks))) where {T,N}
  rks = r_and_d_to_rks(rks.+ℓ,y_tt.ttv_dims;rmax=rmax+ℓ)
  Ω = randn.(y_tt.ttv_rks[2:end],rks[2:end])
  x_tt = zeros_tt(y_tt.ttv_dims,rks)
  y_temp = zeros(maximum(y_tt.ttv_dims),maximum(rks),maximum(y_tt.ttv_rks))
  y_temp[1:y_tt.ttv_dims[1],1:rks[1],1:y_tt.ttv_rks[2]] = copy(y_tt.ttv_vec[1])
  for k in 1:N-1
    @tensor A_temp[iₖ,αₖ₋₁,βₖ] := @view(y_temp[1:y_tt.ttv_dims[k],1:rks[k],1:y_tt.ttv_rks[k+1]])[iₖ,αₖ₋₁,αₖ]*Ω[k][αₖ,βₖ]
    q,_ = qr(reshape(A_temp,y_tt.ttv_dims[k]*rks[k],:))
    x_tt.ttv_vec[k] = reshape(Matrix(q),y_tt.ttv_dims[k],rks[k],:)
    rks[k+1] = size(x_tt.ttv_vec[k],3)
    R_temp = q'[1:rks[k+1],:]*reshape(y_temp[1:y_tt.ttv_dims[k],1:rks[k],1:y_tt.ttv_rks[k+1]],y_tt.ttv_dims[k]*rks[k],:) #size rks[k+1] × y_tt.ttv_rks[k+1]
    @tensor (y_temp[1:y_tt.ttv_dims[k+1],1:rks[k+1],1:y_tt.ttv_rks[k+2]])[iₖ₊₁,αₖ,αₖ₊₁] = R_temp[αₖ,βₖ]*y_tt.ttv_vec[k+1][iₖ₊₁,βₖ,αₖ₊₁]
  end
  x_tt.ttv_vec[N] = y_temp[1:y_tt.ttv_dims[N],1:rks[N],1:1]
  return TTvector{T,N}(N,x_tt.ttv_vec,y_tt.ttv_dims,rks,zeros(N))
end