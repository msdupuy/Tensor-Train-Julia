using TensorOperations
using LinearAlgebra

"""
TT rounding algorithm in https://doi.org/10.1137/21M1451191
Partial contraction defined in Definition 3.1
"""
function partial_contraction(A::TTvector{T,N},B::TTvector{T,N};reverse=true) where {T,N}
  @assert A.dims==B.dims "TT dimensions are not compatible"
  W = zeros.(T,A.rks,B.rks)
  if reverse
    W[N+1] = ones(T,1,1)
    @inbounds for k in N:-1:1
      @tensoropt((a,b,α,β), W[k][a,b] = A.cores[k][z,a,α]*B.cores[k][z,b,β]*W[k+1][α,β]) #size R^A_{k} × R^B_{k} 
    end
  else
    W[1] = ones(T,1,1)
    @inbounds for k in 1:N
      @tensoropt((a,b,α,β), W[k+1][a,b] = A.cores[k][z,α,a]*B.cores[k][z,β,b]*W[k][α,β]) #size R^A_{k} × R^B_{k} 
    end
  end
  return W 
end

"""
Partial contraction for ⟨H*A, B⟩
"""
function partial_contraction(A::TTvector{T,N},H::TToperator{T,N},B::TTvector{T,N};reverse=true) where {T,N}
  @assert A.dims==B.dims "TT dimensions are not compatible"
  W = zeros.(T,A.rks,H.rks,B.rks)
  if reverse
    W[N+1] = ones(T,1,1,1)
    @inbounds for k in N:-1:1
      @tensoropt((a,b,α,β), W[k][a,h,b] = A.cores[k][z,a,α]*H.cores[k][y,z,h,η]*B.cores[k][y,b,β]*W[k+1][α,η,β]) 
    end
  else
    W[1] = ones(T,1,1,1)
    @inbounds for k in 1:N
      @tensoropt((a,b,α,β), W[k+1][a,h,b] = A.cores[k][z,α,a]*H.cores[k][y,z,η,h]*B.cores[k][y,β,b]*W[k][α,η,β]) 
    end
  end
  return W 
end

"""
TT rounding algorithm in https://doi.org/10.1137/21M1451191
Algorithm 3.2 "Randomize then Orthogonalize"
"""
function ttrand_rounding(y_tt::TTvector{T,N};rks=y_tt.rks,rmax=prod(y_tt.dims),orthogonal=true,ℓ=round(Int,0.5*maximum(rks)),khatri_rao=false,good=true) where {T,N}
  rks = r_and_d_to_rks(rks.+ℓ,y_tt.dims;rmax=rmax+ℓ) #rks with oversampling
  if khatri_rao 
    ℜ_tt = rand_tt_khatri_rao(y_tt.dims,rmax+ℓ;spherical=true)
  else
    ℜ_tt = rand_tt(T,y_tt.dims,rks;normalise=true,orthogonal,stable=true,good)
  end
  x_tt = zeros_tt(T,y_tt.dims,rks)
  W = partial_contraction(y_tt,ℜ_tt)
  A_temp = zeros(T,maximum(rks),maximum(y_tt.rks))
  Y_temp = zeros(T,maximum(y_tt.dims),maximum(rks),maximum(y_tt.rks))
  Y_temp[1:y_tt.dims[1],1:1,1:y_tt.rks[2]] = copy(y_tt.cores[1])
  Z_temp = zeros(T,maximum(y_tt.dims),maximum(rks),maximum(rks))
  @inbounds begin
    for k in 1:N-1
      @tensoropt((βₖ,αₖ₋₁,αₖ), Z_temp[1:y_tt.dims[k],1:rks[k],1:rks[k+1]][iₖ,αₖ₋₁,αₖ] = @view(Y_temp[1:y_tt.dims[k],1:rks[k],1:y_tt.rks[k+1]])[iₖ,αₖ₋₁,βₖ]*W[k+1][βₖ,αₖ]) # nₖ × ℓₖ₋₁ × ℓₖ
      @views Qₖ_temp,Rₖ = qr(reshape(Z_temp[1:y_tt.dims[k],1:rks[k],1:rks[k+1]],x_tt.dims[k]*rks[k],:))
      x_tt.cores[k] = reshape(Matrix(Qₖ_temp),y_tt.dims[k],rks[k],:)
      @views A_temp[1:rks[k+1],1:y_tt.rks[k+1]] = Matrix(Qₖ_temp)'*reshape(Y_temp[1:x_tt.dims[k],1:rks[k],1:y_tt.rks[k+1]],x_tt.dims[k]*rks[k],:) # × Rˣₖ
      @tensoropt((βₖ,αₖ₊₁), Y_temp[1:y_tt.dims[k+1],1:rks[k+1],1:y_tt.rks[k+2]][iₖ₊₁,αₖ,αₖ₊₁] = @view(A_temp[1:rks[k+1],1:y_tt.rks[k+1]])[αₖ,βₖ]*y_tt.cores[k+1][iₖ₊₁,βₖ,αₖ₊₁])
    end
    x_tt.cores[N] = Y_temp[1:y_tt.dims[N],1:rks[N],1:1]
  end
  for k in 1:N-1
    x_tt.ot[k] = 1
  end
  return x_tt
end

"""
Rand rounding of A*y-b
"""

function ttrand_rounding(Atto,y_tt::TTvector{T,N},b_tt;rks=y_tt.rks,rmax=prod(y_tt.dims),orthogonal=true,ℓ=round(Int,0.5*maximum(rks))) where {T,N}
  rks = r_and_d_to_rks(rks.+ℓ,y_tt.dims;rmax=rmax+ℓ) #rks with oversampling
  x_tt = zeros_tt(T,y_tt.dims,rks)
  ℜ_tt = rand_tt(T,y_tt.dims,rks;normalise=true,orthogonal=orthogonal,stable=true)
  Wb_right = partial_contraction(b_tt,ℜ_tt)
  WAy_right = partial_contraction(y_tt,Atto,ℜ_tt)
  Wb_left = zeros(T,maximum(b_tt.rks),maximum(rks))
  Wb_lefttemp = zeros(T,maximum(b_tt.rks),maximum(rks))
  WAy_left = zeros(T,maximum(y_tt.rks),maximum(Atto.rks),maximum(rks))
  WAy_lefttemp = zeros(T,maximum(y_tt.rks),maximum(Atto.rks),maximum(rks))
  Wb_left[1,1] = one(T)
  WAy_left[1,1,1] = one(T)
  Z_temp = zeros(T,maximum(y_tt.dims),maximum(rks),maximum(rks))
  @inbounds begin
    for k in 1:N-1
      Wb_leftview = view(Wb_left, 1:b_tt.rks[k], 1:rks[k])
      WAy_leftview = view(WAy_left, 1:y_tt.rks[k], 1:Atto.rks[k], 1:rks[k])
      @tensor Z_temp[1:y_tt.dims[k],1:rks[k],1:rks[k+1]][iₖ,αₖ₋₁,αₖ] = ((WAy_right[k+1][δₖ₊₁,βₖ₊₁,αₖ]*y_tt.cores[k][jₖ,δₖ,δₖ₊₁])*Atto.cores[k][iₖ,jₖ,βₖ,βₖ₊₁])*WAy_leftview[δₖ,βₖ,αₖ₋₁] -  (Wb_right[k+1][γₖ₊₁,αₖ]*b_tt.cores[k][iₖ,γₖ,γₖ₊₁])*Wb_leftview[γₖ,αₖ₋₁] # nₖ × ℓₖ₋₁ × ℓₖ
      #retrieve TT core Rₖ
      @views Qₖ_temp,Rₖ = qr(reshape(Z_temp[1:y_tt.dims[k],1:rks[k],1:rks[k+1]],x_tt.dims[k]*rks[k],:))
      x_tt.cores[k] = reshape(Matrix(Qₖ_temp),y_tt.dims[k],rks[k],:)
      #update left parts
      Wb_lefttemp[1:b_tt.rks[k],1:rks[k]] = Wb_left[1:b_tt.rks[k],1:rks[k]]
      Wb_leftview = view(Wb_left, 1:b_tt.rks[k+1], 1:rks[k+1])
      @tensor Wb_leftview[βₖ,ωₖ] = Wb_lefttemp[1:b_tt.rks[k],1:rks[k]][βₖ₋₁,ωₖ₋₁] * x_tt.cores[k][iₖ,ωₖ₋₁,ωₖ] * b_tt.cores[k][iₖ,βₖ₋₁,βₖ]

      WAy_lefttemp[1:y_tt.rks[k],1:Atto.rks[k],1:rks[k]] = WAy_left[1:y_tt.rks[k],1:Atto.rks[k], 1:rks[k]]
      WAy_leftview = view(WAy_left, 1:y_tt.rks[k+1], 1:Atto.rks[k+1], 1:rks[k+1])
      @tensor WAy_leftview[ηₖ,αₖ,ωₖ] = WAy_lefttemp[1:y_tt.rks[k],1:Atto.rks[k],1:rks[k]][ηₖ₋₁,αₖ₋₁,ωₖ₋₁] * Atto.cores[k][iₖ,jₖ,αₖ₋₁,αₖ] * y_tt.cores[k][jₖ,ηₖ₋₁,ηₖ] * x_tt.cores[k][iₖ,ωₖ₋₁,ωₖ] 
    end
    #last core
    Wb_leftview = view(Wb_left, 1:b_tt.rks[N], 1:rks[N])
    WAy_leftview = view(WAy_left, 1:y_tt.rks[N], 1:Atto.rks[N], 1:rks[N])
    @tensor Z_temp[1:y_tt.dims[N],1:rks[N],1:rks[N+1]][iₖ,αₖ₋₁,αₖ] = ((WAy_right[N+1][δₖ₊₁,βₖ₊₁,αₖ]*y_tt.cores[N][jₖ,δₖ,δₖ₊₁])*Atto.cores[N][iₖ,jₖ,βₖ,βₖ₊₁])*WAy_leftview[δₖ,βₖ,αₖ₋₁] -  (Wb_right[N+1][γₖ₊₁,αₖ]*b_tt.cores[N][iₖ,γₖ,γₖ₊₁])*Wb_leftview[γₖ,αₖ₋₁] # nₖ × ℓₖ₋₁ × ℓₖ

    x_tt.cores[N] = Z_temp[1:y_tt.dims[N],1:rks[N],1:1]
  end
  for k in 1:N-1
    x_tt.ot[k] = 1
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
  Ψ = zeros.(x.dims,L.rks[1:end-1],R.rks[2:end])
  Ω = zeros.(L.rks[2:end-1],R.rks[2:end-1])
  left_contractions = partial_contraction(x,L;reverse=false)
  right_contractions = partial_contraction(x,R;reverse=true)
  for k in eachindex(Ω)
    @tensor (Ω[k][a,b] = left_contractions[k+1][z,a]*right_contractions[k+1][z,b])
  end
  for k in eachindex(Ψ)
    @tensor (Ψ[k][i,α,β] = (x.cores[k][i,y,z]*left_contractions[k][y,α])*right_contractions[k+1][z,β])
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

function stta(y_tt::TTvector{T,N};rks=vcat(1,round.(Int,1.5*y_tt.rks[2:end-1]),1),rmax=maximum(rks),ℓ=round(Int,maximum(rks)),good=true) where {T,N}
  r_rks = r_and_d_to_rks(rks.+ℓ,y_tt.dims;rmax=rmax+ℓ)
  l_rks = r_and_d_to_rks(round.(Int,1.5*(rks.+ℓ)),y_tt.dims;rmax=round(Int,1.5*(rmax+ℓ)))
  L = rand_tt(y_tt.dims,l_rks;normalise=true,orthogonal=true,right=false,stable=true,good)
  R = rand_tt(y_tt.dims,r_rks;normalise=true,orthogonal=true,stable=true,good)
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
  return TTvector{T,N}(N,Ψ,y_tt.dims,rks,zeros(N))
end

"""
Wrong algorithm
"""
function tt_hmt(y_tt::TTvector{T,N};rks=y_tt.rks,rmax=maximum(rks),ℓ=round(Int,maximum(rks))) where {T,N}
  rks = r_and_d_to_rks(rks.+ℓ,y_tt.dims;rmax=rmax+ℓ)
  Ω = randn.(y_tt.rks[2:end],rks[2:end])
  x_tt = zeros_tt(y_tt.dims,rks)
  y_temp = zeros(maximum(y_tt.dims),maximum(rks),maximum(y_tt.rks))
  y_temp[1:y_tt.dims[1],1:rks[1],1:y_tt.rks[2]] = copy(y_tt.cores[1])
  for k in 1:N-1
    @tensor A_temp[iₖ,αₖ₋₁,βₖ] := @view(y_temp[1:y_tt.dims[k],1:rks[k],1:y_tt.rks[k+1]])[iₖ,αₖ₋₁,αₖ]*Ω[k][αₖ,βₖ]
    q,_ = qr(reshape(A_temp,y_tt.dims[k]*rks[k],:))
    x_tt.cores[k] = reshape(Matrix(q),y_tt.dims[k],rks[k],:)
    rks[k+1] = size(x_tt.cores[k],3)
    R_temp = q'[1:rks[k+1],:]*reshape(y_temp[1:y_tt.dims[k],1:rks[k],1:y_tt.rks[k+1]],y_tt.dims[k]*rks[k],:) #size rks[k+1] × y_tt.rks[k+1]
    @tensor (y_temp[1:y_tt.dims[k+1],1:rks[k+1],1:y_tt.rks[k+2]])[iₖ₊₁,αₖ,αₖ₊₁] = R_temp[αₖ,βₖ]*y_tt.cores[k+1][iₖ₊₁,βₖ,αₖ₊₁]
  end
  x_tt.cores[N] = y_temp[1:y_tt.dims[N],1:rks[N],1:1]
  return TTvector{T,N}(N,x_tt.cores,y_tt.dims,rks,zeros(N))
end