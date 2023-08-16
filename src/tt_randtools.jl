using TensorOperations
using LinearAlgebra

"""
TT rounding algorithm in https://doi.org/10.1137/21M1451191
Partial contraction defined in Definition 3.1
"""
function partial_contraction(A::TTvector{T},B::TTvector{T}) where T
  @assert A.ttv_dims==B.ttv_dims "TT dimensions are not compatible"
  A_rks = A.ttv_rks
  B_rks = B.ttv_rks
  L = length(A.ttv_dims)
  W = zeros.(T,A_rks,B_rks)
  W[L+1] = ones(T,1,1)
  @inbounds for k in L:-1:1
    @tensoropt((α,β), W[k][a,b] = A.ttv_vec[k][z,a,α]*B.ttv_vec[k][z,b,β]*W[k+1][α,β]) #size R^A_{k} × R^B_{k} 
  end
  return W 
end

"""
TT rounding algorithm in https://doi.org/10.1137/21M1451191
Algorithm 3.2 "Randomize then Orthogonalize"
"""
function ttrand_rounding(y_tt,r)
  L = length(y_tt.ttv_dims)
  x_tt = zeros_tt(y_tt.ttv_dims,r)
  x_tt.ttv_vec[1] = y_tt.ttv_vec[1]
  ℜ_tt = rand_tt(y_tt.ttv_dims,r;normalise=true)
  W = partial_contraction(y_tt,ℜ_tt)
  A_temp = zeros(maximum(r),maximum(y_tt.ttv_rks))
  Y_temp = zeros(maximum(y_tt.ttv_dims),maximum(r),maximum(y_tt.ttv_rks))
  Y_temp[1:y_tt.ttv_dims[1],1:1,1:y_tt.ttv_rks[2]] = y_tt.ttv_vec[1]
  Z_temp = zeros(maximum(y_tt.ttv_dims),maximum(r),maximum(r))
  for k in 1:L-1
    @tensor Z_temp[1:y_tt.ttv_dims[k],1:r[k],1:r[k+1]][iₖ,αₖ₋₁,αₖ] = (Y_temp[1:y_tt.ttv_dims[k],1:r[k],1:y_tt.ttv_rks[k+1]])[iₖ,αₖ₋₁,βₖ]*W[k+1][βₖ,αₖ] # nₖ × ℓₖ₋₁ × ℓₖ
    Qₖ_temp,Rₖ = qr(reshape(Z_temp[1:y_tt.ttv_dims[k],1:r[k],1:r[k+1]],x_tt.ttv_dims[k]*r[k],:))
    x_tt.ttv_vec[k] = reshape(Matrix(Qₖ_temp),y_tt.ttv_dims[k],r[k],:)
    A_temp[1:r[k+1],1:y_tt.ttv_rks[k+1]] = Matrix(Qₖ_temp)'*reshape(Y_temp[1:x_tt.ttv_dims[k],1:r[k],1:y_tt.ttv_rks[k+1]],x_tt.ttv_dims[k]*r[k],:) # × Rˣₖ
    for iₖ₊₁ in 1:y_tt.ttv_dims[k+1]
      Y_temp[iₖ₊₁,1:r[k+1],1:y_tt.ttv_rks[k+2]] = A_temp[1:r[k+1],1:y_tt.ttv_rks[k+1]]*y_tt.ttv_vec[k+1][iₖ₊₁,:,:]
    end
  end
  x_tt.ttv_vec[L] = Y_temp[1:y_tt.ttv_dims[L],1:r[L],1:1]
  return x_tt
end