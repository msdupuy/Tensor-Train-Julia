using LinearAlgebra

"""
1d-discrete Laplacian
"""
function Δ(n)
  return Matrix(SymTridiagonal(2ones(n),-ones(n-1)))
end 

"""
n^d discrete Laplacian in TTO format with rank 2 of
H = h ⊗ id ⊗ … ⊗ id + ⋯ + id ⊗ ⋯ ⊗ id ⊗ h
"""
function Δ_tto(n,d;h=[Δ(n) for i in 1:d])
  H_vec = Vector{Array{Float64,4}}(undef,d)
  rks = vcat(1,2ones(Int64,d-1),1)
  # first TTO core
  H_vec[1] = zeros(n,n,1,2)
  H_vec[1][:,:,1,1] = h[1]
  H_vec[1][:,:,1,2] = Matrix{Float64}(I,n,n)
  for i in 2:d-1
    H_vec[i] = zeros(n,n,2,2)
    H_vec[i][:,:,1,1] = Matrix{Float64}(I,n,n)
    H_vec[i][:,:,2,1] = h[i]
    H_vec[i][:,:,2,2] = Matrix{Float64}(I,n,n)
  end
  H_vec[d] = zeros(n,n,2,1)
  H_vec[d][:,:,1,1] = Matrix{Float64}(I,n,n)
  H_vec[d][:,:,2,1] = h[d]
  return TToperator{Float64,d}(d,H_vec,Tuple(n*ones(Int64,d)),rks,zeros(Int64,d))
end

"""
  H = -Δ + ∑ₖ₌₁ʳ sₖ|ϕₖ⟩⟨φₖ|
"""
function perturbed_Δ_tto(n,d;hermitian=true,r=1,rks=ones(Int64,d+1))
  H = Δ_tto(n,d)
  s = randn(r)
  for k in 1:r
    ϕₖ = rand_tt(H.tto_dims,rks)
    ϕₖ = 1/norm(ϕₖ)*ϕₖ
    if hermitian 
      H = H+s[k]*outer_product(ϕₖ,ϕₖ)
    else 
      φₖ = rand_tt(H.tto_dims,rks)
      φₖ = 1/norm(φₖ)*φₖ
      H = H+s[k]*outer_product(ϕₖ,φₖ)
    end
  end
  return H
end

function potential(V::TTvector{T,d}) where {T,d}
  out = zeros_tto(T,V.ttv_dims,V.ttv_rks)
  for k in 1:d
    for iₖ in 1:V.ttv_dims[k]
      for jₖ in 1:V.ttv_dims[k]
        if iₖ == jₖ
          out.tto_vec[k][iₖ,iₖ,:,:] = V.ttv_vec[k][iₖ,:,:]
        end
      end
    end
  end
  return out
end