using LinearAlgebra

"""
1d-discrete Laplacian
"""
function Δ(n)
  return Matrix(SymTridiagonal(2ones(n),-ones(n-1)))
end 

"""
n^d discrete Laplacian in TTO format with rank 2
"""
function Δ_tto(n,d)
  h = Δ(n)
  H_vec = Array{Array{Float64,4}}(undef,d)
  rks = vcat(1,2ones(Int64,d-1),1)
  # first TTO core
  H_vec[1] = zeros(n,n,1,2)
  H_vec[1][:,:,1,1] = h
  H_vec[1][:,:,1,2] = Matrix{Float64}(I,n,n)
  for i in 2:d-1
    H_vec[i] = zeros(n,n,2,2)
    H_vec[i][:,:,1,1] = Matrix{Float64}(I,n,n)
    H_vec[i][:,:,2,1] = h
    H_vec[i][:,:,2,2] = Matrix{Float64}(I,n,n)
  end
  H_vec[d] = zeros(n,n,2,1)
  H_vec[d][:,:,1,1] = Matrix{Float64}(I,n,n)
  H_vec[d][:,:,2,1] = h
  return TToperator(d,H_vec,n*ones(Int64,d),rks,zeros(Int64,d))
end