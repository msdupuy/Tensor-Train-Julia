using TensorTrains

function hubbard_2D_arnoldi(w,L;rmax=128,m=3,ε=1e-6,ε_tt=1e-4,Imax=10,r=2,which=:SM,σ=convert(Float64,w*L+2))
  h,V = hubbard_2D(w,L;U=1)
  dims = ntuple(x->2,2w*L)
  v = rand_tt(dims,vcat(1,r*ones(Int64,2w*L-1),1);normalise=true)
  H_tto = hV_to_mpo(h,V,dims,tol=1e-5)
  E,ψ_tt =eig_arnoldi(H_tto,m,v;Imax=Imax,ε=ε,ε_tt=ε_tt,rmax=rmax,which=which,σ=σ) 
  return E,ψ_tt
end