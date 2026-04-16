include("normalization.jl")
include("normal_bond.jl")

function test_normalisation(;
  L = 10,
  n = 3)
  dims = ntuple(x->n,L)
  rks = ones(Int64,L+1)
  for i in 2:L
      rks[i] = min(n^(i-1),n^(L+1-i),1024)
  end

  #TT tests
  Random.seed!(24) 
  x_tt = rand_tt(dims,rks,normalise=true)
  x = ttv_to_tensor(x_tt)
  hsv = tt_svdvals(x_tt)
  x_v = tt_to_vidal(x_tt)

  #Random.seed!(rand(1:100000))
  ytt, cost_list = ttcore_norm_minimization(x_tt,N=60,X=init_X(x_tt.rks;random=true),diagonalise=true)
  y = ttv_to_tensor(ytt)

  #TR tests
  Random.seed!(1234)
  x_tr = rand_tr(dims,6)
  xr = tr_to_tensor(x_tr)

  Random.seed!()
  #yr, ycost = trcore_norm_minimization(x_tr,N=200,X=init_X(x_tr.rks;random=true))
  #yr_tensor = tr_to_tensor(yr)
  return x_tt,x,ytt
end

function test_normal_bond(;
   L = 8,
  n = 3)
  dims = ntuple(x->n,L)
  rks = ones(Int64,L+1)
  for i in 2:L
      rks[i] = min(n^(i-1),n^(L+1-i),1024)
  end

  #TT tests
  Random.seed!() 
  x_tt = rand_tt(dims,rks,normalise=true)
  x = ttv_to_tensor(x_tt)
  hsv = tt_svdvals(x_tt)
  x_v = tt_to_vidal(x_tt)

  #Random.seed!(rand(1:100000))
  ytt, cost_list = bond_cost_als(x_tt,N=1000,X=init_X(x_tt.rks))
  y = ttv_to_tensor(ytt)

  Random.seed!(2604)
  #yr, ycost = trcore_norm_minimization(x_tr,N=200,X=init_X(x_tr.rks;random=true))
  #yr_tensor = tr_to_tensor(yr)
  return x_tt,x,ytt,y
  #return x_tt,x
end