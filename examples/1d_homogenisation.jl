using TensorTrains
using Plots

function homogenisation_1d_tto(δ,d)
  Llow = toeplitz_to_qtto(1,0,-1,d)
  Lup = toeplitz_to_qtto(1,-1,0,d)
  h = 1/(2^d+1)
  ap = 2/3*qtt_polynom([1,1],d;a=h,b=1.0-h)*(1.0+qtt_cos(d;a=h,b=1.0-h,λ=1/δ)*qtt_cos(d;a=h,b=1.0-h,λ=1/δ))
  am =  2/3*qtt_polynom([1,1],d;a=0,b=1.0-2h)*(1.0+qtt_cos(d;a=0,b=1.0-2h,λ=1/δ)*qtt_cos(d;a=0,b=1.0-2h,λ=1/δ))
  A_tto = 1/h*Llow*potential(ap)+1/h*Lup*potential(am)
  overlap = h*toeplitz_to_qtto(4,1,1,d)
  f_rhs = overlap*ones_tt(2,d)
  return A_tto, tt_rounding(f_rhs)
end

"""
Sanity check
A_tto, f_tt = homogenisation_1d_tto(0.1,6)
A6 = qtto_to_matrix(A_tto)
f6 = qtt_to_function(f_tt)
u6 = A6\f6
A_tto, f_tt = homogenisation_1d_tto(0.1,8)
A8 = qtto_to_matrix(A_tto)
f8 = qtt_to_function(f_tt)
u8 = A8\f8
plot(range(1/2^6,1-1/2^6,length=2^6),u6,label="d=6")
plot!(range(1/2^8,1-1/2^8,length=2^8),u8,label="d=8")
"""

A_tto, f_tt = homogenisation_1d_tto(0.1,10)
dmrg_schedule = dmrg_schedule_default(;N=2,rmax=16,nsweeps=2,it_solver=true)
u_dmrg = dmrg_linsolv(A_tto,f_tt,deepcopy(f_tt),schedule=dmrg_schedule)
#u_gmres = tt_gmres(A_tto,f_tt,u_dmrg;Imax=100,tol=1e-8,m=20,hist=false,rmax=256)
α_opt = 1/(2500*4+0.05/4) #works only for δ=0.1

@time x_tt, res_ϵ = gradient_fixed_step(A_tto,f_tt,α_opt,Imax=50,i_trunc=1,eps_tt=1e-4,rand_rounding=false,r_tt=16,ℓ=8)
@time x_rr_tt, res_rr_ϵ = gradient_fixed_step(A_tto,f_tt,α_opt,Imax=50,i_trunc=1,eps_tt=1e-4,rand_rounding=true,r_tt=16,ℓ=8)
scatter(res_ϵ, yscale=:log10, label="TT rounding")
scatter!(res_rr_ϵ, yscale=:log10, label="Rand TT rounding")