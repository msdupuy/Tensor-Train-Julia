include("models.jl")
include("als.jl")
include("ordering_schemes.jl")

using DelimitedFiles

L = 2
w = 4
#h,V = hubbard_2D(w,L,U=2,L_pbc=false,w_pbc=true)
#H_tto = hV_to_mpo(h,V)
#ψ_tt = half_filling(w*L)
#E = Float64[]
#
#for i in [16]
#    global E, ψ_tt = als_eig(H_tto,ψ_tt,sweep_schedule=[2,3],rmax_schedule=[i,i])
##    writedlm("E_w$(w)_L$(L)_R$(i).txt",E)
#end
#
#γ1 = one_rdm(ψ_tt)
#γ2 = two_rdm(ψ_tt)
#sigma_fiedler = fiedler(mutual_information(γ1,γ2))
#site_to_no = eigen(one_prmd(ψ_tt))
sigma_bwpo,prefactor = bwpo_entropy(w*L,2w*L,reverse(site_to_no.vectors,dims=2)[:,1:w*L]',i_cuts=[w*L-2,w*L,w*L+2])