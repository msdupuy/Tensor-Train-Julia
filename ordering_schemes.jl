include("tt_tools.jl")
include("models.jl")

"""
ordering schemes for QC-DMRG or 2D statistical models
"""

#one-orbital rdm
function one_rdm(x_tt::ttvector)
    d = length(x_tt.ttv_dims)
    @assert(2*ones(Int,d)==x_tt.ttv_dims)
    γ = zeros(d,2,2)
    for i in 1:d
        y_tt = mult(one_body_mpo(i,i,d),x_tt)
        γ[i,2,2] = tt_dot(x_tt,y_tt)
        γ[i,1,1] = 1-γ[i,2,2]
    end
    return γ
end

function test_one_rdm()
    C = randn(2,2,2,2)
    C = 1/norm(C)*C
    C_tt =ttv_decomp(C,1)
    γ = one_rdm(C_tt)
    i = rand(1:4)
    A = zeros(2,2)
    A[1,1] = dot(selectdim(C,i,1),selectdim(C,i,1))
    A[2,2] = 1-A[1,1]
    @test(isapprox(γ[i,:,:],A,atol=1e-12))
end

function two_rdm(x_tt::ttvector)
    d = length(x_tt.ttv_dims)
    @assert(2*ones(Int,d)==x_tt.ttv_dims)
    γ = zeros(d,d,2,2,2,2) #(i,j;i,j) occupancy
    for i in 1:d-1
        for j in i+1:d
            γ[i,j,2,2,2,2] = -tt_dot(x_tt,mult(two_body_mpo(i,j,i,j,d),x_tt))
            γ[i,j,1,2,1,2] = -γ[i,j,2,2,2,2] + tt_dot(x_tt,mult(one_body_mpo(j,j,d),x_tt))
            γ[i,j,2,1,2,1] = -γ[i,j,2,2,2,2] + tt_dot(x_tt,mult(one_body_mpo(i,i,d),x_tt))
            γ[i,j,1,1,1,1] = 1.0 -γ[i,j,2,2,2,2] -γ[i,j,2,1,2,1] -γ[i,j,1,2,1,2] 
            γ[i,j,2,1,1,2] = tt_dot(x_tt,mult(one_body_mpo(i,j,d;fermion=false),x_tt))
            γ[i,j,1,2,2,1] = γ[i,j,2,1,1,2]
        end
    end
    return γ
end

function test_two_rdm()
    C = randn(2,2,2,2)
    C = 1/norm(C)*C
    C_tt =ttv_decomp(C,1)
    γ = two_rdm(C_tt)
    i = rand(1:4)
    j = rand(setdiff(1:4,i))
    i,j = min(i,j),max(i,j)
    A = zeros(2,2,2,2)
    A[1,1,1,1] = norm(selectdim(selectdim(C,j,1),i,1))^2
    A[1,2,1,2] = norm(selectdim(selectdim(C,j,2),i,1))^2
    A[2,1,2,1] = norm(selectdim(selectdim(C,j,1),i,2))^2
    A[2,2,2,2] = norm(selectdim(selectdim(C,j,2),i,2))^2
    A[1,2,2,1] = dot(selectdim(selectdim(C,j,2),i,1),selectdim(selectdim(C,j,1),i,2))
    A[2,1,1,2] = A[1,2,2,1]
    #return A,γ[i,j,:,:,:,:]
    @test(isapprox(γ[i,j,:,:,:,:],A,atol=1e-12))
end

function entropy(a,M)
   x = eigvals(M)
   x = x[x.>1e-15]
   if a==Inf
      return -log(maximum(x))
   elseif isapprox(a,1.)
      return -sum(x.*log.(x))
   else
      return 1/(1-a)*log.(sum(x.^a))
   end
end

function mutual_information(γ1,γ2;a=1) #a=1 von neumann entropy
    d = size(γ1,1)
    IM = zeros(d,d)
    s1 = [entropy(a,γ1[i,:,:]) for i in 1:d]
    for i in 1:d-1
        for j in i+1:d
            IM[i,j] = s1[i]+s1[j]-entropy(a,reshape(γ2[i,j,:,:,:,:],4,4))
        end
    end
    return Symmetric(IM)
end

function fiedler(IM)
   L = size(IM)[1]
   Lap = Diagonal([sum(IM[i,:]) for i=1:L]) - IM
   F = eigen(Lap)
   @test isapprox(F.values[1],0.,atol=1e-14)
   return F.vectors[:,2] #to get 2nd eigenvector
end

#best weighted prefactor order
function bwpo_entropy(N,L,V;imax=1000,sigma_current=1:L,sigma_best=sigma_current,temp_chang=0.99,temp_max=1.)
   iter = 0
   x_N = sigma_current[1:N]
   x_diff = setdiff(1:L,x_N)
   prefactor = det(V[1:N,x_N]*Transpose(V[1:N,x_N]))*det(I-V[1:N,x_N]*Transpose(V[1:N,x_N]))
   x = sigma_current
   x_best = sigma_best
   pref_best = det(V[1:N,x_best[1:N]]*Transpose(V[1:N,x_best[1:N]]))*det(I-V[1:N,x_best[1:N]]*Transpose(V[1:N,x_best[1:N]]))
   temp = temp_max
   while iter < imax
      #nouveau voisin
      j = rand(x_N)
      k = rand(x_diff)
      x_temp = sort(vcat(setdiff(x_N,j),k))
      new_prefactor = det(V[1:N,x_temp[1:N]]*Transpose(V[1:N,x_temp[1:N]]))*det(I-V[1:N,x_temp[1:N]]*Transpose(V[1:N,x_temp[1:N]]))
      temp = temp_chang*temp
      if new_prefactor < prefactor
         x_N = x_temp
         x_diff = setdiff(1:L,x_N)
         prefactor = new_prefactor
         if new_prefactor < pref_best
            x_best = vcat(x_N,x_diff)
            pref_best = prefactor
         end
         #etape recuit simule
      elseif exp((prefactor-new_prefactor)/(temp)) > rand()
         x_N = x_temp
         x_diff = setdiff(1:L,x_N)
         prefactor = new_prefactor
      end
      iter = iter+1
   end
   return x_best,vcat(x_N,x_diff)
end