include("als.jl")
using Plots

function Lap(n::Integer,d::Integer) #returns the tensor of the discrete Laplacian in a box [0,1]^d with n equidistant discretization points in each direction
    A = zeros(n^d,n^d)    
    for j in 0:n^d-1
        J = digits(j,base=n,pad=d) #corresponding index as d-tuples
        for k in 0:d-1
            if J[k+1]==0
                A[j+1,j+1] = 2*d
                A[j+1,j+n^k+1] = -1
            elseif J[k+1]==n-1
                A[j+1,j+1] = 2*d
                A[j+1,j-n^k+1] = -1
            else
                A[j+1,j+1] = 2*d
                A[j+1,j-n^k+1] = -1
                A[j+1,j+n^k+1] = -1
            end
        end
    end
    return A
end

function test_als()
    n = 10
    L = Lap(n,3)
    b = ones(n,n,n)
    x0 = randn(n,n,n)
    L_tt = tto_decomp(reshape(L,n,n,n,n,n,n),1)
    b_tt = ttv_decomp(b,1)
    x0_tt = ttv_decomp(x0,1)
    x = L\b[:]
    x_tt = als(L_tt,b_tt,x0_tt)
    y = ttv_to_tensor(x_tt)
    @test isapprox(y[:],x)
end

function test_als_eig()
    n = 10
    L = Lap(n,3)
    E = eigmin(L)
    x0 = randn(n,n,n)
    L_tt = tto_decomp(reshape(L,n,n,n,n,n,n),1)
    x0_tt = ttv_decomp(x0,1)
    E_tt,x_tt = als_eig(L_tt,x0_tt)
    y = ttv_to_tensor(x_tt)
    @test isapprox(E,E_tt)
    E_tt,x_tt = als_eig(L_tt,x0_tt,it_solver=true)
    @test isapprox(E,E_tt,atol=1e-8)
end

function test_als_gen_eig()
    n = 5
    L = randn(n^3,n^3)
    L = L*L'+I
    S = randn(n^3,n^3)
    S = S*S'+I
    F = eigen(L,S)
    x0 = randn(n,n,n)
    L_tt = tto_decomp(reshape(L,n,n,n,n,n,n),1)
    S_tt = tto_decomp(reshape(S,n,n,n,n,n,n),1)
    x0_tt = ttv_decomp(x0,1)
    E_tt,x_tt = als_gen_eig(L_tt,S_tt,x0_tt)
    y = ttv_to_tensor(x_tt)
    @test isapprox(real(F.values[1]),E_tt)
    E_tt,x_tt = als_gen_eig(L_tt,S_tt,x0_tt,it_solver=true)
    @test isapprox(real(F.values[1]),E_tt,atol=1e-8)
end