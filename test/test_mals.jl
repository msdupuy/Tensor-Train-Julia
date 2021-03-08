include("mals.jl")

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

n=6
L = Lap(n,3)
x = L\ones(n^3)

L = reshape(L,n,n,n,n,n,n)
L_tt = tto_decomp(L,1)
b_tt = ttv_decomp(ones(n,n,n),1)
#x_tt = ttv_decomp(randn(n,n,n),1)

x_mals = mals_eig(L_tt,b_tt,tol=1e-12)
