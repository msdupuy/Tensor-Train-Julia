include("tt_tools.jl")
include("gmres.jl") #for qr_hessenberg

"""
TT version of the restarted GMRES algorithm
"""

function history(hist,γ_list,γ)
    if hist
        γ_list = vcat(γ_list,γ)
    end
end

function tt_gmres(A::ttoperator,b::ttvector,x0::ttvector;Imax=500,tol=1e-8,m=30,hist=false,γ_list=Float64[])
    V = Array{ttvector}(undef,m)
    W = Array{ttvector}(undef,m)
    H = zeros(m+1,m)
    r0 = tt_compression_par(tt_add(b,mult_a_tt(-1.0,tt_compression_par(mult(A,x0))))) 
    β = sqrt(tt_dot(r0,r0))
    V[1] = mult_a_tt(1/β,r0)
    W[1] = tt_compression_par(mult(A,V[1]))
    H[1,1] = tt_dot(W[1],V[1])
    W[1] = tt_compression_par(tt_add(W[1],mult_a_tt(-H[1,1],V[1])))
    H[2,1] = sqrt(tt_dot(W[1],W[1]))
    q,r = qr_hessenberg(H[1:2,1])
    γ = abs(β*q[2,1]) #γ = \| Ax_j-b \|_2
    if hist
        γ_list = vcat(γ_list,γ)
    end
    j = 1
    if Imax <=0 || isapprox(H[j+1,j],0.,atol=tol) || isapprox(γ,0,atol=tol)
        if hist
            return x0,γ_list,H[2,1]
        else
            return x0,[γ],H[2,1]
        end
    else
        while j <= min(m-1,Imax) && !isapprox(H[j+1,j],0.,atol=tol) && !isapprox(γ,0,atol=tol)
            V[j+1] = mult_a_tt(1/H[j+1,j],W[j])
            j+=1 
            W[j] = mult(A,V[j])
            W[j] = tt_compression_par(W[j])
            for i in 1:j
                H[i,j] = tt_dot(W[j],V[i])
                W[j] = tt_add(W[j],mult_a_tt(-H[i,j],V[i]))
                W[j] = tt_compression_par(W[j])                        
            end
            H[j+1,j] = sqrt(tt_dot(W[j],W[j]))
            q,r = qr_hessenberg(@view H[1:(j+1),1:j])
            γ = abs(β*q[j+1,1])
            if hist
                γ_list = vcat(γ_list,γ)
            end
        end
        z = r[1:j,1:j]\q[1:j,1]
        for i in 1:j
            x0 = tt_add(x0,mult_a_tt(β*z[i],V[i]))
        end
        x0 = tt_compression_par(x0)
        return tt_gmres(A,b,x0,tol=tol,Imax=Imax-j,m=m,hist=hist,γ_list=γ_list)
    end
end

