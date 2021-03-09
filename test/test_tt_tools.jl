using Test

@testset "tt_up_rks" begin
	x = randn(4,4,4,4)
	x_tt = ttv_decomp(x,2,tol=0.1)
	x = ttv_to_tensor(x_tt)
	y_tt = tt_up_rks(x_tt,20)
	@test isapprox(x,ttv_to_tensor(y_tt),atol=1e-10)
	z_tt = tt_up_rks(x_tt,20,ϵ_wn=1e-10)
	@test isapprox(x,ttv_to_tensor(z_tt),atol=1e-6)
end


@testset "TT operations" begin
    n = 3
    d = 3
    L = randn(n,n,n,n,n,n)
    S = randn(n,n,n,n,n,n)
    L_tto = tto_decomp(L,1)
    S_tto = tto_decomp(S,1)
    x = randn(n,n,n)
    y = randn(n,n,n)
    x_tt = ttv_decomp(x,1)
    y_tt = ttv_decomp(y,1)
    a = randn()
    #mult_a_tt
    @test isapprox(a.*x, ttv_to_tensor(mult_a_tt(a,x_tt)))
    #tt_add
    z_tt = tt_add(x_tt,y_tt)
    @test(isapprox(ttv_to_tensor(z_tt),x+y)) 
    #tt_dot
    @test isapprox(x[:]'*(x+y)[:], tt_dot(x_tt,z_tt))
    #tto_add
    A_tto = tto_add(L_tto,S_tto)
    @test(isapprox(tto_to_tensor(A_tto),L+S)) #tto_add test
    #tto*ttv multiplication
    z = reshape(L,n^d,:)*x[:]
    z_tt = mult(L_tto,x_tt)
    @test(isapprox(ttv_to_tensor(z_tt)[:],z))
    #tto*tto multiplication
    A = reshape(L,n^d,:)*reshape(S,n^d,:)
    A_tto = mult(L_tto,S_tto)
    @test(isapprox(reshape(tto_to_tensor(A_tto),n^d,:),A))
end

@testset "TT rounding" begin
    n=5
    d=3
    L = randn(n,n,n,n,n,n)
    x = randn(n,n,n)
    y = reshape(L,n^d,:)*x[:]
    L_tt = tto_decomp(reshape(L,n*ones(Int,2d)...),1)
    x_tt = ttv_decomp(x,1)
    y_tt = mult(L_tt,x_tt)
    @test(isapprox(ttv_to_tensor(tt_compression_par(y_tt))[:],y))
end