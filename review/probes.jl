using Tatva, SparseArrays, LinearAlgebra, ForwardDiff, StaticArrays, Statistics
println("Review probes on Julia ", VERSION); flush(stdout)

function check_constraints()
    l = Lifter(4, Periodic(2, 1), Periodic(3, 2))
    S = sparse([3], [3], [true], 4, 4)
    P = lifting_matrix(l)
    println("Chained periodic pattern: actual=", Matrix(adapt_sparsity(l, S)), " expected=", Matrix(P' * S * P))
    l2 = Lifter(4, Periodic(2, 1), Periodic([2, 3], [1, 2]))
    P2 = lifting_matrix(l2)
    r = [1., 2., 3., 4.]
    println("Overlapping periodic adjoint: actual=", reduce_adjoint(l2, r), " expected=", P2' * r)
end

function check_projection()
    mesh = unit_square(2, 2; type=:quad)
    op = Operator(mesh, Quad4())
    l = Lifter(num_nodes(mesh), Periodic([7,8,9], [1,2,3]))
    target = map(x -> x[1]^2 + x[2], quads(op))
    energy(u) = integrate(op, (evaluate(op,u) .- target).^2 ./ 2)
    P = lifting_matrix(l)
    M = ForwardDiff.hessian(energy, zeros(num_nodes(mesh)))
    b = -ForwardDiff.gradient(energy, zeros(num_nodes(mesh)))
    expected = P * ((P' * M * P) \ (P' * b))
    actual = project(op, target; lifter=l)
    println("Periodic projection: max error=", maximum(abs, actual-expected), " constrained residual=", norm(P'*(M*actual-b)))
end

function check_alias()
    mesh = unit_square(1,1)
    asm = ElementAssembler(Operator(mesh,Tri3()))
    density(ξ,u) = sum(abs2,u)
    _, _, K = assemble(asm,density,ones(num_nodes(mesh)))
    before = length(asm.pattern.rowval)
    aliases = K.rowval === asm.pattern.rowval && K.colptr === asm.pattern.colptr
    dropzeros!(K)
    println("Assembly structure aliases=", aliases, "; rowval length after dropzeros!: ", before, " -> ", length(asm.pattern.rowval), "; pattern nzval length=", length(asm.pattern.nzval))
end

function check_curved_interpolation()
    el = Tri6()
    X = Matrix(reference_nodes(el))
    # Curved bottom edge y(t) = t*(t-0.4): minimum lies between nodes.
    X[2,4] = 0.05
    X[2,2] = 0.6
    X[2,5] = 0.8
    ξ = SVector(0.2,0.001)
    p = X*shape_function(el,ξ)
    op = Operator(Mesh(X,reshape(collect(1:6),6,1)),el)
    println("Curved point=", p, "; node y min=", minimum(X[2,:]), "; detJ=", last(get_jacobian(el,ξ,X)))
    try
        println("Curved interpolation=", make_interpolate(op,reshape(p,2,1))(X[1,:]))
    catch err
        println("Curved interpolation failed: ", sprint(showerror,err))
    end
end

function check_embedded_quad()
    println("Embedded quad hmax=", hmax(unit_square(1,1; type=:quad,dim=3)), " expected=", sqrt(2))
end

for f in (check_constraints, check_projection, check_alias, check_curved_interpolation, check_embedded_quad)
    try
        f()
    catch err
        showerror(stdout,err,catch_backtrace()); println()
    end
    flush(stdout)
end

function bench(label,f; samples=7)
    f(); f()
    bytes = @allocated f()
    times = [@elapsed f() for _ in 1:samples]
    println(label, ": median_ms=", round(median(times)*1000;digits=3), " bytes=", bytes)
    flush(stdout)
end

function performance()
    for T in (Float64,Float32)
        A=spdiagm(0=>fill(T(2),1000)); b=ones(T,1000)
        bench("spsolve diagonal 1000 $T",()->spsolve(A,b))
    end
    for n in (20,100)
        mesh=unit_square(n,n); op=Operator(mesh,Tri3()); asm=ElementAssembler(op,2)
        u=zeros(2,num_nodes(mesh))
        density(ξ,u,X)=sum(abs2,grad(op.element,ξ,u,X))/2
        energy(u)=integrate(op,sum.(abs2,grad(op,u))./2)
        bench("assemble $n",()->assemble(asm,density,u,mesh.coords))
        bench("assemble_gradient $n",()->assemble_gradient(asm,density,u,mesh.coords))
        bench("assemble_energy $n",()->assemble_energy(asm,density,u,mesh.coords))
        if n==20
            cm=ColoredMatrix(pattern_from_mesh(mesh,2))
            E(v)=energy(reshape(v,2,:))
            gf=jacrev(E); hf=sparse_hessian(E,cm)
            v=vec(u)
            bench("global gradient $n",()->gf(v))
            bench("global Hessian $n",()->hf(v);samples=3)
        end
        l=Lifter(length(u),Fixed(collect(1:2:2*(n+1)),0.))
        _,_,K=assemble(asm,density,u,mesh.coords)
        P=lifting_matrix(l)
        bench("lifting_matrix $n",()->lifting_matrix(l))
        bench("reduce_hessian $n",()->reduce_hessian(l,K))
        bench("cached P reduction $n",()->sparse(P'*K*P))
        @assert K[l.free_dofs,l.free_dofs] ≈ reduce_hessian(l,K)
        bench("Dirichlet-only slicing $n",()->K[l.free_dofs,l.free_dofs])
    end
end
"--correctness-only" in ARGS || performance()
