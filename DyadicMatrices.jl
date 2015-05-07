module DyadicMatrices
import Base: full, size, intersect, step, range, length, show, one
#import writemime
import Base: getindex, setindex!, copy
import Base: *, /, \, -, +, transpose, ctranspose
import Base: triu, tril, istriu, istril, inv

abstract DyadicRing{K, T} <: AbstractMatrix{T} 

export DyadicVector, SDyadic, Dyadic, DUniformScaling, DyadicRing

export len, lvl, full 
export Grid, grid, gridlc, gridrc, binlog, bincld, randsym
export pickup!, pickup, scalec, scalec!, pickupmu, pickupmu!, pickupSigma, addprecision!, addprecision
export drop!, drop, droppedvar
export backsolve, solve, root!, root, inv!, rootcount

export Sigmareg, mureg, fetransf, hat, evalfe

export one

#=
immutable Consec{O}
    x::NTuple(O
end
consec(x) = Consec(x)
eltype{O}(r::Consec{O}) = O
start(it::Consec) = nothing
next(it::Consec, state) = (it.x, nothing)
done(it::Consec, state) = false
Consec(x, n::Int) = take(Consec(x), n)

=#

immutable Grid{T}
   a::T
   b::T
   Grid(a::T, b::T) = new(min(a,b), max(a,b))
end
step(g::Grid) = g.b - g.a
range{T}(g::Grid{T}) = (typemin(T)-rem(typemin(T)+g.a, step(g))):step(g):typemax(T)


Grid{T}(a::T, b::T) = Grid{T}(a,b)

# ceiling-division
#cld(x,y) = fld(x,y) + (mod(x, y) != 0)

function intersect(g::Grid, r::UnitRange)
    (g.a + cld(start(r)-g.a, step(g))*step(g)):step(g):last(r)
end
intersect(r::UnitRange, g::Grid) = intersect(g, r)

function binlog{T<:Integer}(i::T)
    i > 0 || error(DomainError())
    r = zero(i)
    while 0 != (i = i >> 1)
        r += 1
    end
    r
end
@vectorize_1arg Int binlog

immutable BinaryGrid{T}
   a::T
   l::T
   BinaryGrid(a::T, b::T) = new(a, binlog(abs(b-a)))
   BinaryGrid(a::T, _, l) = new(a, l)
end
BinaryGrid{T}(a::T, b::T) = BinaryGrid{T}(a,b)
BinaryGrid{T}(a::T, _, l::T) = BinaryGrid{T}(a, _, l)
step(g::BinaryGrid) = 1<<(g.l)

#grid(c, l) = BinaryGrid(c - 1<<(l-1), c + 1<<(l-1))
#grid(c, l, L) = l>=L ? BinaryGrid(c, c + 1<<(l-1)) : BinaryGrid(c - 1<<(l-1), c + 1<<(l-1))
grid(c, l) = BinaryGrid(c - 1<<(l-1), 0, l)
grid(c, l, L) = l>=L ? BinaryGrid(c, 0, l-1) : BinaryGrid(c - 1<<(l-1), 0, l)

range{T}(g::BinaryGrid{T}) = (typemin(T)-rem(typemin(T)+g.a, step(g))):step(g):typemax(T)

bincld(x,l) = (x >> l) + (x & (1<<l-1) != 0)
#println([bincld(i, j) for i in -20:20, j in 0:15] - [cld(i, 1<<j) for i in -20:20, j in 0:15])

@inline function intersect(g::BinaryGrid, lo, up)
    ((g.a + (bincld(lo-g.a, g.l) << g.l)):step(g):up)
end

function intersect(g::BinaryGrid, r::UnitRange)
    ((g.a + (bincld(start(r)-g.a, g.l) << g.l)):step(g):last(r))
end
intersect(r::UnitRange, g::BinaryGrid) = intersect(g, r)

grid(c, l, L, r) = intersect(grid(c, l, L), r)
grid(c, l, L, lo, up) = intersect(grid(c, l, L), lo, up)
f(l) = 1 << l-1

function gridlc(c, l, r)
    f = 1<<(l-2)
    g = intersect(grid(c, l, l), (first(r)+f):last(r))
    zip(g, g-f)
end
function gridrc(c, l, r)
    f = 1<<(l-2)
    g = intersect(grid(c, l, l), first(r):(last(r)-f))
    zip(g, g+f)
end



randsym(n) = let a = randn(n,n); a+a' end



immutable DyadicVector{K, T} <: AbstractVector{T}
    x::Vector{T}
    L::Int
end
length(x::DyadicVector) = length(x.x)
size(x::DyadicVector) = (length(x),)
DyadicVector{T}(v::AbstractVector{T}, L, K) = DyadicVector{K,T}(v, L)
#summary(S::AbstractVector) =
getindex(x::DyadicVector, i::Integer) = x.x[i]
copy{K,T}(v::DyadicVector{K,T}) = DyadicVector{K,T}(copy(v.x), v.L)
full(v::DyadicVector) = v.x
-{K,T}(v::DyadicVector{K,T}, w::DyadicVector{K,T}) = DyadicVector{K,T}(v.x-w.x, v.L)
+{K,T}(v::DyadicVector{K,T}, w::DyadicVector{K,T}) = DyadicVector{K,T}(v.x+w.x, v.L)
*{K,T}(v::DyadicVector{K,T}, c::Number) = DyadicVector{K,T}(c*v.x, v.L)
/{K,T}(v::DyadicVector{K,T}, c::Number) = DyadicVector{K,T}(v.x/c, v.L)

immutable SDyadic{K,T} <: DyadicRing{K,T}
    S::Matrix{T}
    L::Int
    uplo::Char
end

immutable DyadicScaling{K,T} <: DyadicRing{K, T}
    x::Vector{T}
    L::Int
end
copy{K,T}(S::DyadicScaling{K,T}) = DyadicScaling{K,T}(copy(S.x), S.L)

immutable DUniformScaling{K,T} <: DyadicRing{K, T}
    x::T
end

one{K,T}(::Type{DyadicRing{K, T}}) = DUniformScaling{K,T}(one(T))




SDyadic{T}(S::AbstractMatrix{T}, L, uplo, K) = SDyadic{K,T}(S, L, uplo)
show(io::IO, Sigma::SDyadic) = show(io, Sigma.S)
copy{K,T}(S::SDyadic{K,T}) = SDyadic{K,T}(copy(S.S), S.L, S.uplo)

function getindex{K,T}(Sigma::SDyadic{K, T}, i::Integer, j::Integer) 
    L = Sigma.L
    c = center(L,K)
    if (i-c)*(j-c) < 0 
        return zero(T)
    end
    bi = digits(i-c,2, L)
    bj = digits(j-c,2, L)
    li = findfirst(bi)
    lj = findfirst(bj)
    
    lm = max(li,lj)+1
    
    if lj != 0 && li != 0 && bi[lm:L] != bj[lm:L]
        return zero(T)
    end
    
    
    if li == 0
        li = L+K
    end
    if lj == 0
        lj = L+K
    end
    if Sigma.uplo != 'U' && li >= lj
        Sigma.S[li, j]
    elseif Sigma.uplo != 'L' && lj >= li
        Sigma.S[lj, i]
    else
        zero(T)
    end
end

immutable Dyadic{K,T} <: DyadicRing{K, T}
    H::Matrix{T}
    V::Matrix{T}
    L::Int
end
copy{K,T}(S::Dyadic{K,T}) = Dyadic{K,T}(copy(S.H), (S.V), S.L)
show(io::IO, Sigma::Dyadic) = show(io, Sigma.V, Sigma.H)

function Dyadic{K, T}(Sigma::SDyadic{K,T}) 
    L = Sigma.L
    n = len(L, K)
    H = zeros(L+K,n)
    V = zeros(L+K,n)
    
    c = center(L, K)
    for l in 1:L+K
        f = 1<<(l-1) - 1
        for i in intersect(grid(c, l, L+K), 1:n)
            for l2 in 1:l-1
                for j in intersect(grid(i, l2), max(1, i - f):min(n, i + f))
                    if Sigma.uplo != 'U'
                        H[l, j] = Sigma.S[l, j]
                    end
                    if  Sigma.uplo != 'L' 
                        V[l, j] = Sigma.S[l, j]
                    end
                end
            end
            V[l, i] = Sigma.S[l, i]
        end
    end
    Dyadic{K, T}(H, V, L)
end


function setindex!{K,T}(Sigma::Dyadic{K, T}, x::T, i::Integer, j::Integer) 
    L = Sigma.L
    c = center(L,K)
    if (i-c)*(j-c) < 0 
        return zero(T)
    end
    bi = digits(i-c,2, L)
    bj = digits(j-c,2, L)
    li = findfirst(bi)
    lj = findfirst(bj)
    
    lm = max(li,lj)+1
    
    if lj != 0 && li != 0 && bi[lm:L] != bj[lm:L]
        return zero(T)
    end
    
    
    if li == 0
        li = L+K
    end
    if lj == 0
        lj = L+K
    end
    if lj < li
        Sigma.H[li, j] = x
    else
        Sigma.V[lj, i] = x
    end
end

"""
    unchecked getindex and setindex
"""
function setindex!{K,T}(Sigma::Dyadic{K, T}, x::T, lii, ljj) 
    li, i = lii
    lj, j = ljj
    if lj < li
        Sigma.H[li, j] = x
    else
        Sigma.V[lj, i] = x
    end
end
function getindex{K,T}(Sigma::Dyadic{K, T},  lii, ljj) 
    li, i = lii
    lj, j = ljj
    if lj < li
        Sigma.H[li, j]
    else
        Sigma.V[lj, i]
    end
end



function getindex{K,T}(Sigma::Dyadic{K, T}, i::Integer, j::Integer) 
    L = Sigma.L
    c = center(L,K)
    if (i-c)*(j-c) < 0 
        return zero(T)
    end
    bi = digits(i-c,2, L)
    bj = digits(j-c,2, L)
    li = findfirst(bi)
    lj = findfirst(bj)
    
    lm = max(li,lj)+1
    
    if lj != 0 && li != 0 && bi[lm:L] != bj[lm:L]
        return zero(T)
    end
    
    
    if li == 0
        li = L+K
    end
    if lj == 0
        lj = L+K
    end
    if lj < li
        Sigma.H[li, j]
    else
        Sigma.V[lj, i]
    end
end



typealias SDyadic0 SDyadic{0}
typealias SDyadic1 SDyadic{1}

size(Sigma::SDyadic) = (len(Sigma),len(Sigma))
size(Sigma::SDyadic, d::Integer) = d<1 ? throw(ArgumentError("dimension must be ≥ 1, got $d")) : (d<=2 ? len(Sigma) : 1)

size(Sigma::Dyadic) = (len(Sigma),len(Sigma))
size(Sigma::Dyadic, d::Integer) = d<1 ? throw(ArgumentError("dimension must be ≥ 1, got $d")) : (d<=2 ? len(Sigma) : 1)

function strides(l, K)
    if K == 0
        return (1 << l, 1 << l, 1 << (l-1), 0 )
    else
        return (1, 1 << l, 1 << (l-1), 1 << l)
    end
end

function strides(l, L, K)
    if K == 0 || l < L
	return ((1 << (l-1)) + K, 1 << l, 1 << (l-1), 0 )
    else
        return (1, 1 << (l-1), 1 << (l-2), 1 << (l-2))
    end
end



#function SDyadicMatrix(A::Matrix)  
len{T}(L::T,K::T) = 1<<L - 1 + 2K
len{K}(Sigma::SDyadic{K}) = 1<<Sigma.L - 1 + 2K
len{K}(Sigma::Dyadic{K}) = 1<<Sigma.L - 1 + 2K


function fs(l) 
    1 << (l-1), 1 << l
end

function localgrid(l, l2)
    f, _ = fs(l)
    f2, s2 = fs(l2)
    -f+f2:s2:f-f2
end

function center(L, K)
    1<<(L-1) + K
end



function DyadicVector(x, K)
    n = length(x)
    L = binlog(n-2K+1)
    n == len(L,K) || error("x is not size(n) with n= 2^L-1+2K")
    DyadicVector(x, L, K)    
end


function SDyadic(A, K)
    issym(A) || error("A is not symmetric")
    n,_ = size(A)
    L = binlog(n-2K+1)
    n == len(L,K) || error("A is not size(n,n) with n= 2^L-1+2K")
    
    S = zeros(L+K,n)
    uplo=' '
    Sigma = SDyadic(S, L, uplo, K)
    
    c = center(L, K)
    for li in 1:L+K
        l = min(li, L)
        f, _ = fs(li)
        for i in intersect(grid(c, li, L+K), 1:n)
            for l2 in 1:li-1
                for j in intersect(grid(i, l2), max(1, i - f):min(n, i + f))
                    S[li, j] = A[i, j]
                end
            end
            S[li,i] = A[i, i]
        end
    end
    
    SDyadic(S, L, uplo, K)
end

tril{K}(Sigma::SDyadic{K}) = SDyadic(Sigma.S, Sigma.L, 'L', K)
triu{K}(Sigma::SDyadic{K}) = SDyadic(Sigma.S, Sigma.L, 'U', K)
istril{K}(Sigma::SDyadic{K}) = Sigma.uplo == 'L'
istriu{K}(Sigma::SDyadic{K}) = Sigma.uplo == 'U'
ctranspose{K}(Sigma::SDyadic{K}) = SDyadic(conj(Sigma.S), Sigma.L, istril(Sigma) ? 'U' : istriu(Sigma) ? 'L' : ' ', K)
transpose{K}(Sigma::SDyadic{K}) = SDyadic(Sigma.S, Sigma.L, istril(Sigma) ? 'U' : istriu(Sigma) ? 'L' : ' ', K)



"""
function lvl(L, K = 0)

    Vector of levels
"""
function lvl(L, K = 0)
    n = len(L, K)
    x = zeros(Int, n)
    c = center(L,K)
    for l in 1:L
        for i in grid(c, l, L, 1:n)
            x[i] = l  
        end
    end
    x
end

"""
function pickup!{K}(x::DyadicVector{K})

      Basis change from finite elements to Schauder basis
"""
function pickup!{K}(v::DyadicVector{K})
    L = v.L
    x = v.x
    n = len(L, K)
    c = center(L, K)
    for l in 2:L
        for i in grid(c, l, l, 1:n)
            x[i] /= 2
        end
        for (i,j) in gridlc(c, l, 1:n)
            x[j] -= x[i]
        end
        for (i,j) in gridrc(c, l, 1:n)
            x[j] -= x[i]
        end
    end
    v
end
pickup(v::DyadicVector) = pickup!(copy(v))

"""
function pickupmu!{K}(x::DyadicVector{K})

      Adjoint basis change from finite elements to Schauder basis
"""
function pickupmu!{K}(v::DyadicVector{K})
    L = v.L
    x = v.x
    n = len(L, K)
    c = center(L, K)
    for l in 2:L
        for i in grid(c, l, l, 1:n)
            x[i] *= 2
        end
        for (i,j) in gridlc(c, l, 1:n)
            x[i] += x[j]
        end
        for (i,j) in gridrc(c, l, 1:n)
            x[i] += x[j]
        end
    end
    v
end
pickupmu(v::DyadicVector) = pickupmu!(copy(v))

"""
function drop!{K}(x::DyadicVector{K})

      Basis change from Schauder basis to finite elements
"""
function drop!{K}(v::DyadicVector{K})
    L = v.L
    x = v.x
    n = len(L, K)
    c = center(L, K)
    for l in L:-1:2
        for (i,j) in gridlc(c, l, 1:n)
            x[j] += x[i]
        end
        for (i,j) in gridrc(c, l, 1:n)
            x[j] += x[i]
        end
        for i in grid(c, l, l, 1:n)
            x[i] *= 2
        end
    end
    v
end
drop(v::DyadicVector) = drop!(copy(v))

"""
function dropmu!{K}(x::DyadicVector{K})

      Adjoint basis change from Schauder basis to finite elements
"""
function dropmu!{K}(v::DyadicVector{K})
    L = v.L
    x = v.x
    n = len(L, K)
    c = center(L, K)
    for l in L:-1:2
        for (i,j) in gridlc(c, l, 1:n)
            x[i] -= x[j]
        end
        for (i,j) in gridrc(c, l, 1:n)
            x[i] -= x[j]
        end
        for i in grid(c, l, l, 1:n)
            x[i] /= 2
        end
    end
    v
end
dropmu(v::DyadicVector) = dropmu!(copy(v))


# returns s0, s, f, sh 
# first element of Ell(L-l), stride, distance to children, shift  

function oldstrides(l, K)
    if K == 0
        return (1 << l, 1 << l, 1 << (l-1), 0 )
    else
        return (1, 1 << l, 1 << (l-1), 1 << l)
    end
end


# for reference: full version
function pickupSigma!(Sigma, L, K)
    n = len(L, K)
    for l in 1:L-1
        s0, s, f, sh = oldstrides(l, K)
        for i in s0:s:n
            Sigma[i, :] *= 2
            Sigma[:, i] *= 2
        end
        for i in s0:s:n-1
            Sigma[i, :] += Sigma[i+f, :]
            Sigma[i+sh, :] += Sigma[i+sh-f, :]
            Sigma[:, i] += Sigma[:, i+f]
            Sigma[:, i+sh] += Sigma[:, i+sh-f]
        end
    end
    Sigma
end

# Reemplementing the previous function in place (Kein Zuckerschlecken)
# Previously
# [1] X )
# [2] ( X )
# [3]   ( X )
# [4]     ( X )
# [5]       ( X )
# [6]         ( X )
# [7]           ( X

# [1] o *
# [2] * X * )
# [3]   * o * *
# [4]   ( * X * )
# [5]     * * o *
# [6]       ( * X *
# [6]           * o


# New alternating format
# [1] X ) X ) X ) X
# [2]   X ) X ) X )

# [1] o   o   o   o
# [2] * X * ) * X *
# [3]     * X * ) *

# [1] o   o   o   o   o   o
# [2] * o *   * o *   * o *
# [3] * * * X * * * ) * X *
# [3]         * * * X * * * )


function pickupSigma{T}(B::SymTridiagonal{T}, K::Int)
    n,_ = size(B)
    L = binlog(n-2K+1)
    n == len(L,K) || error("S is not size(n,n) with n= 2^L-1+2K")
    
    S = zeros(L+K,n)
    for i in 1:n-1
        S[1, i] = B[i,i]
        S[2, i+1] = B[i, i+1]
    end
    S[1,n] = B[n,n]
    uplo = ' '
    pickupSigma!(SDyadic{K,T}(S, L, uplo))
end

function pickupSigma!{K}(Sigma::SDyadic{K})
    S = Sigma.S
    L = Sigma.L
    n = len(L, K)
    for i in 2-K:2:n # bring into alternating format
        S[2, i], S[1, i] = S[1, i], S[2, i]
    end
    for l in 1:L-1
        f = 1<<(l-1)
        s = 1<<l
        s0, s, f, sh = oldstrides(l, K)
        for ii in s0:2s:n
            for eb = 2:-1:1  # first go ahead and handle top row
                if K == 0
                    i = (eb == 1 ? ii : ii + s)
                else
                    i = (eb == 1 ? ii - s : ii)
                end

                # save gap
                lgap =  1 <= i <= n ? S[l, i] : 0.
                rgap = 1 <= i+f <= n ? S[l+1, i+f] : 0.
                eb == 2 && (nlgap = lgap)
                1 <= i <= n || continue

                fi = i==1
                la = i==n

                if eb == 2 # cp center to top row (with rgap)
                    for k in max(i-f+1,1):min(n, i+f)
                        S[l+2, k] = S[l+1, k]
                        S[l+1, k] = 0
                    end
                end
                !fi && (S[l+eb, i-f] = lgap)

                # scale mother
                S[l+eb, i] *= 4
                for k in 1:f
                    !la && (S[l+eb, i+k] *= 2)
                    !fi && (S[l+eb, i-k] *= 2)
                end

                # add children (mind the gap)
                for k in 1:s-1
                    !la && (S[l+eb, i+k] += S[l, i+k])
                    !fi && (S[l+eb, i-k] += S[l, i-k])
                end

                # central elements
                S[l+eb, i] += 4lgap + 4rgap
                !fi && (S[l+eb, i] += S[l, i-f])
                !la && (S[l+eb, i] += S[l, i+f])

                # rightmost element
                if i + s <= n
                    S[l+eb, i+s] = S[l, i+f] + 2rgap + (eb == 2 ?  2S[l, i+s] : 2nlgap)
                end
            end
        end
        for i in s0 : s : n
            S[l, i] = 0
        end
    end
    if K == 1
        c = center(L,K)
        S[L+K, c], S[L, 1] = 0., S[L+K,c]
        for i in 1:n
            S[L, i], S[L+K, i] = S[L+K, i], S[L, i]
        end
    end
    Sigma
end

#%  .. function:: dropSigma!(Sigma, L)
#%                sdropSigma!(Sigma, L)
#%
#%      Nonsparse and non-sparse Schauder-FE-transforms for Sigma.
#%      The sparse version only gives an approximation.


function dropSigma!(Sigma, L)
    n = 2^L-1
    for l in L-1:-1:1
        for i in 2^l:2^l:n
         Sigma[i, :] -= Sigma[i-2^(l-1), :] + Sigma[i+2^(l-1), :]
         Sigma[:, i] -= Sigma[:, i-2^(l-1)] + Sigma[:, i+2^(l-1)]
         Sigma[i, :] /= 2
         Sigma[:, i] /= 2
        end
    end
end

function droppedvar{K,T}(Sigma::SDyadic{K,T})
    assert(istril(Sigma))

    S = Sigma.S
    L = Sigma.L
    n = len(L, K)
    c = center(L, K)

    v = zeros(n)
    z = zeros(n)

    for l in 1:L+K
        f = 1<<(l-1) - 1
        for i in grid(c, l, L+K, 1:n)
            z[:] = 0.
            for l2 in 1:l
                for j in intersect(grid(i, l2,l), max(1, i - f):min(n, i + f))
#                    println("$i($l) $j($l2)")
                
                    k = j
                    u = 1<<(l2-1)
                    for kl in l2:l
#                        println("$i($l) $j($l2) $k($kl) ")
                        # k = f + n s -> k' = f' + n/2 s'
                        z[j] += (1<<(min(kl, L)-1) - abs(k-j))*S[l,k]
                        if K == 0
                            k = (((k- u) >> (kl + 1))<<(kl+1)) + 2u  # index of father element in row
                        else
                            k = k - K + 1<<(L-1) 
                            k = (((k- u) >> (kl + 1))<<(kl+1)) + 2u  # index of father element in row
                            k = k + K - 1<<(L-1)
                        end
                        u <<= 1
                    end
                end
            end
            v[:] += z.^2
        end
    end
    DyadicVector{K,T}(v, L)
end




function scalec!{K}(v::DyadicVector{K}, r, C = one(r))
    L = v.L
    n = len(L, K)
    c = center(L, K)
    for l in 1:L+K
        for i in grid(c, l, L+K, 1:n)
            v.x[i] *= C*2.0^(r*(min(l,L)-1))
        end
    end
    v
end
scalec(x, r, C = one(r)) = scalec!(copy(x), r, C)


function scalec!{K}(Sigma::SDyadic{K}, r, C = one(r))
    S = Sigma.S
    L = Sigma.L
    n = len(L, K)
    c = center(L, K)
    for l in 1:L+K
        f = 1<<(l-1) - 1
        for i in grid(c, l, L+K, 1:n)
            for l2 in 1:l-1
                for j in intersect(grid(i, l2), max(1, i - f):min(n, i + f))
                    S[l, j] *= C*2.0^(r*(min(l,L)-1))
                    if Sigma.uplo == ' ' 
                        S[l, j] *= 2.0^(r*(min(l2, L)-1))
                    end
                        
                end
            end
            S[l, i] *= 2.0^(r*(2min(l,L)-2))*C
        end
    end
    Sigma
end

function addprecision!{K}(Sigma::SDyadic{K}, si, beta)
    L = Sigma.L
    assert(Sigma.uplo == ' ')
    n = len(L, K)
    c = center(L, K)
    for l in 1:L + K
        f = 1<<(l-1) - 1
        for i in grid(c, l, L+K, 1:n)
		    Sigma.S[l,i] += 2. ^ (2*(min(l,L)-1) + 2*beta*(L-min(l,L)+2))/(si^2)
        end
    end
    Sigma
end

addprecision(Sigma, si, beta) = addprecision!(copy(Sigma), si, beta)
function full{K}(Sigma::SDyadic{K}, uplos::Symbol = symbol(Sigma.uplo))
    uplo = string(uplos)[1]
    L = Sigma.L
    n = len(L, K)
    R = zeros(n,n)
    c = center(L, K)
    for l in 1:L+K
        f, _ = fs(l)
        for i in intersect(grid(c, l, L+K), 1:n)
            for j in intersect(max(1, i - f):min(n, i + f))
                    S =  Sigma.S[l, j]
                    if uplo != 'U' 
                        R[i, j] = S
                    end
                    if uplo != 'L'
                        R[j, i] = S
                    end
            end
            R[i, i] = Sigma.S[l, i]
        end
    end
    R
end

function full{K}(Sigma::Dyadic{K})
    L = Sigma.L
    n = len(L, K)
    R = zeros(n,n)
    c = center(L, K)
    for l in 1:L+K
        f = 1 << (l-1) - 1
        for i in intersect(grid(c, l, L+K), 1:n)
            for l2 in 1:l-1
                for j in intersect(grid(i, l2), max(1, i - f):min(n, i + f))
                    R[i, j] = Sigma.H[l,i]
                    R[j, i] = Sigma.V[l,i]
                end
            end
            R[i, i] = Sigma.H[l, i]
        end
    end
    R
end

function *{K}(Sigma::SDyadic{K}, v::DyadicVector{K})
    L = Sigma.L
    n = len(L, K)
    y = zeros(n)
    x = v.x
    uplo = Sigma.uplo
    c = center(L, K)
    for l in 1:L+K
        f = 1<<(l-1) - 1
        for i in intersect(grid(c, l, L+K), 1:n)
            for j in max(1, i - f):min(n, i + f)
                    j == i && continue
                    if uplo != 'U' 
                        y[i] += Sigma.S[l, j]*x[j]
                    end
                    if uplo != 'L'
                        y[j] += Sigma.S[l, j]*x[i]
                    end
            end
            y[i] += Sigma.S[l, i]*x[i]
        end
    end
    DyadicVector(y, L, K)
end



"""
Solving R'y = a using backward substition, where R is lower factor
"""
function backsolve{K}(Sigma::SDyadic{K}, v::DyadicVector{K})
    L = Sigma.L
    n = len(L, K)
    assert(Sigma.uplo == 'U')
    R = Sigma.S
    c = center(L, K)
    x = copy(v.x)
    for l in L+K:-1:1
        f = 1<<(l-1) - 1
        for i in intersect(grid(c, l, L+K), 1:n)
            x[i] /= R[l,i]
            for r in (max(1, i - f):i-1, i+1:min(n, i + f))
                for j in r
                    x[j] -= x[i]*R[l, j]
                end
            end

        end
    end
    DyadicVector(x, L, K)
end



"""
Solving R'y = a using backward substition, where R is lower factor
"""
function backsolve{K}(Sigma::SDyadic{K}, A::Dyadic{K})
    L = Sigma.L
    n = len(L, K)
    assert(Sigma.uplo == 'U')
    
    R = Sigma.S
    c = center(L, K)
    X = copy(A)
    for lp in 1:L+K
        fp = 1<<(lp-1) - 1
        for jp in intersect(grid(c, lp, L+K), 1:n)
            for l in lp:-1:1
                f = 1<<(l-1) - 1
                for i in intersect(grid(jp, l, lp), max(1, jp - fp):min(n, jp + fp)) 
                    X[(l,i), (lp, jp)] /= R[l,i]
                    for l2 in 1:l
                        for j in intersect(grid(i, l2, L+K), max(1, i - f):min(n, i + f) ) 
                            X[(l2,j),(lp, jp)] -= X[(l, i),(lp,jp)]*R[l, j]
                        end
                    end

                end
            end
        end
    end
    X
end
function backsolve{K}(Sigma::SDyadic{K}, A::SDyadic{K})
    L = Sigma.L
    n = len(L, K)
    assert(Sigma.uplo == A.uplo == 'U')
    
    R = Sigma.S
    c = center(L, K)
    X = copy(A)
    for lp in 1:L+K
        fp = 1<<(lp-1) - 1
        for jp in intersect(grid(c, lp, L+K), 1:n)
            for l in lp:-1:1
                f = 1<<(l-1) - 1
                for i in intersect(grid(jp, l, lp), max(1, jp - fp):min(n, jp + fp)) 
                    X.S[l, jp] /= R[l,i]
                    for l2 in 1:l
                        for j in intersect(grid(i, l2, L+K), max(1, i - f):min(n, i + f) ) 
                            X.S[l2, jp] -= X.S[l,jp]*R[l, j]
                        end
                    end

                end
            end
        end
    end
    X
end


backsolve{K}(Sigma::SDyadic{K}, A::SDyadic{K}) =

 backsolve(Sigma, Dyadic(A))





"""
Solving Ry = a using forward substition, where R is upper factor
"""
function solve{K}(Sigma::SDyadic{K}, v::DyadicVector{K})
    L = Sigma.L
    n = len(L, K)
    assert(Sigma.uplo == 'L')
    R = Sigma.S
    c = center(L, K)
    x = copy(v.x)
    for l in 1:L+K
        f = 1<<(l-1) - 1
        for i in intersect(grid(c, l, L+K), 1:n)
            xx, x[i] = x[i], 0. #skip x[i]
            for j in max(1, i - f):min(n, i + f) 
                xx -= x[j]*R[l, j]
            end
            x[i] = xx/R[l,i]
        end
    end
    DyadicVector(x, L, K)
end


#=
"""
Solving Ry = a using forward substition, where R is upper factor
"""
function solve{K}(Sigma::SDyadic{K}, A::SDyadic{K})
    L = Sigma.L
    n = len(L, K)
    assert(Sigma.uplo == 'L')
    R = Sigma.S
    c = center(L, K)
    x = copy(v.x)
    X = copy(A)
    for lp in 1:L+K
        fp = 1<<(lp-1) - 1
        for ip in intersect(grid(c, lp, L+K), 1:n)
            for l in lp:-1:1
                f = 1<<(l-1) - 1
                for i in intersect(grid(ip, l, lp), max(1, ip - fp):min(n, ip + fp)) 
                    X.S[l, i] /= R[l,i]

    
    for l in 1:L+K
        f = 1<<(l-1) - 1
        for i in intersect(grid(c, l, L+K), 1:n)
            xx, x[i] = x[i], 0. #skip x[i]
            for j in max(1, i - f):min(n, i + f) 
                xx -= x[j]*R[l, j]
            end
            x[i] = xx/R[l,i]
        end
    end
    DyadicVector(x, L, K)
end=#


function root!{K, T}(Sigma::SDyadic{K, T})
    L = Sigma.L
    n = len(L, K)
    assert(Sigma.uplo == ' ')
    R = SDyadic{K, T}(Sigma.S, L, 'L')
    c = center(L, K)
    @inbounds for l in 1:L+K
        f = 1<<(l-1) - 1
        for i in intersect(grid(c, l, L+K), 1:n)
            for l2 in 1:l-1
                f2 = 1 << (l2 - 1) - 1
                g = grid(i, l2)
                for j in intersect(g, max(1, i - f), min(n, i + f))
                    som, R.S[l,j] = R.S[l, j], 0.
                    if l2 > 1 
                        @simd for k in max(1,j-f2):min(j+f2,n)
                            som -= R.S[l, k]*R.S[l2, k]
                        end 
                    end
                    R.S[l, j] =  som/R.S[l2, j]
                    R.S[l2, i] = 0.
                end
            end

            som = Sigma.S[l,i]
            R.S[l, i] = 0.
            @simd for k in max(1, i - f):min(n, i + f)
                som -= abs2(R.S[l, k])
            end
            R.S[l, i] = sqrt(som)
        end
    end
    R
end

function rootcount{K, T}(Sigma::SDyadic{K, T})
    L = Sigma.L
    n = len(L, K)
    assert(Sigma.uplo == ' ')
    c = center(L, K)
    muls = divs = sqrts = 0
    for l in 1:L+K
        f = 1<<(l-1) - 1
        for i in intersect(grid(c, l, L+K), 1:n)
            for l2 in 1:l-1
                f2 = 1 << (l2 - 1) - 1
                g = grid(i, l2)
                for j in intersect(g, max(1, i - f), min(n, i + f))
                      if l2 > 1 
                        for k in max(1,j-f2):min(j+f2,n)
                            muls += 1
                        end 
                    end
                    divs += 1
                end
            end

            for k in max(1, i - f):min(n, i + f)
                muls+=1
            end
            sqrts += 1
        end
    end
    muls, divs, sqrts
end

root(Sigma::SDyadic) = root!(copy(Sigma))


function \{K}(Sigma::SDyadic{K}, v::DyadicVector{K})
    if istriu(Sigma) 
        backsolve(Sigma, v)
    elseif istril(Sigma)
        solve(Sigma, v)
    else
        R = root(Sigma)
        R'\(R\v)
    end
end        


#%  .. function:: ssinvRoot!(R, L)
#%
#%      Inverting R in place using the hierarchical version of following algorithm
#%
#%      function cholinv!(a)
#%          n = size(a,1)
#%          for i in n:-1:1
#%              a[i,i] = 1/a[i,i]
#%                 for j in i-1:-1:1
#%                 S = 0.0
#%                 for k in j+1:i
#%                     S += a[k,j]*a[i,k]
#%                     end
#%                 a[i,j] = -S/a[j,j]
#%              end
#%          end
#%          a
#%      end
#%

function inv!{K}(Sigma::SDyadic{K})
    R = Sigma.S
    L = Sigma.L
    n = len(L, K)
    assert(Sigma.uplo == 'L')
    c = center(L, K)
    for l in L+K:-1:1
        f = 1<<(l-1) - 1
        for i in intersect(grid(c, l, L+K), 1:n)
            R[l, i] = 1./R[l, i]
            for l2 in l:-1:1
                f2 = 1<<(l2-1) - 1
                for j in intersect(grid(i, l2), max(1, i - f):min(n, i + f))
                    S  = 0.0
                    k = j
                    u = 1 << (l2 - 1)
                    for kl in l2+1:l
                        if K == 1
                            k += -K +  1<<(L-1)
                        end
                        # k = f + n s -> k' = f' + n/2 s'
                        k = (((k- u) >> (kl-1 + 1))<<(kl)) + 2u  # index of father element in row
                        if K == 1
                            k -= -K + 1<<(L-1)
                        end
                        S += R[kl,j]*R[l,k]
                        u <<= 1
                    end
                    R[l, j] = -S/R[l2, j]
                end
            end
        end
    end
    Sigma
end

function inv(Sigma::SDyadic)
    if istriu(Sigma) 
        inv(copy(Sigma)')
    elseif istril(Sigma)
        inv!(copy(Sigma))
    else
        error("Inverse of Sigma not sparse")
    end
end        


"""
hat(x)

    Hat function. Piecewise linear functions with values (-inf,0), (0,0),(0.5,1), (1,0), (inf,0).
    -- ``x`` vector or number
"""
function hat(x)
    max(1. .- abs(2.*x .- 1.),0.)
end

function fetransf(f, a,b, L, K)
    n::Float64 = 2^L-1
    return map!(f, a .+ [1.-K:n+K;]/(n+1.)*(b-a))
end



function evalfe{K}(tt, theta::DyadicVector{K})
    L = theta.L
    n = 1<<(L-1)  #number of even element/elements in lowest level!
    N = length(tt)
    x = zeros(tt)
    s1 = 1 - K

    for t = 1:N
        tn = tt[t]*n
        i = clamp(int(ceil(tn)), 1, n)
        j = clamp(int(ceil(tn .- 0.5)), s1, n-s1)
        x[t] += theta[2i-s1]*hat(tn .- (i - 1))
        x[t] += theta[2j+1-s1]*hat(tn .- (j - 0.5))
    end
    x
end


"""
    Computes mu' from the observations `y` using ``2^L-1`` basis elements
    and returns a vector with ``K`` trailing zeros (in case one wants to customize
    the basis.)
"""
function mureg(tt,y, si, L, K)
    n = 1<<(L-1) #number of even element/elements in lowest level!
    mu = zeros(2n-1 + 2K)

    for i in 1:n
        mu[2i-1 + K] = dot(hat(tt*n .- (i - 1.)),y)
    end
    for i in 1-K:n-1 + K
        mu[2i + K] = dot(hat(tt*n .- (i -.5)),y)
    end
    mu[:] /= si^2
    DyadicVector(mu, K)
end

function Sigmareg(tt, si, L, K)
    n = 1<<(L-1)
    N = length(tt)

    tg = 1./(si^2)

    Sd = zeros(2n-1+2K)
    St = zeros(2n-1+2K-1)
    

    for t = 1:N
        tn = tt[t]*n
        i = clamp(int(ceil(tn)), 1, n)
        j = clamp(int(ceil(tn - 0.5)), 1-K, n-1+K)
        ii = 2i-1+K
        jj = 2j+K
        Sd[ii] += tg*((hat(tn - (i - 1))^2))
        Sd[jj] += tg*((hat(tn - (j - 0.5))^2))
        St[min(ii,jj)] += tg*(hat(tn - (i - 1))*hat(tn - (j - 0.5)))
    end
    SymTridiagonal(Sd, St)
end

function root!{T}(A::AbstractMatrix{T})
    n = Base.LinAlg.chksquare(A)
    @inbounds begin
        for k = 1:n
            for i = 1:k - 1
                A[k,k] -= A[i,k]'A[i,k]
            end
            A[k,k] = root!(A[k,k])'
            for j = k + 1:n
                for i = 1:k - 1
                    A[k,j] -= A[i,k]'A[i,j]
                end
                    C = backsolve(A[k,k],A[k,j])
                    A[k,j] = C

            end
        end
    end
    A
end

end #module
