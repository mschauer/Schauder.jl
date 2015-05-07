#%  .. currentmodule:: Schauder
#%
#%  .. _modschauder:
#%
#%  Module Schauder
#%  ---------------
#%
#%  Introduction
#%  ~~~~~~~~~~~~
#%
#%  In the following ``hat(x)`` is the piecewise linear function taking values
#%  values ``(0,0), (0.5,1), (1,0)`` on the interval ``[0,1]`` and ``0`` elsewhere.
#%
#%  The Schauder basis of level ``L > 0`` in the interval ``[a,b]`` can be defined recursively
#%  from ``n = 2^L-1`` classical finite elements :math:`\psi_i(x)` on the grid
#%
#%  	``a + (1:n)/(n+1)*(b-a)``.
#%
#%
#%  Assume that f is expressed as linear combination
#%
#%  	:math:`f(x) = \sum_{i =1}^n c_i \psi_i(x)`
#%
#%  with
#%
#%  	:math:`\psi_{2j-1}(x) = hat(nx-j + 1)` 	for :math:`j = 1 \dots 2^{L-1}`
#%
#%  and
#%
#%  	:math:`\psi_{2j}(x) = hat(nx-j + 1/2)` 	for :math:`j = 1 \dots 2^{L-1}-1`
#%
#%  Note that these coefficients are easy to find for the finite element basis, just
#%
#%  .. code-block:: matlab
#%
#%  	function fe_transf(f, a,b, L)
#%  	    n = 2^L-1
#%  	    return map(f, a + (1:n)/(n+1)*(b-a))
#%  	end
#%

#%
#%  Then the coefficients of the same function with respect to the Schauder basis
#%
#%  	:math:`f(x) = \sum_{i = 1}^n c_i \phi_i(x)`
#%
#%  where for ``L = 2``
#%
#%  	:math:`\phi_2(x) = 2 hat(x)`
#%
#%  	:math:`\phi_1(x) = \psi_1(x) = hat(2x)`
#%
#%  	:math:`\phi_3(x) = \psi_3(x) = hat(2x-1)`
#%
#%  can be computed directly, but also using the recursion
#%
#%  	:math:`\phi_2(x) = \psi_1(x) + 2\psi_2(x) + \psi_2(x)`
#%
#%  This can be implemented inplace (see ``pickup()``), and is used throughout.
#%

#%
#% .. code-block:: julia
#%
#%  	for l in 1:L-1
#%  		b = sub(c, 2^l:2^l:n)
#%  		b[:] *= 0.5
#%  		a = sub(c, 2^(l-1):2^l:n-2^(l-1))
#%  		a[:] -= b[:]
#%  		a = sub(c, 2^l+2^(l-1):2^l:n)
#%  		a[:] -= b[1:length(a)]
#%  	end
#%
import Base.reverse!, Base.strides

#using Winston
using Distributions
normalcdf(x) = 0.5 + 0.5erf(x/sqrt(2))
normalquant(p) = sqrt(2)*erfinv(2*p-1)

function reverse!(a::Matrix)
    reshape(reverse!(a[:]), size(a)...)
end

displayln0(a...) = begin
    display(a...)
    println()
end
displayln = displayln0

L2n(L) = 1<<L - 1

unit(i, n) = begin x = zeros(n); x[i] = 1; x end

# returns s0, s, f, sh 
# first element of Ell(L-l), stride, distance to children, shift  

function strides(l, K)
    if K == 0
        return (1 << l, 1 << l, 1 << (l-1), 0 )
    else
        return (1, 1 << l, 1 << (l-1), 1 << l)
    end
end

function strides(l,L, K)
    if K == 0 || l < L
    	return ((1 << (l-1)) + K, 1 << l, 1 << (l-1), 0 )
    else
        return (1, 1 << (l-1), 1 << (l-2), 1 << (l-2))
    end
end

len(L, K) = 1 << L - 1 + 2K


# diagonal of inverse of triangular matrix

function diaginv(r)
    n = size(r,1)
    v = zeros(n)
    for i in 1:n
        for j in i:n
            v[i] += r[j,i]*r[j,i]
        end
    end
    v
end



#%  .. function:: pickup!(x)
#%
#%      Basis change finite elements to Schauder basis (1d)
#%

#%  .. function:: drop!(x)
#%
#%      Basis change from Schauder basis to finite elements (1d)
#%

function pickup!(x, L, K = 0)
    n = len(L, K)
    for l in 1:L-1
        s0, s, f, sh = strides(l, K)
        for i in s0:s:n
            x[i] /= 2
        end
        for i in s0:s:n-1
            x[i+f] -= x[i]
            x[i+sh-f] -= x[i+sh]
        end
    end
    x
end
pickup(x, L, K = 0) = pickup!(copy(x),L, K)


function lvl(L, K = 0)
    n = len(L, K)
    x = zeros(Int, n)
    for l in 1:L
        s0, s, f, sh = strides(l, L, K)
        for i in s0:s:n
            x[i] = l  
        end
    end
    x
end


function scalec!(x, r, L, K = 0)
    n = len(L, K)
    for l in 1:L
        s0, s, f, sh = strides(l, L, K)
        for i in s0:s:n
            x[i] *= 2.0^(r*(l-1))
        end
    end
    x
end
scalec(x, r, L, K = 0) = scalec!(copy(x), r, L, K)



function drop!(x, L, K = 0)
    n = len(L, K)
    for l in L-1:-1:1
        s0, s, f, sh = strides(l, K)
        for i in s0:s:n-1
            x[i+f] += x[i]
            x[i+sh-f] += x[i+sh]
        end
        for i in s0:s:n
            x[i] *= 2
        end
    end
    x
end
drop(x,L, K = 0) = drop!(copy(x),L, K)


#%  .. function:: pickupmu!(mu, L, K)
#%
#%      computes mu in schauder basis from mu respective finite elements
#%


#%  .. function:: dropmu!(mu, L, K)
#%
#%      Reverse of pickupmu!(mu, L¸K)
#%


function pickupmu!(x, L, K = 0)
    n = len(L, K)
    for l in 1:L-1
        s0, s, f, sh = strides(l, K)
        for i in s0:s:n
            x[i] *= 2
        end
        for i in s0:s:n-1
            x[i] += x[i+f]
            x[i+sh] += x[i+sh-f]
        end
    end
    x
end
pickupmu(x,L,K = 0) = pickupmu!(copy(x),L, K)

function dropmu!(x, L, K = 0)
    n = len(L, K)
    for l in L-1:-1:1
        s0, s, f, sh = strides(l, K)
        for i in s0:s:n-1
            x[i] -= x[i+f]
            x[i+sh] -= x[i+sh-f]
        end
        for i in s0:s:n
            x[i] /= 2
        end

    end
    x
end
dropmu(x,L,K = 0) = dropmu!(copy(x),L, K)

#%  .. function:: hat(x)
#%
#%      Hat function. Piecewise linear functions with values (-inf,0), (0,0),(0.5,1), (1,0), (inf,0).
#%      -- ``x`` vector or number
function hat(x)
    max(1. .- abs(2.*x .- 1.),0.)
end

function fetransf(f, a,b, L, K = 0)
    n::Float64 = 2^L-1
    return map!(f, a .+ [1.-K:n+K]/(n+1.)*(b-a))
end

function ferot(theta, si, L, K = 0)
    n::Float64 = 2^L-1
    evalfe((si + [1.-K:n+K]/(n+1.)) .% 1.0  , theta, L, K)
end


function evalfe(tt, theta, L, K = 0)
    n = 1<<(L-1)
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

#%  .. function:: musde(y, L, K)
#%
#%      Computes mu' from SDE observations `y` using ``2^L-1`` basis elements
#%      and returns a vector with ``K`` trailing zeros (in case one wants to customize
#%      the basis.)


function musde(y, L, K=0)
    n = 1<<(L-1) #number of even element/elements in lowest level!
    mu = zeros(2n-1 + 2K)
    dy = [diff(y),0]

    for i in 1:n
        mu[2i-1+K] = dot(hat(y*n .- (i - 1)),dy)
    end
    for i in 1-K:n-1+K
        mu[2i+K] = dot(hat(y*n .- (i -.5)),dy)
    end
    mu
end

#%  .. function:: mureg(y, L, K)
#%
#%      Computes mu' from the observations `y` using ``2^L-1`` basis elements
#%      and returns a vector with ``K`` trailing zeros (in case one wants to customize
#%      the basis.)


function mureg(tt,y, si, L, K = 0)
    n = 1<<(L-1) #number of even element/elements in lowest level!
    mu = zeros(2n-1 + 2K)

    for i in 1:n
        mu[2i-1 + K] = dot(hat(tt*n .- (i - 1.)),y)
    end
    for i in 1-K:n-1 + K
        mu[2i + K] = dot(hat(tt*n .- (i -.5)),y)
    end
    mu[:] /= si^2
end


function sfull(Sigma, L, lower = :L)
    n = L2n(L)
    R = zeros(n,n)
    for l in 1:L
        f = 1<<(l-1)
        s = 1<<l
        for i in f:s:n
#            println("[[$i]]")
            for l2 in 1:l-1
                s2 = 1<<l2
                f2 = 1<<(l2-1)
                for j in -f+f2:s2:f-f2
                    R[i, i+j] =  Sigma[l, i+j]
                    if (lower != :L)  R[i+j, i] = R[i, i+j] end
                end
            end
            R[i, i] = Sigma[l, i]
        end
    end
    R
end


#%  .. function:: Sigmareg(y, L, K)
#%                sSigmareg(y, L, K)
#%
#%      Computes Sigma' from the observations `y` using ``2^L-1`` basis elements
#%      The vector y is traversed only once. sSigmareg computes a sparse representation.

function Sigmareg(tt, si, L, K = 0)
    n = 1<<(L-1)
    N = length(tt)

    tg = 1./(si^2)

    S = zeros(2n-1+2K, 2n-1+2K)
    s = 1-K
    for t = 1:N
        tn = tt[t]*n
        i = clamp(int(ceil(tn)), 1, n)
        j = clamp(int(ceil(tn - 0.5)), 1-K, n-1+K)
        ii = 2i-1+K
        jj = 2j+K
        S[ii, ii] += tg*((hat(tn - (i - 1))^2))
        S[jj, jj] += tg*((hat(tn - (j - 0.5))^2))
        S[ii, jj] = S[jj, ii] += tg*(hat(tn - (i - 1))*hat(tn - (j - 0.5)))
    end
    S
end

function sSigmareg(tt, si, L, K = 0)
    n = 1<<(L-1)
    N = length(tt)

    tg = 1./(si^2)

    S = zeros(L+K, 2n-1+2K)

    for t = 1:N
        tn = tt[t]*n
        i = clamp(int(ceil(tn)), 1, n)
        j = clamp(int(ceil(tn - 0.5)), 1-K, n-1+K)
        ii = 2i-1+K
        jj = 2j+K
        S[1, ii] += tg*((hat(tn - (i - 1))^2))
        S[1, jj] += tg*((hat(tn - (j - 0.5))^2))
        S[2, 1+min(ii,jj)] += tg*(hat(tn - (i - 1))*hat(tn - (j - 0.5)))
    end
    S
end


function Sigmasde(y, dt::Float64, L)
    n = 2^(L-1)
    N = length(y)

    S = zeros(2n-1, 2n-1)

    for t = 1:N
        yn = y[t]*n
        i = clamp(ceil(yn), 1, n)
        j = clamp(ceil(yn .- 0.5), 1, n-1)
        S[2i-1, 2i-1] += ((hat(yn .- (i - 1)).^2))*dt
        S[2j, 2j] += ((hat(yn .- (j - 0.5)).^2))*dt
        S[2i-1, 2j] = S[2j, 2i-1] += (hat(yn .- (i - 1)).*hat(yn .- (j - 0.5)))*dt
    end
    S
end

#%  .. function:: pickupSigma!(Sigma, L)
#%                spickupSigma!(Sigma, L)
#%
#%      Nonsparse and sparse FE-Schauder-transforms for Sigma
#%


function pickupSigma!(Sigma, L, K)
    n = len(L, K)
    for l in 1:L-1
        s0, s, f, sh = strides(l, K)
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


function spickupSigma!(Sigma, L)
    K = 0
    n = len(L, K)
    for i in 2:2:n # bring into alternating format
        Sigma[2, i], Sigma[1, i] = Sigma[1, i], Sigma[2, i]
    end
    for l in 1:L-1
        f = 1<<(l-1)
        s = 1<<l
        for ii in s:2s:n
            for eb = 2:-1:1  # first go ahead and handle top row
                i = eb == 1 ? ii : ii + s
                # save gap
                lgap = i <= n ? Sigma[l, i] : 0.
                rgap = i+f <= n ? Sigma[l+1, i+f] : 0.
                eb == 2 && (nlgap = lgap)
                i >= n && continue

                if eb == 2 # cp center to top row (with rgap)
                    for k in i-f+1:i+f
                        Sigma[l+2, k] = Sigma[l+1, k]
                        Sigma[l+1, k] = 0
                    end
                end
                Sigma[l+eb, i-f] = lgap

                # scale mother
                Sigma[l+eb, i] *= 2
                for k in i-f:i+f
                    Sigma[l+eb, k] *= 2
                end

                # add children (mind the gap)
                for k in 1:s-1
                    Sigma[l+eb, i+k] += Sigma[l, i+k]
                    Sigma[l+eb, i-k] += Sigma[l, i-k]
                end

                # central elements
                Sigma[l+eb, i] += (Sigma[l, i-f] + Sigma[l, i+f]) + 4lgap + 4rgap

                # rightmost element
                if i + s <= n
                    Sigma[l+eb, i+s] = Sigma[l, i+f] + 2rgap + (eb == 2 ?  2Sigma[l, i+s] : 2nlgap)
                end
            end
        end
        for i in (1<<l) : (1<<l) : n
            Sigma[l, i] = 0
        end
    end
    Sigma
end


# TODO: traverse lines as in other functions
function spickupSigma!(Sigma, L, K)
    n = len(L, K)
    for i in 2-K:2:n # bring into alternating format
        Sigma[2, i], Sigma[1, i] = Sigma[1, i], Sigma[2, i]
    end
    for l in 1:L-1
        f = 1<<(l-1)
        s = 1<<l
        s0, s, f, sh = strides(l, K)
        for ii in s0:2s:n
            for eb = 2:-1:1  # first go ahead and handle top row
                if K == 0
                    i = (eb == 1 ? ii : ii + s)
                else
                    i = (eb == 1 ? ii - s : ii)
                end

                # save gap
                lgap =  1 <= i <= n ? Sigma[l, i] : 0.
                rgap = 1 <= i+f <= n ? Sigma[l+1, i+f] : 0.
                eb == 2 && (nlgap = lgap)
                1 <= i <= n || continue

                fi = i==1
                la = i==n

                if eb == 2 # cp center to top row (with rgap)
                    for k in max(i-f+1,1):min(n, i+f)
                        Sigma[l+2, k] = Sigma[l+1, k]
                        Sigma[l+1, k] = 0
                    end
                end
                !fi && (Sigma[l+eb, i-f] = lgap)

                # scale mother
                Sigma[l+eb, i] *= 4
                for k in 1:f
#                    println(i, " ", n)
                    !la && (Sigma[l+eb, i+k] *= 2)
                    !fi && (Sigma[l+eb, i-k] *= 2)
                end

                # add children (mind the gap)
                for k in 1:s-1
                    !la && (Sigma[l+eb, i+k] += Sigma[l, i+k])
                    !fi && (Sigma[l+eb, i-k] += Sigma[l, i-k])
                end

                # central elements
                Sigma[l+eb, i] += 4lgap + 4rgap
                !fi && (Sigma[l+eb, i] += Sigma[l, i-f])
                !la && (Sigma[l+eb, i] += Sigma[l, i+f])

                # rightmost element
                if i + s <= n
                    Sigma[l+eb, i+s] = Sigma[l, i+f] + 2rgap + (eb == 2 ?  2Sigma[l, i+s] : 2nlgap)
                end
            end
        end
        for i in s0 : s : n
            Sigma[l, i] = 0
        end
    end
    if K == 1
        c = 1<<(L-1) + K
        Sigma[L+K, c], Sigma[L, 1] = 0., Sigma[L+K,c]
    end

    Sigma
end



function sdropSigma(Sigma, L)
    Sigma = copy(Sigma)
    n = 2^L-1
    for l in L-1:-1:L2
        f = 1<<(l-1)
        s = 1<<l
        for eb = 1:2
            rng = eb == 1 ? (s:2s:n ) : (s+f:2s:n-f)
            for i in rng
                Sigma[l+eb, i] -= (Sigma[l+eb, i-f] + Sigma[l+eb, i+f])
                for k in i-s+1:i+s-1
                    Sigma[l+eb, k] -= Sigma[l, k]
                end
                Sigma[l+eb, i] -= Sigma[l+eb, i-f] + Sigma[l+eb, i+f]
                for k in i-s+1:i+s-1
                    Sigma[l+eb, k] /= 2
                end
                Sigma[l+eb, i] /= 2
            end
        end

    end
    for i in 2-K:2:n
        Sigma[2, i], Sigma[1, i] = Sigma[1, i], Sigma[2, i]
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

#% TODO, only marginals are correct

function sdropSigma(Sigma, L)
    Sigma = copy(Sigma)
    n = 2^L-1
    for l in L-1:-1:1
        f = 1<<(l-1)
        s = 1<<l
        for eb = 1:2
            rng = eb == 1 ? (s:2s:n ) : (s+f:2s:n-f)
            for i in rng
                Sigma[l+eb, i] -= (Sigma[l+eb, i-f] + Sigma[l+eb, i+f])
                for k in i-s+1:i+s-1
                    Sigma[l+1, k] -= Sigma[l, k]
                end
                Sigma[l+eb, i] -= Sigma[l+eb, i-f] + Sigma[l+eb, i+f]
                for k in i-s+1:i+s-1
                    Sigma[l+eb, k] /= 2
                end
                Sigma[l+eb, i] /= 2
            end
        end
    end
    for i in 2:2:n
        Sigma[2, i], Sigma[1, i] = Sigma[1, i], Sigma[2, i]
    end
    for i in 4:4:n
        Sigma[1, i] = 0.5*(Sigma[1, i-1] + Sigma[1, i+1])
    end

    Sigma[1, :][:]
end

#%  .. function:: rootSigma!(Sigma, L)
#%                srootSigma!(Sigma, L)
#%
#%      Computes a root of Sigma using the perfect elimination order.
#%      This root is only triangular after a reordering of rows and columns,
#%      but see below for back/forward substition


function rootSigma!(Sigma, L, L2 = L)
    R = Sigma
    n = L2n(L)
    for l in 1:L2
        f = 1<<(l-1)
        s = 1<<l
        for i in f:s:n
#            println("[[$i]]")
            for l2 in 1:l-1
                s2 = 1<<l2
                f2 = 1<<(l2-1)
                for j in -f+f2:s2:f-f2
#                    println("[ $i + $j = $(i + j) ]")
                    som = Sigma[i, i+j]
                    if l2 > 1 for k in 1:(1<<(l2-1)-1)
#                      print(i+j-k, " ($k) ", i+j+k, ";")
                      som -= R[i, i+j-k]*R[i+j, i+j-k]
                      som -= R[i, i+j+k]*R[i+j, i+j+k]
                    end end
#                    println()
                    R[i, i+j] =  som/R[i+j, i+j]
                    R[i+j, i] = 0
                end
            end
#            println("[ $i + 0 ]")

            som = Sigma[i,i]
            for k in 1:f-1
#                print(i-k, " ", i+k, ";")
                som -= R[i, i-k]^2 + R[i, i+k]^2
            end

            R[i, i] = sqrt(som)
        end
#        displayln(R)
    end
    R
end

function srootSigma!(Sigma, L)
    R = Sigma
#    R = zeros(Sigma)
    n = L2n(L)
#    displayln(Sigma)
    for l in 1:L
        f = 1<<(l-1)
        s = 1<<l
        for i in f:s:n
#            println("[[$i]]")
            for l2 in 1:l-1
                s2 = 1<<l2
                f2 = 1<<(l2-1)
                for j in -f+f2:s2:f-f2
#                    println("[ $i + $j = $(i + j) ]")
                    som = Sigma[l, i+j]
                    if l2 > 1 for k in 1:(1<<(l2-1)-1)
#                      print(i+j-k, " ($k) ", i+j+k, ";")
                      som -= R[l, i+j-k]*R[l2, i+j-k]
                      som -= R[l, i+j+k]*R[l2, i+j+k]
                    end end
#                    println()
                    R[l, i+j] =  som/R[l2, i+j]
                    R[l2, i] = 0
                end
            end
#            println("[ $i + 0 ]")

            som = Sigma[l,i]
            for k in 1:f-1
#                print(i-k, " ", i+k, ";")
                som -= R[l, i-k]^2 + R[l, i+k]^2
            end

            R[l, i] = sqrt(som)
#            displayln([R Sigma])
        end
#        displayln(R)
    end
    R
end


#%  .. function:: ssinvRoot!(R, L)
#%
#%      Inverting R in place using the hierarchical version of following algorithm
#%
#%      function cholinv!(a)
#%          n = size(a,1)
#%          for i in n:-1:1
#%              a[i,i] = 1/a[i,i]
#%                  for j in i-1:-1:1
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

function sinvRoot!(R, L)
    n = L2n(L)
    for l in L:-1:1
        f = 1<<(l-1)
        s = 1<<l
        for i in f:s:n
            R[l, i] = 1./R[l, i]
            for l2 in l-1:-1:1
                s2 = 1<<l2
                f2 = 1<<(l2-1)
                for j in -f+f2:s2:f-f2
                    S  = 0.0
                    k = i+j
                    u = f2
                    for kl in l2+1:l
                        # k = f + n s -> k' = f' + n/2 s'
                        k = (((k- u) >> (kl-1 + 1))<<(kl)) + 2u  # index of father element in row
                        S += R[kl,i+j]*R[l,k]
                        u <<= 1
                    end
                    R[l, i+j] = -S/R[l2, i+j]
                end
            end
        end
    end
    R
end


#%  .. function:: solveR(R, a, L)
#%                ssolveR(R, a, L)
#%
#%      Solving Ry = a using forward substition, where R is given by rootSigma or srootSigma


function solveR(R, x, L)
    x = copy(x)
    n = L2n(L)
    for l in 1:L
        f = 1<<(l-1)
        s = 1<<l
        for i in f:s:n
            for j in 1:(1 << (l-1)-1)
                    x[i] -= x[i-j]*R[i, i-j]
                    x[i] -= x[i+j]*R[i, i+j]
            end
            x[i] /= R[i,i]
        end
    end
    x
end


function ssolveR(R, x, L)
    x = copy(x)
    n = L2n(L)
    for l in 1:L
        f = 1<<(l-1)
        s = 1<<l
        for i in f:s:n
            for j in 1:(1 << (l-1)-1)
                    x[i] -= x[i-j]*R[l, i-j]
                    x[i] -= x[i+j]*R[l, i+j]
            end
            x[i] /= R[l,i]
        end
    end
    x
end

#%  .. function:: backsolveRt(R, a, L)
#%                sbacksolveRt(R, a, L)
#%
#%      Solving R'y = a using backward substition, where R is given by rootSigma or srootSigma

function sbacksolveRt(R, x, L)
    x = copy(x)
    n = L2n(L)
    for l in L:-1:1
        f = 1<<(l-1)
        s = 1<<l
        for j in f:s:n
            x[j] /= R[l,j]
            for i in 1:(1 << (l-1)-1)
                    x[j-i] -= x[j]*R[l, j-i]
                    x[j+i] -= x[j]*R[l, j+i]
            end
        end
    end
    x
end

function backsolveRt(R, x, L)
    x = copy(x)
    n = L2n(L)
    for l in L:-1:1
        f = 1<<(l-1)
        s = 1<<l
        for j in f:s:n
            x[j] /= R[j,j]
            for i in 1:(1 << (l-1)-1)
                    x[j-i] -= x[j]*R[j, j-i]
                    x[j+i] -= x[j]*R[j, j+i]
            end
        end
    end
    x
end

#%  .. function:: solveRtR(R, a, L)
#%                ssolveRtR(R, a, L)
#%
#%      Solving z = R'y, Ry = a.

function ssolveRtR(R, x, L)
         x = ssolveR(R, x, L )
         sbacksolveRt(R, x, L)
end

import Base.dot
dot(x) = norm(x)^2

function svar(R, L)
         n = L2n(L)
         S = sinvRoot!(copy(R), L)
         mapslices(dot, S, 1)[:]
end

function droppedsvar(S, L)
#    println("OLD:")
    n = L2n(L)
    v = zeros(n)
    z = zeros(n)
    for l in 1:L
        f = 1<<(l-1)
        s = 1<<l
        for i in f:s:n
#            println("$i($l)")
            z[:] = 0.
            for l2 in 1:l
                s2 = 1<<l2
                f2 = 1<<(l2-1)
                for j in -f+f2:s2:f-f2
#                    println("$i($l) $(i+j)($l2)")
                    k = i+j
                    u = f2
                    for kl in l2:l
#                      println("$i($l) $(i+j)($l2) $k($kl)  ", (1<<(kl-1) - abs(k-(i+j))), " " ,S[l,k])
                        # k = f + n s -> k' = f' + n/2 s'
                        #z[i+j] += (1<<(l2-1))* S[l,k]
                        z[i+j] += (1<<(kl-1) - abs(k-(i+j)))*S[l,k]#*(1<<(kl))

                        k = (((k- u) >> (kl + 1))<<(kl+1)) + 2u  # index of father element in row
                        u <<= 1
                    end
                end
            end
            v[:] += z.^2
        end
    end
    v
end



function ssampleR(R, mu, w, L)
         x = ssolveR(R, mu, L ) + w
         sbacksolveRt(R, x, L)
end
function solveRtR(R, x, L)
         x = solveR(R, x, L )
         backsolveRt(R, x, L)
end

#%  .. function:: priorsigma(si, beta, L)
#%                priorprecision(si, beta, L)
#%                saddprecision!(Sigma, si, beta, L)
#%      Prior deviation resp. precision of level wise coefficients in Schauder basis.
#%      Corresponds to Brownian bridge if beta = 0.5 and si = 1.
#%      (s)addprecision! adds precision to a existing Sigma matrix

function priorsigma(si, beta, L)
    n = L2n(L)
    xi = zeros(n)
    for l in 1:L
        f = 1<<(l-1)
        s = 1<<l
        for i in f:s:n
		    xi[i] =   si[L-l+1]*2. ^ (-(l-1) - beta*(L-l+2))
        end
    end
    xi
end
function priorprecision(si, beta, L)
    n = L2n(L)
    xi = zeros(n)
    for l in 1:L
        f = 1<<(l-1)
        s = 1<<l
        for i in f:s:n
		    xi[i] =  2. ^ (2*(l-1) + 2*beta*(L-l+2))/(si[L-l+1]^2)
        end
    end
    xi
end


function saddprecision!(Sigma, si, beta, L, L2=L)
    n = L2n(L)
    for l in 1:L2
        f = 1<<(l-1)
        s = 1<<l
        for i in f:s:n
		    Sigma[l,i] += 2. ^ (2*(l-1) + 2*beta*(L-l+2))/(si[L-l+1]^2)
        end
    end
    Sigma
end
function addprecision!(Sigma, si, beta, L, L2=L)
    n = L2n(L)
    for l in 1:L2
        f = 1<<(l-1)
        s = 1<<l
        for i in f:s:n
            Sigma[i,i] += 2. ^ (2*(l-1) + 2*beta*(L-l+2))/(si[L-l+1]^2)
        end
    end
    Sigma
end




function speedTest( L = 5)
    L2 = 5
    n = L2n(L)
    x = (rand(1:2,n))
    x2 = (rand(1:2,n-1))

    if L < 6
    x = 5*[1,2,1,3,2,2,1,1,2,2,2,2,2,1,1,7,1,2,1,3,2,2,1,1,2,2,2,2,2,1,1][1:n]
    x2 =[1,1,1,1,2,0,2,0,2,0,1,1,1,2,0,1,0,2,2,2,1,1,1,1,2,2,2,2,2,1,0][1:n-1]
    end

    S1 = diagm(x) + diagm(x2,1) + diagm(x2,-1)
    S2 = copy(S1)
    #print(x)

    println("@time pickupSigma!(S1, L, L2)")
    @time pickupSigma!(S1, L, L2)
    #reverse!(S2[:])
    #pickupSigma!(S2, L)
    #reverse!(S2[:])



    Sp = zeros(Int, L, n)
    Sp[1, :] = x
    Sp[2,2:n] = x2



    println("@time spickupSigma!(Sp, L)")
    @time spickupSigma!(Sp, L, L2)
#    displayln([S1, Sp])

#    exit()

    rootSigma!(ones(1,1), 0, 0)
    Z = float(S1)
    println("@time R = rootSigma!(Z, L, L)")
    @time R = rootSigma!(Z, L, L)

    srootSigma!(ones(1,1),0,0)
    ZSp = float(Sp)
    println("@time RSp = srootSigma!(ZSp, L, L)")
    @time RSp = srootSigma!(ZSp, L, L)

    Z = float(S1)
    cholfact!(ones(1,1))
    println("@time cholfact!(Z)")
    @time cholfact!(Z)

    Z = sparse(float(S1))
    println("@time cholfact(Z)")
    @time cholfact(Z)

    #displayln(1*(abs(R).>0.01))
    if L <12
        println("@time norm(R*R'-S)")
        @time println("Norm R*R'-S: ",(norm((R*R'-float(S1)))))

    end
    a = rand(n)
    println("@time x = solveR(R, a, L, L)")
    @time x = solveR(R, a, L, L)
    println("@time y = ssolveR(RSp, a, L, L)")
    @time y = ssolveR(RSp, a, L)

    println("@time y2 = backsolveRt(R, a, L)")
    @time x2 = backsolveRt(R, a, L)

    println("@time y2 = sbacksolveRt(RSp, a, L)")
    @time y2 = sbacksolveRt(RSp, a, L)


    if L < 5
        displayln([y x R*x a])
    end
    println("Norm R*x-a: ",norm(R*x- a))
    println("Norm R'x2-a: ",norm(R'*x2 - a))

    #displayln([x R*x a])
    println("Norm x-y: ",norm(x- y))
    println("Norm x2-y2: ",norm(x2- y2))


    #printpatterns(L)
end

#speedTest(4)
function bayes(L)
    srand(4)
    alpha = 0.5
    indlevel = true
    a = 5/2
    b = 5/2 #hyper prior on si0
    a0 = 7/2
    b0 = 7/2 #and on si[i]
    T = 1.
    K = 200
#    f(t) = 2sin(2pi ./(0.9t +.1)) #- 4hat(2t-1)
    f(t) = 2sin(4pi*t)
    tt = linspace(0, T, K)
    sitrue = 2.
    X = f(tt) + sitrue*randn(K)
    #L = 7
    n = L2n(L)
    si = 2.
    si0 = ones(L)
    mu = mureg(tt, X, 1., L);
    mubar = fetransf(f,0,1,7);
    Sigma0 = sSigmareg(tt, 1., L)
    pickupmu!(mu, L)
    spickupSigma!(Sigma0, L)
    if L < 5 displayln(Sigma0) end

#    displayln = identity
    R0 = copy(Sigma0)
    R0[:] /= si^2
    saddprecision!(R0, si0, alpha, L)
    srootSigma!(R0, L)
    theta0 = ssolveRtR(R0, mu/si^2, L)
    x0 = copy(theta0)
    drop!(x0, L)

    println("Res: ", round(std(X.-evalfe(tt,x0,L)),2))

    if false # empirical bayes for si0 and si
    Sigma = similar(Sigma0)
    S = 0.
    S0 = zeros(si0)
    IT2 = 1000
    theta = similar(mu)
    for i = 1:IT2
        if indlevel
            for l in 1:L
                Sigma[:] = Sigma0/si^2
                saddprecision!(Sigma, si0, alpha, L)
                srootSigma!(Sigma, L)
                theta[:] = ssampleR(Sigma, mu/si^2,randn(n), L)
                th0 = theta ./ priorsigma(ones(L),alpha,L)
                s = 1<<(l-1)
                thi = th0[s:s:n]
                si0[L-l+1] = sqrt(posterior_rand((0., InverseGamma(a0, b0)), Normal, thi))
            end
            print("$i si0 ", round(si0,1)," si ", round(si,1),"    \r")
        else
            Sigma[:] = Sigma0/si^2
            saddprecision!(Sigma, si0, alpha, L)
            srootSigma!(Sigma, L)
            theta[:] = ssampleR(Sigma, mu/si^2,randn(n), L)
            th0 = theta ./ priorsigma(ones(L),alpha,L)
            si0[:] = sqrt(posterior_rand((0., InverseGamma(a0, b0)), Normal, th0))
            print("$i si0 ", round(si0[1],1)," si ", round(si,1),"     \r")
        end

        Sigma[:] = Sigma0/si^2
        saddprecision!(Sigma, si0, alpha, L)
        srootSigma!(Sigma, L)
        theta[:] = ssampleR(Sigma, mu/si^2, randn(n), L)
        drop!(theta, L)
        si = sqrt(posterior_rand((0., InverseGamma(a, b)), Normal, X.-evalfe(tt,theta,L)))
        S += si
        S0[:] += si0
#        if norm(si0- si0old) <0.1 break end
    end
        si0[:] = S0/IT2
        si0 *= 2
        si = S / IT2
    end
    println("si0 ", round(si0,2), " si ", round(si,2))

    # scale mu, Sigma0
    mu .*= 1./(si*si)
    Sigma0 .*= 1./(si*si)


    #final verdict
    Sigma = copy(Sigma0)
    saddprecision!(Sigma, si0, alpha, L)

    R = copy(Sigma)
    #1.64 sigma band
    srootSigma!(R, L)
    theta = ssolveRtR(R, mu, L)
    x = copy(theta)
    drop!(x, L)

    println("Res: ", round(std(X.-evalfe(tt,x,L)),2))
    S = sinvRoot!(copy(R), L)
    band = 1.64*sqrt(droppedsvar(S, L))

    thetar = ssampleR(R, mu,randn(n), L)
    drop!(thetar, L)
#    display(round([mubar x theta],2))



    band0 = fetransf(t ->  sqrt((1-t)*t),0,1,L); #if beta = 0.5
    if false # if no data, band should be scale invariant
        extremalnoise = copy(band0)
        pickup!(extremalnoise, L)
        extremalnoise[:] ./= priorsigma(ones(n), alpha, L)
        extremalnoise += 2ones(n)
        band = sbacksolveRt(R, extremalnoise, L)
        drop!(band, L)
    end
    for w in (mubar, x0, x, band)
        n2 = length(w)
        resize!(w, n2+2)
        w[2:n2+1] = w[1:n2]
        w[1] = 0
    end
    n2 = n + 2

    display(plot(tt, X, ".", linspace(0, 1, length(mubar) ), mubar, "r", linspace(0, 1, n2), x0, "y-", linspace(0, 1, n2), x, "g-",
        linspace(0,1,n2), x+band, "b--", linspace(0,1,n2), x-band, "b--", yrange=[-6,6], linewidth=1))
    print("Ok")
end
