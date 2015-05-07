####
#
#   EXAMPLE
#
####

using DyadicMatrices
using Winston

DM = DyadicMatrices
DV = DyadicVector

const L = 8
const K = 0

#srand(4)
    alpha = 0.5
    a = 5/2
    b = 5/2 #hyper prior on si0
    T = 1.
    N = 500
#    f(t) = 2sin(2pi ./(0.9t +.1)) #- 4hat(2t-1)
    f(t) = 2sqrt(sin(4pi*t)+1) - 2t
    tt = linspace(0, T, N)
    sitrue = 2.
    X = f(tt) + sitrue*randn(N)
    n = len(L, K)
    si0 = 4
    si = std(diff(X)/sqrt(2)) # simple estimate for empirical bayes
    mu = DM.mureg(tt, X, si, L, K)
    mubar = DM.fetransf(f, 0.0, T, 8, K)
    Sigma = DM.Sigmareg(tt, si, L, K)
    mu = DM.pickupmu!(mu)
    Sigma = pickupSigma(Sigma, K)
    W = addprecision(Sigma, si0, alpha)


    R = root(W)
    theta0 = R'\(R\mu)
    x = DM.drop(theta0)

    println("Res: ", round(std(X.-evalfe(tt,x)),2))


    println("si0 ", round(si0,2), " si ", round(si,2))

    S = inv!(copy(R))
    band = 1.64*sqrt(droppedvar(S))

    thetar = R'\(R\mu + DV(randn(n),K))
    xr = DM.drop(thetar)

    n2 = n + 2 - 2K
    if K == 0
        dp = T/n:T/(n+1):T*(n-1)/n
    else
        dp = linspace(0, T, n)
    end
    

    if true
    display(plot(tt, X, ".", 
        linspace(0, 1, length(mubar) ),  mubar, "r", 
        dp, x.x, "g-", 
        dp, xr.x, "g.",
        dp, x.+band, "b--", dp, x.-band, "b--", 
        yrange=[-6,6], linewidth=1))
    print("Ok")
    end
#	readline()
