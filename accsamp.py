import scipy.stats
import pandas as pd
import numpy as np
import scipy.optimize as opt

def accsamp(n, N, alpha = 0.05, P = 0.99, AQL = 0.01, RQL = 0.02):
    '''
Acceptance Sampling

Description
    Provides an upper bound on the number of acceptable rejects or 
    nonconformities in a process. This is similar to a 1-sided upper tolerance 
    bound for a hypergeometric random variable.
    
    accsamp(n, N, alpha = 0.05, P = 0.99, AQL = 0.01, RQL = 0.02)
        
Parameters
----------
    n: int
        The sample size to be drawn from the inventory.
    
    N: int
        The total inventory (or lot) size.
    
    alpha: float, optional
        1-alpha is the confidence level for bounding the probability of 
        accepting the inventory. The default is 0.05.
    
    P: float, optional
        The proportion of items in the inventory which are to be accountable. 
        The default is 0.99.
    
    AQL: float, optional
        The acceptable quality level, which is the largest proportion of 
        defects in a process considered acceptable. Note that 0 < AQL < 1. 
        The default is 0.01.
    
    RQL: float, optional
        The rejectable quality level, which is the largest proportion of 
        defects in an independent lot that one is willing to tolerate. Note 
        that AQL < RQL < 1. The default is 0.02.
    
Returns
-------
  accsamp returns a data frame with the following qualities:
        
    acceptance.limit: 
        The number of items in the sample which may be unaccountable, yet 
        still be able to attain the desired confidence level 1-alpha.
        
    lot.size: 
        The total inventory (or lot) size N.
        
    confidence:	
        The confidence level 1-alpha.
        
    P:	
        The proportion of accountable items specified by the user.
        
    AQL:	
        The acceptable quality level as specified by the user. If the sampling 
        were to be repeated numerous times as a process, then this quantity 
        specifies the proportion of missing items considered acceptable from 
        the process as a whole. Conditioning on the calculated value for 
        acceptance.limit, the AQL is used to estimate the producer's risk 
        (see prod.risk below).

    RQL:	
        The rejectable quality level as specified by the user. This is the 
        proportion of individual items in a sample one is willing to tolerate 
        missing. Conditioning on the calculated value for acceptance.limit, 
        the RQL is used to estimate the consumer's risk (see cons.risk below).

    sample.size:
        The sample size drawn as specified by n.

    prod.risk:
        The producer's risk at the specified AQL. This is the probability of 
        rejecting an audit of a good inventory (also called the Type I error). 
        A good inventory can be rejected if an unfortunate random sample is
        selected (e.g., most of the missing items happened to be selected for 
        the audit). 1-prod.risk gives the confidence level of this sampling 
        plan for the specified AQL and RQL. If it is lower than the confidence 
        level desired (e.g., because the AQL is too high), then a warning 
        message will be displayed.

    cons.risk:	
        The consumer's risk at the specified RQL. This is the probability of 
        accepting an audit of a bad inventory (also called the Type II error). 
        A bad inventory can be accepted if a fortunate random sample is 
        selected (e.g., most of the missing items happened to not be selected 
        for the audit).

References
----------
    Derek S. Young (2010). tolerance: An R Package for Estimating Tolerance 
        Intervals. Journal of Statistical Software, 36(5), 1-39. 
        URL http://www.jstatsoft.org/v36/i05/.
        
    Montgomery, D. C. (2005), Introduction to Statistical Quality Control, 
        Fifth Edition, John Wiley & Sons, Inc.
    
Examples
--------
    ## A 90%/90% acceptance sampling plan for a sample of 450 drawn from a lot 
    size of 960.
    
        acc.samp(n = 450, N = 960, alpha = 0.10, P = 0.90, AQL = 0.07,
         RQL = 0.10)
    '''
    if (RQL-AQL < 4e-08):
        return 'RQL must be greater than AQL!'
    D = N - (N*P)
    mh = D
    nh = N - D
    def ff(c, k, m, n):
        #produces flaoting point error relative to R
        r = alpha - scipy.stats.hypergeom(M=n,n=m,N=k).cdf(c)
        return r
    #regression analysis came up with a B1 of 0.9962390
    # did this to have results as accurate to R as possible
    # there is some discrepancy. As n increases, so does the error
    B1 = 0.9962390
    try:
        c = np.floor(B1*opt.brentq(ff,a=0,b=np.ceil(D),args=(n,mh,nh)))
    except ValueError:
        c = 0
    if scipy.stats.hypergeom(M=nh,n=mh,N=n).cdf(c) > alpha:
        c = max(c-1,0)
    prob_rej_good = 1 - scipy.stats.hypergeom(M=N - np.floor(AQL *N), n=np.floor(AQL * N), N=n).cdf(c)
    prob_rej_bad = 1 - scipy.stats.hypergeom(M=(N - np.floor(RQL *N)), n=np.floor(RQL * N), N = n).cdf(c)
    temp = pd.DataFrame({"acceptance.limit":[round(c,0)], "lot.size":[round(N,0)], "confidence":[1-round(alpha,4)], "P":[round(P,4)], "AQL":[round(AQL,4)],
                      "RQL":[round(RQL,4)], "sample.size":[round(n,0)], "prod.risk":[round(prob_rej_good,4)], "cons.risk":[round((1-prob_rej_bad),4)]}).T
    if round(prob_rej_good,4) > alpha:
        print("Warning: Desired confidence level not attained!\n")
    if n == N:
        print("Sample size and lot size are the same.  This is just a census!\n")
    return temp

accsa
#print(accsamp(n = 400, N = 960000))

#hypergeom.cdf(x, M, n, N)
#g = 75
#k = 59
#m = 611
#N = 13588
#n = N-m
#x = 19
#print(scipy.stats.hypergeom(M=N,n=m,N=k).sf(x-1))
#equivalent to 
#1-phyper(q=x -1, m=m, n=n, k=k)

#print(scipy.stats.hypergeom(M=950.4,n=9.6,N=450).cdf(2))

# x = np.zeros(221)
# x[0] = accsamp(n = 450,N = 100_000)
# for i in range(1,len(x)):
#     x[i] = accsamp(n = 450*i,N = 100_000)

# print(list(x))