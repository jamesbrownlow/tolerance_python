import numpy as np
from scipy.stats import rayleigh
from scipy.optimize import brentq
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def length(x):
    if type(x) == float or type(x) == int or type(x) == np.int32 or type(x) == np.float64 or type(x) == np.float32 or type(x) == np.int64:
        return 1
    return len(x)

#WORKS CITED:
    #On Construction of Two-Sided Tolerance Intervals and Confidence Intervals for Probability Content
    #   Hoang-Nguyen-Thuy, Ngan.   University of Louisiana at Lafayette ProQuest Dissertations Publishing,  2020. 27959915.
    #   https://www.proquest.com/openview/eaab2073101c1082445e2611c08c376e/1?pq-origsite=gscholar&cbl=18750&diss=y

def RaylMLEScens(xc, n):
    r = length(xc)
    x = list(xc.copy())
    x.extend([xc[-1],]*(n-r))
    x = np.array(x)
    xb = np.mean(xc)
    s = np.std(xc, ddof = 1)
    bhat = np.sqrt(2/(4-np.pi))*s
    def fn(a):
        ssq = sum((x-a)**2)
        y = 2*r*sum(x-a)/ssq-sum(1/(np.array(xc)-a))
        return y
    a0 = x[0]-15*bhat/np.sqrt(r)
    a1 = x[0]
    aMLE = brentq(fn, a = a0, b = a1, xtol = 1e-5, maxiter = 20)
    bMLE = np.sqrt(0.5*sum((x-aMLE)**2)/r)
    return np.array([aMLE, bMLE])

def RaylMLESuncens(x):
    n = length(x)
    xmin = min(x)
    bh = np.sqrt(2/(4-np.pi))*np.std(x, ddof=1)
    a0 = xmin-15*bh/np.sqrt(n)
    a1 = xmin
    def ha(a):
        sxx = sum((np.array(x)-a)**2)
        hs = 2*n**2*(np.mean(x)-a)/sxx-sum(1/(np.array(x)-a))
        return (hs)
    aMLE = brentq(ha, a = a0, b = a1, xtol = 1e-5, maxiter = 30)
    bMLE = np.sqrt(sum((np.array(x)-aMLE)**2)/2/n)
    return np.array([aMLE, bMLE])

def RaylMLES(x, n, censored):
    if censored:
        mles = RaylMLEScens(x, n)
    else:
        mles = RaylMLESuncens(x)
    return mles

def RayOneSidedFac(nr, n, r, P, alpha, censored):
    al = 1-alpha
    qupp = (np.sqrt(-2*np.log(1-P)))
    qlow = (np.sqrt(-2*np.log(P)))
    u = np.random.uniform(size = int(nr*n))
    #u = np.linspace(0.01,0.99, int(nr*n))
    xm = np.sqrt(-2*np.log(u)).reshape(n,nr).T
    xm = pd.DataFrame(np.array(list(map(np.sort,xm))))
    xc = xm.iloc[:,0:r]
    mles = np.zeros(length(xc.iloc[:,0]),dtype = 'object')
    for i in range(length(xc.iloc[:,0])):
        mles[i] = RaylMLES(xc.iloc[i].values, n, censored)
    mles0 = [x[0] for x in mles]
    mles1 = [x[1] for x in mles]
    mles = pd.DataFrame(np.array([mles0,mles1]))
    ahs = mles.iloc[0].values
    bhs = mles.iloc[1].values
    pivL = np.sort((qlow-ahs)/bhs)
    pivU = np.sort((qupp-ahs)/bhs)
    if int(nr*al) == 0:
        return [min(pivL),pivU[int(nr*(1-al))-1]]
    else:
        Low = pivL[int(nr*al)-1]
        Upp = pivU[int(nr*(1-al))-1]
    return [Low, Upp]

def RaylTF(nr, n, r, P, alpha, censored, tails):
    p = (1+P)/2
    gam = (1+alpha)/2
    qupp = np.sqrt(-2*np.log(1-p))
    qlow = np.sqrt(-2*np.log(p))
    u = np.random.uniform(size = int(nr*n))
    #u = np.linspace(0.01,0.99, int(nr*n))
    xm = np.sqrt(-2*np.log(u)).reshape(n,nr).T
    xm = pd.DataFrame(np.array(list(map(np.sort,xm))))
    xc = xm.iloc[:,0:r]
    mles = np.zeros(length(xc.iloc[:,0]))
    for i in range(length(xc.iloc[:,0])):
        mles[i] = RaylMLES(xc.iloc[i].values, n, censored)
    mles0 = [x[0] for x in mles]
    mles1 = [x[1] for x in mles]
    mles = pd.DataFrame(np.array([mles0,mles1]))
    ahs = mles.iloc[0].values
    bhs = mles.iloc[1].values
    pivL = np.sort((qlow-ahs)/bhs)
    pivU = np.sort((qupp-ahs)/bhs)
    def fn(x):
        if tails == 'equal-tailed':
            al = 1-(1+x)/2
            if int(nr*al) == 0:
                return "Number of Runs, nr, must be larger."
            Lfac = pivL[int(nr*al)-1]
            Ufac = pivU[int(nr*(1-al))-1]
            LowLim = ahs+Lfac*bhs
            UppLim = ahs+Ufac*bhs
            cont = (np.where(LowLim <= qlow) and np.where(qupp <= UppLim))[0]
            covr = np.mean(cont >= P)
            return(covr-alpha)
        elif tails == '1' or tails == '2':
            facL = np.percentile(pivL, (1-(1+x)/2)*100)
            facU = np.percentile(pivU, ((1+x)/2)*100)
            LowLim = ahs+facL*bhs
            UppLim = ahs+facU*bhs
            cont = rayleigh.cdf(UppLim,0,1) - rayleigh.cdf(LowLim,0,1)
            covr = np.mean(cont >= P)
            return covr-alpha
    xl = 0.3
    xr = alpha
    k = 1
    while True:
        fl = fn(xl)
        fr = fn(xr)
        xm = (xl+xr)/2
        fm = fn(xm)
        if abs(fm) < 1e-5 or k > 50:
            break
        if fl*fm > 0:
            xl = xm
        else:
            xr = xm
        k = k+1
    als = 1-(1+xm)/2
    if tails == 'equal-tailed':
        Lfac = pivL[int(nr*als)]
        Ufac = pivU[int(nr*(1-als))]
    else:
        Lfac = pivL[int(nr*als)-1]
        Ufac = pivU[int(nr*(1-als))-1]
    return (Lfac, Ufac)

def rayleightolint(x, alpha = 0.05, P= 0.99, side = 1, nr = 1000, censored = False, printMLES = False, printFactors = False):
    '''
Description
    Rayleigh Tolerance Interval

Usage
    rayleightolint(x, alpha = 0.05, P= 0.99, side = 1, nr = 1000, 
                   censored = False, printMLES = False, printFactors = False):

Parameters
----------
    x : list
        A vector of Rayleigh distributed data. 
        
    alpha : float, optional
        The level chosen such that 1-alpha is the confidence level. The 
        default is 0.05.
        
    P : float, optional
        The proportion of the population to be covered by this tolerance 
        interval. The default is 0.99.
        
    side : 1, 2, or 'equal-tailed', optional
        Whether a 1-sided, 2-sided, or equal-tailed tolerance interval is 
        required (determined by side = 1 or side = 2, respectively). The 
        default is 1.
        
    nr : int, optional
        The number of simulations. The default is 1000.
        
    censored : bool, optional
        If True, the the value of a measurement or observation is only 
        partially known. The default is False.
        
    printMLES : bool, optional
        Prints the Maximum Likelihood Estimators if True. The default is False.
        
    printFactors : TYPE, optional
        Prints the tolerance factors if True. The default is False.

Returns
-------
    rayleightolint returns a data frame with items:

        alpha	
            The specified significance level.
        
        P	
            The proportion of the population covered by this tolerance 
            interval.
        
        1-sided.lower	
            The 1-sided lower tolerance bound. This is given only if side = 1.
        
        1-sided.upper	
            The 1-sided upper tolerance bound. This is given only if side = 1.
        
        2-sided.lower	
            The 2-sided lower tolerance bound. This is given only if side = 2.
        
        2-sided.upper	
            The 2-sided upper tolerance bound. This is given only if side = 2.
            
        equal-tailed.lower
            The equal-tailed lower tolerance bound. This is given only if 
            side = 'equal-tailed'.
            
        equal-tailed.upper
            The equal-tailed upper tolerance bound. This is given only if 
            side = 'equal-tailed'.

References
----------
    On Construction of Two-Sided Tolerance Intervals and Confidence Intervals 
        for Probability Content Hoang-Nguyen-Thuy, Ngan. University of 
        Louisiana at Lafayette ProQuest Dissertations Publishing, 2020. 
        27959915.

Examples
--------
    x = rayleigh.rvs(size = 100)
    
    rayleightolint(x, alpha = 0.01, P = 0.99, side = 1)
    
    rayleightolint(x, alpha = 0.05, P = 0.95, side = 2)
    
    rayleightolint(x, alpha = 0.1, P = 0.9, side = 'equal-tailed')
    '''
    alpha = 1-alpha
    n = length(x)
    r = length(x)
    mles = RaylMLES(x, n, censored)
    ah0 = mles[0]
    bh0 = mles[1]
    if printMLES:
        print(ah0,bh0)
    if side == 1:
        osfac = RayOneSidedFac(nr,n,r,P,alpha,censored)
        if printFactors:
            print(f'One-sided Factors are: {np.array(osfac)}')
        OSLow = ah0 + osfac[0]*bh0
        OSUpp = ah0 + osfac[1]*bh0
        return pd.DataFrame({'alpha': [1-alpha], 'P': [P], '1-sided.lower':OSLow, '1-sided.upper':OSUpp})
    elif side == 2:
        tsfac = RaylTF(nr,n,r,P,alpha,censored, tails = '2')
        if printFactors:
            print(f'Two-sided Factors are: {np.array(tsfac)}')
        TSLow = ah0 + tsfac[0]*bh0
        TSUpp = ah0 + tsfac[1]*bh0
        return pd.DataFrame({'alpha': [1-alpha], 'P': [P], '2-sided.lower':TSLow, '2-sided.upper':TSUpp})
    elif side == 'equal-tailed':
        eqfac = RaylTF(nr, n, r, P, alpha, censored, tails = 'equal-tailed')
        if printFactors:
            print(f'Equal-Tailed Factors are: {np.array(eqfac)}')
        EQLow = ah0 + eqfac[0]*bh0
        EQUpp = ah0 + eqfac[1]*bh0
        return pd.DataFrame({'alpha': [1-alpha], 'P': [P], 'equal-tailed.lower':EQLow, 'equal-tailed.upper':EQUpp})
    
## Tests
# x = rayleigh.rvs(size = 100000)
# print(rayleightolint(x,0.05,0.95, 1, nr = 100),'\n')
# print(rayleightolint(x,0.05,0.95, 2),'\n')
# print(rayleightolint(x,0.05,0.95, 'equal-tailed'))

## True Percentile Values
# print(rayleigh.ppf(0.95))
# print(rayleigh.ppf(0.05))
#print(rayleigh.ppf([0.025,0.975]))

## Notes
## Rayl.cdf is equivalent to rayleigh.cdf()
## Rayl.rand is equivalent to rayleigh.rvs()
