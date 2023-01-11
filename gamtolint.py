import numpy as np
import scipy.stats
import pandas as pd
import scipy.optimize as opt
import math
import scipy.stats
import scipy.integrate as integrate
import warnings
warnings.filterwarnings('ignore')

def Kfactor(n, f = None, alpha = 0.05, P = 0.99, side = 1, method = 'HE', m=50):
    K=None
    if f == None:
        f = n-1
    if (len((n,)*1)) != len((f,)*1) and (len((f,)*1) > 1):
        return 'Length of \'f\' needs to match length of \'n\'!'
    if (side != 1) and (side != 2):
        return 'Must specify one sided or two sided procedure'
    if side ==1:
        zp = scipy.stats.norm.ppf(P)
        ncp = np.sqrt(n)*zp
        ta = scipy.stats.nct.ppf(1-alpha,df = f, nc=ncp) #students t noncentralized
        K = ta/np.sqrt(n)
    else:
        def Ktemp(n, f, alpha, P, method, m):
            chia = scipy.stats.chi2.ppf(alpha, df = f)
            k2 = np.sqrt(f*scipy.stats.ncx2.ppf(P,df=1,nc=(1/n))/chia) #noncentralized chi 2 (ncx2))
            if method == 'HE':
                def TEMP4(n, f, P, alpha):
                    chia =  scipy.stats.chi2.ppf(alpha, df = f)
                    zp = scipy.stats.norm.ppf((1+P)/2)
                    za = scipy.stats.norm.ppf((2-alpha)/2)
                    dfcut = n**2*(1+(1/za**2))
                    V = 1 + (za**2)/n + ((3-zp**2)*za**4)/(6*n**2)
                    K1 = (zp * np.sqrt(V * (1 + (n * V/(2 * f)) * (1 + 1/za**2))))
                    G = (f-2-chia)/(2*(n+1)**2)
                    K2 = (zp * np.sqrt(((f * (1 + 1/n))/(chia)) * (1 + G)))
                    if f > dfcut:
                        K = K1
                    else:
                        K = K2
                        if K == np.nan or K == None:
                            K = 0
                    return K
                #TEMP5 = np.vectorize(TEMP4())
                K = TEMP4(n, f, P, alpha)
                return K
                
            elif method == 'HE2':
                zp = scipy.stats.norm.ppf((1+P)/2)
                K = zp * np.sqrt((1+1/n)*f/chia)
                return K
            
            elif method == 'WBE':
                r = 0.5
                delta = 1
                while abs(delta) > 0.00000001:
                    Pnew = scipy.stats.norm.cdf(1/np.sqrt(n)+r) - scipy.stats.norm.cdf(1/np.sqrt(n)-r)
                    delta = Pnew-P
                    diff = scipy.stats.norm.pdf(1/np.sqrt(n)+r) + scipy.stats.norm.pdf(1/np.sqrt(n)-r)
                    r = r-delta/diff
                K = r*np.sqrt(f/chia)
                return K
            
            elif method == 'ELL':
                if f < n**2:
                    print("The ellison method should only be used for f appreciably larger than n^2")
                    return None
                r = 0.5
                delta = 1
                zp = scipy.stats.norm.ppf((1+P)/2)
                while abs(delta) > 0.00000001:
                    Pnew = scipy.stats.norm.cdf(zp/np.sqrt(n)+r) - scipy.stats.norm.cdf(zp/np.sqrt(n)-r)
                    delta = Pnew - P
                    diff =  scipy.stats.norm.pdf(zp/np.sqrt(n)+r) +  scipy.stats.norm.pdf(zp/np.sqrt(n)-r)
                    r = r-delta/diff
                K = r*np.sqrt(f/chia)
                return K
            elif method == 'KM':
                K = k2
                return K
            elif method == 'OCT':
                delta = np.sqrt(n)*scipy.stats.norm.ppf((1+P)/2)
                def Fun1(z,P,ke,n,f1,delta):
                    return (2 * scipy.stats.norm.cdf(-delta + (ke * np.sqrt(n * z))/(np.sqrt(f1))) - 1) * scipy.stats.chi2.pdf(z,f1) 
                def Fun2(ke, P, n, f1, alpha, m, delta):
                    return integrate.quad(Fun1,a = f1 * delta**2/(ke**2 * n), b = 1000 * n, args=(P,ke,n,f1,delta),limit = m)
                def Fun3(ke,P,n,f1,alpha,m,delta):
                    f = Fun2(ke = ke, P = P, n = n, f1 = f1, alpha = alpha, m = m, delta = delta)
                    return abs(f[0] - (1-alpha))
                K = opt.minimize(fun=Fun3, x0=k2, args=(P,n,f,alpha,m,delta), method = 'L-BFGS-B')['x']
                return float(K)
            elif method == 'EXACT':
                print('This method prodcues slightly different results',
                      'when compared to R. Take these results with',
                      'a grain of salt. The range of error',
                      'is between approximately (1e-3,2.0).',
                      'If this method is abosolutely needed',
                      'use R instead.')
                def fun1(z,df1,P,X,n):
                    k = (scipy.stats.chi2.sf(df1*scipy.stats.chi2.ppf(P,1,z**2)/X**2,df=df1)*np.exp(-0.5*n*z**2))
                    return k
                def fun2(X,df1,P,n,alpha,m):
                    return integrate.quad(fun1,a =0, b = 5, args=(df1,P,X,n),limit=m)
                def fun3(X,df1,P,n,alpha,m):
                    return np.sqrt(2*n/np.pi)*fun2(X,df1,P,n,alpha,m)[0]-(1-alpha)
                
                K = opt.brentq(f=fun3,a=0,b=k2+(1000)/n, args=(f,P,n,alpha,m))
                return K
        #TEMP = np.vectorize(Ktemp)
        K = Ktemp(n=n,f=f,alpha=alpha,P=P,method=method,m=m)
    return K

def length(x):
    if type(x) == int or type(x) == float or type(x) == np.float64:
        return 1
    else:
        return len(x)

def mean(x,n):
    #using this because for some reason it's the same as R
    x = sum(x)
    return x/(n)

def gamtolint(x,alpha=0.05,P=0.99,side=1,method = 'HE',m=50,loggamma=False):
    '''
Description:
    Provides 1-sided or 2-sided tolerance intervals for data distributed 
    according to either a gamma distribution or log-gamma distribution
    
    gamtolint(x, alpha = 0.05, P = 0.99, side = 1, 
           method = c("HE", "HE2", "WBE", "ELL", "KM", "EXACT", 
           "OCT"), m = 50, log.gamma = FALSE)
Parameters
    ----------
    x : list
        A vector of data which is distributed according to either a gamma 
        distribution or a log-gamma distribution.

    alpha : float, optional
        The level chosen such that 1-alpha is the confidence level. 
        The default is 0.05.
        
    P : float, optional
        The proportion of the population to be covered by this tolerance 
        interval. The default is 0.99.
        
    side : 1 or 2, optional
        Whether a 1-sided or 2-sided tolerance interval is required 
        (determined by side = 1 or side = 2, respectively). The default is 1.
        
    method : string, optional
        The method for calculating the k-factors. The k-factor for the 1-sided 
        tolerance intervals is performed exactly and thus is the same for the 
        chosen method. "HE" is the Howe method and is often viewed as being
        extremely accurate, even for small sample sizes. "HE2" is a second 
        method due to Howe, which performs similarly to the Weissberg-Beatty
        method, but is computationally simpler. "WBE" is the Weissberg-Beatty 
        method (also called the Wald-Wolfowitz method), which performs 
        similarly to the first Howe method for larger sample sizes. "ELL" is 
        the Ellison correction to the Weissberg-Beatty method when f is 
        appreciably larger than n^2. A warning message is displayed if f is 
        not larger than n^2. "KM" is the Krishnamoorthy-Mathew approximation 
        to the exact solution, which works well for larger sample sizes. 
        "EXACT" computes the k-factor exactly by finding the integral solution 
        to the problem via the integrate function. Note the computation time 
        of this method is largely determined by m. "OCT" is the Owen approach 
        to compute the k-factor when controlling the tails so that there is 
        not more than (1-P)/2 of the data in each tail of the distribution. 
        
    m : int, optional
        The maximum number of subintervals to be used in the integrate 
        function. This is necessary only for method = "EXACT" and method = 
        "OCT". The larger the number, the more accurate the solution. Too low 
        of a value can result in an error. A large value can also cause the 
        function to be slow for method = "EXACT".
        
    log.gamma	
        If TRUE, then the data is considered to be from a log-gamma 
        distribution, in which case the output gives tolerance intervals for 
        the log-gamma distribution. The default is FALSE.

Returns
-------
    gamtolint returns a dataframe with items:
        alpha: 
            The specified significance level.
        
        P: 
            The proportion of the population covered by this tolerance 
            interval. 
        
        1-sided.lower: 
            The 1-sided lower tolerance bound. This is given 
            only if side = 1.
        
        1-sided.upper: 
            The 1-sided upper tolerance bound. This is given 
            only if side = 1.
        
        2-sided.lower: 
            The 2-sided lower tolerance bound. This is given 
            only if side = 2.
        
        2-sided.upper: 
            The 2-sided upper tolerance bound. This is given 
            only if side = 2.

Details
-------
    Recall that if the random variable X is distributed according to a 
    log-gamma distribution, then the random variable Y = ln(X) is distributed 
    according to a gamma distribution.

Note
----
    R uses the Newton miniminzation method and this code uses the SLSQP
    minimization method. Both methods are non linear minimization methods. 
    Their results have a minor discrepancy. 

References
----------
    Derek S. Young (2010). tolerance: An R Package for Estimating Tolerance 
        Intervals. Journal of Statistical Software, 36(5), 1-39. 
        URL http://www.jstatsoft.org/v36/i05/.
        
    Krishnamoorthy, K., Mathew, T., and Mukherjee, S. (2008), Normal-Based 
        Methods for a Gamma Distribution: Prediction and Tolerance Intervals 
        and Stress-Strength Reliability, Technometrics, 50, 69–78.

Examples
--------
    ## 99%/99% 1-sided gamma tolerance intervals for a sample of size 50. 
        
        x = np.random.gamma(size=50)
        gamtolint(x = x, alpha = 0.01, P = 0.99, side = 1, method = "HE")

    '''
    if side != 1 and side != 2:
        return 'Must specify a one-sided or two-sided procedure'
    if method == 'ELL':
        return 'Must specify the HE, WBE, or EXACT method'
    if loggamma:
        x = np.log(x)
    n = len(x)
    xbar = np.mean(x)
    x2bar = np.mean(np.array(x)**2) #np.mean(np.array(x)**2)
    inits = [xbar**2/(x2bar-xbar**2),(x2bar-xbar**2)/xbar]
    x = np.array(x)
    def gammall(pars,x):
        return sum(-scipy.stats.gamma.logpdf(x,a=pars[0],scale=pars[1])) #for y in x])
    out = (opt.minimize(gammall,x0 = inits, args = (x),method = 'SLSQP')['x'])
    ahat = out[0]
    bhat = out[1]
    x = x**(1/3)
    xbar = bhat**(1/3) * (np.exp(math.lgamma(ahat + (1/3))-math.lgamma(ahat)))
    s = np.sqrt(bhat**(2/3) * (np.exp(math.lgamma(ahat + (2/3))-math.lgamma(ahat))) - (xbar)**2)
    K = Kfactor(n = n, alpha = alpha, P = P, side = side, method = method, m = m)
    lower = max(0,(xbar-s*K)**3)
    upper = (xbar+s*K)**3
    if loggamma:
        lower = np.exp(lower)
        upper = np.exp(upper)
        xbar = np.exp(xbar)
    if side == 1:
        return pd.DataFrame({"alpha":[alpha], "P":[P], "1-sided.lower":lower, "1-sided.upper":upper})
    else:
        return (pd.DataFrame({"alpha":[alpha], "P":[P], "2-sided.lower":lower, "2-sided.upper":upper}))


#x = [1,2,3,4,5,6,7,8,9,9,8,4,6,2,1,6,8,4,3,2,4,6,8,4,2,4,6,7,8,2,3,5,7,5,2,3,4,0]
#x = [1,2]
#x = [31,20,20,27,26,26,30,25,24,29,111,2,3,4,2,5,6,777,3,223,425,151,100]
#print(gamtolint(x,side=2))
gam