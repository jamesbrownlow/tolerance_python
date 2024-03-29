import scipy.stats 
import scipy.integrate as integrate
import numpy as np
import scipy.optimize as opt
import pandas as pd
import warnings
import statistics
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
                    print("Warning Message:\nThe ellison method should only be used for f appreciably larger than n^2")
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
                    if n < 75:
                        return integrate.quad(Fun1,a = f1 * delta**2/(ke**2 * n), b = np.inf, args=(P,ke,n,f1,delta),limit = m)
                    else:
                        return integrate.quad(Fun1,a = f1 * delta**2/(ke**2 * n), b = n*1000, args=(P,ke,n,f1,delta),limit = m)
                def Fun3(ke,P,n,f1,alpha,m,delta):
                    f = Fun2(ke = ke, P = P, n = n, f1 = f1, alpha = alpha, m = m, delta = delta)
                    return abs(f[0] - (1-alpha))
                K = opt.minimize(fun=Fun3, x0=k2,args=(P,n,f,alpha,m,delta), method = 'L-BFGS-B')['x']
                return float(K)
            elif method == 'EXACT':
                def fun1(z,df1,P,X,n):
                    k = (scipy.stats.chi2.sf(df1*scipy.stats.ncx2.ppf(P,1,z**2)/X**2,df=df1)*np.exp(-0.5*n*z**2))
                    return k
                def fun2(X,df1,P,n,alpha,m):
                    return integrate.quad(fun1,a =0, b = 5, args=(df1,P,X,n),limit=m)
                def fun3(X,df1,P,n,alpha,m):
                    return np.sqrt(2*n/np.pi)*fun2(X,df1,P,n,alpha,m)[0]-(1-alpha)
                K = opt.brentq(f=fun3,a=0,b=k2+(1000)/n, args=(f,P,n,alpha,m))
                return K
        K = Ktemp(n=n,f=f,alpha=alpha,P=P,method=method,m=m)
    return K

def normtolint(x, alpha = 0.05, P = 0.99, side = 1, method = 'HE', m = 50, lognorm = False):
    '''
    normtolint(x, alpha = 0.05, P = 0.99, side = 1, method = ["HE", "HE2", "WBE", "ELL", "KM", "EXACT", "OCT"], m = 50, lognorm = False):
        
Parameters
----------
    x: list
        A vector of data which is distributed according to either a normal 
        distribution or a log-normal distribution.
    
    alpha: float, optional
        The level chosen such that 1-alpha is the confidence level.
        The default is 0.05.
    
    P: float, optional
        The proportion of the population to be covered by this tolerance 
        interval. The default is 0.99.
    
    side: 1 or 2, optional
        Whether a 1-sided or 2-sided tolerance interval is required 
        (determined by side = 1 or side = 2, respectively). The default is 1.
    
    method: string, optional
        The method for calculating the k-factors. The k-factor for the 1-sided 
        tolerance intervals is performed exactly and thus is the same for the 
        chosen method. 
        
            "HE" is the Howe method and is often viewed as being extremely 
            accurate, even for small sample sizes. 
        
            "HE2" is a second method due to Howe, which performs similarly to the 
            Weissberg-Beatty method, but is computationally simpler. 
        
            "WBE" is the Weissberg-Beatty method 
            (also called the Wald-Wolfowitz method), which performs similarly to 
            the first Howe method for larger sample sizes. 
            
            "ELL" is the Ellison correction to the Weissberg-Beatty method when f 
            is appreciably larger than n^2. A warning message is displayed if f is
            not larger than n^2. "KM" is the Krishnamoorthy-Mathew approximation 
            to the exact solution, which works well for larger sample sizes. 
            
            "EXACT" computes the k-factor exactly by finding the integral solution 
            to the problem via the integrate function. Note the computation time 
            of this method is largely determined by m. 
            
            "OCT" is the Owen approach to compute the k-factor when controlling 
            the tails so that there is not more than (1-P)/2 of the data in each 
            tail of the distribution.
            
        The default is "HE"
    
    m: int, optional 
        The maximum number of subintervals to be used in the integrate 
        function. This is necessary only for method = "EXACT" and method = 
        "OCT". The larger the number, the more accurate the solution. Too low 
        of a value can result in an error. A large value can also cause the 
        function to be slow for method = "EXACT". The default is m = 50.

    lower: float, optional
        If TRUE, then the data is considered to be from a log-normal 
        distribution, in which case the output gives tolerance intervals for 
        the log-normal distribution. The default is False.
    
Details
    Recall that if the random variable X is distributed according to a 
    log-normal distribution, then the random variable Y = ln(X) is distributed 
    according to a normal distribution.
    
Returns
-------
  normtolint returns a data frame with items:
        
    alpha: 
        The specified significance level.
    P: 
        The proportion of the population covered by this tolerance interval.
    mean:
        The sample mean.
    1-sided.lower: 
        The 1-sided lower tolerance bound. This is given only if side = 1.
    1-sided.upper: 
        The 1-sided upper tolerance bound. This is given only if side = 1.
    2-sided.lower: 
        The 2-sided lower tolerance bound. This is given only if side = 2.
    2-sided.upper: 
        The 2-sided upper tolerance bound. This is given only if side = 2.

References
----------
    Derek S. Young (2010). tolerance: An R Package for Estimating Tolerance 
        Intervals. Journal of Statistical Software, 36(5), 1-39. 
        URL http://www.jstatsoft.org/v36/i05/.
    
    Howe, W. G. (1969), Two-Sided Tolerance Limits for Normal Populations - 
        Some Improvements, Journal of the American Statistical Association, 
        64, 610–620.

    Wald, A. and Wolfowitz, J. (1946), Tolerance Limits for a Normal 
        Distribution, Annals of Mathematical Statistics, 17, 208–215.

    Weissberg, A. and Beatty, G. (1969), Tables of Tolerance Limit Factors 
        for Normal Distributions, Technometrics, 2, 483–500.
        
Examples
--------
    ## 95%/95% 2-sided normal tolerance intervals for a sample of size 100. 
    
        x = np.random.normal(size=100)
        
        normtolint(x, alpha = 0.05, P = 0.95, side = 2, 
                    method = "HE", log.norm = FALSE)
    '''
    if lognorm:
        x = np.log(x)
    xbar = np.mean(x)
    s = statistics.stdev(x)
    n = len(x)
    K = Kfactor(n, alpha=alpha, P=P, side = side, method= method, m = m)
    lower = xbar-s*K
    upper = xbar+s*K
    if(lognorm):
        lower = np.exp(lower)
        upper = np.exp(upper)
        xbar = np.exp(xbar)
    if side == 1:
        temp = pd.DataFrame([[alpha,P, xbar,lower,upper]],columns=['alpha','P','mean','1-sided.lower','1-sided.upper'])
        return temp
    else:
        temp = pd.DataFrame([[alpha,P, xbar,lower,upper]],columns=['alpha','P','mean','2-sided.lower','2-sided.upper'])
        return temp
    
def bonftolint(fn, P1 = 0.005, P2 = 0.005, alpha = 0.05, *args, **kwargs):
    '''
Approximate 2-Sided Tolerance Intervals that Control the Tails Using 
Bonferroni's Inequality

Description
    This function allows the user to control what proportion of the population 
    is to be in the tails of the given distribution for a 2-sided tolerance 
    interval. The result is a conservative approximation based on Bonferroni's 
    inequality.
    
Usage
    bonftoint(fn, P1 = 0.005, P2 = 0.005, alpha = 0.05, *args, **kwargs)

Parameters
----------
    fn : tolerance interval function (pandas.DataFrame)
        The function name for the 2-sided tolerance interval to be calculated.
        
    P1 : flaot, optional
        The proportion of the population not covered in the lower tail of the 
        distribution. The default is 0.005.
        
    P2 : float, optional
        The proportion of the population not covered in the upper tail of the 
        distribution. The default is 0.005.
        
    alpha : float, optional
        The level chosen such that 1-alpha is the confidence level. The 
        default is 0.05.
        
    *args and **kwargs : additional arguments
        Additional arguments passed to fn, including the data. All arguments 
        that would be specified in fn must also be specified here.


Returns
-------
    bonftolint returns a dataframe with items:
        The results for the 2-sided tolerance interval procedure are reported. 
        See the corresponding help file for fn about specific output. 
        Note that the (minimum) proportion of the population to be covered by
        this interval is 1 - (P1 + P2).
        
Note
----
    This function can be used with any 2-sided tolerance interval function, 
    including the regression tolerance interval functions.

References
----------
    Derek S. Young (2010). tolerance: An R Package for Estimating Tolerance 
        Intervals. Journal of Statistical Software, 36(5), 1-39. 
        URL http://www.jstatsoft.org/v36/i05/.

    Jensen, W. A. (2009), Approximations of Tolerance Intervals for Normally 
        Distributed Data, Quality and Reliability Engineering International, 
        25, 571–580.

    Patel, J. K. (1986), Tolerance Intervals - A Review, Communications in 
        Statistics - Theory and Methodology, 15, 2719–2762.

Examples
-------- 
    ## 95%/97% tolerance interval for normally distributed data controlling 1% 
    of the data is in the lower tail and 2% of the data in the upper tail.

        x = [0,1,1,1,2,2,2,2,2,3,3,3,3,3,3,3,4,4,4,4,4,5,5,5,6]
    
        bonftolint(normtolint, P1 = 0.01, P2 = 0.02, alpha = 0.05,
                   method = 'HE',x = x)

    '''
    P = 1 - (P1+P2)
    if P < 0:
        return "Invalid values for P1 and P2. 1-(P1+P2) should be >= 0."
    lower = fn(P = 1 - P1, side = 1, alpha = alpha,*args, **kwargs)
    #lower = pd.concat([fn(P = 1 - P1, side = 1, alpha = alpha,*args, **kwargs),fn(P = 1 - P2, side = 1, alpha = alpha,*args, **kwargs)])
    upper = fn(P = 1 - P2, side = 1, alpha = alpha,*args, **kwargs)
    #upper = pd.concat([fn(P = P1, side = 1, alpha = alpha,*args, **kwargs),fn(P = P2, side = 1, alpha = alpha,*args, **kwargs)])
    print(f"These are {(1 - alpha) * 100}%/{P * 100}% 2-sided tolerance limits controlling {P1 * 100}% in the lower tail and {P2 * 100}% in the upper tail.")
    if type(lower) != list:
        lower = lower.drop(columns = '1-sided.upper') #drop only column '1-sided.upper'
        upper = upper[['1-sided.upper']] #drop all columns except '1-sided.upper'
        out = pd.concat([lower,upper],axis=1) 
        out = out.rename(columns = {'1-sided.lower':'2-sided.lower','1-sided.upper':'2-sided.upper'})
        out.iloc[:,1] = P 
        return out
    else:
        s = '\nThe inputted function returns a list, if this is ever happens email me at baileyarzate@gmail.com so I can add code here to handle it. I couldn\'t find a case while testing'
        return s
    
# x = [0,1.1,1,1,2,2,2,2,2,3,3,3,3,3,3,3,4,4,4,4,4,5,5,5,6,2]

# print(bonftolint(normtolint, P1 = 0.01, P2 = 0.02, alpha = 0.05,method = 'HE',x = x))
# print(bonftolint(normtolint, P1 = 0.01, P2 = 0.02, alpha = 0.05,method = 'HE2',x = x))
# print(bonftolint(normtolint, P1 = 0.01, P2 = 0.02, alpha = 0.05,method = 'WBE',x = x))
# print(bonftolint(normtolint, P1 = 0.01, P2 = 0.02, alpha = 0.05,method = 'ELL',x = x))
# print(bonftolint(normtolint, P1 = 0.01, P2 = 0.02, alpha = 0.05,method = 'KM',x = x))
# print(bonftolint(normtolint, P1 = 0.01, P2 = 0.02, alpha = 0.05,method = 'EXACT',x = x))
# print(bonftolint(normtolint, P1 = 0.01, P2 = 0.02, alpha = 0.05,method = 'OCT',x = x))