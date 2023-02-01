import pandas as pd
import numpy as np
import scipy.stats
import scipy.optimize as opt
import scipy.integrate as integrate
import warnings
from math import sqrt
import statistics
warnings.filterwarnings('ignore')

def length(x):
    if type(x) == float or type(x) == int or type(x) == np.int32 or type(x) == np.float64 or type(x) == np.float32 or type(x) == np.int64:
        return 1
    return len(x)

def Kfactor(n, f = None, alpha = 0.05, P = 0.99, side = 1, method = 'HE', m=50):
    K=None
    #n=np.array(n)
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
                    return integrate.quad(Fun1,a = f1 * delta**2/(ke**2 * n), b = np.inf, args=(P,ke,n,f1,delta),limit = m)
                def Fun3(ke,P,n,f1,alpha,m,delta):
                    f = Fun2(ke = ke, P = P, n = n, f1 = f1, alpha = alpha, m = m, delta = delta)
                    return abs(f[0] - (1-alpha))
                K = opt.minimize(fun=Fun3, x0=k2, args=(P,n,f,alpha,m,delta), method = 'L-BFGS-B')['x']
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
        #TEMP = np.vectorize(Ktemp)
        K = Ktemp(n=n,f=f,alpha=alpha,P=P,method=method,m=m)
    return K

def bayesnormtolint(x = None, normstats = {'xbar':np.nan,'s':np.nan,'n':np.nan},
                    alpha = 0.05, P = 0.99, side = 1, method = 'HE', m = 50, 
                    hyperpar = {'mu0':None,'sig20':None,'m0':None,'n0':None}):
    '''
Bayesian Normal Tolerance Intervals

Description
    Provides 1-sided or 2-sided Bayesian tolerance intervals under the 
    conjugate prior for data distributed according to a normal distribution.
    
    bayesnormtol.int(x = None, normstats = {'xbar':np.nan,'s':np.nan,'n':np.nan},
                    alpha = 0.05, P = 0.99, side = 1, method = ("HE", "HE2", "WBE", 
                 "ELL", "KM", "EXACT", "OCT"), m = 50,
                 hyperpar = {'mu0':None,'sig20':None,'m0':None,'n0':None})
        
Parameters
----------
    x: list
        A vector of data which is distributed according to a normal distribution.
    
    normstats: dictionary
        An optional dictionary of statistics that can be provided in-lieu of 
        the full dataset. If provided, the user must specify all three 
        components: the sample mean (xbar), the sample standard deviation (s), 
        and the sample size (n). The default values are np.nan.
    
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
            
            "HE2" is a second method due to Howe, which performs similarly to 
            the Weissberg-Beatty method, but is computationally simpler. 
            
            "WBE" is the Weissberg-Beatty method (also called the Wald-
            Wolfowitz method), which performs similarly to the first Howe 
            method for larger sample sizes. 
            
            "ELL" is the Ellison correction to the Weissberg-Beatty method 
            when f is appreciably larger than n^2. A warning message is 
            displayed if f is not larger than n^2. "KM" is the 
            Krishnamoorthy-Mathew approximation to the exact solution, which 
            works well for larger sample sizes. 
            
            "EXACT" computes the k-factor exactly by finding the integral 
            solution to the problem via the integrate function. Note the 
            computation time of this method is largely determined by m. 
            
            "OCT" is the Owen approach to compute the k-factor when 
            controlling the tails so that there is not more than (1-P)/2 of 
            the data in each tail of the distribution.
        
        The default is 'HE'.
    
    m: int, optional 
        The maximum number of subintervals to be used in the integrate 
        function. This is necessary only for method = "EXACT" and method = 
        "OCT". The larger the number, the more accurate the solution. Too low 
        of a value can result in an error. A large value can also cause the 
        function to be slow for method = "EXACT". The default is m = 50.

    hyperpar: dictionary
        A dictionary consisting of the hyperparameters for the conjugate 
        prior: the hyperparameters for the mean (mu0 and n0) and the
        hyperparameters for the variance (sig20 and m0).
    
Details
    Note that if one considers the non-informative prior distribution, then 
    the Bayesian tolerance intervals are the same as the classical solution, 
    which can be obtained by using normtol.int.
    
Returns
-------
  bayesnormtolint returns a data frame with items:
        
    alpha: 
        The specified significance level.
    P: 
        The proportion of the population covered by this tolerance interval.
    xbar:
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
    Aitchison, J. (1964), Bayesian Tolerance Regions, Journal of the Royal 
        Statistical Society, Series B, 26, 161–175.
    
    Derek S. Young (2010). tolerance: An R Package for Estimating Tolerance 
        Intervals. Journal of Statistical Software, 36(5), 1-39. 
        URL http://www.jstatsoft.org/v36/i05/.
    
    Guttman, I. (1970), Statistical Tolerance Regions: Classical and Bayesian, 
        Charles Griffin and Company.

    Young, D. S., Gordon, C. M., Zhu, S., and Olin, B. D. (2016), Sample Size 
        Determination Strategies for Normal Tolerance Intervals Using 
        Historical Data, Quality Engineering, 28, 337–351.
        
Examples
--------
    ## 95%/85% 1-sided Bayesian normal tolerance limits for a sample of size 100.
    
        x = np.random.normal(size=100)
        
        test_dict = {'mu0':'','sig20':'','m0':'','n0':''}
        
        test_list = [1,2,3,4]
        
        test_dict = dict(zip(test_dict, test_list))
        
        bayesnormtolint(x, alpha = 0.05, P = 0.85, 
                        side = 1, method = "OCT", 
                        hyperpar = test_dict)
        
    ## A similar method to fill normstats if desired.
    '''
    if(side != 1 and side != 2):
        return "must be one or two sided only"
    if x == None:
        xbar = normstats['xbar']
        s = normstats['s']
        n = normstats['n']
    else:
        xbar = np.mean(x)
        s = statistics.stdev(x)
        n = length(x)
    
    #checks to see if 0 None, all None, or between 0 and all None in hyperpar
    checklist = list(hyperpar.values())
    boollist = []
    for i in range(len(checklist)):
        if checklist[i] == None:
            boollist.append(1)        
            
    if len(boollist) == len(checklist):
        K = Kfactor(n=n,alpha=alpha,P=P,side=side,method=method,m=m)
        if K == None:
            return ''
        lower = xbar - s*K
        upper = xbar + s*K
    elif len(boollist) > 0 and len(boollist) != len(checklist):
        return 'All or 0 hyperparameters must be specified.'
    else:
        mu0 = hyperpar['mu0']
        sig20 = hyperpar['sig20']
        m0 = hyperpar['m0']
        n0 = hyperpar['n0']
        K = Kfactor(n=n0+n,f=m0+n-1,alpha=alpha,P=P,side=side,method=method,m=m)
        if K == None:
            return ''
        xbarbar = (n0*mu0+n*xbar)/(n0+n)
        q2 = (m0*sig20+(n-1)*s**2+(n0*n*(xbar-mu0)**2)/(n0+n))/(m0+n-1)
        lower = xbarbar - np.sqrt(q2)*K
        upper = xbarbar + np.sqrt(q2)*K
    if side == 2:
        temp = pd.DataFrame([[alpha,P,lower,upper]],columns=['alpha','P','2-sided.lower','2-sided.upper'])
    else:
        temp = pd.DataFrame([[alpha,P,lower,upper]],columns=['alpha','P','1-sided.lower','1-sided.upper'])   
    return temp

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

def normss(x = None, alpha = 0.05, P = 0.99, delta = None, Pprime = None, side = 1, m = 50, spec = [None,None], mu0 = None, sig20 = None, m0 = None, n0 = None, method = 'DIR', fast = False):
    '''
Sample Size Determination for Normal Tolerance Intervals  

Description
    Provides minimum sample sizes for a future sample size when constructing 
    normal tolerance intervals. Various strategies are available for 
    determining the sample size, including strategies that incorporate known 
    specification limits.
    
Usage
    norm.ss(x = None, alpha = 0.05, P = 0.99, delta = None, Pprime = None, 
        side = 1, m = 50, spec = [None, None], mu0 = None, sig20 = None, 
        m0 = None, n0 = None, method = ["DIR", "FW", "YGZO"], fast = False)

Parameters
----------
    x : list, optional
        A vector of current data that is distributed according to a normal 
        distribution. This is only required for method = "YGZO". The default 
        is None.
        
    alpha : float, optional
        The level chosen such that 1-alpha is the confidence level. The 
        default is 0.05.
        
    P : float, optional
        The proportion of the population to be covered by this tolerance 
        interval. The default is 0.99.
        
    delta : float, optional
        The precision measure for the future tolerance interval as specified 
        under the Faulkenberry-Weeks method. The default is None.
        
    Pprime : float, optional
        The proportion of the population (greater than P) such that the 
        tolerance interval of interest will only exceed Pprime by the 
        probability given by delta. The default is None.
        
    side : 1 or 2, optional
        Whether a 1-sided or 2-sided tolerance interval is required 
        (determined by side = 1 or side = 2, respectively). The default is 1.
        
    m : int, optional
        The maximum number of subintervals to be used in the integrate 
        function, which is used for the underlying exact method for 
        calculating the normal tolerance intervals. The default is 50.
        
    spec : list of length 2, optional
        A vector of length 2 given known specification limits. These are 
        required when method = "DIR" or method = "YGZO". By default, the
        values are None. The two elements of the vector are for the lower and 
        upper specification limits, respectively. If side = 1, then only one 
        of the specification limits must be specified. If side = 2, then both 
        specification limits must be specified. The default is [None,None].
        0
        
    mu0, sig20, m0, n0 : optional
        Necessary parameter values for the different methods. If 
        method = "DIR" or method = "YGZO", then mu0 and sig20 must be 
        specified, which correspond to the assumed population mean and 
        variance of the underlying normal distribution, which further pertains 
        to the historical data for method = "YGZO". If method = "YGZO" and the
        sample size is to be determined using Bayesian normal tolerance 
        intervals, then this is a required list consisting of the 
        hyperparameters for the conjugate prior – the hyperparameters for the 
        mean (mu0 and n0) and the hyperparameters for the variance 
        (sig20 and m0).
        
    method : string, optional
        The method for performing the sample size determination. "DIR" is the 
        direct method (intended as a simple calculation for planning purposes) 
        where the mean and standard deviation are taken as truth and the 
        sample size is determined with respect to the given specification 
        limits. "FW" is for the traditional Faulkenberry-Weeks approach for 
        sample size determination. "YGZO" is for the Young-Gordon-Zhu-Olin 
        approach, which incorporates historical data and specification limits 
        for determining the value of delta and/or Pprime in the 
        Faulkenberry-Weeks approach. Note that for "YGZO", at least one of 
        delta and Pprime must be None. The default is 'DIR'.
        
    fast : bool, optional
        Specifies the computational complexity. If the user wants a fast 
        output, but slightly inaccurate response relative to R, use 
        fast = True. Otherwise, if the user wants a completely accurate 
        response relative to R, use fast = False. The default is False.

Returns
-------
    normss returns a data frame items:
        alpha :
            The specified significance level
        
        P :
            The proportion of the population covered by this tolerance 
            interval.
        
        delta :
            The user-specified or calculated precision measure. Not returned 
            if method = "DIR".
            
        Pprime : 
            The user-specified or calculated closeness measure. Not returned 
            if method = "DIR".
        
        n :
            The minimum sample size determined using the conditions specified 
            for this function.

References
    Derek S. Young (2010). tolerance: An R Package for Estimating Tolerance 
        Intervals. Journal of Statistical Software, 36(5), 1-39. 
        URL http://www.jstatsoft.org/v36/i05/.
        
    Faulkenberry, G. D. and Weeks, D. L. (1968), Sample Size Determination for 
        Tolerance Limits, Technometrics, 10, 343–348.

    Young, D. S., Gordon, C. M., Zhu, S., and Olin, B. D. (2016), Sample Size 
        Determination Strategies for Normal Tolerance Intervals Using 
        Historical Data, Quality Engineering, 28, 337–351.
        
Examples
    ## Sample size determination for 95%/95% 2-sided normal tolerance 
    intervals using the direct method.
       
        norm.ss(alpha = 0.05, P = 0.95, side = 2, spec = [-3, 3], 
        method = "DIR", mu0 = 0, sig20 = 1) 
        
    ## Sample size determination for 95%/95% 2-sided normal tolerance 
    intervals using the Faulkenberry-Weeks method.
        
        normss(x=[1,2],alpha = 0.05, P = 0.95, side = 2, spec = [-4,4],
               method = 'FW', mu0 = 1, sig20 = 1.1, m0 = 12, n0 = 30, 
               delta = .62, Pprime = .998)
    '''
    if side != 1 and side != 2:
        return "Must specify a one-sided or two-sided procedure."
    if spec[0] == None:
        specL = None
    else:
        specL = spec[0]
    if spec[1] == None:
        specU = None
    else:
        specU = spec[1]
    if method == 'DIR':
        if (mu0 == None or sig20 == None) or (specL == None and specU == None):
            return "Must specify mu0 and sig20 as well as the appropriate spec limit(s)!"
        s0 = np.sqrt(sig20)
        def f1(n,mu,sigma,alpha,P,side,specU):
            return specU - (mu+Kfactor(n=n,alpha=alpha,P=P,side=side,method='OCT',m=m)*sigma)
        def f2(n,mu,sigma,alpha,P,side,specL):   
            return (mu-Kfactor(n=n,alpha=alpha,P=P,side=side,method='OCT',m=m)*sigma)-specL

        if side == 1:
            if specL == None:
                try:
                    fcalc = f1(2,mu=mu0,sigma=s0,alpha=alpha,P=P,side=side,specU=specU)
                    n = np.ceil(opt.brentq(f1,a=2,b=1e10,args=(mu0,s0,alpha,P,side,specU),maxiter = 1000))
                except:
                    if fcalc <0:
                        n = np.inf
                    elif fcalc >= 0 :
                        n = 2
            else:
                fcalc = f2(2,mu=mu0,sigma=s0,alpha=alpha,P=P,side=side,specL=specL)
                try:
                    n = np.ceil((opt.brentq(f2,a=2,b=1e10,args=(mu0,s0,alpha,P,side,specL),maxiter = 1000)))
                    if n > 520:
                        print('Warning: sigma significantly different than mu. Python slightly underestimates n relative to R. The bigger n is, the more Python underestimates. For n < 521, Python and R return the same value.')
                except:
                    if fcalc < 0:
                        n = np.inf
                    elif fcalc >= 0:
                        n = 2
        else:
            try:
                dL = abs(mu0-specL)
                dU = abs(mu0-specU)
            except:
                return 'The function must have the argument spec, a vector of length 2, inputted.'
            
            try:
                if dL <= dU:
                    fcalc = f2(2,mu0,s0,alpha,P,1,specL)
                    n = np.ceil(opt.brentq(f2,a=2,b=1e10,args=(mu0,s0,alpha,P,1,specL),maxiter=1000))
                else:
                    fcalc = f1(2,mu0,s0,alpha,P,1,specU)
                    n = np.ceil(opt.brentq(f1,a=2,b=1e10,args=(mu0,s0,alpha,P,1,specU),maxiter=1000))
            except:
                if fcalc < 0:
                    n = np.inf
                elif fcalc >= 0:
                    n = 2    
            else:
                TI01 = mu0 -1*Kfactor(n=1e100,alpha=alpha,P=P,side=2,method='HE')*s0
                TI02 = mu0 +1*Kfactor(n=1e100,alpha=alpha,P=P,side=2,method='HE')*s0
                if (TI01 <= specL or TI02 >= specU):
                    n = np.inf
                else:
                    withinspec=[False]
                    newn=n
                    def TITest(x,L,U):
                        return [x[0].any()>=L and x[1].any()<=U]
                    inc = 1
                    while(np.sum(withinspec)==0 and n < np.inf):
                        newn = int(newn)
                        newn = list(range(newn,(newn+inc*1000)+1))
                        if length(newn) == 1:
                            K2 = Kfactor(n = newn, alpha = alpha, P = P, side = 2, method = 'HE')
                        else:
                            K2 = np.zeros(length(newn))
                            for i in range(len(K2)):
                                K2[i] = Kfactor(n = newn[i], alpha = alpha, P = P, side = 2, method = 'HE')
                        TI1 = mu0 - K2*s0
                        TI2 = mu0 + K2*s0
                        withinspec.extend(TITest(x=[TI1,TI2],L=specL,U=specU))
                        if sum(withinspec) == 0:
                            newn = newn[-1]+1
                        else:
                            n = newn[np.min(np.where(withinspec))]
                        inc = inc+1
                        if inc > 500:
                            n = np.inf
                    withinspec = False
                    nold = n
                    n = max(1,n-8)
                    brk = True
                    if fast == False:
                        print('fast = False by default, the results are identical to R. However, if you are in a rush and you are fine with error relatvive to R, you can set fast = True in the function argument to exponentially speed up computiton.')
                    while (not withinspec) and brk:
                        n = n+1
                        try:
                            if fast == False:
                                TI1 = mu0-1*Kfactor(n=n,alpha=alpha,P=P,side=2,method='OCT',m=m)*s0
                                TI2 = mu0+1*Kfactor(n=n,alpha=alpha,P=P,side=2,method='OCT',m=m)*s0
                            else:
                                TI1 = mu0-1*Kfactor(n=n,alpha=alpha,P=P,side=2,method='HE',m=m)*s0
                                TI2 = mu0+1*Kfactor(n=n,alpha=alpha,P=P,side=2,method='HE',m=m)*s0
                        except:
                            n = nold
                            brk=False
                        else:
                            withinspec = TI1>=specL and TI2 <=specU
    else:                        
        if method == "YGZO":
                if type(x) == None:
                    return 'Data must be provided to use this method.'
                if specL == None and specU == None:
                    return 'Must specify the appropriate spec limit(s) for this method.'
                if m0 == None and n0 == None:
                    TIout = list(normtolint(x=x,alpha=alpha,P=P,side=side,method = "EXACT",m=m).iloc[0][3:])
                else:
                    TIout = list(bayesnormtolint(x=x,alpha=alpha,P=P,side=side,method = "EXACT",m=m).iloc[0][3:])
                s0 = sqrt(sig20)
                if delta == None or Pprime == None:
                    if side == 1 and (specL == None ^ specU == None):
                        return 'You must specify a single value for one (and only one) of specL or specU.'
                    if side == 2 and (specL== None or specU == None):
                        return 'Valyes for both specL and specU must be specified'
                    if Pprime == None:
                        if type(specU) == None and type(specL) == None:
                            Pprime = (1+P)/2
                        else:
                            if side == 2:
                                Pprime = scipy.stats.norm.cdf(specU, loc = mu0, scale = s0) - scipy.stats.norm.cdf(specL,loc = mu0, scale  = s0)
                                if Pprime <= P or Pprime >=1:
                                    Pprime = (1+P)/2
                            else:
                                if not specL == None:
                                    Pprime = scipy.stats.norm.sf(specL, loc = mu0, scale = s0)
                                else:
                                    Pprime = scipy.stats.norm.cdf(specU, loc = mu0, scale = s0)
                                if Pprime <= P or Pprime >= 1:
                                    Pprime = (1+P)/2
                if delta == None:
                    if side == 1:
                        if not type(specL) == None:
                            cont = scipy.stats.norm.sf(TIout[0],loc = mu0, scale = s0)
                            delta = abs(cont-P)/P
                        else:
                            cont = scipy.stats.norm.cdf(TIout[1],loc=mu0,scale=s0)
                            delta = abs(cont-P)/P
                    else:
                        if specL == None and specU == None:
                            return 'Must specify both spec limits.'
                        cont = np.diff(scipy.stats.norm.cdf(TIout, loc = mu0, scale = s0))
                        delta = abs(cont-P)/P
        if method == 'FW' and (delta == None or Pprime == None):
            return "You must specify delta and Pprime."
        if side == 1:
            def norm1(n,P,alpha,Pprime,delta):
                return Kfactor(n=n,P=P,alpha=alpha,side=2,method="HE")-Kfactor(n=n,P=Pprime,alpha=1-delta,side=2,method="HE")
            newn = np.floor(opt.brentq(norm1, a = 2, b = 1e10, args=(P,alpha,Pprime,delta),maxiter = 1000))
        else:
            def norm2(n,P,alpha,Pprime,delta):
                return Kfactor(n=n,P=P,alpha=alpha,side=2,method="HE")-Kfactor(n=n,P=Pprime,alpha=1-delta,side=2,method="HE")
            def norm2ex(n,P,alpha,Pprime,delta):
                return Kfactor(n=n,P=P,alpha=alpha,side=2,method="EXACT")-Kfactor(n=n,P=Pprime,alpha=1-delta,side=2,method="EXACT")
            nstar = np.ceil(opt.brentq(norm2, a = 2, b = 1e10, args=(P,alpha,Pprime,delta),maxiter = 1000)) 
            newn = nstar + np.array(range(-2,3))
            newn = newn[np.where(newn>3)]
            try:
                df1 = pd.DataFrame(newn)
                # if length(newn) > 1:
                #     df2 = []
                #     df3 = []
                #     for i in range(length(newn)):
                #         df2.append(Kfactor(n=newn[i],P=P,alpha=alpha,side=2,method="EXACT",m=m))
                #         df3.append(Kfactor(n=newn[i],P=Pprime,alpha=1-delta,side=2,method="EXACT",m=m))
                #     df2 = pd.DataFrame(df2)
                #     df3 = pd.DataFrame(df3)
                #     out = pd.concat([df1,df2,df3],axis=1)
                # else:
                df2 = Kfactor(n=newn,P=P,alpha=alpha,side=2,method="EXACT",m=m)
                df3 = Kfactor(n=newn,P=Pprime,alpha=1-delta,side=2,method="EXACT",m=m) 
                df2 = pd.DataFrame(float(df2))
                df3 = pd.DataFrame(float(df3))
                out = pd.concat([df1,df2,df3],axis=1)
            except:
                newn = nstar
            else:
                diff = out.iloc[:,1]-out.iloc[:,2]
                if sum(diff<0) == 0:
                    newn = nstar + np.array(range(2,7))
                    df1 = pd.DataFrame(newn)
                    df2 = pd.DataFrame(Kfactor(n=newn,P=P,alpha=alpha,side=2,method="EXACT",m=m))
                    df3 = pd.DataFrame(Kfactor(n=newn,P=Pprime,alpha=1-delta,side=2,method="EXACT",m=m))
                    out = pd.concat([df1,df2,df3],axis=1)
                    diff2 = out.iloc[:,1]-out.iloc[:,2]
                    if sum(diff2<0)==0:
                        newn = newn[-1]
                        tst=1
                        while tst > 0:
                            newn = newn+20
                            tst =  Kfactor(n=newn,P=P,alpha=alpha,side=2,method="EXACT",m=m)-Kfactor(n=newn,P=Pprime,alpha=1-delta,side=2,method="EXACT",m=m)
                        tst=1
                        newn = newn-20
                        while tst > 0:
                            newn = newn+1
                            tst = Kfactor(n=newn,P=P,alpha=alpha,side=2,method="EXACT",m=m)-Kfactor(n=newn,P=Pprime,alpha=1-delta,side=2,method="EXACT",m=m)
                    else:
                        min_diff2_idx = diff2[diff2<0].idxmin()
                        out = out.iloc[:,0]
                        newn = out.iloc[min_diff2_idx]
                elif np.min(diff<0) == 1:
                    newn = nstar + np.array(range(-6,-1))
                    diff2 = out.iloc[:,1]-out.iloc[:,2]
                    df1 = pd.DataFrame(newn)
                    # if length(newn) > 1:
                    #     df2 = []
                    #     df3 = []
                    #     for i in range(length(newn)):
                    #         df2.append(Kfactor(n=newn[i],P=P,alpha=alpha,side=2,method="EXACT",m=m))
                    #         df3.append(Kfactor(n=newn[i],P=Pprime,alpha=1-delta,side=2,method="EXACT",m=m))
                    #     df2 = pd.DataFrame(df2)
                    #     df3 = pd.DataFrame(df3)
                    #     out = pd.concat([df1,df2,df3],axis=1)
                    # else:
                    df2 = Kfactor(n=newn,P=P,alpha=alpha,side=2,method="EXACT",m=m)
                    df3 = Kfactor(n=newn,P=Pprime,alpha=1-delta,side=2,method="EXACT",m=m)
                    df2 = pd.DataFrame(float(df2))
                    df3 = pd.DataFrame(float(df3))
                    out = pd.concat([df1,df2,df3],axis=1)
                    min_diff2_idx = diff2[diff2<0].idxmin()
                    out = out.iloc[:,0]
                    newn = out.iloc[min_diff2_idx]     
                else:
                    print(out)
                    min_diff_idx = diff[diff<0].idxmin()
                    out = out.iloc[:,0]
                    newn = out.iloc[min_diff_idx]
                    
    if method == 'DIR':
        Pprime = ''
        delta = ''
    else:
        n = newn
        delta = float(delta)
    return pd.DataFrame({'alpha':[alpha],'P':[P],'delta':[delta],'P.prime':[Pprime],'n':[int(n)]})


print(normss(alpha = 0.05, P = 0.95, side = 2, spec = [-4,4],method = 'DIR', mu0 = 1, sig20 = 1.1,m0=12,n0=30, delta = .62, Pprime = .998,fast = True))







