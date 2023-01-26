import scipy.stats
import numpy as np
import pandas as pd
import scipy.integrate as integrate
import scipy.optimize as opt
import statistics as st
import warnings
warnings.filterwarnings('ignore')

def length(x):
    if type(x) == int or type(x) == float:
        return 1
    else:
        return len(x)

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
        s = st.stdev(x)
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

# #need this if you want to initialize hyperpar
# test_dict = {'mu0':'','sig20':'','m0':'','n0':''}
# test_list = [1,2,3,4]
# test_dict = dict(zip(test_dict, test_list))

# #need this if you want to initialize normstats
# stats_dict = {'xbar':'','s':'','n':''}
# stats_list = [1,2,3]
# stats_dict = dict(zip(stats_dict,stats_list))

# sims = 50
# lower = np.zeros(sims)
# upper = np.zeros(sims)
# #for i in range(sims):
# #    lower[i], upper[i] = bayesnormtolint(x=np.random.normal(size=100), hyperpar = test_dict).iloc[0][2:4]

# print(bayesnormtolint(x=[1,2,3,4],method = 'OCT', side = 2, hyperpar = test_dict))
# print(bayesnormtolint(x=[1,2,3,4],method = 'OCT', side = 2))#,hyperpar = test_dict))
# print(bayesnormtolint(normstats = stats_dict, method = 'OCT', side = 2,hyperpar=test_dict))#,hyperpar = test_dict))
    
    
    
    
    
    
    
    