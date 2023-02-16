import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def length(x):
    if type(x) == float or type(x) == int or type(x) == np.int32 or type(x) == np.float64 or type(x) == np.float32 or type(x) == np.int64:
        return 1
    try:
        return len(x)
    except:
        return 0

import scipy.stats
#import pandas as pd
import scipy.integrate as integrate
#import statistics as st
import warnings
warnings.filterwarnings('ignore')

import scipy.optimize as opt

def KfactorP(P, n = 10, alpha = 0.05,  side = 1, method = 'HE', m=50,k=0):
    K=None
    #if f == None:
    f = n-1
    # if (length((n,)*1)) != length((f,)*1) and (length((f,)*1) > 1):
    #     return 'Length of \'f\' needs to match length of \'n\'!'
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
        #TEMP = np.vectorize(Ktemp)
        K = Ktemp(n=n,f=f,alpha=alpha,P=P,method=method,m=m)
    return k-K

def Kfactoralpha(alpha, n = 10, P = 0.05,  side = 1, method = 'HE', m=50,k=0):
    K=None
    #if f == None:
    f = n-1
    # if (length((n,)*1)) != length((f,)*1) and (length((f,)*1) > 1):
    #     return 'Length of \'f\' needs to match length of \'n\'!'
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
    return k-K

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
                    return integrate.quad(Fun1,a = f1 * delta**2/(ke**2 * n), b = np.inf, args=(P,ke,n,f1,delta),limit = m)
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

def Ktable(n, alpha, P, side = 1, f = None, method = 'HE', m = 50, byarg = 'n'):
    '''
Tables of K-factors for Tolerance Intervals Based on Normality

Description
    Tabulated summary of k-factors for tolerance intervals based on normality. 
    The user can specify multiple values for each of the three inputs.
    
    K.table(n, alpha, P, side = 1, f = NULL, method = ["HE", 
        "HE2", "WBE", "ELL", "KM", "EXACT", "OCT"], m = 50,
        by.arg = ["n", "alpha", "P"]) 
    
Parameters
----------
    n : list
        A vector of (effective) sample sizes.
        
    alpha : float or list
        The level chosen such that 1-alpha is the confidence level. Can be a 
        vector.

    P : float or list
        The proportion of the population to be covered by this tolerance 
        interval. Can be a vector.
    
    f : int, optional
        The number of degrees of freedom associated with calculating the
        estimate of the population standard deviation. If NULL, then f is 
        taken to be n-1. Only a single value can be specified for f. The
        default is None. 
        
    side : 1 or 2, optional
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
        
    m : int, optional
        The maximum number of subintervals to be used in the integrate 
        function. This is necessary only for method = "EXACT". The larger the 
        number, the more accurate the solution. Too low of a value can result 
        in an error. A large value can also cause the function to be slow for 
        method = "EXACT". The default is 50.
        
    byarg : string, optional
            How you would like the output organized. If by.arg = "n", then the 
            output provides a list of matrices sorted by the values specified 
            in n. The matrices have rows corresponding to the values specified 
            by 1-alpha and columns corresponding to the values specified by P. 
            If by.arg = "alpha", then the output provides a list of matrices 
            sorted by the values specified in 1-alpha. The matrices have rows 
            corresponding to the values specified by n and columns 
            corresponding to the values specified by P. If by.arg = "P", then
            the output provides a list of matrices sorted by the values 
            specified in P. The matrices have rows corresponding to the values 
            specified by 1-alpha and columns corresponding to the values 
            specified by n. The default is 'n'.

Details
-------
    The method used for estimating the k-factors is that due to Howe as it is 
    generally viewed as more accurate than the Weissberg-Beatty method.

Returns
-------
    Ktable returns a list with a structure determined by the argument by.arg 
    described above. There is no 'return' value, the values 'returned' are 
    not returned but printed inside of the function. You should not assign 
    this function to a variable. 
    
References
----------
    Derek S. Young (2010). tolerance: An R Package for Estimating Tolerance 
        Intervals. Journal of Statistical Software, 36(5), 1-39. 
        URL http://www.jstatsoft.org/v36/i05/.
    
    Howe, W. G. (1969), Two-Sided Tolerance Limits for Normal Populations - 
        Some Improvements, Journal of the American Statistical Association, 
        64, 610–620.

    Weissberg, A. and Beatty, G. (1969), Tables of Tolerance Limit Factors for 
        Normal Distributions, Technometrics, 2, 483–500.

Examples
--------
## Tables generated for each value of the sample size.

    Ktable(n=[2,3,4,5], alpha=[.7,.3,.4], P=[.99,.98,.97,.96,.95], 
           side = 1, f = None, method = 'HE', m = 50, byarg = 'alpha'
                 
    Ktable(n=2, alpha=.05, P=[.99,.98], side = 1, f = None, method = 'HE', 
           m = 50, byarg = 'alpha')
    '''
    nn = length(n)
    na = length(alpha)
    nP = length(P)
    if length(n) == 1:
        n = [n]
    if length(P) == 1:
        P = [P]
    if length(alpha) == 1:
        alpha = [alpha]
    if byarg == 'alpha':
        for l in range(na):
            temp = None
            for i in range(nn):
                t = pd.DataFrame([None])
                for j in range(nP):
                    K = pd.DataFrame([Kfactor(n=n[i],alpha=alpha[l],P = P[j],side=side,method=method,f=f,m=m)])
                    t = pd.concat([t,K],axis = 1)
                    t1 = pd.DataFrame([t.iloc[0].values[1:]])
                temp = pd.concat([temp,t1])
            temp.index = n
            temp.columns = P
            #print(f'\nalpha = {round(1-alpha[l],3)}')
            return temp
            
    elif byarg == 'n':
        for l in range(nn):
            temp = None
            for i in range(na):
                t = pd.DataFrame([None])
                for j in range(nP):
                    K = pd.DataFrame([Kfactor(n=n[l],alpha=alpha[i],P = P[j],side=side,method=method,f=f,m=m)])
                    t = pd.concat([t,K],axis = 1)
                    t1 = pd.DataFrame([t.iloc[0].values[1:]])
                temp = pd.concat([temp,t1])
            if length(alpha) == 1:
                try:
                    alpha = 1-alpha
                except:
                    alpha = [1-a for a in alpha[0]]
            else:
                alpha = [1-a for a in alpha]
            temp.index = alpha
            temp.columns = P
            #print(f'\nn = {round(n[l],1)}')
            #alpha = [1-a for a in alpha]
            return temp
        
    elif byarg == 'P':
        for l in range(nP):
            temp = None
            for i in range(na):
                t = pd.DataFrame([None])
                for j in range(nn):
                    K = pd.DataFrame([Kfactor(n=n[j],alpha=alpha[i],P = P[l],side=side,method=method,f=f,m=m)])
                    t = pd.concat([t,K],axis = 1)
                    t1 = pd.DataFrame([t.iloc[0].values[1:]])
                temp = pd.concat([temp,t1])
            alpha = [1-a for a in alpha]
            temp.index = alpha
            temp.columns = n
            #print(f'\nP = {round(P[l],3)}')
            #alpha = [1-a for a in alpha]
            return temp
    else:
        return 'Must specify index for table!'
    return ''

def flatten(A):
    rt = []
    for i in A:
        if isinstance(i,list): rt.extend(flatten(i))
        else: rt.append(i)
    return rt

def normOC(n, k = None, alpha = None, P = None, side = 1, method = 'HE', m = 50):
    '''
Operating Characteristic (OC) Curves for K-Factors for Tolerance Intervals 
Based on Normality

Description
Provides OC-type curves to illustrate how values of the k-factors for normal 
tolerance intervals, confidence levels, and content levels change as a 
function of the sample size.

Usage
    normOC(n, k = None, alpha = None, P = None, side = 1, method = "HE", 
           m = 50)
    
Parameters
----------
    n: list of ints
        A sequence of sample sizes to consider. This must be a vector of at 
        least length 2 since all OC curves are constructed as functions of n.
    k: int, optional
        If wanting OC curves where the confidence level or content level is on 
        the y-axis, then a single positive value of k must be specified. This 
        would be the target k-factor for the desired tolerance interval. If 
        k = None, then OC curves will be constructed where the k-factor value 
        is found for given levels of alpha, P, and n. The default is None.

    alpha: list of floats, optional
        The set of levels chosen such that 1-alpha are confidence levels. If 
        wanting OC curves where the content level is being calculated, then 
        each curve will correspond to a level in the set of alpha. If a set of
        P values is specified, then OC curves will be constructed where the 
        k-factor is found and each curve will correspond to each combination 
        of alpha and P. If alpha = NULL, then OC curves will be constructed to
        find the confidence level for given levels of k, P, and n. The default
        is None. 

    P: list of floats, optional
        The set of content levels to be considered. If wanting OC curves where
        the confidence level is being calculated, then each curve will 
        correspond to a level in the set of P. If a set of alpha values is
        specified, then OC curves will be constructed where the k-factor is 
        found and each curve will correspond to each combination of alpha and 
        P. If P = None, then OC curves will be constructed to find the content 
        level for given levels of k, alpha, and n. The default is None.

    side: 1 or 2, optional
        Whether a 1-sided or 2-sided tolerance interval is required 
        (determined by side = 1 or side = 2, respectively).

    method: string, optional
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
        The default is 'HE'.

    m: int, optional
        The maximum number of subintervals to be used in the integrate 
        function, which is used for the underlying exact method for 
        calculating the normal tolerance intervals.

Returns
-------
    normOC returns a figure with the OC curves constructed using the 
    specifications in the arguments.

References
    Derek S. Young (2010). tolerance: An R Package for Estimating Tolerance 
        Intervals. Journal of Statistical Software, 36(5), 1-39. 
        URL http://www.jstatsoft.org/v36/i05/.
        
    Young, D. S. (2016), Normal Tolerance Interval Procedures in the tolerance 
        Package, The R Journal, 8, 200–212.

Note
----
    For n sufficiently large, min(n) > 1000, the results approach the 
    y asymptote. The graphs may be misleading for sufficiently large n.

Examples
    ## The three types of OC-curves that can be constructed with the normOC 
    function.
        
        ## Example 1, finding k-factors
    
            alphalist = np.arange(start=0.01,stop=0.1,step=0.005)
        
            Plist = np.arange(start=0.9,stop=0.99,step=0.005)
        
            normOC(k = None, alpha = alphalist, P = Plist, 
                   n = list(range(10,21)), side = 2)
        
        ## Example 2, finding alpha
        
            Plist = [0.985,0.995,.98,.99]
        
            normOC(k = 4, alpha = None, P = Plist, n = list(range(10,21)), 
                   side = 2)
        
        ## Example 3, finding P
        
            alphalist = [0.01,0.02,0.03,0.04,0.05]
        
            normOC(k = 4, alpha = alphalist, P = None, n = list(range(10,21)), 
                   side = 2)
    '''
    if side != 1 and side != 2:
        return 'Must specify a one-sided or two-sided procedure'
    if length(n)<2:
        return "\'n\' needs to be a vector of at least length 2 to produce an OC curve"
    n = sorted(n)
    if length(P) == 1 and type(P) is float:
        P = [P]
    if length(alpha) == 1 and type(alpha) is float:
        alpha = [alpha]
    colblind = ["#000000", "#E69F00", "#56B4E9", "#009E73", 
              "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
              "#7FFF00","#7D26CD"]
    if P is None:
        if length(k) != 1 or length(alpha)<1:
            return "Check values specified for k, n, and alpha"
        if length(alpha)>10:
            print("Too many values of alpha specified! Using only the first 10 values.")
        alpha = np.array(sorted(alpha))[:min(length(alpha),10)]
        tmpobj = f'({k=})'
        allP = []
        for i in range(length(alpha)):
            for j in range(length(n)):
                allP.append(opt.brentq(KfactorP, a = 1e-10, b = 1-1e-10, args = (n[j],alpha[i],side,method,m,k)))
        Pmin = min(allP)
        if Pmin > 0.99:
            allP = np.array([Pmin,]*length(allP)).reshape(length(alpha), length(n)).T
        else:
            allP = np.array(allP).reshape(length(alpha), length(n)).T
        allP = pd.DataFrame(allP)
        plt.figure(1)
        plt.plot(n,[0,]*length(n))
        plt.title(f'Normal Tolerance Interval OC Curve for P {tmpobj}')
        if Pmin > 0.99:
            plt.ylim([0.999,1.001])
        else:
            plt.ylim([Pmin,1])
        plt.xlabel('n')
        plt.ylabel('P')
        labels = []
        for i in range(length(alpha)):
            plt.plot(n, allP.iloc[:,i],ls = '-', color = colblind[i],label = round(1-alpha[i],4), marker = 'o', ms = 3, lw = 0.5)
            labels.append(round(1-alpha[i],8))
        plt.legend(reversed(plt.legend().legendHandles), reversed(labels), loc=0, title = "1-α", bbox_to_anchor=(1.04, 1))
        plt.show()
    elif alpha is None:
        if length(k) != 1 or length(P)<1:
            return "Check values specified for k, n,  and P!"
        if length(P)>10:
            print("Too many values of P specified! Using only the first 10 values.")
        P = np.array(sorted(P))[:min(length(P), 10)]
        tmpobj = f'({k=})'
        allalpha = []
        for i in range(length(P)):
            for j in range(length(n)):
                allalpha.append(opt.brentq(Kfactoralpha, a = 1e-10, b = 1-1e-10, args = (n[j],P[i],side,method,m,k)))
        Amin = min(1-np.array(allalpha))
        allalpha = np.array(allalpha).reshape(length(P), length(n)).T
        allalpha = pd.DataFrame(allalpha)
        plt.figure(2)
        plt.plot(n,[0,]*length(n))
        plt.title(f'Normal Tolerance Interval OC Curve for 1-α {tmpobj}')
        plt.ylim([Amin,1])
        plt.xlabel('n')
        plt.ylabel('alpha')
        labels = []
        for i in range(length(P)):
            plt.plot(n, 1-allalpha.iloc[:,i],ls = '-', color = colblind[i],label = round(P[i],4), marker = 'o', ms = 3, lw = 0.5)
            labels.append(round(P[i],8))
        plt.legend(loc=0, title = "P", bbox_to_anchor=(1.04, 1))
        plt.show()
    elif k is None:
        if length(P)*length((alpha))>10:
            print("Too many combinations of α and P specified! Using only the first 10 such combinations.")
        alpha = sorted(alpha)[:min(length(alpha),10)]
        P = sorted(P)[:min(length(P),10)]
        tmp = []
        for i in range(length(n)):
            tmp.append(Ktable(n=n[i],alpha=alpha,P=P,method=method,m=m,side=side))
        allk = []
        for i in range(length(tmp)):
            allk.append(flatten(tmp[i].T.values.tolist()))
        allk = pd.DataFrame(allk).T
        allk.columns = n
        allk = allk.iloc[0:min(length(allk.iloc[:,0]),10)]
        if length(alpha) == 1:
            alpha = [alpha]
        if length(P) == 1:
            P = [P]
        alpha = np.array(alpha)
        P = np.array(P)
        #tmpalpha = np.array([1-alpha,]*length(P)).flatten()
        #tmpP = np.round(sorted(np.array([P,]*length(alpha)).flatten()),8)
        plt.figure(3)
        plt.plot(n,[0,]*length(n))
        plt.title('Normal Tolerance Interval OC Curve for k and n')
        plt.ylim([0,allk.values.max()+1e-01])
        plt.xlabel('n')
        plt.ylabel('k')
        labels = []
        Palist = sorted(np.array([P,]*length(alpha)).ravel())
        alpha = alpha.ravel()
        for i in range(length(allk.iloc[:,0])):
            plt.plot(n, allk.iloc[i],ls = '-', color = colblind[i],label = [np.round(1-alpha[i%length(alpha)],8),np.round(P[i%length(P)],8)], marker = 'o', ms = 3, lw = 0.5)
            labels.append([np.round(1-alpha[i%length(alpha)],8),np.round(Palist[i],8)])
        plt.legend(plt.legend().legendHandles,labels, loc=0, title = "(1-α,P)", bbox_to_anchor=(1.04, 1))
        plt.show()
    else:
        print("Check values specified for k, n, alpha, and P!")
        
# ## Example 1, finding k-factors

# alphalist = np.arange(start=0.01,stop=0.1,step=0.005)

# Plist = np.arange(start=0.9,stop=0.99,step=0.005)

# normOC(k = None, alpha = alphalist, P = Plist, 
#         n = list(range(10,21)), side = 1)

# ## Example 2, finding alpha

# Plist = [0.985,0.995,.98,.99]

# normOC(k = 4, alpha = None, P = Plist, n = list(range(10,21)), 
#         side = 1)

# ## Example 3, finding P

# alphalist = [0.01,0.02,0.03,0.04,0.05]

# normOC(k = 4, alpha = alphalist, P = None, n = list(range(10,21)), 
#         side = 1)