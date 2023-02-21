import pandas as pd
import numpy as np
import scipy.stats
import scipy.integrate as integrate
import scipy.optimize as opt

def length(x):
    if type(x) == float or type(x) == int or type(x) == np.int32 or type(x) == np.float64 or type(x) == np.float32 or type(x) == np.int64:
        return 1
    return len(x)

import scipy.stats
import numpy as np
#import pandas as pd
import scipy.integrate as integrate
#import statistics as st
import warnings
warnings.filterwarnings('ignore')

import scipy.optimize as opt

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
           side = 1, f = None, method = 'HE', m = 50, byarg = 'alpha')
                 
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
            print(f'\nalpha = {round(1-alpha[l],3)}')
            print(temp)
            
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
            alpha = [1-a for a in alpha]
            temp.index = alpha
            temp.columns = P
            print(f'\nn = {round(n[l],1)}')
            print(temp)
            alpha = [1-a for a in alpha]
        
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
            print(f'\nP = {round(P[l],3)}')
            print(temp)
            alpha = [1-a for a in alpha]
    else:
        return 'Must specify index for table!'
    return ''

