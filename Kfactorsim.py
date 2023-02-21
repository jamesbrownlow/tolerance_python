import numpy as np
import scipy.stats 
import scipy.integrate as integrate
import scipy.optimize as opt

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

def length(x):
    if type(x) == float or type(x) == int:
        return 1
    return len(x)

def Kfactorsim(n, l = None, alpha = 0.05, P = 0.99, side = 1, method = 'EXACT', m = 50):
    '''
Estimating K-factors for Simultaneous Tolerance Intervals Based on Normality

Description
    Estimates k-factors for simultaneous tolerance intervals based on normality.
    
    K.factor.sim(n, l = None, alpha = 0.05, P = 0.99, side = 1, 
         method = ["EXACT", "BONF"], m = 50)
    

Parameters
----------
    n : int or list
        If method = "EXACT", this is the sample size of each of the l groups. 
        If method = "BONF", then n can be a vector of different sample sizes 
        for the l groups.
        
    l : int, optional
        The number of normal populations for which the k-factors will be 
        constructed simultaneously. If NULL, then it is taken to be the length 
        of n. The default is None.
        
    alpha : float, optional
        The level chosen such that 1-alpha is the confidence level. The 
        default is 0.05.
        
    P : float, optional
        The proportion of the population to be covered by the tolerance 
        interval. The default is 0.99.
        
    side : 1 or 2, optional
        Whether a k-factor for a 1-sided or 2-sided tolerance interval is 
        required (determined by side = 1 or side = 2, respectively).
        
    method : string, optional
        The method for calculating the k-factors. "EXACT" is an exact method 
        that can be used when all l groups have the same sample size. "BONF" 
        is an approximate method using the Bonferroni inequality, which can be 
        used when the l groups have different sample sizes. The default is 
        'EXACT'.
        
    m : int, optional
        The maximum number of subintervals to be used in the integrate 
        function. This is necessary only for method = "EXACT". The larger the 
        number, the more accurate the solution. Too low of a value can result 
        in an error. A large value can also cause the function to be slow for 
        method = "EXACT". The default is 50.

Returns
-------
    K.factor returns the k-factor for simultaneous tolerance intervals based 
    on normality with the arguments specified above.
    
Note
-----
    For larger combinations of n and l when side = 2 and method = "EXACT", the 
    calculation can be slow. For larger sample sizes when side = "BONF", there 
    may be some accuracy issues with the 1-sided calculation since it depends 
    on the noncentral t-distribution. The code is primarily intended to be 
    used for moderate values of the noncentrality parameter. It will not be 
    highly accurate, especially in the tails, for large values.
    
References
----------
    Derek S. Young (2010). tolerance: An R Package for Estimating Tolerance 
        Intervals. Journal of Statistical Software, 36(5), 1-39. 
        URL http://www.jstatsoft.org/v36/i05/.
    
    Krishnamoorthy, K. and Mathew, T. (2009), Statistical Tolerance Regions: 
        Theory, Applications, and Computation, Wiley.

    Mee, R. W. (1990), Simultaneous Tolerance Intervals for Normal Populations 
        with Common Variance, Technometrics, 32, 83-92.

Examples
--------
    Kfactorsim(n=4, alpha = 0.05, P = 0.99, side = 1, m = 50,method = 'BONF')
    '''
    if side != 1 and side != 2:
        return 'Must specify a one-sided or two-sided procedure!'
    if method.upper() == "EXACT":
        if l == None:
            l = 1
        if length(n) != 1:
            print('Sample size n cannot be a list of sample sizes.')
        df = n*l-l
        chia = scipy.stats.chi2.ppf(alpha,df)
        k2 = np.sqrt(df*scipy.stats.chi2.ppf(P,1,1/n)/chia)
        if side == 1:
            def Ksimfun(K,n,P,alpha,l,m):
                def fun_temp(z,K,n,P,l):
                    df = n*l-l
                    zp = scipy.stats.norm.ppf(P)
                    inside = np.sqrt(n) * (K*np.sqrt(z)/np.sqrt(df)-zp)
                    return scipy.stats.chi2.pdf(z,df)*scipy.stats.norm.cdf(inside)**l
                return integrate.quad(fun_temp, a = 0, b = n*l*10, args = (K,n,P,l),limit = m)[0] - (1-alpha)
            K = opt.brentq(Ksimfun, a=0, b = k2+100, args=(n,P,alpha,l,m), xtol=np.finfo(float).eps**0.5)
        else:
            def Ksimfun(K,n,P,alpha,l,m):
                def fun_temp(z,K,n,P,l):
                    df = n*l-l
                    P1 = scipy.stats.chi2.sf(df*scipy.stats.ncx2.ppf(P,df=1,nc=z**2/n)/(K**2),df) #cdf lower tail = False
                    P2 = (2*scipy.stats.norm.cdf(z)-1)**(l-1)
                    return scipy.stats.norm.pdf(z)*P1*P2
                return 2*l*integrate.quad(fun_temp, a = 0, b = n*l*10, args = (K,n,P,l), limit = m)[0] - (1-alpha)
            K = opt.brentq(Ksimfun, a = 0, b = k2+100, args=(n,P,alpha,l,m), xtol=np.finfo(float).eps**0.5)
    else:
        if length(n) > 1 or l == None:
            l = length(n)
        if side == 1:
            if length(n) != 1:
                K = []
                for i in range(len(n)):
                    K.append(Kfactor(n[i],f=sum(n)-l,alpha=alpha/l,P=P,side = 1))
            else:
                K = Kfactor(n,f=n-l,alpha=alpha/l,P=P,side = 1)
        else:
            if length(n) != 1:
                K = []
                for i in range(len(n)):
                    K.append(Kfactor(n[i],f=sum(n)-l,alpha=alpha/l,P=P,side=2,method="EXACT",m=m))
            else:
                K = Kfactor(n,f=n-l,alpha=alpha/l,P=P,side=2,method="EXACT",m=m)
    return K
