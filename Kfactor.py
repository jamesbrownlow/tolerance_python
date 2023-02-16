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

# print(Kfactor(15,f=12,method = 'HE',side = 2,m=100, alpha = 0.02, P = 0.98))
# print(Kfactor(15,f=12,method = 'HE2',side = 2,m=100, alpha = 0.02, P = 0.98))
# print(Kfactor(15,f=12,method = 'WBE',side = 2,m=100, alpha = 0.02, P = 0.98))
# print(Kfactor(15,f=12,method = 'ELL',side = 2,m=100, alpha = 0.02, P = 0.98))
# print(Kfactor(15,f=12,method = 'KM',side = 2,m=100, alpha = 0.02, P = 0.98))
# print(Kfactor(15,f=12,method = 'EXACT',side = 2,m=100, alpha = 0.02, P = 0.98))
# print(Kfactor(15,f=12,method = 'OCT',side = 2,m=100, alpha = 0.02, P = 0.98))