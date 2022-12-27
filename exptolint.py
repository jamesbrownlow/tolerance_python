import pandas as pd
import numpy as np
import scipy.stats 
def exptolint(x, alpha = 0.05, P = 0.99, side = 1, type2 = False):
    if side != 1 and side != 2:
        return 'Must specify a one-sided or two-sided procedure'
    if side == 2:
        alpha = alpha/2
    n = len(x)
    lhat = np.mean(x)
    if type2:
        mx = max(x)
        r = n - np.sum(x == mx)
    else:
        r = n
    if side == 2:
        lower = 2*r*lhat*np.log(2/(1+P))/scipy.stats.chi2.ppf(1-alpha,df=2*r)
        upper = 2*r*lhat*np.log(2/(1-P))/scipy.stats.chi2.ppf(alpha,df=2*r)
        alpha = 2*alpha
        data = {'alpha':[alpha],'P':[P],'lambda.hat':[lhat],'2-sided.lower':[lower],'2-sided.upper':[upper]}
    else:
        lower = 2*r*lhat*np.log(1/P)/scipy.stats.chi2.ppf(1-alpha,df=2*r)
        upper = 2*r*lhat*np.log(1/(1-P))/scipy.stats.chi2.ppf(alpha,df=2*r)
        data = {'alpha':[alpha],'P':[P],'lambda.hat':[lhat],'1-sided.lower':[lower],'1-sided.upper':[upper]}

    return pd.DataFrame(data=data)
