import numpy as np
import scipy.stats
import pandas as pd
import tolinternalfunc as tif
import distfreeest as dfe

def nptolint(x,alpha = 0.05, P = 0.99, side = 1, method = 'WILKS', upper = None, lower = None):
    '''
    nptolint(x,alpha = 0.05, P = 0.99, side = 1, method = ('WILKS','WALD','HM','YM'), upper = None, lower = None):
Parameters
----------
    x: list
        A vector of data which no distributional assumptions are made. 
        The data is only assumed to come from a continuous distribution.
    
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
        The method for determining which indices of the ordered observations will be used for the tolerance intervals. 
        "WILKS" is the Wilks method, which produces tolerance bounds symmetric 
        about the observed center of the data by using the beta distribution. 
        
        "WALD" is the Wald method, which produces (possibly) multiple tolerance 
        bounds for side = 2 (each having at least the specified confidence level), 
        but is the same as method = "WILKS" for side = 1. 
        
        "HM" is the Hahn-Meeker method, which is based on the binomial distribution, 
        but the upper and lower bounds may exceed the minimum and maximum of the sample data. 
        For side = 2, this method will yield two intervals if an odd number of 
        observations are to be trimmed from each side. 
        
        "YM" is the Young-Mathew method for performing interpolation or 
        extrapolation based on the order statistics. 
        See below for more information on this method.
        
        The default is "WILKS"
    
    upper: float, optional 
        The upper bound of the data. When None, then the maximum of x is used. 
        If method = "YM" and extrapolation is performed, then upper will be 
        greater than the maximum. The default value is None.
    
    lower: float, optional
        The lower bound of the data. When None, then the minimum of x is used. 
        If method = "YM" and extrapolation is performed, then lower will be 
        less than the minimum. The default value is None.
    
Details
    For the Young-Mathew (YM) method, interpolation or extrapolation is performed. 
    When side = 1, two intervals are given: one based on linear 
    interpolation/extrapolation of order statistics (OS-Based) and one based 
    on fractional order statistics (FOS-Based). When side = 2, only an interval 
    based on linear interpolation/extrapolation of order statistics is given.
    
Returns
-------
  nptolint returns a data frame with items:
        
    alpha: The specified significance level.
    
    P: The proportion of the population covered by this tolerance interval.
    
    1-sided.lower: The 1-sided lower tolerance bound. This is given only if side = 1.
    
    1-sided.upper: The 1-sided upper tolerance bound. This is given only if side = 1.
    
    2-sided.lower: The 2-sided lower tolerance bound. This is given only if side = 2.
    
    2-sided.upper: The 2-sided upper tolerance bound. This is given only if side = 2.

References
----------
    Derek S. Young (2010). tolerance: An R Package for Estimating Tolerance Intervals. 
        Journal of Statistical Software, 36(5), 1-39. URL http://www.jstatsoft.org/v36/i05/.
    
    Bury, K. (1999), Statistical Distributions in Engineering, Cambridge University Press.

    Hahn, G. J. and Meeker, W. Q. (1991), Statistical Intervals: A Guide for 
        Practitioners, Wiley-Interscience.

    Wald, A. (1943), An Extension of Wilks' Method for Setting Tolerance Limits, 
        The Annals of Mathematical Statistics, 14, 45–55.

    Wilks, S. S. (1941), Determination of Sample Sizes for Setting Tolerance 
        Limits, The Annals of Mathematical Statistics, 12, 91–96.

    Young, D. S. and Mathew, T. (2014), Improved Nonparametric Tolerance 
        Intervals Based on Interpolated and Extrapolated Order Statistics, Journal 
        of Nonparametric Statistics, 26, 415–432.
Examples
--------
    ## 90%/95% 2-sided nonparametric tolerance intervals for a sample of size 20. 

    nptol.int(x = x, alpha = 0.10, P = 0.95, side = 1, method = "WILKS", upper = NULL, lower = NULL)
    '''
    n = len(x)
    if n < 2:
        return 'cannot do anything with less than 2 datapoints'
    xsort = np.sort(x)
    if(upper == None):
        upper = np.max(x)
    if(lower == None):
        lower = np.min(x)
    if(method == "WILKS"):
        if(side == 2):
            if(np.floor((n+1)/2) == (n+1)/2):
                up = ((n + 1)/2) - 1
            else:
                up = np.floor((n + 1)/2)
            r = np.arange(1,up+1)
            #r = np.array([1,2,3,4,.001,.002])
            out2 = 1-scipy.stats.beta.cdf(P, n - 2 * r + 1, 2 * r) - (1-alpha)
            ti2 =pd.DataFrame([r,out2])
            ti2 = ti2.T #transpose the dataframe to make it easier to work with
            temp2 = ti2[ti2[1]>0] #Gets all rows where col2 > 0
            if len(temp2) == 0:
                lower = lower
                upper = upper
            else:
                mins2 = min(temp2[1])
                temp2 = temp2[temp2[1]==mins2]
                r = int(temp2[0])
                lower = xsort[r]
                upper = xsort[n-r+1]
            d = {'alpha': [alpha], 'P': [P], '2-sided lower':lower, '2-sided upper':upper}
            temp = pd.DataFrame(data=d)
                
        if(side ==1):
            r = scipy.stats.binom.ppf(alpha, n= n, p=1-P)
            s = n-r+1
            if(r<1):
                lower = lower
            else:
                lower = xsort[int(r)]
            if (s > n):
                upper = upper
            else:
                upper = xsort[int(s-1)]
            d = {'alpha': [alpha], 'P': [P], '1-sided lower':lower, '1-sided upper':upper}
            temp = pd.DataFrame(data=d)
    if(method == "WALD"): #needs to be made more effient for side == 1 and side == 2
        t = []
        s = []
        for i in range(2,n+1):
            s.extend(list(range(1,i)))
            t.extend((i,)*(i-1))
            
        if side == 1: #Make this code more efficient
            r = scipy.stats.binom.ppf(alpha, n = n, p = 1-P)
            s = n-r+1
            if r < 1:
                lower = lower
            else:
                lower = xsort[int(r)]
            if s > n:
                upper = upper
            else:
                upper = xsort[int(s)]
            d = {'alpha': [alpha], 'P': [P], '1-sided lower':lower, '1-sided upper':upper}
            temp = pd.DataFrame(data = d)
        else: #Make this code more efficient
            out3 = []
            for i in range(len(t)):
                out3.append(1 - scipy.stats.beta.cdf(P, int(t[i]-s[i]), int(n-t[i]+s[i]+1))-(1-alpha))
            #ti3 = pd.DataFrame({'s':s,'t':t,'out3':out3}).T
            ti3 =pd.DataFrame([s,t,out3])
            ti3 = ti3.T #transpose the dataframe to make it easier to work with
            temp3 = ti3[ti3[2] > 0] #should be >
            if len(temp3) == 0:
                lower = lower
                upper = upper
            else:
                mins3 = min(temp3[2])
                out5 = temp3[temp3[2]==mins3]
                s = out5[0]
                t = out5[1]
                s = s.tolist()
                t = t.tolist()
                for i in range(len(s)):
                    t[i] = t[i]-1
                    s[i] = s[i]-1
                lower = np.zeros(len(s))
                upper = np.zeros(len(s))
                for i in range(len(t)):
                    lower[i] = xsort[int(s[i])]
                    upper[i] = xsort[int(t[i])]
            if len(lower) == 1 and len(upper == 1):
                d = {'alpha': [alpha], 'P': [P], '2-sided lower':lower, '2-sided upper':upper}
                temp = pd.DataFrame(data = d)
            else:
                d = {'alpha': [alpha], 'P': [P], '2-sided lower':lower[0], '2-sided upper':upper[0]}
                d = pd.DataFrame(data=d)
                for i in range(1,len(lower)):
                    d.loc[len(d.index)] = [alpha,P,lower[i],upper[i]]
                temp = d
    if (method == 'HM'):
        ind = range(n+1)
        out = scipy.stats.binom.cdf(ind, n = n, p = P) - (1-alpha)
        ti = pd.DataFrame([ind,out]).T
        temp = ti[ti[1] > 0]
        mins = min(temp[1])
        HMind = int(temp[temp[1] == mins][0])
        diff = n - HMind
        if side == 2:
            if diff == 0 or int(np.floor(diff/2)) == 0:
                if lower != None:
                    xsort = np.insert(xsort, 0, lower)#pd.DataFrame([lower, xsort])
                if upper != None:
                    xsort = np.insert(xsort, len(xsort), upper)#pd.DataFrame([xsort, upper]) #come back to this area when done
                HM = [1] + [len(xsort)]                
                d = {'alpha': [alpha], 'P': [P], '2-sided lower': lower, '2-sided upper': upper}#xsort[int(HM.loc[0][1])]}
                temp = pd.DataFrame(data=d)
            else:
                if np.floor(diff/2) == diff/2:
                    v1 = diff/2 #scalar
                    v2 = diff/2
                else:
                    v1 = [np.floor(diff/2), np.ceil(diff/2)] #list
                    v2 = [sum(x) for x in zip(v1, [1,-1])] #add v1 to [1,-1] element-wise
                if type(v1) == list:
                    #%%
                    #you can make this block more effient
                    data = {'v1': [v1[0]], 'v2*': [n- v2[0] +1]}
                    HM = pd.DataFrame(data = data)
                    for i in range(1,len(v1)):
                        HM.loc[len(HM.index)] = [v1[i], n-v2[i]+1]
                    
                    d = {'alpha': [alpha], 'P': [P], '2-sided lower': xsort[int(HM.loc[0][0])], '2-sided upper': xsort[int(HM.loc[0][1]-1)]}#xsort[int(HM.loc[0][1])]}
                    d = pd.DataFrame(data = d)
                    for i in range(1,len(HM)):
                        d.loc[len(d.index)] = [alpha,P,xsort[int(HM.loc[i][0])],xsort[int(HM.loc[i][1]-1)]]    
                    temp = d
                    #%%
                else:
                    data = {'v1': [v1], 'v2*': [n-v2+1]}
                    HM = pd.DataFrame(data = data)
                    d = {'alpha': [alpha], 'P': [P], '2-sided lower': xsort[int(HM.loc[0][0])], '2-sided upper': xsort[int(HM.loc[0][1])-1]}
                    temp = pd.DataFrame(data = d)
                if len(HM) == 2 and len(HM.loc[0]) == 2: #is the row dim 2 and col dim 2? T/F
                    if xsort[int(HM.loc[0][0])] == xsort[int(HM.loc[1][0])] and xsort[int(HM.loc[0][1])-1] == xsort[int(HM.loc[1][1])-1]:
                        temp = temp.loc[0]
                        temp = pd.DataFrame(temp).T
        else:
            l = pd.DataFrame(range(n+1),columns=['l'])
            lp = pd.DataFrame((1-scipy.stats.binom.cdf(l-1,n,1-P))-(1-alpha),columns=[''])
            lowtemp = pd.concat([l,lp],axis=1)
            u =  pd.DataFrame(range(1,n+2),columns=['u'])
            up = pd.DataFrame((scipy.stats.binom.cdf(u-1,n,P))-(1-alpha),columns=[''])
            uptemp = pd.concat([u,up],axis=1)
            l = lowtemp[lowtemp.loc[:,'']>0]
            l = max(l.loc[:,'l'])
            if l > 0:
                lower = xsort[l-1]
            u = uptemp[uptemp.loc[:,'']>0]
            u = min(u.loc[:,'u'])
            if u < n+1:
                upper = xsort[u-1]
            d = {'alpha': [alpha], 'P': [P], '1-sided lower': lower, '1-sided upper': upper}
            temp = pd.DataFrame(data=d)
    if method == 'YM':
        nmin = int(np.array(dfe.distfreeest(alpha = alpha, P=P, side = side).iloc[0]))
        temp = None
        if side == 1:
            if n < nmin:
                temp = tif.extrap(x=x,alpha=alpha,P=P)
            else:
                temp = tif.interp(x=x,alpha=alpha,P=P)
        else:
            temp = tif.twosided(x=x,alpha=alpha,P=P)
    return temp
