import numpy as np
import scipy.stats
import scipy.optimize as opt
import pandas as pd 
#import tolinternalfunc as tif

def length(x):
    if type(x) == float or type(x) == int or type(x) == np.int32 or type(x) == np.float64 or type(x) == np.float32 or type(x) == np.int64:
        return 1
    return len(x)

def f(n,alpha,P):
    return alpha - (n * P**(n - 1) - (n - 1) * P**n)

def bisection(a,b,n,alpha,tol=1e-8):
    xl = a
    xr = b
    while np.abs(xl-xr) >= tol:
        c = (xl+xr)/2
        prod = f(n=n,alpha=alpha,P=xl)*f(n=n,alpha=alpha,P=c)
        if prod > tol:
            xl = c
        else:
            if prod < tol:
                xr = c
    return c

def distfreeest2(n = None, alpha = None, P = None, side = 1):
    temp = 0
    if n == None:
        temp += 1
    if alpha == None:
        temp +=1
    if P == None:
        temp += 1
    if temp > 1:
        return 'Must specify values for any two of n, alpha, and P'
    if (side != 1 and side != 2):
        return 'Must specify a 1-sided or 2-sided interval'
    if side == 1:
        if n == None:
            ret = int(np.ceil(np.log(alpha)/np.log(P)))
        if P == None:
            ret = np.exp(np.log(alpha)/n)
            ret = float(f'{ret:.4f}')
        if alpha == None:
            ret = 1-P**n
    else:
        if alpha == None:
            ret = 1-(np.ceil((n*P**(n-1)-(n-1)*P**n)*10000))/10000
        if n == None:
            ret = int(np.ceil(opt.brentq(f,a=0,b=1e100,args=(alpha,P),maxiter=1000)))
        if P == None:
            ret = np.ceil(bisection(0,1,alpha =alpha, n = n, tol = 1e-8)*10000)/10000    
    return ret

def distfreeest(n = None, alpha = None, P = None, side = 1):
    if n == None:
        if type(alpha) == float:
            alpha = [alpha]
        if type(P) == float:
            P = [P]
        A = length(alpha)
        B = length(P)
        column_names = np.zeros(B)
        row_names = np.zeros(A)
        matrix = np.zeros((A,B))
        for i in range(A): 
            row_names[i] = alpha[i]
            for j in range(B):
                column_names[j] = P[j]
                matrix[i,j] = distfreeest2(alpha=alpha[i],P=P[j],side=side)
        out = pd.DataFrame(matrix,columns = column_names, index = row_names)
        
    if alpha == None:
        if type(n) == float or type(n) == int:
            n = [n]
        if type(P) == float:
            P = [P]
        A = length(n)
        B = length(P)
        column_names = np.zeros(B)
        row_names = np.zeros(A)
        matrix = np.zeros((A,B))
        for i in range(A): 
            row_names[i] = n[i]
            for j in range(B):
                column_names[j] = P[j]
                matrix[i,j] = distfreeest2(n=n[i],P=P[j],side=side)
        out = pd.DataFrame(matrix,columns = column_names, index = row_names)
        
    if P == None:
        if type(alpha) == float:
            alpha = [alpha]
        if type(n) == float or type(n) == int:
            n = [n]
        A = length(alpha)
        B = length(n)
        print(f'length of alpha = {A}',f'length of n = {B}')
        column_names = np.zeros(B)
        row_names = np.zeros(A)
        matrix = np.zeros((A,B))
        for i in range(A): 
            row_names[i] = alpha[i]
            for j in range(B):
                column_names[j] = n[j]
                matrix[i,j] = distfreeest2(alpha=alpha[i],n=n[j],side=side)
        out = pd.DataFrame(matrix,columns = column_names, index = row_names)
    return out

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
            if length(lower) == 1 and length(upper) == 1:
                d = {'alpha': [alpha], 'P': [P], '2-sided lower':[lower], '2-sided upper':[upper]}
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
        nmin = int(np.array(distfreeest(alpha = alpha, P=P, side = side).iloc[0]))
        temp = None
        if side == 1:
            if n < nmin:
                temp = extrap(x=x,alpha=alpha,P=P)
            else:
                temp = interp(x=x,alpha=alpha,P=P)
        else:
            temp = twosided(x=x,alpha=alpha,P=P)
    return temp

def fl(u1,u,n,alpha):
    #error of 1e-5 compared to R
    return scipy.stats.beta.cdf(u,(n+1)*u1,(n+1)*(1-u1))-1+alpha

def fu(u2,u,n,alpha):
    #error of 1e-5 compared to R
    return scipy.stats.beta.cdf(u,(n+1)*u2,(n+1)*(1-u2))-alpha

def eps(u,n):
    return (n+1)*u-np.floor((n+1)*u)

def neps(u,n):
    return -((n+1)*u-np.floor((n+1)*u))

def LSReg(y,x,gamma):  
    
    xbar = sum(x)/len(x)
    ybar = sum(y)/len(y)
    sumx = []
    sumx2 = []
    sumy = []
    sumxy = []
    for i in range(len(x)):
        sumx.append(x[i])
        sumx2.append(x[i]**2)
        sumy.append(y[i])
        sumxy.append(y[i]*x[i])
    ssxx = sum(sumx2) - (1/len(x))*sum(sumx)**2
    ssxy = sum(sumxy) - (1/len(y))*sum(sumy)*sum(sumx)
    B1 = ssxy/ssxx
    B0 = ybar - B1*xbar
    return B0+B1*gamma #regression equation with xi = gamma      

def interp(x,alpha,P):
    n = len(x)
    x = sorted(x)
    gamma = 1-alpha
    out = list(nptolint(range(n+1),alpha=alpha,P=P,side=1)[['1-sided lower','1-sided upper']].loc[0])
    s = out[0]
    r = out[1]
    ###Beran-Hall
    pil = (gamma-scipy.stats.binom.cdf(n-s-1,n,P))/scipy.stats.binom.pmf(n-s,n,P)
    piu = (gamma-scipy.stats.binom.cdf(r-2,n,P))/scipy.stats.binom.pmf(r-1,n,P)
    if s == n:
        Ql = x[s]
    else:
        Ql = pil*x[s+1]+(1-pil)*x[s]
    if r == 1:
        Qu = x[r-1]
    else:
        Qu = piu*x[r-1] + (1-piu)*x[r-2]
    ###Hutson
    u1 = opt.brentq(fl, a = 0.00001,b=0.99999,args=(1-P,n,alpha))
    u2 = opt.brentq(fu, a = 0.00001,b=0.99999,args=(P,n,alpha))
    if s == n:
        Qhl = x[s]
    else:
        Qhl = (1 - eps(u1, n)) * x[s] + eps(u1, n) * x[s + 1]
    if r == 1:
        Qhu = x[r-1]
    else:
        Qhu = (1 - eps(u2, n)) * x[r - 2] + eps(u2, n) * x[r-1]
    names = ['alpha','P','1-sided.lower','1-sided.upper']
    temp = pd.DataFrame([[alpha,P,Ql,Qu],[alpha,P,Qhl,Qhu]],columns=names)
    temp.index = ['OS-Based','FOS-Based']
    #return value is slightly different than R due to rounding. and scipy.stats.beta.cdf() vs pbeta()
    return temp

def extrap(x,alpha,P):
    n = len(x)
    x = sorted(x)
    gamma = 1-alpha
    out = list(nptolint(range(n+1),alpha=alpha,P=P,side=1)[['1-sided lower','1-sided upper']].loc[0])
    pib = -(gamma-scipy.stats.binom.cdf(n-1,n,P))/(scipy.stats.binom.pmf(n-1,n,P))
    Qexpl = pib*x[1]+(1-pib)*x[0]
    Qexpu = pib*x[n-2]+(1-pib)*x[n-1]
    u1b = opt.brentq(fl, a = 0.00001,b=0.99999,args=(1-P,n,alpha))
    u2b = opt.brentq(fu, a = 0.00001,b=0.99999,args=(P,n,alpha))
    Qhexpl = (1-neps(u1b,n))*x[0]+neps(u1b,n)*x[1]
    Qhexpu = (1-neps(u2b,n))*x[n-1]+neps(u2b,n)*x[n-2]
    names = ['alpha','P','1-sided.lower','1-sided.upper']
    temp = pd.DataFrame([[alpha,P,Qexpl,Qexpu],[alpha,P,Qhexpl,Qhexpu]],columns=names)
    temp.index = ['OS-Based','FOS-Based']
    return temp

def twosided(x,alpha,P):
    n = len(x)
    x = sorted(x)
    gamma = 1-alpha
    out = nptolint(range(n+1),alpha=alpha,P=P,side=2,method='HM')[['2-sided lower','2-sided upper']]
    r = np.ravel(np.array(out[['2-sided lower']]).T)
    s = np.ravel(np.array(out[['2-sided upper']]).T)
    r = [int(x) for x in r]
    s = [int(x) for x in s]
    if (len(out.index) == 2): #around 430,000 datapoints needed for this to be true
        X1L = np.array([x[r[0]],x[r[0]+1]])
        X2L = np.array([x[r[1]],x[r[1]+1]])
        X1U = np.array([x[s[0]],x[s[0]-1]])
        X2U = np.array([x[s[1]],x[s[1]-1]])
        g = np.ravel(np.array([(scipy.stats.binom.cdf(s[0]-r[0]-1,n,P),(scipy.stats.binom.cdf(s[0]-(r[0]+1)-1,n,P)))]))
        #predict using X1L and g, you are here
        out1L = LSReg(X1L,g,gamma)
        out2L = LSReg(X2L,g,gamma)
        out1U = LSReg(X1U,g,gamma)
        out2U = LSReg(X2U,g,gamma)
        temp1 = pd.DataFrame({'0':[out1L,out2L,x[r[0]],x[r[1]]]})
        temp2 = pd.DataFrame({'1':[x[s[0]],x[s[1]],out1U,out2U]})
        temp3 = pd.DataFrame({'2':[x[s[0]]-out1L,x[s[1]]-out2L,out1U-x[r[0]],out2U-x[r[0]]]})
        temp = pd.concat([temp1,temp2,temp3],axis=1)
        if scipy.stats.binom.cdf(s[1]-r[1]-1,n,P) >= gamma:
            indtemp = list(temp['2'])
            ind = indtemp.index(max(indtemp))
            temp = list(temp.iloc[ind,0:2])
            if ind==1 or ind==3:
                ord1 = 1
            else:
                indtemp = list(temp['2'])
                ind = indtemp.index(max(indtemp))
                temp = list(temp.iloc[ind,0:2])
                if ind==1 or ind ==3:
                    ord1 = 1
                else:
                    ord1 = 2
    else:
        XL = np.array([x[r[0]],x[r[0]+1]])
        if s[0] == length(x):
            XU = np.array([x[s[0]-1],x[s[0]-2]])
            print(s[0]-(r[0]+1)-1)
            g = np.ravel(np.array([(scipy.stats.binom.cdf(s[0]-(r[0]+1)-1,n,P)),(scipy.stats.binom.cdf(s[0]-(r[0]+1)-2,n,P))]))
            print(g)
            outL = LSReg(XL,g,gamma)
            outU = LSReg(XU,g,gamma)
            temp1 = pd.DataFrame({'0':[outL,x[r[0]]]})
            temp2 = pd.DataFrame({'1':[x[s[0]-1],outU]})
            temp3 = pd.DataFrame({'2':[x[s[0]-1]-outL,outU-x[r[0]]]})
            temp = pd.concat([temp1,temp2,temp3],axis=1)
        else:
            XU = np.array([x[s[0]],x[s[0]+1]])
            g = np.ravel(np.array([(scipy.stats.binom.cdf(s[0]-r[0]-1,n,P),(scipy.stats.binom.cdf(s[0]-(r[0]+1)-1,n,P)))]))
            outL = LSReg(XL,g,gamma)
            outU = LSReg(XU,g,gamma)
            temp1 = pd.DataFrame({'0':[outL,x[r[0]]]})
            temp2 = pd.DataFrame({'1':[x[s[0]],outU]})
            temp3 = pd.DataFrame({'2':[x[s[0]]-outL,outU-x[r[0]]]})
            temp = pd.concat([temp1,temp2,temp3],axis=1)
        if scipy.stats.binom.cdf(s[0]-r[0]-1,n,P) >= gamma:
            indtemp = list(temp['2'])
            ind = indtemp.index(min(indtemp))
            temp = list(temp.iloc[ind,0:2])
        else:
            temp = list([outL,outU])
    temp = pd.DataFrame({'alpha':[alpha], 'P':[P],'2-sided.lower':temp[0],'2-sided.upper':temp[1]},['OS-Based'])
    return temp


#x = np.array([12,21,4,2,5,6,7,3,31,23,34,21,22,20,3,4,2,6,89,23,45,6,2,4,6,24,
#               6,34,23,54,65,7,32,42,42,1,7,89,56,54,23,15,87,8,9,56,12,3,5,69,
#               8,7,41,56,100,59,80,69,52,46,78,90,78,45,46,49,1,56,26,36,32,55,
#               44,88,74,9,6,55,90,32,39,40,19,21,24,5,62,14,13,56,75,23,5,77,12,
#               78,45,12,56,98,78,45,12,35])

#print(nptolint(x, side = 2,method = "WILKS"))
# print(nptolint(x, side = 2,method = "HM"))
#print(nptolint(x,alpha=0.05, P = 0.99, side = 2,method = "YM"))

