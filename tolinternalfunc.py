#internal functions
import numpy as np
import scipy.stats
import pandas as pd
import scipy.optimize as opt
import warnings
warnings.filterwarnings('ignore')

def nptolint(x,alpha = 0.05, P = 0.99, side = 1, method = 'WILKS', upper = None, lower = None):
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

