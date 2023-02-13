import pandas as pd
import numpy as np
import scipy.stats as st
from scipy.spatial.distance import mahalanobis

def length(x):
    if type(x) == float or type(x) == int or type(x) == np.int32 or type(x) == np.float64 or type(x) == np.float32 or type(x) == np.int64:
        return 1
    return len(x)

def npmvtolregion(x, depthfn, alpha = None, P = None, Beta = None, adjust = "no",
                           type1 = ["central"], slower = None, scenter = None, supper = None, 
                           L = -np.inf, U = np.inf):
    '''
Nonparametric Multivariate Hyperrectangular Tolerance Regions

Description
    Provides depth-based multivariate central or semi-space nonparametric 
    tolerance regions. These can be calculated for any continuous multivariate 
    data set. Either (P, 1-alpha) tolerance regions or beta-expectation 
    tolerance regions can be specified.

Usage
    npmvtol.region(x, alpha = None, P = None, Beta = None, depthfn, 
               adjust = ["no", "floor", "ceiling"], 
               type1 = ["central", "semispace"], 
               slower = None, center = None, upper = None), 
               L = -np.inf, U = np.inf)
Parameters
----------
    x: dataframe
        An nxp matrix of data assumed to be drawn from a p-dimensional
        multivariate distribution. n pertains to the sample size.

    alpha: float
        The level chosen such that 1-alpha is the confidence level. Note that 
        if a (P, 1-alpha) tolerance region is required, then both alpha and P
        must be specified, but Beta must be set to None.

    P: float
        The proportion of the population to be covered by this tolerance 
        interval. Note that if a (P, 1-alpha) tolerance region is required, 
        then both alpha and P must be specified, but Beta must be set to None.

    Beta: float
        The confidence level for a beta-expectation tolerance region. Note 
        that if a beta-expectation tolerance region is required, then Beta 
        must be specified, but both alpha and P must be set to None.

    depthfn: function
        The data depth function used to perform the ordering of the multivariate data. Thus function must be coded in such a way that the first argument is multivariate data for which to calculate the depth values and the second argument is the original multivariate sample, x. For the purpose of this tolerance region calculation, these two arguments should both be the original multivariate sample.

    adjust: string, optional
        Whether an adjustment should be made during an intermediate 
        calculation for determining the number of points that need to be 
        included in the multivariate region. If adjust = "no", the default,
        then no adjustment is made during the intermediate calculation. If 
        adjust = "floor", then the intermediate calculation is rounded down to
        the next nearest integer. If adjust = "ceiling", then the intermediate
        calculation is rounded up to the next nearest integer.

    type1: string, optional
        The type of multivariate hyperrectangular region to calculate. If 
        type = "central", then two-sided intervals are reported for each 
        dimension of the data x. If type = "semispace", then a combination of
        one-sided intervals and two-sided intervals are reported for the
        dimensions of x. Which interval is calculated for each dimension in 
        this latter setting is dictated by the semi.order argument.

    supper, scenter, slower: 1,2, or 3, unique
        If type = "semispace", then this argument must be specified. Each 
        element gives the indices of the dimensions of x for which the type of
        interval should be calculated. Indices specified for the element of 
        lower will return one-sided lower limits for those dimensions, indices 
        specified for the element of center will return two-sided intervals 
        for those dimensions, and indices specified for the element of upper
        will return one-sided upper limits for those dimensions.

    L: float, optional
        If type = "semispace", these are the lower limits for any dimensions 
        for which one requests one-sided upper limits.

    U: float, optional
        If type = "semispace", these are the upper limits for any dimensions
        for which one requests one-sided lower limits.

Returns
-------
    npmvtolregion returns a px2 matrix where the columns give the lower and 
    upper limits, respectively, of the multivariate hyperrectangular tolerance 
    region.

References
    Young, D. S. and Mathew, T. (2020+), Nonparametric Hyperrectangular 
    Tolerance and Prediction Regions for Setting Multivariate Reference 
    Regions in Laboratory Medicine, Submitted.
    
Examples
    x = np.array([[1,4,10,10.45],[9,4,11,11.1],[4,7,9,9.8]]).T
    
    x = pd.DataFrame(x)
    
    npmvtolregion(x, depthfn=1,alpha = 0.05, P = 0.99, type1 ='semispace', 
                  adjust = 'no',slower = 2, supper = 1, scenter = 3)
    '''
    if not ((alpha is None and P is None and Beta is not None) or (alpha is not None and P is not None and Beta is None)):
        print("Either alpha and P, (exclusive) or Beta must be specified!")
    if adjust not in ['no','floor','ceiling']:
        return "adjust must be one of 'no','floor', or 'ceiling'."
    n = length(x.iloc[:,0])
    p = length(x.iloc[0])
    typevec = [type1,]*p
    semiorder = np.array([slower,scenter,supper])
    slower -=1
    scenter -=1
    supper -=1
    if type1 == 'semispace':
        if length(list(set(semiorder))) != length(semiorder):
            return "All indices must be uniquely specified in the list semiorder!"
        if 1 not in semiorder:
            return "slower, supper, and scenter must any permutation of 1,2,3"
        if 2 not in semiorder:
            return "slower, supper, and scenter must any permutation of 1,2,3"
        if 3 not in semiorder:
            return "slower, supper, and scenter must any permutation of 1,2,3"
        typevec[slower] = 'lower'
        typevec[scenter] = 'central'
        typevec[supper] = 'upper'
    if p<2:
        return "This procedure only works when x has two or more dimensions"
    if type1 == 'central':
        side = int(2*p)
    else:
        side = sum(np.array(list(map(length,semiorder)))*[1,2,1])
    if Beta is None:
        rn= n*P + st.norm.ppf(1-alpha)*np.sqrt(n*(P*(1-P)))
        rn = [np.floor(rn),np.ceil(rn)]
        if adjust == 'no':
            if n+1-rn[0] == 0:
               n1rn0 = n+1-rn[0]+1e-14
            else:
               n1rn0 = n+1-rn[0]
            if n+1-rn[1] == 0:
               n1rn1 = n+1-rn[1]+1e-14
            else:
               n1rn1 = n+1-rn[1]
            Pb = np.array([st.beta.sf(P, rn[0], n1rn0),st.beta.sf(P, rn[1], n1rn1)])
            m = n - (rn[np.max(np.where(np.abs((1-alpha)-Pb)))])-side
        elif adjust == 'floor':
            m = n - rn[0] - side
        else:
            m = n - rn[1] - side
    else:
        rn = (n+1)*Beta
        rn = np.array([np.floor(rn),np.ceil(rn)])
        if adjust == 'no':
            m = n - rn[int(np.min(np.abs(Beta-rn/(n+1))))]
        elif adjust == 'floor':
            m = n - rn[0]
        else:
            m = n - rn[1]
    #when m > 0, there's unknown issues. 
    if m > 0:
        dx = np.array(depthfn(x,x))
        dx = pd.DataFrame(dx)
        maxdepth = dx[dx == max(dx.iloc[:,0])]
        maxdepth = maxdepth.dropna().index
        y = pd.DataFrame(x.iloc[maxdepth]).T
        y.columns = range(length(maxdepth))
        y = np.array(y)
        yy = []
        for i in range(length(y[0])):
            for j in range(length(y)):
                yy.append(y[j][i])
        y = pd.DataFrame(y)
        yy = np.array(yy).reshape(length(y.iloc[:,0]),length(y.iloc[0])).T
        yy = pd.DataFrame(yy)
        centx = []
        for i in range(length(yy.iloc[0])):
            centx.append(np.mean(yy.iloc[:,i]))
        euclidean = pd.DataFrame(np.sqrt((scale(x, c = centx, sc=False)**2).T.sum().values))
        allx = []
        for i in range(p):
            temp = pd.concat([x,dx,pd.DataFrame(range(1,n+1)),euclidean],axis = 1)
            emp = ['']*p
            tcolumns = [emp,'dx','range','euclidean']
            tcolumns = flatten(tcolumns)
            temp.columns = tcolumns
            if typevec[i] == 'lower':
                idxs = (-x.iloc[:,i]).argsort()
            else:
                idxs = (x.iloc[:,i]).argsort()
            temp.index = idxs
            temp = temp.sort_index()
            allx.append(temp)
        newallxred = []
        allxred = []
        
        #fix this###########################################################
        for i in range(int(m-1)):
            for j in range(p):
                idxx = []
                if (typevec[j] == 'central'):
                    idxx.append(allx[j].iloc[0*(typevec[j]=='central')])
                    idxx.append(allx[j].iloc[n-(i+1)])
                    allxred.append((pd.concat(idxx,axis = 1).T))
                else:
                    idxx.append(allx[j].iloc[n-(i+1)])
                    allxred.append(pd.concat(idxx,axis = 1).T)
        ####################################################################
        
        allxred = pd.concat(allxred)
        for j in range(int(length(allxred)/(m-1))):
            newallxred.append(pd.concat([allxred.iloc[j*int(m-1):int(m-1)*(j+1)]]))

        #print(newallxred)
        for i in range(length(newallxred)):
            tmpind = np.where(newallxred[i].iloc[:,p] == min(newallxred[i].iloc[:,p]))[0]
            if length(tmpind) > 1:
                newallxred[i] = newallxred[i].iloc[tmpind]
                ind2remove = newallxred[i].iloc[np.max(np.where(newallxred[i].iloc[:,p+2])),p+1]
            else:
                ind2remove = newallxred[i].iloc[np.min(np.where(newallxred[i].iloc[:,p])),p+1]
            for j in range(p):
                which = np.where(allx[j].iloc[:,p+1] == ind2remove)[0]
                try:
                    allx[j] = allx[j].drop(which)
                except:
                    break
        #print(allx)
        LIMITS = allx[0].iloc[:,:p].T
        #print(LIMITS)
    else:
        LIMITSmin = []
        LIMITSmax = []
        for i in range(length(x.iloc[0])):
            LIMITSmin.append(min(x.iloc[:,i]))
            LIMITSmax.append(max(x.iloc[:,i]))
        LIMITSmin = pd.DataFrame(LIMITSmin)
        LIMITSmax = pd.DataFrame(LIMITSmax)
        LIMITS = pd.concat([LIMITSmin,LIMITSmax],axis=1)
    LIMITS.columns = ['Lower','Upper']
    if type1 == 'semispace':
        LIMITS.iloc[supper,0] = L
        LIMITS.iloc[slower,1] = U
    Lim_idx = []
    for i in range(length(LIMITS.iloc[:,0])):
        Lim_idx.append(f'X{i+1}')
    LIMITS.index = [Lim_idx]
    return LIMITS
   
def flatten(A):
    rt = []
    for i in A:
        if isinstance(i,list): rt.extend(flatten(i))
        else: rt.append(i)
    return rt
        
def scale(y, c=True, sc=True):
    x = y.copy()
    if c is True:
        x -= x.mean()
    if type(c) == list:
        x = x.apply(lambda x: x-c, axis = 1)
    if sc and c:
        x /= x.std()
    elif sc:
        x /= np.sqrt(x.pow(2).sum().div(x.count() - 1))
    return x    
        
        
def mahalanobisR(X,meanCol,IC):
    m = []
    for i in range(X.shape[0]):
        m.append(mahalanobis(X.iloc[i,:],meanCol,IC) ** 2)
    return m

def mdepth(pts,x):
    return mahalanobisR(pts, meanCol=np.array([0,0,0]), IC=np.diag([1,1,1]))

# x = np.array([[1,4,10,10.45],[9,4,11,11.1],[4,7,9,9.8]]).T
# #x = np.array([[1,4,10.45],[9,4,11.1],[4,7,9.8]]).T
# x = pd.DataFrame(x)
# #print(npmvtolregion(x, depthfn=mdepth, Beta = -.1, type1 ='semispace', adjust = 'no',slower = 2, supper = 1, scenter = 3))
# print(npmvtolregion(x, depthfn=1,alpha = 0.05, P = 0.99, type1 ='semispace', adjust = 'no',slower = 2, supper = 1, scenter = 3))
print("There is an error in this function when m > 0.")
