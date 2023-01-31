import pandas as pd
import numpy as np
import scipy.stats as st

def rwishart(df, p):
    '''
Random Wishart Distributed Matrices

Usage
    rwishart(df, p)
    

Parameters
----------
    df : int
        The degrees of freedom for the Wishart matrix to be generated.
    p : int
        The dimension of the random Wishart matrix.

Returns
-------
    X : matrix
        Random generation of Wishart matrices.
        
References
    Derek S. Young (2010). tolerance: An R Package for Estimating Tolerance 
        Intervals. Journal of Statistical Software, 36(5), 1-39. 
        URL http://www.jstatsoft.org/v36/i05/.
        
    Yee, T. (2010), The VGAM Package for Categorical Data Analysis, Journal of 
        Statistical Software, 32, 1–34.

Example:
    ## Generate a 4x4 wishart matrix with 10 degrees of freedom.
        
        rwishart(df = 10, p = 4)
    '''
    X = np.zeros(p*p)
    shape = (p,p)
    X = X.reshape(shape)
    #X = pd.DataFrame(X)
    #print(X.iloc[0][1])
    chi2rvs = []
    for i in range(p):
        chi2rvs.append(np.sqrt(st.chi2.rvs(size = 1,df = df-i)))
    np.fill_diagonal(X,chi2rvs)
    if p > 1:
        a = []
        for i in range(1,p):
            pseq = list(range(1,p))
            a.extend(np.repeat(4*pseq[i-1], i))
        # a is equivalent to rep(p*pseq,pseq)
        #X = pd.DataFrame(X)
        for i in range(p-1):
            for j in range((p-1)-i):
                X[i][j+1+i] = st.norm.rvs(size = 1)
        X = np.dot(X.T,X)
        return X


def sort(x,decreasing=False):
    if type(x) == float or type(x) == int or type(x) == np.int32 or type(x) == np.float64 or type(x) == np.float32 or type(x) == np.int64:
        return x
    return sorted(x,reverse=decreasing,key=abs)

def length(x):
    if type(x) == float or type(x) == int or type(x) == np.int32 or type(x) == np.float64 or type(x) == np.float32 or type(x) == np.int64:
        return 1
    return len(x)

def tolfun(alpha, P, inptol,*args):
    if length(alpha) > 1:
        if length(P) > 1:
            tol = [sort(a,decreasing=False) for a in inptol]
            tol = pd.DataFrame(np.array(tol),columns=sort(alpha,decreasing=True))
        else:
            tol = sort(inptol,decreasing=(False))
            tol = pd.DataFrame([tol],columns=sort(alpha,decreasing=True))
    else:
        if length(P) > 1:
            tol = np.array([sort(a,decreasing=True) for a in inptol])
            tol = pd.DataFrame(tol,columns = [alpha])
        else:
            tol = pd.DataFrame({f"{alpha}":[inptol]})
    tol.index = [P]
    return tol

def mvtolregion(x, alpha = 0.05, P = 0.99, B = 1000, M = 1000, method = 'KM'):
    '''
Multivariate Normal Tolerance Regions

Description
    Determines the appropriate tolerance factor for computing multivariate 
    normal tolerance regions based on Monte Carlo methods or other 
    approximations.

Usage
    mvtolregion(x, alpha = 0.05, P = 0.99, B = 1000, M = 1000, method = ["KM",
                "AM", "GM", "HM", "MHM", "V11", "HM.V11", "MC"]) 

    Parameters
    ----------
    x : matrix
        An nxp matrix of data assumed to be drawn from a p-dimensional
        multivariate normal distribution. n pertains to the sample size.
        
    alpha : float or list, optional
        The level chosen such that 1-alpha is the confidence level. A vector 
        of alpha values may be specified. The default is 0.05.
        
    P : float or list, optional
        The proportion of the population to be covered by this tolerance 
        region. A vector of P values may be specified. The default is 0.99.
        
    B : int, optional
        The number of iterations used for the Monte Carlo algorithms (i.e., 
        when method = "KM" or "MC"), which determines the tolerance factor. 
        The number of iterations should be at least as large as the default 
        value of 1000. The default is 1000.
        
    M : int, optional
    The number of iterations used for the inner loop of the Monte Carlo 
    algorithm specified through method = "MC". The number of iterations should 
    be at least as large as the default value of 1000. Note that this is not 
    required for method = "KM" since that algorithm handles the eigenvalues 
    differently in the estimation of the tolerance factor. The default is 1000.
    
    method : string, optional
        The method for estimating the tolerance factors. "KM" is the 
        Krishnamoorthy-Mondal method, which is the method implemented in 
        previous versions of the tolerance package. It is one of the more 
        accurate methods available. "AM" is an approximation method based on 
        the arithmetic mean. "GM" is an approximation method based on the 
        geometric mean. "HM" is an approximation method based on the harmonic 
        mean. "MHM" is a modified approach based on the harmonic mean. "V11" 
        is a method that utilizes a certain partitioning of a Wishart random 
        matrix for deriving an approximate tolerance factor. "HM.V11" is a 
        hybrid method of the "HM" and "V11" methods. "MC" is a simple Monte
        Carlo approach to estimating the tolerance factor, which is 
        computationally expensive as the values of B and M increase. 
        The default is 'KM'.
        
        It is possible "MC" method has an error. Through testing, the results
        seem very similar to R. 

Details
    All of the methods are outlined in the references that we provided. In 
    practice, we recommend using the Krishnamoorthy-Mondal approach. A basic 
    sketch of how the Krishnamoorthy-Mondal algorithm works is as follows:

        (1) Generate independent chi-square random variables and Wishart
        random matrices.

        (2) Compute the eigenvalues of the randomly generated Wishart matrices.

        (3) Iterate the above steps to generate a set of B sample values such 
        that the 100(1-alpha)-th percentile is an approximate tolerance factor.
        
Returns
-------
    mvtolregion returns a dataframe where the rows pertain to each confidence 
    level 1-alpha specified and the columns pertain to each proportion level P 
    specified.

References
----------
    Derek S. Young (2010). tolerance: An R Package for Estimating Tolerance 
        Intervals. Journal of Statistical Software, 36(5), 1-39. 
        URL http://www.jstatsoft.org/v36/i05/.
        
    Krishnamoorthy, K. and Mathew, T. (1999), Comparison of Approximation 
        Methods for Computing Tolerance Factors for a Multivariate Normal 
        Population, Technometrics, 41, 234–249.
    
    Krishnamoorthy, K. and Mondal, S. (2006), Improved Tolerance Factors for 
        Multivariate Normal Distributions, Communications in Statistics - 
        Simulation and Computation, 35, 461–478.
        
Examples
--------
   ## 90%/90% bivariate normal tolerance region
    
        x1 = pd.DataFrame(st.norm.rvs(size=100,loc=0,scale = 0.2))
        
        x2 = pd.DataFrame(st.norm.rvs(size=100,loc=0,scale = 0.5))
        
        x = pd.concat([x1,x2],axis=1)
        
        mvtolregion(x, alpha = 0.1, P = 0.9, B = 1000, method = 'KM')
    
   ## 99%/94%, 99%/95%, 98%/94%, 98%/95% bivariate normal tolerance regions
   
        Plist = [0.99,0.98]
        
        alphalist = [0.06,0.05]
        
        x = pd.DataFrame({'X1':[1,2,4],'X2':[5,6,7]})
        
        mvtolregion(x, alpha = alphalist, P = Plist, B = 4, method = 'AM')
    '''
    P = sort(P)
    alpha = sort(alpha)
    if length(alpha) == 1:
        alpha = alpha
    else:
        alpha = np.array(alpha)
    if length(P) == 1:
        P = P
    else:
        P = np.array(P)
    n = len(x) #columns
    p = len(x.iloc[0]) #ROWS
    if method == 'KM':
        qsquared = st.chi2.rvs(df=1,size = p*B)/n
        shape = (int(p*B/2),p)
        qsquared = qsquared.reshape(shape)
        #qsquared = pd.DataFrame(qsquared)
        L = []
        for i in range(B):
            R = rwishart(df=n-1,p=p)
            eigenValues = np.linalg.eig(R)[0]
            idx = eigenValues.argsort()[::-1]   
            eigenValues = eigenValues[idx]
            L.append(eigenValues)
        c1 = np.array([sum(a) for a in (1+qsquared)/np.array(L)])
        c2 = np.array([sum(a) for a in (1+2*qsquared)/np.array(L)**2])
        c3 = np.array([sum(a) for a in (1+3*qsquared)/np.array(L)**3])
        a = c2**3/c3**2
        if length(P) > 1:
            T = np.array([(n - 1) * (np.sqrt(c2/a) * (st.chi2.ppf(p, a) - a) + c1) for p in P]).T
        else:
            T = np.array((n - 1) * (np.sqrt(c2/a) * (st.chi2.ppf(P, a) - a) + c1)).T
        T = pd.DataFrame(T)
        tol = []
        if length(alpha) == 1:
            for i in range(len(T.iloc[0])):
                tol.append(np.quantile(T.iloc[:,i],1-alpha))
            tol = pd.DataFrame(np.array(tol),columns = [alpha])
        else:
            for i in range(len(T.iloc[0])): 
                tol.append(np.quantile(T.iloc[:,i],1-np.array(sort(alpha,decreasing=True))))
            tol = pd.DataFrame(np.array(tol),columns=sort(alpha,decreasing=True))
        tol.index = [P]
        
    elif method == 'AM':
        if length(alpha) > 1:   
            tol = np.array([sort((p*(n-1)*st.ncx2.ppf(P,df = p,nc = p/n))/(st.chi2.ppf(alpha,(n-1)*p)),decreasing=False) for alpha in alpha]).T
            tol = tolfun(alpha,P,tol)
        else:
            tol = np.array(sort((p*(n-1)*st.ncx2.ppf(P,df = p,nc = p/n))/(st.chi2.ppf(alpha,(n-1)*p)),decreasing=False))
            tol = tolfun(alpha,P,tol)
    elif method == 'GM':
        g1 = (p/2)*(1-((p-1)*(p-2))/(2*n))**(1/p)
        if length(alpha) > 1:
            tol = np.array([sort(g1*(n-1)*st.ncx2.ppf(P,p,p/n)/st.gamma.ppf(alpha,p*(n-p)/2),decreasing=False) for alpha in alpha]).T
            tol = tolfun(alpha,P,tol)
        else:
            tol = np.array(sort(g1*(n-1)*st.ncx2.ppf(P,p,p/n)/st.gamma.ppf(alpha,p*(n-p)/2),decreasing=False))
            tol = tolfun(alpha,P,tol)
    elif method == 'HM':
        if length(alpha) > 1:
            tol = np.array([sort((p*(n-1)*st.ncx2.ppf(P,p,p/n))/(st.chi2.ppf(alpha,(n-1)*p-p*(p+1)+2)),decreasing=False) for alpha in alpha]).T
            tol = tolfun(alpha,P,tol)
        else:
            tol = np.array(sort((p*(n-1)*st.ncx2.ppf(P,p,p/n))/(st.chi2.ppf(alpha,(n-1)*p-p*(p+1)+2)),decreasing=False))
            tol = tolfun(alpha,P,tol)
    elif method == 'MHM':
        b = (p*(n-p-1)*(n-p-4)+4*(n-2))/(n-2)
        a = (p*(b-2))/(n-p-2)
        if length(alpha) > 1:
            tol = np.array([sort((a*(n-1)*st.ncx2.ppf(P, p, p/n))/(p*st.chi2.ppf(alpha,b)),decreasing=False) for alpha in alpha]).T
            tol = tolfun(alpha,P,tol)
        else:
            tol = np.array(sort((a*(n-1)*st.ncx2.ppf(P, p, p/n))/(p*st.chi2.ppf(alpha,b)),decreasing=False))
            tol = tolfun(alpha,P,tol)
    elif method == 'V11':
        if length(alpha) > 1:
            tol = np.array([sort((n-1)*st.ncx2.ppf(P,p,p/n)/st.chi2.ppf(alpha,n-p),decreasing=False) for alpha in alpha]).T
            tol = tolfun(alpha,P,tol)
        else:
            tol = np.array(sort((n-1)*st.ncx2.ppf(P,p,p/n)/st.chi2.ppf(alpha,n-p),decreasing=False))
            tol = tolfun(alpha,P,tol)
    elif method == 'HMV11':
        e = (4*p*(n-p-1)*(n-p)-12*(p-1)*(n-p-2))/(3*(n-2)+p*(n-p-1))
        d = (e-2)/(n-p-2)
        if length(alpha) > 1:
            tol = np.array([sort(d*(n-1)*st.ncx2.ppf(P,p,p/n)/st.chi2.ppf(alpha,e),decreasing=False) for alpha in alpha]).T
            tol = tolfun(alpha,P,tol)
        else:
            tol = np.array(sort(d*(n-1)*st.ncx2.ppf(P,p,p/n)/st.chi2.ppf(alpha,e),decreasing=False))
            tol = tolfun(alpha,P,tol)
    elif method == 'MC':
        if length(P) < 2 or length(alpha) < 2:
            return 'The length of P and alpha must both be greater than 1 for this method.'
        U = np.array([st.norm.rvs(size = int(B*p), loc = 0, scale = 1/n)]).reshape((B,p))
        V = []
        Y = []
        RES = []
        tmp = []
        for i in range(B):
            V.append(np.linalg.inv(rwishart(df = n-1,p = p)))
            Y.append(np.array([st.norm.rvs(size = int(M*p), loc = 0, scale = 1)]).reshape((p,M)))
            #RES.append(Y[i] -  U[i,:])
        for i in range(B):
            for j in range(n-1):
                tmp.append(Y[i][j]-U[i][j])
                if j == n - 1 - 1:
                    RES.append(tmp)
                    tmp = []
        quants = []
        for i in range(B):
            for j in range(length(P)):
                quants.append(np.quantile((n-1)*(np.dot(np.dot(np.array(RES[i]).T,V[i]),np.array(RES[i]))).T,P[j]))
        T = np.array(quants).reshape(B,length(P))
        T = pd.DataFrame(T)
        tmp = []
        tol = []
        for i in range(length(alpha)):
            for j in range(length(T.iloc[0])):
                tmp.append(np.quantile(T.iloc[j,i],1-alpha[i])) 
                if j == length(T.iloc[0])-1:
                    tol.append(sort(tmp,decreasing=False))
                    tmp = []
        tol = pd.DataFrame(tol,columns = sort(alpha,decreasing=True))
        tol.index = [P]
        
    return tol


