import numpy as np
import scipy.stats as st

#Internal Function

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
