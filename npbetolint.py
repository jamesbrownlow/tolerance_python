import numpy as np
import pandas as pd

def length(x):
    if type(x) == int or type(x) == float or type(x) == np.float64:
        return 1
    else:
        return len(x)

def npbetolint(x, Beta = 0.95, side = 1, upper = None, lower = None):
    '''
Nonparametric Beta-Expectation Tolerance Intervals

Description:
    Provides 1-sided or 2-sided nonparametric (i.e., distribution-free) 
    beta-expectation tolerance intervals for any continuous data set. These 
    are equivalent to nonparametric prediction intervals based on order 
    statistics.
    
    npbetol.int(x, Beta = 0.95, side = 1, upper = None, lower = None)
Parameters
    ----------
    x : list
        A vector of data which no distributional assumptions are made. The 
        data is only assumed to come from a continuous distribution.

    Beta : float, optional
        The confidence level. The default is 0.95
        
    side : 1 or 2, optional
        Whether a 1-sided or 2-sided tolerance interval is required 
        (determined by side = 1 or side = 2, respectively). The default is 1.
        
    upper : int, optional
        The upper bound of the data. When None, then the maximum of x is used.
        The default is None.

    lower : int, optional
        The lower bound of the data. When None, then the minimum of x is used.
        The default is None. 

Returns
-------
    npbetolint returns a dataframe with items:
        
        Beta: 
            The specified confidence level.
        
        1-sided.lower: 
            The 1-sided lower tolerance bound. This is given 
            only if side = 1.
        
        1-sided.upper: 
            The 1-sided upper tolerance bound. This is given 
            only if side = 1.
        
        2-sided.lower: 
            The 2-sided lower tolerance bound. This is given 
            only if side = 2.
        
        2-sided.upper: 
            The 2-sided upper tolerance bound. This is given 
            only if side = 2.
        
References
----------
    Beran, R. and Hall, P (1993), Interpolated Nonparametric Prediction 
        Intervals and Confidence Intervals, Journal of the Royal Statistical 
        Society, Series B, 55, 643–652.
        
    Derek S. Young (2010). tolerance: An R Package for Estimating Tolerance 
        Intervals. Journal of Statistical Software, 36(5), 1-39. 
        URL http://www.jstatsoft.org/v36/i05/.
        


Examples
--------
    ## Nonparametric 90%-expectation tolerance intervals for a sample of size 
    100. 
        
        x = [12,32,45,5,35,34,23,1,24,56,6,5,57,5,43,35,2,36,547,57]
        npbetol.int(x = x, Beta = 0.90, side = 2)

    '''
    n = length(x)
    x = sorted(x)
    ne = min(np.ceil(Beta*(n+1)),n)
    ne2 = max(np.floor((n-ne)/2),1)
    if side == 1:
        if upper == None:
            upper = x[int(ne-1)]
        if lower == None:
            lower = x[int(max(n-ne,0))]
        return pd.DataFrame({'beta':[Beta],'1-sided.lower':lower,'1-sided.upper':upper})
    else:
        if upper == None:
            upper = x[int(min([ne+ne2-1,n-1]))]
        if lower == None:
            lower = x[int(ne2-1)]
        return pd.DataFrame({'beta':[Beta],'2-sided.lower':lower,'2-sided.upper':upper})
    
npbet