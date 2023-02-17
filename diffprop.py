import numpy as np
import scipy.stats
import scipy.special
import scipy.integrate
import scipy.optimize as opt

#possible errors

# def A1simple(u, a, b, bprime, c, x, y):
#     try:
#         return u**(a-1)*(1-u)**(c-a-1)*(1-u*x)**(-b)*(1-u*y)**(-bprime)
#     except OverflowError:
#         return 1

def F1(a, b, bprime, c, x, y,*args):
    def A1simple(u, a, b, bprime, c, x, y):
        try:
            return u**(a-1)*(1-u)**(c-a-1)*(1-u*x)**(-b)*(1-u*y)**(-bprime)
        except OverflowError:
            return 1
    return scipy.special.gamma(c)/(scipy.special.gamma(a)*scipy.special.gamma(c-a))*scipy.integrate.quad(A1simple,a=0,b=1,args=(a,b,bprime,c,x,y))[0]
        

def length(x):
    if type(x) == int or type(x) == float:
        return 1
    else:
        return len(x)

def ddiffprop2(d,k1,k2,n1,n2,a1,a2):
           c1 = k1+a1
           c2 = k2+a2
           b1 = n1-k1+a1
           b2 = n2-k2+a2
           K = scipy.special.beta(c1,b1)*scipy.special.beta(c2,b2)
           if scipy.special.beta(c1,b1) < 0 or scipy.special.beta(c2,b2) < 0:
               print('NaNs produced')
               return 0
           if d >= -1 and d <= 0:
               try:
                   out = scipy.special.beta(c1,b2)*F1(b2, c1 + c2 + b1 + b2 - 2, 1 - c2, c1 + b2, 1 + d, 1 - d**2)*((-d)**(b1 + b2 - 1)*(1 + d)**(c1 + b2 - 1))/K
               except ZeroDivisionError:
                   out = 0
               except OverflowError:
                   out = 0
           elif d > 0 and d <= 1:
               try:
                   out = scipy.special.beta(c2, b1)*F1(b1, c1 + c2 + b1 + b2 - 2, 1 - c1, c2 + b1, 1 - d, 1 - d**2)*(d**(b1 + b2 - 1)*(1 - d)**(c2 + b1 - 1))/K
               except ZeroDivisionError:
                   out = 0
               except OverflowError:
                   out = 0
           else:
               out = 0
           return out
       
def ddiffprop(x,k1,k2,n1,n2,a1=0.5,a2=0.5,log = False):
    '''
Difference Between Two Proportions Distribution

Description
    Density (mass). This is determined by taking the difference between two 
    independent beta distributions.

Usage
    ddiffprop(x, k1, k2, n1, n2, a1 = 0.5, a2 = 0.5, log = FALSE)
    
Parameters
----------
    x: float or list of floats
        Vector of quantiles.

    k1, k2: int	
        The number of successes drawn from groups 1 and 2, respectively.

    n1, n2: int
        The sample sizes for groups 1 and 2, respectively.

    a1, a2: float	
        The shift parameters for the beta distributions. For the fiducial 
        approach, we know that the lower and upper limits are set at 
        a1 = a2 = 0 and a1 = a2 = 1, respectively, for the true p1 and p2. 
        While computations can be performed on real values outside the unit 
        interval, a warning message will be returned if such values are 
        specified. For practical purposes, the default value of 0.5 should be 
        used for each parameter.

    log: bool
        Logical vectors. If TRUE, then the probabilities are given as log.

Details
    The difference between two proportions distribution has a fairly 
    complicated functional form. Please see the article by Chen and Luo (2011)
    , who corrected a typo in the article by Nadarajah and Kotz (2007), for 
    the functional form of this distribution.

Returns
    ddiffprop gives the density (mass)

References
----------
    Chen, Y. and Luo, S. (2011), A Few Remarks on 'Statistical Distribution 
        of the Difference of Two Proportions', Statistics in Medicine, 30, 
        1913–1915.
    
    Derek S. Young (2010). tolerance: An R Package for Estimating Tolerance 
        Intervals. Journal of Statistical Software, 36(5), 1-39. 
        URL http://www.jstatsoft.org/v36/i05/.

    Nadarajah, S. and Kotz, S. (2007), Statistical Distribution of the 
        Difference of Two Proportions, Statistics in Medicine, 26, 3518–3523.

Examples
    ## Randomly generated data from the difference between
    ## two proportions distribution.
        ddiffprop(x = [1,2,3], k1 = 2, k2 = 10, n1 = 17, n2 = 13)
    '''
    if (a1<0 or a1>1) or (a2<0 or a2 >1):
        return "a1 and a2 should both be between 0 and 1 for this fiducial approach!"
    d = np.zeros(length(x))
    temp = np.zeros(length(x))
    for i in range(length(x)):
        if length(x) == 1:
            d = x
            temp[i] = ddiffprop2(d=d, k1=k1, k2=k2, n1=n1, n2=n2, a1=a1, a2=a2)         
        else:
            d[i] = x[i]
            temp[i] = ddiffprop2(d=d[i], k1=k1, k2=k2, n1=n1, n2=n2, a1=a1, a2=a2)
        if log:
            temp[i] = np.log(temp[i])
    return temp

def pdiffprop2(x,k1,k2,n1,n2,a1,a2):
        if x <= -1:
            out = 0
        elif x >= 1:
            out = 1
        else:
            out = scipy.integrate.quad(ddiffprop,a=-1,b=x,args=(k1,k2,n1,n2,a1,a2))[0]
        return out

def pdiffprop(q,k1,k2,n1,n2,a1=0.5,a2=0.5,lowertail = True, logp= False):
    '''
Difference Between Two Proportions Distribution

Description
    Distribution function. This is determined by taking the difference between
    two independent beta distributions.

Usage
    pdiffprop(q, k1, k2, n1, n2, a1 = 0.5, a2 = 0.5, lowertail = True, logp = 
              FALSE)
    
Parameters
----------
    q: float or list of floats
        Vector of quantiles.

    k1, k2: int	
        The number of successes drawn from groups 1 and 2, respectively.

    n1, n2: int
        The sample sizes for groups 1 and 2, respectively.

    a1, a2: float	
        The shift parameters for the beta distributions. For the fiducial 
        approach, we know that the lower and upper limits are set at 
        a1 = a2 = 0 and a1 = a2 = 1, respectively, for the true p1 and p2. 
        While computations can be performed on real values outside the unit 
        interval, a warning message will be returned if such values are 
        specified. For practical purposes, the default value of 0.5 should be 
        used for each parameter.
        
    lowertail: bool
        Logical vector. If TRUE, then probabilities are P[X≤ x], else P[X>x].

    logp: bool
        Logical vectors. If TRUE, then the probabilities are given as logp.

Details
    The difference between two proportions distribution has a fairly 
    complicated functional form. Please see the article by Chen and Luo (2011)
    , who corrected a typo in the article by Nadarajah and Kotz (2007), for 
    the functional form of this distribution.

Returns
    pdiffprop gives the distribution function.

References
----------
    Chen, Y. and Luo, S. (2011), A Few Remarks on 'Statistical Distribution 
        of the Difference of Two Proportions', Statistics in Medicine, 30, 
        1913–1915.
    
    Derek S. Young (2010). tolerance: An R Package for Estimating Tolerance 
        Intervals. Journal of Statistical Software, 36(5), 1-39. 
        URL http://www.jstatsoft.org/v36/i05/.

    Nadarajah, S. and Kotz, S. (2007), Statistical Distribution of the 
        Difference of Two Proportions, Statistics in Medicine, 26, 3518–3523.

Examples
    ## Randomly generated data from the difference between
    ## two proportions distribution.
        pdiffprop(q = [1,2,3], k1 = 2, k2 = 10, n1 = 17, n2 = 13)
    '''
    if (a1<0 or a1>1) or (a2<0 or a2 >1):
        return "a1 and a2 should both be between 0 and 1 for this fiducial approach!"
    x = np.zeros(length(q))
    temp = np.zeros(length(q))
    for i in range(length(q)):
        if length(q) == 1:
            x = q
            temp[i] = pdiffprop2(x=x,k1=k1,k2=k2,n1=n1,n2=n2,a1=a1,a2=a2)
        else:
            x[i] = q[i]
            temp[i] = pdiffprop2(x=x[i],k1=k1,k2=k2,n1=n1,n2=n2,a1=a1,a2=a2)
        if lowertail == False:
            temp[i] = 1-temp[i]
        if logp:
            temp[i] = np.log(temp[i])
    return temp


 
def qdiffprop2(p, k1, k2, n1, n2, a1 = 0.5, a2 = 0.5):
    if p <= 0:
        out = -1
    elif p >= 1:
        out = 1
    else:
        def tempfun(d, p, k1, k2, n1, n2, a1, a2):
                return p - scipy.integrate.quad(ddiffprop, a=-1,b=d, args=(k1,k2,n1,n2,a1,a2))[0]
        try:
            out = opt.brentq(tempfun, a = -1, b = 1, args=(p,k1,k2,n1,n2,a1,a2))
        except:
            temp2 = [abs(tempfun(-1,p,k1,k2,n1,n2,a1,a2)), abs(tempfun(1,p,k1,k2,n1,n2,a1,a2))]
            if min(temp2) == temp2[0]:
                out = -1
            else:
                out = 1
    return out
    
def qdiffprop(p, k1, k2, n1, n2, a1 = 0.5, a2 = 0.5, lowertail = True, logp = False):
    '''
Difference Between Two Proportions Distribution

Description
    Quantile function This is determined by taking the difference between two 
    independent beta distributions.

Usage
    qdiffprop(p, k1, k2, n1, n2, a1 = 0.5, a2 = 0.5, lowertail = True, logp = 
              FALSE)
    
Parameters
----------
    p: float or list of floats
        Vector of probabilities.

    k1, k2: int	
        The number of successes drawn from groups 1 and 2, respectively.

    n1, n2: int
        The sample sizes for groups 1 and 2, respectively.

    a1, a2: float	
        The shift parameters for the beta distributions. For the fiducial 
        approach, we know that the lower and upper limits are set at 
        a1 = a2 = 0 and a1 = a2 = 1, respectively, for the true p1 and p2. 
        While computations can be performed on real values outside the unit 
        interval, a warning message will be returned if such values are 
        specified. For practical purposes, the default value of 0.5 should be 
        used for each parameter.

    lowertail: bool
        Logical vector. If TRUE, then probabilities are P[X≤ x], else P[X>x].
        
    logp: bool
        Logical vectors. If TRUE, then the probabilities are given as logp.
        
Details
    The difference between two proportions distribution has a fairly 
    complicated functional form. Please see the article by Chen and Luo (2011)
    , who corrected a typo in the article by Nadarajah and Kotz (2007), for 
    the functional form of this distribution.

Returns
    qdiffprop gives the quantile function

References
----------
    Chen, Y. and Luo, S. (2011), A Few Remarks on 'Statistical Distribution 
        of the Difference of Two Proportions', Statistics in Medicine, 30, 
        1913–1915.
    
    Derek S. Young (2010). tolerance: An R Package for Estimating Tolerance 
        Intervals. Journal of Statistical Software, 36(5), 1-39. 
        URL http://www.jstatsoft.org/v36/i05/.

    Nadarajah, S. and Kotz, S. (2007), Statistical Distribution of the 
        Difference of Two Proportions, Statistics in Medicine, 26, 3518–3523.

Examples
    ## Randomly generated data from the difference between
    ## two proportions distribution.
        pdiffprop(q = [1,2,3], k1 = 2, k2 = 10, n1 = 17, n2 = 13)
    '''
    if (a1<0 or a1>1) or (a2<0 or a2 >1):
        return "a1 and a2 should both be between 0 and 1 for this fiducial approach!"
    temp = np.zeros(length(p))
    for i in range(length(temp)):
        if length(p) == 1:
            if logp:
                p = np.exp(p)
            if not lowertail:
                p = 1-p
            temp[i] = qdiffprop2(p=p,k1=k1,k2=k2,n1=n1,n2=n2,a1=a1,a2=a2)
        else:
            if logp:
                p[i] = np.exp(p[i])
            if not lowertail:    
                p[i] = 1-p[i]
            temp[i] = qdiffprop2(p=p[i],k1=k1,k2=k2,n1=n1,n2=n2,a1=a1,a2=a2)
    return temp
    

def rdiffprop(n, k1, k2, n1, n2, a1 = 0.5, a2 = 0.5):
    '''
Difference Between Two Proportions Distribution

Description
    Random generation for the difference between two proportions. This is 
    determined by taking the difference between two independent beta 
    distributions.

Usage
    rdiffprop(n, k1, k2, n1, n2, a1 = 0.5, a2 = 0.5, lowertail = True, logp = 
              FALSE)
    
Parameters
----------
    n: int	
        The number of observations. If length>1, then the length is taken to 
        be the number required.

    k1, k2: int	
        The number of successes drawn from groups 1 and 2, respectively.

    n1, n2: int
        The sample sizes for groups 1 and 2, respectively.

    a1, a2: float	
        The shift parameters for the beta distributions. For the fiducial 
        approach, we know that the lower and upper limits are set at 
        a1 = a2 = 0 and a1 = a2 = 1, respectively, for the true p1 and p2. 
        While computations can be performed on real values outside the unit 
        interval, a warning message will be returned if such values are 
        specified. For practical purposes, the default value of 0.5 should be 
        used for each parameter.

Details
    The difference between two proportions distribution has a fairly 
    complicated functional form. Please see the article by Chen and Luo (2011)
    , who corrected a typo in the article by Nadarajah and Kotz (2007), for 
    the functional form of this distribution.

Returns
    rdiffprop generates random deviates.

References
----------
    Chen, Y. and Luo, S. (2011), A Few Remarks on 'Statistical Distribution 
        of the Difference of Two Proportions', Statistics in Medicine, 30, 
        1913–1915.
    
    Derek S. Young (2010). tolerance: An R Package for Estimating Tolerance 
        Intervals. Journal of Statistical Software, 36(5), 1-39. 
        URL http://www.jstatsoft.org/v36/i05/.

    Nadarajah, S. and Kotz, S. (2007), Statistical Distribution of the 
        Difference of Two Proportions, Statistics in Medicine, 26, 3518–3523.

Examples
    ## Randomly generated data from the difference between
    ## two proportions distribution.
        rdiffprop(n = 100, k1 = 2, k2 = 10, n1 = 17, n2 = 13)
    '''
    if ((a1 < 0 or a1 > 1) or (a2 < 0 or a2 > 1)):
        return "a1 and a2 should both be between 0 and 1 for this fiducial approach!"
    out=np.random.beta(size=int(n), a = k1 + a1, b = n1 - k1 + a1) - np.random.beta(size=int(n), a = k2 + a2, b = n2 - k2 + a2)
    return out

#x = rdiffprop(n=100,k1=2,k2=10,n1=17,n2=13)
#x1 = sorted(x)
#print(ddiffprop(x=x1, k1=2, k2=10, n1=17, n2=13,a1=0.4,a2=0.9))
#print(ddiffprop(x=1,k1=.2,k2=.1,n1=.1,n2=.2,a1=0.4,a2=0.9)) #works
#print(ddiffprop((.335,.44,.20,.1289,.45),k1 = 12, k2 = 11, n1 = 13, n2 = 120,log=True,a1=0.7,a2=0.4))
#print(pdiffprop(q = x, k1 = 2, k2 = 10, n1 = 17, n2 = 13,lowertail = False,logp = True))
#print(qdiffprop(p = (0.650,.32), k1 = 2, k2 = 10, n1 = 17, n2 = 13))
#print(rdiffprop(n = 100, k1 = 2, k2 = 10, n1 = 17, n2 = 13))