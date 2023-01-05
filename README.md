# tolerance_python
Python package for tolerance intervals. Derived from: Derek S. Young (2010). tolerance: An R Package for Estimating Tolerance Intervals. Journal of Statistical Software, 36(5), 1-39. URL http://www.jstatsoft.org/v36/i05/.

The package answers the practical question: "I have (1-α)*100% confidence that (P)*100% of my population falls within certain bounds."

gamtolint file is slightly different than the R equivalent, this is due to R using the Newton minimization method and the Python code uses the CG minimization method. 
