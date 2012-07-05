### empca: Weighted Expectation Maximization Principal Component Analysis ###

Classic PCA is great but it doesn't know how to handle noisy or missing
data properly.  This module provides Weighted Expectation Maximization PCA,
an iterative method for solving PCA while properly weighting data.
Missing data is simply the limit of weight=0.

Given data[nobs, nvar] and weights[nobs, nvar],

    m = empca(data, weights, options...)

Returns a Model object m, from which you can inspect the eigenvectors,
coefficients, and reconstructed model, e.g.

    pylab.plot( m.eigvec[0] )
    pylab.plot( m.data[0] )
    pylab.plot( m.model[0] )
    
For comparison, two alternate methods are also implemented which also
return a Model object:

    m = lower_rank(data, weights, options...)
    m = classic_pca(data)  #- but no weights or even options...

Everything is self contained in empca.py .  Just put that into your
PYTHONPATH and "pydoc empca" for more details.  For a quick test
on toy example data, run

    python empca.py

Requires numpy and scipy; will make plots if you have pylab installed.

A paper describing the underlying math is in the works,
but not available for citation yet.
If you use this code in an academic paper, please include an
acknowledgement such as

> This work uses the Weighted EMPCA code by Stephen Bailey, available
> at https://github.com/sbailey/empca/

Stephen Bailey, Summer 2012

