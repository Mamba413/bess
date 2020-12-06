# BeSS 1.0.0

## Improvments

- The majority of code in the package is now written in C++ for fast implementation, 
in which linear algebra is supported by the Eigen3 library for portable, high-performance computation. 
With Rcpp and RcppEigen,
the C++ program can be called from R by user-friendly interfaces. 

## New features

- It supports the best-subset ridge regression model (BSRR), which is a more flexible model. The advantage of BSRR is that it can adapt to the low signal-to-noise ratio and high-multicollinearity setting, tackle the non-identifiable problem in BSS when $p>n$,
and often outperforms the best subset selection as well as other variable selection algorithms in terms of prediction accuracy, while maintain the model's parsimony at the same time; To realize a BSRR model, set the `type` `bsrr`.

- The Poisson model is added as a complement for the case where the response is a count value. 

- The BeSS package now is capable of selecting variables with group structures (e.g., a group of dummy variables corresponding to a multilevel variable). The option `group.index` indicates the group index for each variable.

- Sure independent screening can be carried out through the option `screening.num` in consideration of dealing with the ultra-high dimensional data.