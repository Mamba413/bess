# BeSS: An R Package for Best Subset Selection and Best subset ridge regression

## Introduction

One of the main tasks of statistical modeling is to exploit the association between
a response variable and multiple predictors. The linear model (LM), as a simple parametric
regression model, is often used to capture linear dependence between response and
predictors. The generalized linear model (GLM) can be considered as
the extension of the linear model, depending on the types of responses. Parameter estimation in these models
can be computationally intensive when the number of predictors is large. Meanwhile,
Occam's razor is widely accepted as a heuristic rule for statistical modeling,
which balances the goodness of fit and model complexity. This rule leads to a relatively small subset of important predictors. 

**BeSS** package provides solutions for best subset selection problem and the best subset ridge regression for sparse LM,
and GLM models.

We consider a primal-dual active set (PDAS) approach to exactly solve the best subset
selection problem and the best subset ridge regression for sparse LM and GLM models. The PDAS algorithm for linear 
least-squares problems was first introduced by [Ito and Kunisch (2013)](https://iopscience.iop.org/article/10.1088/0266-5611/30/1/015001)
and later discussed by [Jiao, Jin, and Lu (2015)](https://arxiv.org/abs/1403.0515) and [Huang, Jiao, Liu, and Lu (2017)](https://arxiv.org/abs/1701.05128). 
It utilizes an active set updating strategy and fits the sub-models through the use of
complementary primal and dual variables. We generalize the PDAS algorithm for general convex loss functions with the best subset constraint. For the best subset selection problem we further extend the PDAS algorithm to support both sequential and golden section search strategies
for optimal k determination. Besides a grid search method, the two tuning parameters in the best ridge regression are determined through Powell's conjugate direction method.

## Installation

To download and install `msaenet` from CRAN:

```r
install.packages("BeSS")
```

Or try the development version on GitHub:

```r
# install.packages("devtools")
devtools::install_github("Mamba413/bess")
```

## Functions and examples

In the **BeSS** package, we have the following functions:

* bess:  Best subset selection/Best subset ridge regression with a specified model size for generalized linear models and Cox's proportional hazard model.

* bess.one:  Best subset selection/Best subset ridge regression with a specified model size (and a specified shrinkage parameter for the best subset ridge regression) for generalized linear models and Cox's proportional hazard model.

* coef.bess:  This function provides estimated coefficients from a fitted "bess" object.

* deviance.bess:  This function provides deviance from a fitted "bess" object.

* logLik.bess: This function provides loglikelihood from a fitted "bess" object.

* summary.bess: This function provides a summary for "bess" object.

* plot.bess:  Produces a coefficient profile plot of the coefficient or loss paths for a fitted "bess" object.

* predict.bess.R:  This function provides predictions from a fitted "bess" object.

* print.bess.R:  Print the primary elements of the "bess" object.

Here is an example of how to use this package to solve a linear model. The vignette can be opened with `vignette("BeSS")` in R for a quick-start.

```r
# Generate simulated data
n <- 200
p <- 20
k <- 5
rho <- 0.4
seed <- 10
Tbeta <- rep(0, p)
Tbeta[1:k*floor(p/k):floor(p/k)] <- rep(1, k)
Data <- gen.data(n, p, k, rho, family = "gaussian", beta = Tbeta, seed = seed)

x <- Data$x[1:140, ]
y <- Data$y[1:140]
x_new <- Data$x[141:200, ]
y_new <- Data$y[141:200]

# solve the best subset selection problem
lm.bss <- bess(x, y)

# solve the best subset ridge regression problem
lambda.list <- exp(seq(log(5), log(0.1), length.out = 10))
lm.bsrr <- bess(x, y, type = "bsrr", method = "pgsection")

# Best subset selection with a specified model size
lm.bss.one <- bess(x, y, s = 5)

# Best subset ridge regression with a specified model size and a specified shrinkage parameter
lm.bsrr.one <- bess(x, y, s = 5, lambda = 0.01)

# get the estimated coefficients of type dgCMatrix
coef(lm.bss)
coef(lm.bsrr)

# obtain the deviance and loglikelihood
deviance(lm.bss)
deviance(lm.bsrr)
logLik(lm.bss)
logLik(lm.bsrr)

# get summaries
summary(lm.bss)
summary(lm.bsrr)

# generate plots
plot(lm.bss, type = "both", breaks = TRUE)
plot(lm.bsrr)

# make predictions on new data
pred.bss <- predict(lm.bss, newx = x_new)
pred.bsrr <- predict(lm.bsrr, newx = x_new)

# print the primary information of the calls
print(lm.bss)
print(lm.bsrr)
```


## Reference

- Wen, C. , Zhang, A. , Quan, S. , & Wang, X. . (2017). [Bess: an r package for best subset selection in linear, logistic and coxph models](https://arxiv.org/pdf/1709.06254.pdf)


