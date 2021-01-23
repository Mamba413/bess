# BeSS: A Python/R Package for Best Subset Selection <img src='https://raw.githubusercontent.com/Mamba413/git_picture/master/BeSS.png' align="right" height="120" />


## Introduction

One of the main tasks of statistical modeling is to exploit the association between
a response variable and multiple predictors. Linear model (LM), as a simple parametric
regression model, is often used to capture linear dependence between response and
predictors. Generalized linear model (GLM) can be considered as
the extensions of linear model, depending on the types of responses. Parameter estimation in these models
can be computationally intensive when the number of predictors is large. Meanwhile,
Occam's razor is widely accepted as a heuristic rule for statistical modeling,
which balances goodness of fit and model complexity. This rule leads to a relative 
small subset of important predictors. 

**BeSS** package provides solutions for best subset selection problem for sparse LM,
and GLM models.

We consider a primal-dual active set (PDAS) approach to exactly solve the best subset
selection problem for sparse LM and GLM models. 
It utilizes an active set updating strategy and fits the sub-models through use of
complementary primal and dual variables. We generalize the PDAS algorithm for 
general convex loss functions with the best subset constraint.


## Installation

### Python 

The package has been publish in PyPI. You can easy install by:
```sh
$ pip install bess
```

### R

To download and install **BeSS** from CRAN:

```r
install.packages("BeSS")
```

Or try the development version on GitHub:

```r
# install.packages("devtools")
devtools::install_github("Mamba413/bess/R")
```



## Reference

- Wen, C., Zhang, A., Quan, S., & Wang, X. (2020). BeSS: An R Package for Best Subset Selection in Linear, Logistic and Cox Proportional Hazards Models. Journal of Statistical Software, 94(4), 1 - 24. doi:http://dx.doi.org/10.18637/jss.v094.i04

## Bug report

Please send an email to Jiang Kangkang(jiangkk3@mail2.sysu.edu.cn).

