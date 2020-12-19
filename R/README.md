# BeSS: An R Package for Best Subset Selection and Best subset ridge regression

## Introduction

There are at
least three challenges for regression methods under the high dimensional setting:

- How to find
models with good prediction performance?

- How to discover the
true “sparsity pattern”?

- How to find models combining the above-mentioned two abilities?

The best subset selection is up to these challenges, which enjoy the following admirable advantages:

- It obtains an unbiased estimator as long as the true active set is discovered.

- It ranks highest in terms of model interpretation.

- It provides an objective way to reduce the number of variables.

By introducing a shrinkage on the coefficients the best subset ridge regression provides a more sophisticated trade-off between model parsimony and prediction on the based of the best subset selection



Softwares
----------
### R package

To download and install **BeSS** from CRAN:

```r
install.packages("BeSS")
```

Or try the development version on GitHub:

```r
# install.packages("devtools")
devtools::install_github("Mamba413/bess/R")
```

Compared with selective R packages available for datasets in metric spaces:
| |[leaps](https://cran.r-project.org/package=leaps)|[lmSubset](https://cran.r-project.org/web/packages/lmSubsets/index.html) |[bestglm](https://cran.r-project.org/package=bestglm)|[glmuti](https://cran.r-project.org/package=glmulti)|[BeSS](https://cran.r-project.org/web/packages/BeSS/index.html)
| :-------------------------------- | :----------------------------------------------------------: | :--------------------------------------------------------: | :--------------------------------------------------: | :----------------------------------------------------: | :--------------------------------------------------: | :----------------------------------------------------: |
| Solve linear regression models|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:     |:heavy_check_mark:     |
| Solve logistic regression models|:x:|:x:|:heavy_check_mark:     |:heavy_check_mark:     |:heavy_check_mark:     |
| Solve poisson regression models|:x:|:x:|:heavy_check_mark:     |:heavy_check_mark:     |:heavy_check_mark:     |
| Solve CoxPH regression models|:x:|:x:|:x:     |:heavy_check_mark:     |:heavy_check_mark:     |
| group variable selection|:x:|:x:|:x:|:x:|:heavy_check_mark:     |
|  Feature screening |:x:|:x:|:x:  |:x:   |:heavy_check_mark:     |
| Tuning parameter determination on information criterion |:x:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
| Tuning parameter determination on cross-validation |:x:|:x:|:heavy_check_mark:|:x:|:heavy_check_mark:|
| Include specified variables|:x:|:heavy_check_mark:|:x:|:x:|:heavy_check_mark:|
| Options for coefficient shrinkage|:x:|:x:|:x:|:x:|:heavy_check_mark:|
| Computational efficiency          | :walking::walking::walking: |:walking::walking::running:|:walking::walking::running: (impossible for glm with variables number greater than 15)|:walking::walking::running: (impossible for glm with variables number greater than 32)|:running::running::running:|


See the following documents for more details about the **[BeSS](https://cran.r-project.org/web/packages/BeSS/index.html)** package:

- [github page](https://github.com/Mamba413/bess/tree/master/R) (short)

- vignette can be opened with `vignette("BeSS")` in R (moderate)

- [JSS paper](https://www.jstatsoft.org/v094/i04) (detailed)

References
----------
- Wen, C., Zhang, A., Quan, S., & Wang, X. (2020). BeSS: An R Package for Best Subset Selection in Linear, Logistic and Cox Proportional Hazards Models. Journal of Statistical Software, 94(4), 1 - 24. doi:http://dx.doi.org/10.18637/jss.v094.i04

