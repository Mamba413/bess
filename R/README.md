# BeSS: An R Package for Best Subset Selection and Best subset ridge regression

## Introduction

There are at
least three challenges for regression methods under the high dimensional setting:

- How to find
models with good prediction performance?

- How to discover the
true “sparsity pattern”?

- How to find models combining the above mentioned two abilities?

The best subset selection is up to these challenge, which enjoy following admirable advantages:

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
devtools::install_github("Mamba413/bess")
```

Compared with selective R packages available for datasets in metric spaces:
| |[leaps](https://cran.r-project.org/package=leaps)|[lmSubset](https://cran.r-project.org/web/packages/lmSubsets/index.html) |[L0learn](https://cran.r-project.org/package=L0Learn)|[BeSS](https://cran.r-project.org/web/packages/BeSS/index.html)
| :-------------------------------- | :----------------------------------------------------------: | :--------------------------------------------------------: | :--------------------------------------------------: | :----------------------------------------------------: |
| Solve generalized linear models|:x:|:x:|:x:|:heavy_check_mark:     |
|  Feature screening |:x:|:x:|:heavy_check_mark:     |:heavy_check_mark:     |
| Tuning parameter determination on information criterion |:x:|:heavy_check_mark:|:x:|:heavy_check_mark:|
| Tuning parameter determination on cross-validation |:x:|:x:|:heavy_check_mark:|:heavy_check_mark:|
| Include specified variables|:x:|:heavy_check_mark:|:x:|:heavy_check_mark:|
| Options for coefficient shrinkage|:x:|:x:|:heavy_check_mark:|:heavy_check_mark:|
| Computational efficiency          | :walking::walking::walking: |:walking::walking::running:|:running::running::running:|:running::running::walking:|


See the following documents for more details about the **[BeSS](https://cran.r-project.org/web/packages/BeSS/index.html)** package:

- [github page](https://github.com/Mamba413/bess/tree/master/R) (short)

- vignette can be opened with `vignette("BeSS")` in R (moderate)

- [JSS paper](https://www.jstatsoft.org/v094/i04) (detailed)

References
----------
- Wen, C. , Zhang, A. , Quan, S. , & Wang, X. . (2017). [Bess: an r package for best subset selection in linear, logistic and coxph models](https://arxiv.org/pdf/1709.06254.pdf)

