# BeSS: An R Package for Best Subset Selection and Best Subset Ridge Regression


Introdution
----------

The advance in modern technology, including computing power and storage, brings about more and more high-dimensional data in which the number of features can be much larger than the number of observations (Hastie et al. 2009). Examples include gene, microarray, and proteomics data, high-resolution images, high-frequency financial data, e-commerce data, warehouse data, resonance imaging, signal processing, among many others (Fan et al. 2011). 

Since it is not easy to explain the relationship between the response and the variables if the model is too complicated, associated with a lot of predictors for example, and reducing the number of variables resorting to subjective approaches can be influenced by one's interests and hypotheses. There are at least three challenges for regression methods under the high dimensional setting:

- How to find
models with good prediction performance?

- How to discover the
true “sparsity pattern”?

- How to find models combining the above-mentioned two abilities?

The best subset selection is up to these challenges, which enjoy the following admirable advantages:

- It obtains an unbiased estimator as long as the true active set is discovered.

- It ranks highest in terms of model interpretation.

- It provides an objective way to reduce the number of variables.

- By excluding irrelative variables, the best subset selection earns improved out-of-sample accuracy and avoids overfitting in some sence.

By introducing a shrinkage on the coefficients the best subset ridge regression provides a more sophisticated trade-off between model parsimony and prediction on the based of the best subset selection.



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
Following are comparisons with some R packages aiming at best subset selection in several metrics:
| |[leaps](https://cran.r-project.org/package=leaps)|[lmSubset](https://cran.r-project.org/package=lmSubsets) |[bestglm](https://cran.r-project.org/package=bestglm)|[glmuti](https://cran.r-project.org/package=glmulti)|[BeSS](https://cran.r-project.org/package=BeSS)
| :-------------------------------- | :----------------------------------------------------------: | :--------------------------------------------------------: | :--------------------------------------------------: | :----------------------------------------------------: | :--------------------------------------------------: | 
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
| Computational efficiency          | :walking::walking: |:walking::running:|:walking::walking:(impossible for glm with variable number greater than 15)|:walking::running: (impossible for glm with variable number greater than 32) |:running::running:|


See the following documents for more details about the **[BeSS](https://cran.r-project.org/package=BeSS)** package:

<!--- - [github page](https://github.com/Mamba413/bess/tree/master/R) (short) -->

- vignette can be opened with `vignette("BeSS")` in R (moderate)

- [JSS paper](https://www.jstatsoft.org/v094/i04) (detailed)

References
----------
- Wen, C., Zhang, A., Quan, S., & Wang, X. (2020). BeSS: An R Package for Best Subset Selection in Linear, Logistic and Cox Proportional Hazards Models. Journal of Statistical Software, 94(4), 1 - 24. doi:http://dx.doi.org/10.18637/jss.v094.i04

