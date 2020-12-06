#' Provides estimated coefficients from a fitted "bess" object.
#'
#' This function provides estimated
#' coefficients from a fitted "\code{bess}" object.
#'
#'
#' @param object A "\code{bess}" project.
#' @param sparse Logical or NULL, specifying whether the coefficients should be
#' presented as sparse matrix or not.
#' @param \dots Other arguments.
#' @author Canhong Wen, Aijun Zhang, Shijie Quan, Liyuan Hu, Kangkang Jiang, Yanhang Zhang, Jin Zhu and Xueqin Wang.
#' @seealso \code{\link{bess}}, \code{\link{print.bess}}.
#' @references Wen, C., Zhang, A., Quan, S. and Wang, X. (2020). BeSS: An R
#' Package for Best Subset Selection in Linear, Logistic and Cox Proportional
#' Hazards Models, \emph{Journal of Statistical Software}, Vol. 94(4).
#' doi:10.18637/jss.v094.i04.
#' @examples
#'
#' # Generate simulated data
#' n <- 200
#' p <- 20
#' k <- 5
#' rho <- 0.4
#' SNR <- 10
#' cortype <- 1
#' seed <- 10
#' Data <- gen.data(n, p, k, rho, family = "gaussian", cortype = cortype, snr = SNR, seed = seed)
#' x <- Data$x[1:140, ]
#' y <- Data$y[1:140]
#' x_new <- Data$x[141:200, ]
#' y_new <- Data$y[141:200]
#' lm.bss <- bess(x, y, method = "sequential")
#' lambda.list <- exp(seq(log(5), log(0.1), length.out = 10))
#' lm.bsrr <- bess(x, y, type = "bsrr", method = "pgsection")
#' coef(lm.bss)
#' coef(lm.bsrr)
#'
#'
#'
coef.bess <- function(object, sparse=TRUE, ...)
{
  if(!is.null(object$coef0)){
    beta<-c(intercept=object$coef0, object$beta)
    names(beta)[1] <- "(intercept)"
  } else beta<-object$beta
  if(sparse==TRUE)
  {
    beta<-matrix(beta,byrow =TRUE, dimnames = list(names(beta)))
    beta<-Matrix(beta,sparse = TRUE)
    return(beta)
  }else return(beta)
}
