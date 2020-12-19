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
#' seed <- 10
#' Tbeta <- rep(0, p)
#' Tbeta[1:k*floor(p/k):floor(p/k)] <- rep(1, k)
#' Data <- gen.data(n, p, k, rho, family = "gaussian", beta = Tbeta, seed = seed)
#' lm.bss <- bess(Data$x, Data$y, method = "sequential")
#' lambda.list <- exp(seq(log(5), log(0.1), length.out = 10))
#' lm.bsrr <- bess(Data$x, Data$y, type = "bsrr", method = "pgsection")
#' coef(lm.bss)
#' coef(lm.bsrr)
#' @export
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
