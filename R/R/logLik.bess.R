#' Extract the log-likelihood from a "bess.one" object.
#'
#' This function returns the log-likelihood for the fitted models.
#'
#' The log-likelihood for the best model chosen by a certain information criterion
#' or cross-validation corresponding to the call in \code{bess} or the best models
#' with model size and \eqn{\lambda} in the original \code{s.list} and \code{lambda.list}
#' (or the in the iteration path) can be returned.
#' For "lm" fits it is assumed that the scale has been estimated
#'  (by maximum likelihood or REML),
#'  and all the constants in the log-likelihood are included.
#'
#' @param object A "\code{bess}" object.
#' @param best.model Whether only return the log-likelihood of the best model. Default is \code{TRUE}.
#' If \code{best.model = FALSE}, the log-likelihood of the best models with model size and
#'  \eqn{\lambda} in the original \code{s.list} and \code{lambda.list} (for \code{method = "sequential"})
#'  or in the iteration path (for \code{method = "gsection"}, \code{method = "pgsection"},
#'  and \code{method = "psequential"}) is returned.
#' @param \dots additional arguments
#' @return A matrix or vector containing the log-likelihood for each model is returned.
#' For \code{bess} objects fitted by \code{sequantial} method, values in each row in the
#' returned matrix corresponding to the model size in \code{s.list}, and each column the shrinkage parameters
#' in \code{lambda.list}.
#'
#' For \code{bess} objects fitted by \code{gsection}, \code{pgsection} and \code{psequential}, the returned vector
#' contains log-likelihood for fitted models in each iteration. The coefficients of those model can be extracted
#' from \code{beta.all} and \code{coef0.all} in the \code{bess} object.
#' @author Canhong Wen, Aijun Zhang, Shijie Quan, Liyuan Hu, Kangkang Jiang, Yanhang Zhang, Jin Zhu and Xueqin Wang.
#' @seealso \code{\link{bess}}, \code{\link{summary.bess}}.
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
#' Tbeta <- rep(0, p)
#' Tbeta[1:k*floor(p/k):floor(p/k)] <- rep(1, k)
#' Data <- gen.data(n, p, k, rho, family = "gaussian", cortype = cortype, snr = SNR, seed = seed)
#' lm.bss <- bess(Data$x, Data$y, method = "sequential")
#'
#' logLik(lm.bss)
#' logLik(lm.bss, best.model = FALSE)
#'
#'@export
#'@export logLik.bess
#'@method logLik bess


logLik.bess <- function(object, best.model = TRUE,...){
  n=object$nsample
  if(best.model){
    if(object$family!="gaussian"){
      Loglik <- -object$loss/2
    }else{
      #Loglik=-n*log(object$loss)/2
      Loglik <- -n/2 *(log(2*pi) + log(object$loss) + 1)
    }
    names(Loglik) <- 'Loglik'
    class(Loglik) <- 'logLik'
    return(Loglik)
  } else{
    if(!is.null(object$bess.one)) stop("Please set best.model = TRUE for bess objects from bess.one function.")
    deviance <- deviance(object, best.model= FALSE)
    if(object$family!="gaussian"){
      Loglik <- -deviance/2
    }else{
      deviance <- deviance(object, best.model=FALSE)
      loss <- exp(deviance/n)*2
      Loglik <- -n/2 *(log(2*pi) + log(loss) + 1)
    }
    return(Loglik)
  }
}
