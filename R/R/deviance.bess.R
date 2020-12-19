# Various imports
#' @importFrom Rcpp evalCpp
#' @importFrom stats runif
#' @importFrom survival coxph
#' @importFrom survival Surv
#' @importFrom stats glm
#' @importFrom stats lm
#' @importFrom stats rbinom
#' @importFrom stats rpois
#' @importFrom stats rnorm
#' @importFrom stats rexp
#' @importFrom stats deviance
#' @importFrom stats logLik
#' @importFrom stats quantile
#' @importFrom stats coef
#' @importFrom MASS mvrnorm
#' @importFrom Matrix Matrix
#' @importFrom pheatmap pheatmap
#' @importFrom graphics abline
#' @importFrom graphics axis
#' @importFrom graphics box
#' @importFrom graphics grid
#' @importFrom graphics layout
#' @importFrom graphics lines
#' @importFrom graphics mtext
#' @importFrom graphics par
#' @importFrom graphics plot
#' @importFrom graphics plot.new
#' @importFrom graphics plot.window
#' @importFrom graphics text
#' @importFrom graphics title
NULL

#' Extract the deviance from a "bess.one" object.
#'
#' Similar to other deviance methods, which returns deviance from a fitted "\code{bess.one}" object.
#'
#'
#' @param object A "\code{bess}" object.
#' @param best.model Whether only return the loglikelihood of the best model. Default is \code{TRUE}.
#' If \code{best.model = FALSE}, the loglikelihood of the best models with model size and
#'  \eqn{\lambda} in the original \code{s.list} and \code{lambda.list} (for \code{method = "sequential"})
#'  or in the iteration path (for \code{method = "gsection"}, \code{method = "pgsection"},
#'  and \code{method = "psequential"}) is returned.
#' @return A matrix or vector containing the deviance for each model is returned.
#' For \code{bess} object fitted by \code{sequantial} method, values in each row in the
#' returned matrix corresponding to the model size in \code{s.list}, and each column the shrinkage parameters
#' in \code{lambda.list}.
#'
#' For \code{bess} object fitted by \code{gsection}, \code{pgsection} and \code{psequential}, the returned vector
#' contains deviance for fitted models in each iteration. The coefficients of those model can be extracted
#' from \code{beta.all} and \code{coef0.all} in the \code{bess} object.
#' @param \dots additional arguments
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
#' seed <- 10
#' Tbeta <- rep(0, p)
#' Tbeta[1:k*floor(p/k):floor(p/k)] <- rep(1, k)
#' Data <- gen.data(n, p, k, rho, family = "gaussian", seed = seed)
#' lm.bss <- bess(Data$x, Data$y, method = "sequential")
#'
#' deviance(lm.bss)
#' deviance(lm.bss, best.model = FALSE)
#'@method deviance bess
#' @export

deviance.bess <- function(object, best.model = TRUE,...)
{
  n=object$nsample
  if(best.model){
    if(object$family!="gaussian"){
      deviance= object$loss
    }else{
      deviance=n*log(object$loss/2)
    }
    names(deviance)='deviance'
    return(deviance)
  }else{
    if(!is.null(object$bess.one)) stop("Please set best.model = TRUE for bess objects from bess.one function.")
    if(object$method == "sequential"){
      train_loss_all <- matrix(unlist(object$loss.all), nrow = length(object$s.list), byrow = F) # orgininally, the train_loss_all is a list
      deviance <- deviance.all(object, train_loss_all)
    }else{
      deviance <- as.vector(deviance.all(object, object$loss.all))
    }
    return(deviance)
  }
}

deviance.all = function(object, training_error){
  n=object$nsample
  if(object$family!="gaussian"){
    deviance=training_error
  }else{
    deviance=n*(unlist(training_error))
  }
  return(deviance)
}


