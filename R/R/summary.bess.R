#' summary method for a "bess.one" object
#'
#' Print a summary of the "bess.one" object.
#'
#'
#' @param object a "bess" object
#' @param \dots additional print arguments
#' @author Liyuan Hu.
#' @seealso \code{\link{bess}}.
#' @references Wen, C., Zhang, A., Quan, S. and Wang, X. (2020). BeSS: An R
#' Package for Best Subset Selection in Linear, Logistic and Cox Proportional
#' Hazards Models, \emph{Journal of Statistical Software}, Vol. 94(4).
#' doi:10.18637/jss.v094.i04.
#' @examples
#'
#'
#' #-------------------linear model----------------------#
#' # Generate simulated data
#' n = 200
#' p = 20
#' k = 5
#' rho = 0.4
#' SNR = 10
#' cortype = 1
#' seed = 10
#' Data = gen.data(n, p, k, rho, family = "gaussian", cortype=cortype, SNR=SNR, seed=seed)
#' x = Data$x[1:140, ]
#' y = Data$y[1:140]
#' x_new = Data$x[141:200, ]
#' y_new = Data$y[141:200]
#' lm.pdas = bess(x, y)
#' lambda.list = exp(seq(log(5), log(0.1), length.out = 10))
#' lm.l0l2 = bess(x, y, type = "bsrr", lambda.list = lambda.list)
#' summary(lm.pdas)
#' summary(lm.l0l2)
#'
#'
#'
summary.bess <-function(object, ...){
  df <- sum(object$beta != 0)
  predictors <- names(which(object$beta!=0))
  a <- rbind(predictors, object$beta[predictors])
    cat("----------------------------------------------------------------------------------\n")
  if(object$algorithm_type != "L0L2")
    {
    cat("    Primal-dual active algorithm with tuning parameter determined by",object$method, "method", "\n\n")
    }else {
      if(object$method == "sequential"){
    cat("    Penalized Primal-dual active algorithm", "\n")
    cat("    with tuning parameter determined by",object$method, "method", "\n\n")
      } else{
    cat("    Penalized Primal-dual active algorithm with tuning parameter determined by","\n")
    cat("    powell method using",object$line.search,"for line search","\n\n")
      }
    }
  if(object$algorithm_type == "PDAS")
    cat("    Best model with k =", df, "includes predictors:", "\n\n") else cat("    Best model with k =", df,"lambda =",object$lambda, "includes predictors:", "\n\n")
  print(object$beta[predictors])
  cat("\n")
  if(logLik(object)>=0)
    cat("    log-likelihood:   ", logLik(object),"\n") else cat("    log-likelihood:  ", logLik(object),"\n")

  if(deviance(object)>=0)
    cat("    deviance:         ", deviance(object),"\n") else cat("    deviance:        ", deviance(object),"\n")
  if(object$ic_type %in% c("AIC", "BIC", "GIC")){
    if(object$ic>=0)
      cat("    ", object$ic_type, ":               ", object$ic,"\n", sep = "") else cat("    ",object$ic_type,":             ", object$ic,"\n", sep = "")

  } else if(object$ic_type == "EBIC"){
    if(object$ic>=0)
      cat("    EBIC:             ", object$ic,"\n") else cat("    EBIC:            ", object$ic,"\n")
  } else{
    if(object$cvm >= 0)
      cat("    cv loss:          ", object$cvm,"\n") else cat("    cv loss:         ", object$cvm,"\n")
  }
  cat("----------------------------------------------------------------------------------\n")
}

