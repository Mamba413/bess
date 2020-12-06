#' print method for a "bess" object
#'
#' Print the primary elements of the "\code{bess}" object.
#'
#' prints the fitted model and returns it invisibly.
#'
#' @param x A "\code{bess}" object.
#' @param digits Minimum number of significant digits to be used.
#' @param nonzero Whether the output should only contain the non-zero coefficients.
#' @param \dots additional print arguments
#' @author Canhong Wen, Aijun Zhang, Shijie Quan, Liyuan Hu, Kangkang Jiang, Yanhang Zhang, Jin Zhu and Xueqin Wang.
#' @seealso \code{\link{bess}}, \code{\link{plot.bess}},
#' \code{\link{summary.bess}}.
#' @references Wen, C., Zhang, A., Quan, S. and Wang, X. (2020). BeSS: An R
#' Package for Best Subset Selection in Linear, Logistic and Cox Proportional
#' Hazards Models, \emph{Journal of Statistical Software}, Vol. 94(4).
#' doi:10.18637/jss.v094.i04.
#' @examples
#'
#'
#' # Generate simulated data
#' n = 200
#' p = 20
#' k = 5
#' rho = 0.4
#' SNR = 10
#' cortype = 1
#' seed = 10
#' Data = gen.data(n, p, k, rho, family = "gaussian", cortype=cortype, snr=SNR, seed=seed)
#' x = Data$x[1:140, ]
#' y = Data$y[1:140]
#' x_new = Data$x[141:200, ]
#' y_new = Data$y[141:200]
#' lm.bss = bess(x, y, method = "sequential")
#' lambda.list = exp(seq(log(5), log(0.1), length.out = 10))
#' lm.bsrr = bess(x, y, type = "bsrr", lambda.list = lambda.list, method = "sequential")
#'
#' print(lm.bss)
#' print(lm.bsrr)
#'
#'@method print bess
#'@export
#'@export print.bess
#'
print.bess<-function(x, digits = max(5, getOption("digits") - 5), nonzero = FALSE,...)
{
  # if(x$method != "sequential"){
  #   if(x$algorithm_type != "L0L2"){
  #     df <- apply(matrix(unlist(x$beta_all), nrow = length(x$beta), byrow = F), 2, function(x){sum(ifelse(abs(x) < 1e-8, 0, 1))})
  #     if(x$ic_type == "cv") {
  #       print(cbind(Df=df, deviance = unlist(deviance_all(x, x$train_loss_all)),  cvm = as.vector(x$cvm_all)))
  #     }else{
  #       ic <- x$ic_type
  #       print(cbind(Df=df, deviance =unlist(deviance_all(x, x$train_loss_all)),ic = as.vector(x$ic_all)))
  #     }
  #   } else{
  #     df <- apply(x$beta_all, 2, function(x){sum(ifelse(abs(x) < 1e-8, 0, 1))})
  #     if(x$ic_type == "cv"){
  #       print(cbind(DF = df, lambda = x$lambda_all, deviance = as.vector(deviance_all(x, x$train_loss_all)), cvm = as.vector(x$cvm_all)))
  #     } else{
  #       ic <- x$ic_type
  #       print(cbind(DF = df, lambda = x$lambda_all, deviance = as.vector(deviance_all(x, x$train_loss_all)), ic = as.vector(x$ic_all)))
  #     }
  #   }
  # } else{
  #   if(x$algorithm_type != "L0L2"){
  #     if(x$ic_type == "cv") {
  #       print(cbind(Df=x$s.list,deviance=unlist(deviance_all(x, x$train_loss_all)), cvm = as.vector(x$cvm_all)))
  #     }else{
  #       ic = x$ic_type
  #       print(cbind(Df=x$s.list, deviance=unlist(deviance_all(x, x$train_loss_all)),ic = as.vector(x$ic_all)))
  #     }
  #   } else{
  #     train_loss_all <- matrix(unlist(x$train_loss_all), nrow = length(x$s.list), byrow = F)
  #     deviance <- deviance_all(x, train_loss_all)
  #     # rownames(train_loss_all) = x$s.list
  #     # colnames(train_loss_all) = x$lambda_list
  #     if(x$ic_type == "cv") {
  #       cv_all <- x$cvm_all
  #       #rownames(cv_all) <- x$s.list
  #       #colnames(cv_all) <- x$lambda_list
  #       print(list(deviance = deviance, cvm = cv_all))
  #     }else{
  #       ic_all <- x$ic_all
  #       # rownames(ic_all) = x$s.list
  #       #colnames(ic_all) = x$lambda_list
  #       ic <- x$ic_type
  #       print(list(deviance = deviance, ic = ic_all))
  #     }
  #   }
  # }

    cat("Call:\n", paste(deparse(x$call), sep = "\n", collapse = "\n"), "\n\n", sep =
          "")
  if(!nonzero){
    print(round(coef(x),  digits), ...)
  } else{
    coefx <- round(coef(x, sparse = FALSE)[which(coef(x, sparse = FALSE)!=0)], digits)
    print(coefx, ...)
  }
    cat("\n")
    invisible(x)


}
