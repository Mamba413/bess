#' make predictions from a "bess" object.
#'
#' Similar to other predict methods, which returns predictions from a fitted
#' "\code{bess}" object.
#'
#' For "gaussian" family, \eqn{\hat{y} = X \beta} is returned.
#'
#' For "binomial" family,\deqn{\hat{Prob}(Y = 1) = exp(X \beta + \epsilon)/(1 +
#' exp(X \beta)) is returned. For "cox" family, \eqn{\eta = X \beta}} is
#' returned.
#'
#' @param object Output from the \code{bess} function or the \code{bess}
#' function.
#' @param newx New data used for prediction.
#' @param type Type "link" gives the linear predictors for "binomial",
#' , "poisson" or "cox" models; for "gaussian" models it gives the
#' fitted values. Type "response" gives the fitted probabilities for
#' "binomial", fitted mean for "poisson" and the fitted relative-risk for
#' "cox";For "gaussian", \code{type = "response"} is equivalent to \code{type = "link"}
#' @param \dots Additional arguments affecting the predictions produced.
#' @return The object returned depends on the types of family.
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
#' Data = gen.data(n, p, k, rho, family = "gaussian", cortype = cortype, SNR = SNR, seed = seed)
#' x = Data$x[1:140, ]
#' y = Data$y[1:140]
#' x_new = Data$x[141:200, ]
#' y_new = Data$y[141:200]
#' lm.pdas = bess(x, y, method = "sequential")
#' lambda.list = exp(seq(log(5), log(0.1), length.out = 10))
#' lm.l0l2 = bess(x, y, type = "bsrr")
#' pred.pdas = predict(lm.pdas, newx = x_new)
#' pred.l0l2 = predict(lm.l0l2, newx = x_new)
#' pred.pdas = predict(lm.pdas, newx = x_new, type = "response")
#' pred.l0l2 = predict(lm.l0l2, newx = x_new, type = "response")
#' #-------------------logistic model----------------------#
#' #Generate simulated data
#' Data = gen.data(n, p, k, rho, family = "binomial", cortype = cortype, SNR = SNR, seed = seed)
#'
#' x = Data$x[1:140, ]
#' y = Data$y[1:140]
#' x_new = Data$x[141:200, ]
#' y_new = Data$y[141:200]
#' logi.pdas = bess(x, y, family = "binomial", method = "sequential", tune = "cv")
#' lambda.list = exp(seq(log(5), log(0.1), length.out = 10))
#' logi.l0l2 = bess(x, y, type = "bsrr", tune="cv",
#'                  family = "binomial", lambda.list = lambda.list, method = "sequential")
#' pred.pdas = predict(logi.pdas, newx = x_new)
#' pred.l0l2 = predict(logi.l0l2, newx = x_new)
#'
#' #-------------------coxph model----------------------#
#' #Generate simulated data
#' Data = gen.data(n, p, k, rho, family = "cox", scal = 10)
#'
#' x = Data$x[1:140, ]
#' y = Data$y[1:140, ]
#' x_new = Data$x[141:200, ]
#' y_new = Data$y[141:200, ]
#' cox.pdas = bess(x, y, family = "cox", method = "sequential")
#' lambda.list = exp(seq(log(5), log(0.1), length.out = 10))
#' cox.l0l2 = bess(x, y, type = "bsrr", family = "cox", lambda.list = lambda.list)
#' pred.pdas = predict(cox.pdas, newx = x_new)
#' pred.l0l2 = predict(cox.l0l2, newx = x_new)
#'
#'
#'
predict.bess <- function(object, newx, type = c("link", "response"), ...)
{
  # if(!is.null(object$factor)){
  #   factor <- c(object$factor)
  #   if(!is.data.frame(newx)) newx <- as.data.frame(newx)
  #   newx[,factor] <- apply(newx[,factor,drop=FALSE], 2, function(x){
  #     return(as.factor(x))
  #   })
  #   newx <- model.matrix(~., data = newx)[,-1]
  # }
  if(missing(newx)){
    newx = object$x
  }
  if(is.null(colnames(newx))) {
    newx <- as.matrix(newx)
  }else{
    vn <- names(object$beta)
    if(any(is.na(match(vn, colnames(newx))))) stop("names of newx don't match training data!")
    newx <- as.matrix(newx[,vn])
  }
  type <- match.arg(type)
  if(object$family=="gaussian")
  {
    betas <- object$beta
    coef0 <- object$coef0

    y <- drop(newx %*% betas)+coef0
    return(y)
  }
  if(object$family == "binomial")
  {
    betas <- object$beta
    coef <- object$coef0
    # class = as.numeric(exp(newx%*%betas+coef)/(1+exp(newx%*%betas+coef))>0.5)
    # class[is.na(class)] = 1
    # if(!is.null(object$y_names))
    # {
    #   class[which(class == 0,arr.ind = T)] = object$y_names[1]
    #   class[which(class == 1,arr.ind = T)] = object$y_names[2]
    # }
    #if(sum(is.infinite(exp(newx%*%betas+coef))) > 0) print(sum(is.infinite(exp(newx%*%betas+coef))))
    if(type  == "link"){
        link <- newx%*%betas+coef
        return(drop(link))
    } else{
        prob <- ifelse(is.infinite(exp(newx%*%betas+coef)), 1, exp(newx%*%betas+coef)/(1+exp(newx%*%betas+coef)))
        return(drop(prob))
    }

  }
  if(object$family == "poisson"){
    betas <- object$beta

    eta <- newx%*%betas
    if(type == "link"){
      return(eta)
    }else{
      expeta <- exp(eta)
      return(drop(expeta))
    }

  }
  if(object$family == "cox")
  {
    betas <- object$beta

    eta <- newx%*%betas;
    if(type == "link"){
      return(eta)
    } else{
      expeta <- exp(eta)
      return(drop(expeta))
    }

  }

}

