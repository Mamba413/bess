#' make predictions from a "bess" object.
#'
#' Returns predictions from a fitted
#' "\code{bess}" object.
#'
#' @param object Output from the \code{bess} function.
#' @param newx New data used for prediction. If omitted, the fitted linear predictors are used.
#' @param type \code{type = "link"} gives the linear predictors for \code{"binomial"},
#' \code{"poisson"} or \code{"cox"} models; for \code{"gaussian"} models it gives the
#' fitted values. \code{type = "response"} gives the fitted probabilities for
#' \code{"binomial"}, fitted mean for \code{"poisson"} and the fitted relative-risk for
#' \code{"cox"}; for \code{"gaussian"}, \code{type = "response"} is equivalent to \code{type = "link"}
#' @param \dots Additional arguments affecting the predictions produced.
#' @return The object returned depends on the types of family.
#' @author Canhong Wen, Aijun Zhang, Shijie Quan, Liyuan Hu, Kangkang Jiang, Yanhang Zhang, Jin Zhu and Xueqin Wang.
#' @seealso \code{\link{bess}}.
#' @references Wen, C., Zhang, A., Quan, S. and Wang, X. (2020). BeSS: An R
#' Package for Best Subset Selection in Linear, Logistic and Cox Proportional
#' Hazards Models, \emph{Journal of Statistical Software}, Vol. 94(4).
#' doi:10.18637/jss.v094.i04.
#' @examples
#'
#' #-------------------linear model----------------------#
#' # Generate simulated data
#' n <- 200
#' p <- 20
#' k <- 5
#' rho <- 0.4
#' seed <- 10
#' Tbeta <- rep(0, p)
#' Tbeta[1:k*floor(p/k):floor(p/k)] <- rep(1, k)
#' Data <- gen.data(n, p, k, rho, family = "gaussian", beta = Tbeta, seed = seed)
#' x <- Data$x[1:140, ]
#' y <- Data$y[1:140]
#' x_new <- Data$x[141:200, ]
#' y_new <- Data$y[141:200]
#' lm.bss <- bess(x, y, method = "sequential")
#' lambda.list <- exp(seq(log(5), log(0.1), length.out = 10))
#' lm.bsrr <- bess(x, y, type = "bsrr", method = "pgsection")
#'
#' pred.bss <- predict(lm.bss, newx = x_new)
#' pred.bsrr <- predict(lm.bsrr, newx = x_new)
#'
#' #-------------------logistic model----------------------#
#' #Generate simulated data
#' Data <- gen.data(n, p, k, rho, family = "binomial", beta = Tbeta, seed = seed)
#'
#' x <- Data$x[1:140, ]
#' y <- Data$y[1:140]
#' x_new <- Data$x[141:200, ]
#' y_new <- Data$y[141:200]
#' logi.bss <- bess(x, y, family = "binomial", method = "sequential", tune = "cv")
#' lambda.list <- exp(seq(log(5), log(0.1), length.out = 10))
#' logi.bsrr <- bess(x, y, type = "bsrr", tune="cv",
#'                  family = "binomial", lambda.list = lambda.list, method = "sequential")
#'
#' pred.bss <- predict(logi.bss, newx = x_new)
#' pred.bsrr <- predict(logi.bsrr, newx = x_new)
#'
#' #-------------------coxph model----------------------#
#' #Generate simulated data
#' Data <- gen.data(n, p, k, rho, family = "cox", beta = Tbeta, scal = 10)
#'
#' x <- Data$x[1:140, ]
#' y <- Data$y[1:140, ]
#' x_new <- Data$x[141:200, ]
#' y_new <- Data$y[141:200, ]
#' cox.bss <- bess(x, y, family = "cox", method = "sequential")
#' lambda.list <- exp(seq(log(5), log(0.1), length.out = 10))
#' cox.bsrr <- bess(x, y, type = "bsrr", family = "cox", lambda.list = lambda.list)
#'
#' pred.bss <- predict(cox.bss, newx = x_new)
#' pred.bsrr <- predict(cox.bsrr, newx = x_new)
#'
#'#-------------------group selection----------------------#
#'beta <- rep(c(rep(1,2),rep(0,3)), 4)
#'Data <- gen.data(200, 20, 5, rho=0.4, beta = beta, seed =10)
#'x <- Data$x
#'y <- Data$y
#'
#'group.index <- c(rep(1, 2), rep(2, 3), rep(3, 2), rep(4, 3),
#'                 rep(5, 2), rep(6, 3), rep(7, 2), rep(8, 3))
#'lm.group <- bess(x, y, s.min=1, s.max = 8, type = "bss", group.index = group.index)
#'lm.groupbsrr <- bess(x, y, type = "bsrr", s.min = 1, s.max = 8, group.index = group.index)
#'
#'pred.group <- predict(lm.group, newx = x_new)
#'pred.groupbsrr <- predict(lm.groupbsrr, newx = x_new)
#'
#'@method predict bess
#'@export
#'@export predict.bess
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

