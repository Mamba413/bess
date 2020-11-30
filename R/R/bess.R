#' Best subset selection
#'
#' Best subset selection for generalized linear model and Cox's proportional
#' model.
#'
#' The best subset selection problem with model size \eqn{s} is
#' \deqn{\min_\beta -2 logL(\beta) \;\;{\rm s.t.}\;\; \|\beta\|_0 \leq s.} In
#' the GLM case, \eqn{logL(\beta)} is the log-likelihood function; In the Cox
#' model, \eqn{logL(\beta)} is the log partial likelihood function.
#'
#' The best ridge regression problem with model size \eqn{s} is
#' \deqn{\min_\beta -2 logL(\beta) + \lambda\Vert\beta\Vert_2^2 \;\;{\rm
#' s.t.}\;\; \|\beta\|_0 \leq s.} In the GLM case, \eqn{logL(\beta)} is the
#' log-likelihood function; In the Cox model, \eqn{logL(\beta)} is the log
#' partial likelihood function.
#'
#' For each candidate model size and \eqn{\lambda}, the best subset selection
#' problem is solved by the primal-dual active set (PDAS) algorithm, see Wen et
#' al(2017) for details. And the best ridge regression is solved by the
#' \eqn{L_2} penalized primal-dual active set algorithm. These algorithms
#' utilize an active set updating strategy via primal and dual variables and
#' fits the sub-model by exploiting the fact that their support sets are
#' non-overlap and complementary. For the case of method = "\code{sequential}"
#' if \code{is_warms_start} = "TRUE", we run the PDAS algorithm for a list of
#' sequential model sizes and use the estimate from the last iteration as a
#' warm start. For the case of method = "\code{gsection}" of the best subset
#' selection problem, a golden section search technique is adopted to
#' efficiently determine the optimal model size. And for the case of method =
#' "\code{powell}" of the best ridge regression problem, the powell method is
#' used for parameters determination, the line search method of which is
#' sepecified by \code{line_search} = "sequential" or "gsection".
#'
#' @param x Input matrix, of dimension n x p; each row is an observation
#' vector.
#' @param y The response variable, of length n. For family="binomial" should be
#' a factor with two levels. For family= "cox", y should be a two-column matrix
#' with columns named 'time' and 'status'.
#' @param type One of the two types of problems. Options are "bss" and "bsrr".
#' @param family One of the GLM or Cox models. Either "gaussian", "binomial",
#' "poisson", or "cox", depending on the response.
#' @param method Methods to be used to select the optimal model size. For
#' \code{method} = "\code{sequential}", we solve the best subset selection
#' problem for each \eqn{s} in \eqn{1,2,\dots,s_{max}}. For \code{method} =
#' "\code{gsection}", which is only valid for method "PDAS" and "GPDAS", we
#' solve the best subset selection problem with a range non-continuous model
#' sizes. For \code{method} = "\code{pgsection}" and \code{psequential}, the powell method is used to
#' solve the "L0L2" problem.
#' @param tune The criterion for choosing the sparsity or L_2 penalty
#' parameters. Available options are "gic", "ebic", "bic", "aic" and "cv".
#' Default is "GIC".
#' @param s.list An increasing list of sequential value representing the model
#' sizes. Only used for method = "sequential".Default is (1,\eqn{\min{p,
#' n/\log(n)}}).
#' @param lambda.list A lambda sequence for "\code{L0L2}". Default is
#' \code{exp(seq(log(100), log(0.01), length.out = 100))}.
#' @param s.min The minimum value of model sizes. Only used for method =
#' "\code{gsection}". Default is 1.
#' @param s.max The maximum value of model sizes. Only used for method =
#' "\code{gsection}". Default is \eqn{\min{p, n/\log(n)}}.
#' @param lambda.min The minimum value of lambda. Only used for method =
#' "\code{powell}". Default is 0.01.
#' @param lambda.max The maximum value of lambda. Only used for method =
#' "\code{powell}". Default is 100.
#' @param nlambda The number of lambdas for the powell path.
#' @param screening.num The number of screening variables.
#' @param normalize Options for data mean-substraction or normalization. "0" for
#' no data process.  Entering "1" will
#' only substract the mean of columns of X.
#' 2 for scaling the columns of x to have \eqn{\sqrt n} norm.
#' 3 for centralizing the columns of X and y, and also
#' normalizing the colnums of X to have \eqn{\sqrt n} norm. Entering other number will normalize the
#' columns of x. If \code{NULL}, \code{normalize} will be "1" for "gaussian",
#' "2" for "binomial", "3" for "cox".
#' @param weight Observation weights. Default is 1 for each observation.
#' @param max.iter The maximum number of iterations in the bess function. In
#' linear regression, only a few steps can guarantee the convergence. Default
#' is 20.
#' @param warm.start Whether to use the last solution as a warm start. Default
#' is \code{TRUE}.
#' @param nfolds The number of folds in cross-validation. Default is 5.
#' @param group.index A vector indicating the group index for each variable.
#' @return A list with class attribute 'bess' and named components:
#' \item{beta}{The best fitting coefficients.} \item{coef0}{The best fitting
#' intercept.} \item{train_loss}{The training loss of the best fitting model.}
#' \item{ic}{The information criterion of the best fitting model when model
#' selection is based on a certain information criterion.} \item{cvm}{The mean
#' cross-validated error for the best fitting model when model selection is
#' based on the cross-validation.}
#'
#' \item{lambda}{the lambda chosen for the best fitting model}
#' \item{beta_all}{A list of the best fitting coefficients of size
#' \eqn{s=0,1,\dots,p} and \eqn{\lambda} in \code{lambda.list} with the
#' smallest loss function. For example, the fitting coefficients of the
#' \eqn{i^{th} \lambda} and the \eqn{j^{th}} \code{s} is at the \eqn{i^{th}}
#' list component's \eqn{j^{th}} column.} \item{coef0_all}{The best fitting
#' intercepts of size \eqn{s=0,1,\dots,p} and \eqn{\lambda} in
#' \code{lambda.list} with the smallest loss function.} \item{train_loss_all}{A
#' list of the training loss the best fitting intercepts of model size
#' \eqn{s=0,1,\dots,p} and \eqn{\lambda} in \code{lambda.list}. For example,
#' the training loss of the \eqn{i^{th} \lambda} and the \eqn{j^{th}} \code{s}
#' is at the \eqn{i^{th}} list component's \eqn{j^{th}} entry.} \item{ic_all}{A
#' matrix of the mean cross-validated error of model size \eqn{s=0,1,\dots,p}
#' and \eqn{\lambda} in \code{lambda.list} with the smallest loss function. For
#' example, the training loss of the \eqn{i^{th} \lambda} and the \eqn{j^{th}}
#' \code{s} is at the \eqn{i^{th}} row \eqn{j^{th}} column. Only available when
#' model selection is based on a certain information criterion.}
#'
#' \item{cvm_all}{A matrix of the information criteria of model size
#' \eqn{s=0,1,\dots,p} and \eqn{\lambda} in \code{lambda.list} with the
#' smallest loss function. For example, the training loss of the \eqn{i^{th}
#' \lambda} and the \eqn{j^{th}} \code{s} is at the \eqn{i^{th}} row
#' \eqn{j^{th}} column. Only available when model selection is based on the
#' cross-validation.} \item{lambda_all}{The lambda chosen for each step.}
#' \item{family}{Types of the model: "\code{gaussian}" for linear
#' model,"\code{binomial}" for logistic model,"\code{poisson}" for poisson
#' model, and "\code{cox}" for Cox model.} \item{factor}{Which variable to be
#' factored. Should be NULL or a numeric vector.} \item{s.list}{The input
#' \code{s.list}.} \item{nsample}{The sample size.} \item{tyoe}{One of the
#' three algorithm types, "PDAS", "GPDAS", and "L0L2".} \item{method}{One of
#' the three methods, "sequential", "gsection", and "powell".}
#' \item{ic_type}{The criterion of model selection. Either "cv", "AIC", "BIC",
#' "GBIC", or "EBIC".}
#' @author Liyuan Hu, Aijun Zhang, Shijie Quan, and Xueqin Wang.
#' @seealso \code{\link{plot.bess}}, \code{\link{summary.bess}},
#' \code{\link{coef.bess}}, \code{\link{predict.bess}}.
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
#' lm.l0l2 = bess(x, y, type = "bsrr", method = "pgsection")
#' coef(lm.pdas)
#' coef(lm.l0l2)
#' print(lm.pdas)
#' print(lm.l0l2)
#' pred.pdas = predict(lm.pdas, newx = x_new)
#' pred.l0l2 = predict(lm.l0l2, newx = x_new)
#'
#' # Plot the solution path and the loss function of PDAS
#' plot(lm.pdas, type = "both", breaks = TRUE)
#' plot(lm.l0l2)
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
#' coef(logi.pdas)
#' coef(logi.l0l2)
#' print(logi.pdas)
#' print(logi.l0l2)
#' pred.pdas = predict(logi.pdas, newx = x_new)
#' pred.l0l2 = predict(logi.l0l2, newx = x_new)
#'
#' # Plot the solution path and the loss function of PDAS
#' plot(logi.pdas, type = "both", breaks = TRUE)
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
#' coef(cox.pdas)
#' coef(cox.l0l2)
#' print(cox.pdas)
#' print(cox.l0l2)
#' pred.pdas = predict(cox.pdas, newx = x_new)
#' pred.l0l2 = predict(cox.l0l2, newx = x_new)
#'
#' # Plot the solution path and the loss function of PDAS
#' plot(cox.pdas, type = "both", breaks = TRUE)
#' plot(cox.l0l2)
#' plot(cox.l0l2)
#'
#'
bess <- function(x, y, family = c("gaussian", "binomial", "poisson", "cox"), type = c("bss", "bsrr"),
                 method = c("gsection", "sequential", "pgsection", "psequential"),
                 tune = c("gic", "ebic", "bic", "aic", "cv"),
                 s.list, lambda.list = 0,
                 s.min, s.max,
                 lambda.min = 0.001, lambda.max = 100, nlambda = 100,
                 screening.num = NULL,
                 normalize = NULL, weight = NULL,
                 max.iter = 20, warm.start = TRUE,
                 nfolds = 5,
                 group.index =NULL){

  if(missing(s.list)) s.list <- 1:min(ncol(x),round(nrow(x)/log(nrow(x))))
  if(missing(s.min)) s.min <- 1
  if(missing(s.max)) s.max <- min(ncol(x),round(nrow(x)/log(nrow(x))))

  tune <- match.arg(tune)
  ic_type <- switch(tune,
                   "aic" = 1,
                   "bic" = 2,
                   "gic" = 3,
                   "ebic" = 4,
                   "cv" = 1)
  is_cv <- ifelse(tune == "cv", TRUE, FALSE)
  type <- match.arg(type)
  if(!is.null(group.index)){
    g_index <- group.index - 1
    algorithm_type = switch(type,
                            "bss" = "GPDAS",
                            "bsrr" = "GL0L2")
  } else{
    algorithm_type = switch(type,
                            "bss" = "PDAS",
                            "bsrr" = "L0L2")
  }
  family <- match.arg(family)
  model_type <- switch(family,
                      "gaussian" = 1,
                      "binomial" = 2,
                      "poisson" = 3,
                      "cox" = 4)
  # type <- match.arg(type)
  # algorithm_type <- switch(type,
  #                          "L0" = "PDAS",
  #                          "L0L2" = "L0L2",
  #                          "groupL0" = "GPDAS",
  #                          "groupL0L2" = "GL0L2")
  # if((algorithm_type == "GPDAS" | algorithm_type == "GL0L2") & is.null(g_index)) stop(" \"g_index\" is needed for \"GPDAS\".")
  method <- match.arg(method)
  # if(algorithm_type == "L0L2" & method == "gsection") method <- "powell"
  # path_type <- switch(method,
  #                    "sequential" = 1,
  #                    "gsection" = 2,
  #                    "pgsection" = 2,
  #                    "psequential" = 2)
  if(method == "pgsection"){
    path_type = 2
    line.search = 1
  } else if(method == "psequential") {
    path_type = 2
    line.search = 2
  } else if(method == "sequential"){
    path_type = 1
    line.search = 1
  } else{
    path_type = 2
    line.search=1
  }

  # line.search <- match.arg(line.search)
  # line.search <- switch(line.search,
  #                      'gsection' = 1,
  #                      "sequential" = 2)
  if(path_type == 1 & s.list[length(s.list)] > ncol(x)) stop("The maximum one in s.list is too large!")
  if(path_type == 2 & s.max > ncol(x)) stop("s.max is too large")
  if(ncol(x)==1|is.vector(x)) stop("x should be two columns at least!")

  if(family=="binomial")
  {
    if(is.factor(y)){
      y <- as.character(y)
    }
    if(length(unique(y)) != 2)  stop("Please input binary variable!") else
      if(setequal(y_names <- unique(y), c(0,1)) == FALSE)
      {
        y[which(y==unique(y)[1])] = 0
        y[which(y==unique(y)[2])] = 1
        y<-as.numeric(y)
      }
  }
  if(family=="cox")
  {
    if(!is.matrix(y)) y <- as.matrix(y)
    if(ncol(y) != 2) stop("Please input y with two columns!")
  }
  if(is.vector(y))
  {
    if(nrow(x) != length(y)) stop("Rows of x must be the same as length of y!")
  }else{
    if(nrow(x) != nrow(y)) stop("Rows of x must be the same as rows of y!")
  }
  if(is.null(normalize)){
    is_normal = TRUE
    normalize <- switch(family,
                        "gaussian" = 1,
                        "binomial" = 2,
                        "poisson" = 1,
                        "cox" = 3)
  } else if(normalize !=0){
    normalize = as.character(normalize)
    normalize = switch (normalize,
      '1' = 2,
      '2' = 3,
      '3' = 1
    )
    is_normal = TRUE
  } else{
    is_normal = FALSE
    normalize = 0
  }
  # if(!is.null(factor)){
  #   if(is.null(colnames(x))) colnames(x) <- paste0("X",1:ncol(x),"g")
  # }else{
  #   if(is.null(colnames(x))) colnames(x) <- paste0("X",1:ncol(x))
  # }
  if(!is.matrix(x)) x <- as.matrix(x)
  # if(!is.null(factor)){
  #   if(!is.data.frame(x)) x <- as.data.frame(x)
  #   x[,factor] <- apply(x[,factor,drop=FALSE], 2, function(x){
  #     x <- as.factor(x)
  #   })
  #   group <- rep(1, ncol(x))
  #   names(group) <- colnames(x)
  #   group[factor] <- apply(x[,factor,drop=FALSE], 2, function(x) {length(unique(x))})-1
  #   Gi <- rep(1:ncol(x), times = group)
  #   beta0 <- rep(beta0, times = group)
  #   x <- model.matrix(~., data = x)[,-1]
  # }
  vn <- colnames(x)
  if(is.null(weight)) weight <- rep(1, nrow(x))
  if(is.null(screening.num)){
    screening <- FALSE
    screening.num <- ncol(x)
  } else{
    screening <- TRUE
    if(screening.num > ncol(x)) stop("The number of screening features must be less than that of the column of x!")
    if(path_type == 1){
      if(screening.num < s.list[length(s.list)]) stop("The number of screening features must be less than the maximum one in s.list!")
    } else{
      if(screening.num < s.max) stop("The number of screening features must be less than the s.max!")
    }
  }
  # if(is.null(screening.num)){
  #   screening.num <- switch(path_type,
  #                          "1" = pmax(nrow(x)/log(nrow(x)), s.list[length(s.list)]),
  #                          "2" = pmax(nrow(x)/log(nrow(x)), s.max))
  # } else{
  #   if(screening.num > ncol(x)) stop("The number of screening features must be less than that of the column of x!")
  #   if(path_type == 1){
  #     if(screening.num < s.list[length(s.list)]) stop("The number of screening features must be less than the maximum one in s.list!")
  #   } else{
  #     if(screening.num < s.max) stop("The number of screening features must be less than the s.max!")
  #   }
  # }
  if(algorithm_type == "PDAS"){
    if(model_type == 4){
      sort_y <- order(y[, 1])
      y <- y[sort_y, ]
      x <- x[sort_y, ]
      y <- y[, 2]
    }
    res.pdas <- bessCpp(x, y, data_type = normalize, weight, is_normal = is_normal, algorithm_type = 1, model_type =  model_type,
                       max_iter = max.iter, exchange_num = 2, path_type = path_type, is_warm_start = warm.start,
                       ic_type = ic_type, is_cv = is_cv, K = nfolds, state = rep(2,10), sequence = s.list, lambda_seq = 0,
                       s_min = s.min, s_max = s.max, K_max = 10, epsilon = 10,
                       lambda_max = 0, lambda_min = 0 , nlambda = nlambda, is_screening = screening, screening_size = screening.num,
                       powell_path = 1, g_index=(1:ncol(x)-1))
    beta.pdas <- res.pdas$beta
    names(beta.pdas) <- vn
    res.pdas$beta <- beta.pdas
    if(is_cv == TRUE) {
      names(res.pdas)[which(names(res.pdas) == "ic")] <- "cvm"
      names(res.pdas)[which(names(res.pdas) == "ic_all")] <- "cvm_all"
    }
    res.pdas$x <- x
    res.pdas$y <- y
    res.pdas$family <- family
    res.pdas$factor <- factor
    res.pdas$s.list <- s.list
    res.pdas$nsample <- nrow(x)
    res.pdas$algorithm_type <- "PDAS"
    res.pdas$method <- method
    res.pdas$ic_type <- ifelse(is_cv == TRUE, "cv", c("AIC", "BIC", "GIC", "EBIC")[ic_type])
    res.pdas$call <- match.call()
    class(res.pdas) <- 'bess'
    return(res.pdas)
  }
  if(algorithm_type == "GPDAS"){
    if(model_type == 4){
      sort_y <- order(y[, 1])
      y <- y[sort_y, ]
      x <- x[sort_y, ]
      y <- y[, 2]
    }
    res.gpdas <- bessCpp(x, y, data_type = normalize, weight, is_normal, algorithm_type = 2, model_type =  model_type,
                       max_iter = max.iter, exchange_num = 2, path_type = path_type, is_warm_start = warm.start,
                       ic_type = ic_type, is_cv = is_cv, K = nfolds, state = rep(2,10), sequence = s.list, lambda_seq = 0,
                       s_min = s.min, s_max = s.max, K_max = 10, epsilon = 10,
                       lambda_max = 0, lambda_min = 0 , nlambda = nlambda, is_screening = screening, screening_size = screening.num, powell_path = 1, g_index = g_index)
    beta.gpdas <- res.gpdas$beta
    names(beta.gpdas) <- vn
    res.gpdas$beta <- beta.gpdas
    if(is_cv == TRUE) {
      names(res.gpdas)[which(names(res.gpdas) == "ic")] <- "cvm"
      names(res.gpdas)[which(names(res.gpdas) == "ic_all")] <- "cvm_all"
    }
    res.gpdas$x <- x
    res.gpdas$y <- y
    res.gpdas$family <- family
    res.gpdas$factor <- factor
    res.gpdas$s.list <- s.list
    res.gpdas$nsample <- nrow(x)
    res.gpdas$algorithm_type <- "GPDAS"
    res.gpdas$method <- method
    res.gpdas$ic_type <- ifelse(is_cv == TRUE, "cv", c("AIC", "BIC", "GIC", "EBIC")[ic_type])
    res.gpdas$call <- match.call()
    class(res.gpdas) <- "bess"
    return(res.gpdas)
  }
  if(algorithm_type == "GL0L2"){
    if(model_type == 4){
      sort_y <- order(y[, 1])
      y <- y[sort_y, ]
      x <- x[sort_y, ]
      y <- y[, 2]
    }
    res.gl0l2 <- bessCpp(x, y, data_type = normalize, weight, is_normal, algorithm_type = 3, model_type =  model_type,
                         max_iter = max.iter, exchange_num = 2, path_type = path_type, is_warm_start = warm.start,
                         ic_type = ic_type, is_cv = is_cv, K = nfolds, state = rep(2,10), sequence = s.list, lambda_seq = lambda.list,
                         s_min = s.min, s_max = s.max, K_max = 10, epsilon = 10,
                         lambda_max = lambda.max, lambda_min = lambda.min, nlambda = nlambda, is_screening = screening, screening_size = screening.num, powell_path = 1, g_index = g_index)
    beta.gl0l2 <- res.gl0l2$beta
    names(beta.gl0l2) <- vn
    res.gl0l2$beta <- beta.gl0l2
    if(is_cv == TRUE) {
      names(res.gl0l2)[which(names(res.gl0l2) == "ic")] <- "cvm"
      names(res.gl0l2)[which(names(res.gl0l2) == "ic_all")] <- "cvm_all"
    }
    res.gl0l2$x <- x
    res.gl0l2$y <- y
    res.gl0l2$family <- family
    res.gl0l2$factor <- factor
    res.gl0l2$s.list <- s.list
    res.gl0l2$nsample <- nrow(x)
    res.gl0l2$algorithm_type <- "GL0L2"
    res.gl0l2$method <- method
    res.gl0l2$ic_type <- ifelse(is_cv == TRUE, "cv", c("AIC", "BIC", "GIC", "EBIC")[ic_type])
    res.gl0l2$call <- match.call()
    class(res.gl0l2) <- "bess"
    return(res.gl0l2)
  }
  # L0L2
  if(algorithm_type == "L0L2"){
    if(model_type == 4){
      sort_y <- order(y[, 1])
      y <- y[sort_y, ]
      x <- x[sort_y, ]
      y <- y[, 2]
    }

    if(path_type == 1 & lambda.list[1] == 0 & length(lambda.list) == 1) lambda.list <- exp(seq(log(100), log(0.01), length.out = 100))
    res.l0l2 <- bessCpp(x, y, data_type = normalize, weight, is_normal, algorithm_type = 5, model_type =  model_type,
                       max_iter = max.iter, exchange_num = 2, path_type = path_type, is_warm_start = warm.start,
                       ic_type = ic_type, is_cv = is_cv, K = nfolds, state = rep(2,10), sequence = s.list, lambda_seq = lambda.list,
                       s_min = s.min, s_max = s.max, K_max = 10, epsilon = 10,
                       lambda_max = lambda.max, lambda_min = lambda.min, nlambda = nlambda, is_screening = screening, screening_size = screening.num,
                       powell_path = line.search, g_index = (1:ncol(x) -1))
    beta.l0l2 <- res.l0l2$beta
   # length(which.max(beta.l0l2!=0))
    names(beta.l0l2) <- vn
    res.l0l2$beta <- beta.l0l2
    if(is_cv == TRUE) {
      names(res.l0l2)[which(names(res.l0l2) == "ic")] <- "cvm"
      names(res.l0l2)[which(names(res.l0l2) == "ic_all")] <- "cvm_all"
    }
    res.l0l2$x <- x
    res.l0l2$y <- y
    res.l0l2$family <- family
    res.l0l2$factor <- factor
    res.l0l2$s.list <- s.list
    res.l0l2$lambda.list <- lambda.list
    res.l0l2$nsample <- nrow(x)
    res.l0l2$algorithm_type <- "L0L2"
    res.l0l2$method <- method
    res.l0l2$line.search <- ifelse(line.search == 1, "gsection", "sequential")
    res.l0l2$ic_type <- ifelse(is_cv == TRUE, "cv", c("AIC", "BIC", "GIC", "EBIC")[ic_type])
    res.l0l2$lambda.max <- lambda.max
    res.l0l2$lambda.min <- lambda.min
    res.l0l2$s.max <- s.max
    res.l0l2$s.min <- s.min
    res.l0l2$nlambda <- nlambda

    res.l0l2$call <- match.call()
    class(res.l0l2) <- "bess"
    return(res.l0l2)
  }
}


