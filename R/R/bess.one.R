#' Best subset selection/Best subset ridge regression with a
#' specified model size and a shrinkage parameter
#'
#' Best subset selection with a specified model size for generalized
#' linear models and Cox's proportional hazard model.
#'
#'  Given a model size \eqn{s}, we consider the following best subset selection problem:
#'\deqn{\min_\beta -2 \log L(\beta) ;{ s.t.} \|\beta\|_0 = s.}
#'And given a model size \eqn{s} and a shrinkage parameter \eqn{\lambda}
#', consider the following best subset ridge regression problem:
#'\deqn{\min_\beta -2 \log L(\beta) + \lambda \Vert\beta \Vert_2^2; { s.t.} \|\beta\|_0 = s.}
#'
#'In the GLM case, \eqn{\log L(\beta)} is the log likelihood function;
#' In the Cox model, \eqn{\log L(\beta)} is the log partial likelihood function.
#'
#'The best subset selection problem is solved by the primal dual active set algorithm,
#'see Wen et al. (2017) for details. This algorithm utilizes an active set updating strategy
#'via primal and dual variables and fits the sub-model by exploiting the fact that their
#' support set are non-overlap and complementary.
#'
#' @param x Input matrix, of dimension \eqn{n \times p}; each row is an observation
#' vector and each column is a predictor/feature/variable.
#' @param y The response variable, of \code{n} observations. For \code{family = "binomial"} should be
#' a factor with two levels. For \code{family="poisson"}, \code{y} should be a vector with positive integer.
#'  For \code{family = "cox"}, \code{y} should be a two-column matrix
#' with columns named \code{time} and \code{status}.
#' @param type One of the two types of problems.
#' \code{type = "bss"} for the best subset selection,
#' and \code{type = "bsrr"} for the best subset ridge regression.
#' @param family One of the following models: \code{"gaussian"}, \code{"binomial"},
#' \code{"poisson"}, or \code{"cox"}. Depending on the response.
#' @param s A specified model size
#' @param lambda A shrinkage parameter for \code{"bsrr"}.
#' @param always.include A vector containing the index of variables that should always be included in the model.
#' @param screening.num Users can pre-exclude some irrelevant variables according to maximum marginal likelihood estimators before fitting a
#' model by passing an integer to \code{screening.num} and the sure independence screening will choose a set of variables of this size.
#' Then the active set updates are restricted on this subset.
#' @param normalize Options for normalization. \code{normalize = 0} for
#' no normalization. Setting \code{normalize = 1} will
#' only subtract the mean of columns of \code{x}.
#' \code{normalize = 2} for scaling the columns of \code{x} to have \eqn{\sqrt n} norm.
#' \code{normalize = 3} for subtracting the means of the columns of \code{x} and \code{y}, and also
#' normalizing the columns of \code{x} to have \eqn{\sqrt n} norm.
#' If \code{normalize = NULL}, by default, \code{normalize} will be set \code{1} for \code{"gaussian"},
#' \code{2} for \code{"binomial"} and \code{"poisson"}, \code{3} for \code{"cox"}.
#' @param weight Observation weights. Default is \code{1} for each observation.
#' @param max.iter The maximum number of iterations in the bess function.
#' In most of the case, only a few steps can guarantee the convergence. Default
#' is \code{20}.
#' @param group.index A vector of integers indicating the which group each variable is in.
#' For variables in the same group, they should be located in adjacent columns of \code{x}
#' and their corresponding index in \code{group.index} should be the same.
#' Denote the first group as \code{1}, the second \code{2}, etc.
#' If you do not fit a model with a group structure,
#' please set \code{group.index = NULL}. Default is \code{NULL}.
#' @return A list with class attribute 'bess' and named components:
#' \item{beta}{The best fitting coefficients.} \item{coef0}{The best fitting
#' intercept.}
#' \item{loss}{The training loss of the fitting model.}
#' \item{s}{The model size.}
#' \item{lambda}{The shrinkage parameter.}
#' \item{family}{Type of the model.}
#' \item{nsample}{The sample size.}
#' \item{type}{Either \code{"bss"} or \code{"bsrr"}.}
#' @author Canhong Wen, Aijun Zhang, Shijie Quan, Liyuan Hu, Kangkang Jiang, Yanhang Zhang, Jin Zhu and Xueqin Wang.
#' @seealso \code{\link{bess}}, \code{\link{summary.bess}}
#' \code{\link{coef.bess}}, \code{\link{predict.bess}}.
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
#' lm.bss <- bess.one(x, y, s = 5)
#' lm.bsrr <- bess.one(x, y, type = "bsrr", s = 5, lambda = 0.01)
#' coef(lm.bss)
#' coef(lm.bsrr)
#' print(lm.bss)
#' print(lm.bsrr)
#' summary(lm.bss)
#' summary(lm.bsrr)
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
#' logi.bss <- bess.one(x, y, family = "binomial", s = 5)
#' logi.bsrr <- bess.one(x, y, type = "bsrr", family = "binomial", s = 5, lambda = 0.01)
#' coef(logi.bss)
#' coef(logi.bsrr)
#' print(logi.bss)
#' print(logi.bsrr)
#' summary(logi.bss)
#' summary(logi.bsrr)
#' pred.bss <- predict(logi.bss, newx = x_new)
#' pred.bsrr <- predict(logi.bsrr, newx = x_new)
#'
#'#-------------------poisson model----------------------#
#' Data <- gen.data(n, p, k, rho=0.3, family = "poisson", beta = Tbeta, seed = seed)
#'
#' x <- Data$x[1:140, ]
#' y <- Data$y[1:140]
#' x_new <- Data$x[141:200, ]
#' y_new <- Data$y[141:200]
#' poi.bss <- bess.one(x, y, family = "poisson", s=5)
#' lambda.list <- exp(seq(log(5), log(0.1), length.out = 10))
#' poi.bsrr <- bess.one(x, y, type = "bsrr", family = "poisson", s = 5, lambda = 0.01)
#' coef(poi.bss)
#' coef(poi.bsrr)
#' print(poi.bss)
#' print(poi.bsrr)
#' summary(poi.bss)
#' summary(poi.bsrr)
#' pred.bss <- predict(poi.bss, newx = x_new)
#' pred.bsrr <- predict(poi.bsrr, newx = x_new)
#'
#' #-------------------coxph model----------------------#
#' #Generate simulated data
#' Data <- gen.data(n, p, k, rho, family = "cox", beta = Tbeta, scal = 10)
#'
#' x <- Data$x[1:140, ]
#' y <- Data$y[1:140, ]
#' x_new <- Data$x[141:200, ]
#' y_new <- Data$y[141:200, ]
#' cox.bss <- bess.one(x, y, family = "cox", s = 5)
#' cox.bsrr <- bess.one(x, y, type = "bsrr", family = "cox", s = 5, lambda = 0.01)
#' coef(cox.bss)
#' coef(cox.bsrr)
#' print(cox.bss)
#' print(cox.bsrr)
#' summary(cox.bss)
#' summary(cox.bsrr)
#' pred.bss <- predict(cox.bss, newx = x_new)
#' pred.bsrr <- predict(cox.bsrr, newx = x_new)
#'#----------------------High dimensional linear models--------------------#
#'\dontrun{
#' data <- gen.data(n, p = 1000, k, family = "gaussian", seed = seed)
#'
#'# Best subset selection with SIS screening
#' fit <- bess.one(data$x, data$y, screening.num = 100, s = 5)
#'}
#'
#'#-------------------group selection----------------------#
#'beta <- rep(c(rep(1,2),rep(0,3)), 4)
#'Data <- gen.data(200, 20, 5, rho=0.4, beta = beta, seed =10)
#'x <- Data$x
#'y <- Data$y
#'
#'group.index <- c(rep(1, 2), rep(2, 3), rep(3, 2), rep(4, 3),
#'                 rep(5, 2), rep(6, 3), rep(7, 2), rep(8, 3))
#'lm.group <- bess.one(x, y, s = 5, type = "bss", group.index = group.index)
#'lm.groupbsrr <- bess.one(x, y, type = "bsrr", s = 5, lambda = 0.01, group.index = group.index)
#'coef(lm.group)
#'coef(lm.groupbsrr)
#'print(lm.group)
#'print(lm.groupbsrr)
#'summary(lm.group)
#'summary(lm.groupbsrr)
#'pred.group <- predict(lm.group, newx = x_new)
#'pred.groupl0l2 <- predict(lm.groupbsrr, newx = x_new)
#'#-------------------include specified variables----------------------#
#'Data <- gen.data(n, p, k, rho, family = "gaussian", beta = Tbeta, seed = seed)
#'lm.bss <- bess.one(Data$x, Data$y, s = 5, always.include = 2)
#'
#'\dontrun{
#'#-------------------code demonstration in doi: 10.18637/jss.v094.i04----------------------#
#'Tbeta <- rep(0, 20)
#'Tbeta[c(1, 2, 5, 9)] <- c(3, 1.5, -2, -1)
#'
#'data <- gen.data(n = 200, p = 20, family = "gaussian", beta = Tbeta,
#'rho = 0.2, seed = 123)
#'fit.one <- bess.one(data$x, data$y, s = 4, family = "gaussian")
#'print(fit.one)
#'summary(fit.one)
#'coef(fit.one, sparse = FALSE)
#'pred.one <- predict(fit.one, newdata = data$x)
#'bm.one <- fit.one$bestmodel
#'summary(bm.one)
#'}
#' @export


bess.one <- function(x, y, family = c("gaussian", "binomial", "poisson", "cox"), type = c("bss", "bsrr"),
                             s, lambda= 0, always.include = NULL,
                             screening.num = NULL,
                             normalize = NULL, weight = NULL,
                             max.iter = 20,
                             group.index =NULL){
  if(length(s)>1) stop("bess.one needs only a single value for s.")
  if(length(lambda) > 1) stop("bess.one needs only a single value for lambda.")
  family <- match.arg(family)
  type <- match.arg(type)

  res <- bess(x, y, family = family, type = type,
              method ="sequential",
              tune = "gic",
              s.list=s, lambda.list = lambda,
              s.min=s, s.max=s,
              lambda.min = lambda, lambda.max = lambda, nlambda = 1,
              always.include = always.include,
              screening.num = screening.num,
              normalize = normalize, weight = weight,
              max.iter = max.iter, warm.start = TRUE,
              nfolds = 5,
              group.index =group.index,
              seed=NULL)
  res$s <- s
  res$bess.one <- TRUE
  res$call <- match.call()
  res$beta.all <- NULL
  res$lambda.list <- NULL
  res$s.list <- NULL
  if(type == 'bsrr'){
    res$ic.all <- NULL
    res$s.list <- NULL
    res$loss.all <- NULL
    res$beta.all <- NULL
    res$coef0.all <- NULL
    res$lambda.list <- NULL
    res$algorithm_type <- "L0L2"
    res$method <- NULL
    res$line.search <- NULL
    res$ic.type <- NULL
    res$lambda.max <- NULL
    res$lambda.min <- NULL
    if(!is.null(res$lambda.all)){
     res$lambda.all <- NULL
    }
    res$s.max <- NULL
    res$s.min <- NULL
    res$nlambda <-  NULL
  }else{
    res$ic.all <- NULL
    res$loss.all <- NULL
    res$beta.all <- NULL
    res$coef0.all <- NULL
    res$s.list <- NULL
    res$algorithm_type <- "PDAS"
    res$type <- type
    res$ic.type <- NULL
    res$s.max <- NULL
    res$s.min <- NULL
  }
  return(res)
}
