#' Best subset selection
#'
#' Best subset selection for generalized linear model and Cox's proportional
#' model.
#'
#' The best subset selection problem with model size \eqn{s} is
#' \deqn{\min_\beta -2 \log L(\beta) \;\;{\rm s.t.}\;\; \|\beta\|_0 \leq s.} In
#' the GLM case, \eqn{\log L(\beta)} is the log-likelihood function; In the Cox
#' model, \eqn{\log L(\beta)} is the log partial likelihood function.
#'
#' The best ridge regression problem with model size \eqn{s} is
#' \deqn{\min_\beta -2 \log L(\beta) + \lambda\Vert\beta\Vert_2^2 \;\;{\rm
#' s.t.}\;\; \|\beta\|_0 \leq s.} In the GLM case, \eqn{\log L(\beta)} is the
#' log likelihood function; In the Cox model, \eqn{\log L(\beta)} is the log
#' partial likelihood function.
#'
#' For each candidate model size and \eqn{\lambda}, the best subset selection and the best subset ridge regression
#' problems are solved by the primal-dual active set (PDAS) algorithm, see Wen et
#' al (2020) for details. This algorithm
#' utilizes an active set updating strategy via primal and dual variables and
#' fits the sub-model by exploiting the fact that their support sets are
#' non-overlap and complementary. For the case of \code{method = "sequential"}
#' if \code{warm.start = "TRUE"}, we run the PDAS algorithm for a list of
#' sequential model sizes and use the estimate from the last iteration as a
#' warm start. For the case of \code{method = "gsection"} of the best subset
#' selection problem, a golden section search technique is adopted to
#' determine the optimal model size efficiently. And for the case of
#' \code{ method = "psequential"} and \code{method = "pgsection"}of the best ridge regression problem, the Powell method
#' using a sequential line search method or a golden section search technique is
#' used for parameters determination.
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
#' \code{"poisson"}, or \code{"cox"}. Depending on the response. Any unambiguous substring can be given.
#' @param method The method to be used to select the optimal model size and \eqn{L_2} shrinkage. For
#' \code{method = "sequential"}, we solve the best subset selection and the best subset ridge regression
#' problem for each \code{s} in \code{1,2,...,s.max} and \eqn{\lambda} in \code{lambda.list}. For \code{method =
#' "gsection"}, which is only valid for \code{type = "bss"},
#' we solve the best subset selection problem with model size ranged between s.min and s.max,
#' where the specific model size to be considered is determined by golden section. we
#' solve the best subset selection problem with a range of non-continuous model
#' sizes. For \code{method = "pgsection"} and \code{"psequential"}, the Powell method is used to
#' solve the best subset ridge regression problem. Any unambiguous substring can be given.
#' @param tune The criterion for choosing the model size and \eqn{L_2} shrinkage
#' parameters. Available options are \code{"gic"}, \code{"ebic"}, \code{"bic"}, \code{"aic"} and \code{"cv"}.
#' Default is \code{"gic"}.
#' @param s.list An increasing list of sequential values representing the model
#' sizes. Only used for \code{method = "sequential"}. Default is \code{1:min(p,
#' round(n/log(n)))}.
#' @param lambda.list A lambda sequence for \code{"bsrr"}. Default is
#' \code{exp(seq(log(100), log(0.01), length.out = 100))}.
#' @param s.min The minimum value of model sizes. Only used for \code{method =
#' "gsection"}, \code{"psequential"} and \code{"pgsection"}. Default is 1.
#' @param s.max The maximum value of model sizes. Only used for \code{method =
#' "gsection"}, \code{"psequential"} and \code{"pgsection"}. Default is \code{min(p, round(n/log(n)))}.
#' @param lambda.min The minimum value of lambda. Only used for \code{method =
#' "powell"}. Default is \code{0.001}.
#' @param lambda.max The maximum value of lambda. Only used for \code{method =
#' "powell"}. Default is \code{100}.
#' @param nlambda The number of \eqn{\lambda}s for the Powell path with sequential line search method.
#' Only valid for \code{method = "psequential"}.
#' @param always.include An integer vector containing the indexes of variables that should always be included in the model.
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
#' @param warm.start Whether to use the last solution as a warm start. Default
#' is \code{TRUE}.
#' @param nfolds The number of folds in cross-validation. Default is \code{5}.
#' @param group.index A vector of integers indicating the which group each variable is in.
#' For variables in the same group, they should be located in adjacent columns of \code{x}
#' and their corresponding index in \code{group.index} should be the same.
#' Denote the first group as \code{1}, the second \code{2}, etc.
#' If you do not fit a model with a group structure,
#' please set \code{group.index = NULL}. Default is \code{NULL}.
#' @param seed Seed to be used to devide the sample into K cross-validation folds. Default is \code{NULL}.
#' @return A list with class attribute 'bess' and named components:
#' \item{beta}{The best fitting coefficients.}
#'  \item{coef0}{The best fitting
#' intercept.}
#' \item{bestmodel}{The best fitted model for \code{type = "bss"}, the class of which is \code{"lm"}, \code{"glm"} or \code{"coxph"}.}
#' \item{loss}{The training loss of the best fitting model.}
#' \item{ic}{The information criterion of the best fitting model when model
#' selection is based on a certain information criterion.} \item{cvm}{The mean
#' cross-validated error for the best fitting model when model selection is
#' based on the cross-validation.}
#'
#' \item{lambda}{The lambda chosen for the best fitting model}
#' \item{beta.all}{For \code{bess} objects obtained by \code{gsection}, \code{pgsection}
#' and \code{psequential}, \code{beta.all} is a matrix with each column be the coefficients
#' of the model in each iterative step in the tuning path.
#' For \code{bess} objects obtained by \code{sequential} method,
#' A list of the best fitting coefficients of size
#' \code{s=0,1,...,p} and \eqn{\lambda} in \code{lambda.list} with the
#' smallest loss function. For \code{"bess"} objects of \code{"bsrr"} type, the fitting coefficients of the
#' \eqn{i^{th} \lambda} and the \eqn{j^{th}} \code{s} are at the \eqn{i^{th}}
#' list component's \eqn{j^{th}} column.}
#' \item{coef0.all}{For \code{bess} objects obtained from \code{gsection}, \code{pgsection} and \code{psequential},
#' \code{coef0.all} contains the intercept for the model in each iterative step in the tuning path.
#' For \code{bess} objects obtained from \code{sequential} path,
#' \code{coef0.all} contains the best fitting
#' intercepts of size \eqn{s=0,1,\dots,p} and \eqn{\lambda} in
#' \code{lambda.list} with the smallest loss function.}
#' \item{loss.all}{For \code{bess} objects obtained from \code{gsection}, \code{pgsection} and \code{psequential},
#' \code{loss.all} contains the training loss of the model in each iterative step in the tuning path.
#' For \code{bess} objects obtained from \code{sequential} path, this is a
#' list of the training loss of the best fitting intercepts of model size
#' \eqn{s=0,1,\dots,p} and \eqn{\lambda} in \code{lambda.list}. For \code{"bess"} object obtained by \code{"bsrr"},
#' the training loss of the \eqn{i^{th} \lambda} and the \eqn{j^{th}} \code{s}
#' is at the \eqn{i^{th}} list component's \eqn{j^{th}} entry.}
#' \item{ic.all}{For \code{bess} objects obtained from \code{gsection}, \code{pgsection} and \code{psequential},
#' \code{ic.all} contains the values of the chosen information criterion of the model in each iterative step in the tuning path.
#' For \code{bess} objects obtained from \code{sequential} path, this is a
#' matrix of the values of the chosen information criterion of model size \eqn{s=0,1,\dots,p}
#' and \eqn{\lambda} in \code{lambda.list} with the smallest loss function. For \code{"bess"} object obtained by \code{"bsrr"},
#' the training loss of the \eqn{i^{th} \lambda} and the \eqn{j^{th}}
#' \code{s} is at the \eqn{i^{th}} row \eqn{j^{th}} column. Only available when
#' model selection is based on a certain information criterion.}
#'
#' \item{cvm.all}{For \code{bess} objects obtained from \code{gsection}, \code{pgsection} and \code{psequential},
#' \code{cvm.all} contains the mean cross-validation error of the model in each iterative step in the tuning path.
#' For \code{bess} objects obtained from \code{sequential} path, this is a
#'  matrix of the mean cross-validation error of model size
#' \eqn{s=0,1,\dots,p} and \eqn{\lambda} in \code{lambda.list} with the
#' smallest loss function. For \code{"bess"} object obtained by \code{"bsrr"}, the training loss of the \eqn{i^{th}
#' \lambda} and the \eqn{j^{th}} \code{s} is at the \eqn{i^{th}} row
#' \eqn{j^{th}} column. Only available when model selection is based on the
#' cross-validation.}
#' \item{lambda.all}{The lambda chosen for each step in \code{pgsection} and \code{psequential}.}
#' \item{family}{Type of the model.}
#' \item{s.list}{The input
#' \code{s.list}.} \item{nsample}{The sample size.}
#' \item{type}{Either \code{"bss"} or \code{"bsrr"}.}
#' \item{method}{Method used for tuning parameters selection.}
#' \item{ic.type}{The criterion of model selection.}
#' @author Canhong Wen, Aijun Zhang, Shijie Quan, Liyuan Hu, Kangkang Jiang, Yanhang Zhang, Jin Zhu and Xueqin Wang.
#' @seealso \code{\link{plot.bess}}, \code{\link{summary.bess}},
#' \code{\link{coef.bess}}, \code{\link{predict.bess}}, \code{\link{bess.one}}.
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
#' lm.bss <- bess(x, y)
#' lambda.list <- exp(seq(log(5), log(0.1), length.out = 10))
#' lm.bsrr <- bess(x, y, type = "bsrr", method = "pgsection")
#' coef(lm.bss)
#' coef(lm.bsrr)
#' print(lm.bss)
#' print(lm.bsrr)
#' summary(lm.bss)
#' summary(lm.bsrr)
#' pred.bss <- predict(lm.bss, newx = x_new)
#' pred.bsrr <- predict(lm.bsrr, newx = x_new)
#'
#' # generate plots
#' plot(lm.bss, type = "both", breaks = TRUE)
#' plot(lm.bsrr)
#' #-------------------logistic model----------------------#
#' #Generate simulated data
#' Data <- gen.data(n, p, k, rho, family = "binomial", beta = Tbeta, seed = seed)
#'
#' x <- Data$x[1:140, ]
#' y <- Data$y[1:140]
#' x_new <- Data$x[141:200, ]
#' y_new <- Data$y[141:200]
#' logi.bss <- bess(x, y, family = "binomial")
#' lambda.list <- exp(seq(log(5), log(0.1), length.out = 10))
#' logi.bsrr <- bess(x, y, type = "bsrr", family = "binomial", lambda.list = lambda.list)
#' coef(logi.bss)
#' coef(logi.bsrr)
#' print(logi.bss)
#' print(logi.bsrr)
#' summary(logi.bss)
#' summary(logi.bsrr)
#' pred.bss <- predict(logi.bss, newx = x_new)
#' pred.bsrr <- predict(logi.bsrr, newx = x_new)
#'
#' # generate plots
#' plot(logi.bss, type = "both", breaks = TRUE)
#' plot(logi.bsrr)
#'#-------------------poisson model----------------------#
#'Data <- gen.data(n, p, k, rho=0.3, family = "poisson", beta = Tbeta, seed = seed)
#'
#'x <- Data$x[1:140, ]
#' y <- Data$y[1:140]
#' x_new <- Data$x[141:200, ]
#' y_new <- Data$y[141:200]
#' poi.bss <- bess(x, y, family = "poisson")
#' lambda.list <- exp(seq(log(5), log(0.1), length.out = 10))
#' poi.bsrr <- bess(x, y, type = "bsrr",
#'                  family = "poisson", lambda.list = lambda.list)
#' coef(poi.bss)
#' coef(poi.bsrr)
#' print(poi.bss)
#' print(poi.bsrr)
#' summary(poi.bss)
#' summary(poi.bsrr)
#' pred.bss <- predict(poi.bss, newx = x_new)
#' pred.bsrr <- predict(poi.bsrr, newx = x_new)
#'
#' # generate plots
#' plot(poi.bss, type = "both", breaks = TRUE)
#' plot(poi.bsrr)
#' #-------------------coxph model----------------------#
#' #Generate simulated data
#' Data <- gen.data(n, p, k, rho, family = "cox", scal = 10, beta = Tbeta)
#'
#' x <- Data$x[1:140, ]
#' y <- Data$y[1:140, ]
#' x_new <- Data$x[141:200, ]
#' y_new <- Data$y[141:200, ]
#' cox.bss <- bess(x, y, family = "cox")
#' lambda.list <- exp(seq(log(5), log(0.1), length.out = 10))
#' cox.bsrr <- bess(x, y, type = "bsrr", family = "cox", lambda.list = lambda.list)
#' coef(cox.bss)
#' coef(cox.bsrr)
#' print(cox.bss)
#' print(cox.bsrr)
#' summary(cox.bss)
#' summary(cox.bsrr)
#' pred.bss <- predict(cox.bss, newx = x_new)
#' pred.bsrr <- predict(cox.bsrr, newx = x_new)
#'
#' # generate plots
#' plot(cox.bss, type = "both", breaks = TRUE)
#' plot(cox.bsrr)
#'
#'#----------------------High dimensional linear models--------------------#
#'\dontrun{
#' data <- gen.data(n, p = 1000, k, family = "gaussian", seed = seed)
#'
#'# Best subset selection with SIS screening
#' lm.high <- bess(data$x, data$y, screening.num = 100)
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
#'lm.group <- bess(x, y, s.min=1, s.max = 8, type = "bss", group.index = group.index)
#'lm.groupbsrr <- bess(x, y, type = "bsrr", s.min = 1, s.max = 8, group.index = group.index)
#'coef(lm.group)
#'coef(lm.groupbsrr)
#'print(lm.group)
#'print(lm.groupbsrr)
#'#'summary(lm.group)
#'summary(lm.groupbsrr)
#'pred.group <- predict(lm.group, newx = x_new)
#'pred.groupl0l2 <- predict(lm.groupbsrr, newx = x_new)
#'#-------------------include specified variables----------------------#
#'Data <- gen.data(n, p, k, rho, family = "gaussian", beta = Tbeta, seed = seed)
#'lm.bss <- bess(Data$x, Data$y, always.include = 2)
#'
#'\dontrun{
#'#-------------------trim32 data analysis in doi: 10.18637/jss.v094.i04----------------------#
#'# import trim32 data by:
#'load(url('https://github.com/Mamba413/bess/tree/master/data/trim32.RData'))
#'# or manually downloading trim32.RData in the github page:
#'# "https://github.com/Mamba413/bess/tree/master/data/" and read it by:
#'load('trim32.RData')
#'
#'X <- trim32$x
#'Y <- trim32$y
#'dim(X)
#'
#'# running bess with argument method = "sequential".
#' fit.seq <- bess(X, Y, method = "sequential")
#' summary(fit.seq)
#'
#' # the bess function outputs an 'lm' type of object bestmodel associated
#' # with the selected best model
#' bm.seq <- fit.seq$bestmodel
#' summary(bm.seq)
#' pred.seq <- predict(fit.seq, newdata = data$x)
#' plot(fit.seq, type = "both", breaks = TRUE)
#'
#' # We now call the function bess with argument method = "gsection"
#' fit.gs <- bess(X, Y, family = "gaussian", method = "gsection")
#' bm.gs <- fit.gs$bestmodel
#' summary(bm.gs)
#' beta <- coef(fit.gs, sparse = TRUE)
#' class(beta)
#' pred.gs <- predict(fit.gs, newdata = X)
#'}
#' @export
bess <- function(x, y, family = c("gaussian", "binomial", "poisson", "cox"), type = c("bss", "bsrr"),
                 method = c("gsection", "sequential", "pgsection", "psequential"),
                 tune = c("gic", "ebic", "bic", "aic", "cv"),
                 s.list, lambda.list = 0,
                 s.min, s.max,
                 lambda.min = 0.001, lambda.max = 100, nlambda = 100,
                 always.include = NULL,
                 screening.num = NULL,
                 normalize = NULL, weight = NULL,
                 max.iter = 20, warm.start = TRUE,
                 nfolds = 5,
                 group.index =NULL,
                 seed=NULL){
  set.seed(seed)

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
  family <- match.arg(family)
  # FAMILY <- c("gaussian", "binomial", "poisson", "cox")
  # family <- pmatch(family, FAMILY)
  # if(is.na(family)) stop("invalid family")
  # if(family == -1) stop("ambigous family")
  # family <- c("gaussian", "binomial", "poisson", "cox")[family]
  model_type <- switch(family,
                       "gaussian" = 1,
                       "binomial" = 2,
                       "poisson" = 3,
                       "cox" = 4)
  method <- match.arg(method)
  # METHOD <- c("gsection", "sequential", "pgsection", "psequential")
  # method <- pmatch(method, METHOD)
  # if(is.na(method)) stop("invalid method")
  # if(method == -1) stop("ambigous method")
  # method <- c("gsection", "sequential", "pgsection", "psequential")[method]
  if(method == "pgsection"){
    path_type <- 2
    line.search <- 1
  } else if(method == "psequential") {
    path_type <- 2
    line.search <- 2
  } else if(method == "sequential"){
    path_type <- 1
    line.search <- 1
  } else{
    path_type <- 2
    line.search <- 1
  }
  if(!is.null(group.index)){
    if(path_type == 1 & s.list[length(s.list)] > length(group.index)) stop("The maximum one s.list should not be larger than the number of groups!")
    if(path_type == 2 & s.max > length(group.index)) stop("s.max is too large. Should be smaller than the number of groups!")
  } else{
    if(path_type == 1 & s.list[length(s.list)] > ncol(x)) stop("The maximum one in s.list is too large!")
    if(path_type == 2 & s.max > ncol(x)) stop("s.max is too large")
  }

  if(!is.null(group.index)){
    gi <- unique(group.index)
    g_index <- match(gi, group.index)-1
    g_df <- c(diff(g_index), length(group.index) - g_index[length(g_index)])
    # g_df <- NULL
    # g_index <- NULL
    # group_set <- unique(group.index)
    # j <- 1
    # k <- 0
    # for(i in group_set){
    #   while(group.index[j] != i){
    #     j <- j+1
    #     k <- k+1
    #   }
    #   g_index <- c(g_index, j - 1)
    #   g_df <- c(g_df, k)
    # }
    algorithm_type = switch(type,
                            "bss" = "GPDAS",
                            "bsrr" = "GL0L2")
  } else{
    algorithm_type = switch(type,
                            "bss" = "PDAS",
                            "bsrr" = "L0L2")
  }
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
    is_normal <- TRUE
    normalize <- switch(family,
                        "gaussian" = 1,
                        "binomial" = 2,
                        "poisson" = 2,
                        "cox" = 3)
  } else if(normalize !=0){
    # normalize <- as.character(normalize)
    # normalize <- switch (normalize,
    #                      '1' <- 2,
    #                      '2' <- 3,
    #                      '3' <- 1
    # )
    if(normalize == 1){
      normalize <- 2
    }else if(normalize == 2){
      normalize <- 3
    }else{
      normalize <- 1
    }
    is_normal <- TRUE
  } else{
    is_normal <- FALSE
    normalize <- 0
  }
  # if(!is.null(factor)){
  #   if(is.null(colnames(x))) colnames(x) <- paste0("X",1:ncol(x),"g")
  # }else{
  #   if(is.null(colnames(x))) colnames(x) <- paste0("X",1:ncol(x))
  # }
  if(!is.matrix(x)) x <- as.matrix(x)
  # if(!is.null(factor) & length(which(diff(group.index)!=1))>0){
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
  if(is.null(vn)) vn <- paste("x", 1:ncol(x), sep = "")
  if(is.null(weight)) weight <- rep(1, nrow(x))
  if(is.null(screening.num)){
    screening <- FALSE
    screening.num <- ncol(x)
  } else{
    screening <- TRUE
    if(screening.num > ncol(x)) stop("The number of screening features must be equal or less than that of the column of x!")
    if(path_type == 1){
      if(screening.num < s.list[length(s.list)]) stop("The number of screening features must be equal or greater than the maximum one in s.list!")
    } else{
      if(screening.num < s.max) stop("The number of screening features must be equal or greater than the s.max!")
    }
  }
  if(is.null(always.include)) {
    always.include <- numeric(0)
  }else{
    if(is.na(sum(as.integer(always.include)))) stop("always.include should be an integer vector")
    if(sum(always.include <= 0)) stop("always.include should be an vector containing variable indexes which is possitive.")
    always.include <- as.integer(always.include) - 1
    if(length(always.include) > screening.num) stop("The number of variables in always.include should not exceed the sc")
    if(path_type == 1){
      if(length(always.include) > s.list[length(s.list)]) stop("always.include containing too many variables. The length of it should not exceed the maximum in s.list.")
    }else{
      if(length(always.include)>s.max) stop("always.include containing too many variables. The length of it should not exceed the s.max.")
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
      ys <- y
      xs <- x
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
                        powell_path = 1, g_index=(1:ncol(x)-1), always_select=always.include, tao=1.1)
    beta.pdas <- res.pdas$beta
    names(beta.pdas) <- vn
    res.pdas$beta <- beta.pdas
    if(is_cv == TRUE) {
      names(res.pdas)[which(names(res.pdas) == "ic")] <- "cvm"
      names(res.pdas)[which(names(res.pdas) == "ic_all")] <- "cvm.all"
    } else{
      names(res.pdas)[which(names(res.pdas)=='ic_all')] <- 'ic.all'
    }
    res.pdas$x <- ifelse(family == "cox", xs, x)
    res.pdas$y <- ifelse(family == "cox", ys, y)
    res.pdas$family <- family
    names(res.pdas)[which(names(res.pdas)=="train_loss")] <- "loss"
    names(res.pdas)[which(names(res.pdas)=="train_loss_all")] <- "loss.all"
    names(res.pdas)[which(names(res.pdas)=='beta_all')] <- 'beta.all'
    names(res.pdas)[which(names(res.pdas)=="coef0_all")] <- 'coef0.all'
    res.pdas$s.list <- s.list
    res.pdas$nsample <- nrow(x)
    res.pdas$algorithm_type <- "PDAS"
    res.pdas$method <- method
    res.pdas$type <- type
    res.pdas$ic.type <- ifelse(is_cv == TRUE, "cv", c("AIC", "BIC", "GIC", "EBIC")[ic_type])
    res.pdas$s.max <- s.max
    res.pdas$s.min <- s.min
    if(screening) res.pdas$screening_A <-  res.pdas$screening_A + 1;
    res.pdas$call <- match.call()
    class(res.pdas) <- 'bess'
    #res.pdas$beta_all <- res.pdas$beta.all
    res.pdas$beta.all <- recover(res.pdas, F)
    if(family == "gaussian"){
      xbest <- x[,which(beta.pdas!=0)]
      bestmodel <- lm(y~xbest, weights = weight)
    }else if(family == "cox"){
      xbest <- xs[,which(beta.pdas!=0)]
      bestmodel <- coxph(Surv(ys[,1],ys[,2])~xbest, iter.max=max.iter, weights=weight)
    }else{
      xbest <- x[,which(beta.pdas!=0)]
      bestmodel=glm(y~xbest, family=family, weights=weight)
    }

    res.pdas$bestmodel <- bestmodel

    set.seed(NULL)
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
                         lambda_max = 0, lambda_min = 0 , nlambda = nlambda, is_screening = screening,
                         screening_size = screening.num, powell_path = 1, g_index = g_index, always_select=always.include, tao=1.1)
    beta.gpdas <- res.gpdas$beta
    names(beta.gpdas) <- vn
    res.gpdas$beta <- beta.gpdas
    if(is_cv == TRUE) {
      names(res.gpdas)[which(names(res.gpdas) == "ic")] <- "cvm"
      names(res.gpdas)[which(names(res.gpdas) == "ic_all")] <- "cvm.all"
    }else{
      names(res.gpdas)[which(names(res.gpdas)=='ic_all')] <- 'ic.all'
    }
    res.gpdas$x <- x
    res.gpdas$y <- y
    res.gpdas$family <- family
    names(res.gpdas)[which(names(res.gpdas)=="train_loss")] <- "loss"
    names(res.gpdas)[which(names(res.gpdas)=="train_loss_all")] <- "loss.all"
    names(res.gpdas)[which(names(res.gpdas)=='beta_all')] <- 'beta.all'
    names(res.gpdas)[which(names(res.gpdas)=="coef0_all")] <- 'coef0.all'
    res.gpdas$s.list <- s.list
    res.gpdas$nsample <- nrow(x)
    res.gpdas$algorithm_type <- "GPDAS"
    res.gpdas$method <- method
    res.gpdas$type <- type
    res.gpdas$ic.type <- ifelse(is_cv == TRUE, "cv", c("AIC", "BIC", "GIC", "EBIC")[ic_type])
    res.gpdas$s.max <- s.max
    res.gpdas$s.min <- s.min
    res.gpdas$group.index <- group.index
    res.gpdas$g_index <- g_index
    res.gpdas$g_df <- g_df
    if(screening) res.gpdas$screening_A <-  res.gpdas$screening_A + 1;
    res.gpdas$call <- match.call()
    class(res.gpdas) <- "bess"
    #res.gpdas$beta_all <- res.gpdas$beta.all
    res.gpdas$beta.all <- recover(res.gpdas, F)
    xbest <- x[,which(beta.gpdas!=0)]
    if(family == "gaussian"){
      xbest <- x[,which(beta.gpdas!=0)]
      bestmodel <- lm(y~xbest, weights = weight)
    }else if(family == "cox"){
      xbest <- xs[,which(beta.gpdas!=0)]
      bestmodel <- coxph(Surv(ys[,1],ys[,2])~xbest, iter.max=max.iter, weights=weight)
    }else{
      xbest <- x[,which(beta.gpdas!=0)]
      bestmodel=glm(y~xbest, family=family, weights=weight)
    }
    res.gpdas$bestmodel <- bestmodel

    set.seed(NULL)
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
                         lambda_max = lambda.max, lambda_min = lambda.min, nlambda = nlambda,
                         is_screening = screening, screening_size = screening.num, powell_path = 1,
                         g_index = g_index, always_select=always.include, tao=1.1)
    beta.gl0l2 <- res.gl0l2$beta
    names(beta.gl0l2) <- vn
    res.gl0l2$beta <- beta.gl0l2
    if(is_cv == TRUE) {
      names(res.gl0l2)[which(names(res.gl0l2) == "ic")] <- "cvm"
      names(res.gl0l2)[which(names(res.gl0l2) == "ic_all")] <- "cvm.all"
    }else{
      names(res.gl0l2)[which(names(res.gl0l2)=='ic_all')] <- 'ic.all'
    }
    res.gl0l2$x <- x
    res.gl0l2$y <- y
    res.gl0l2$family <- family
    res.gl0l2$s.list <- s.list
    names(res.gl0l2)[which(names(res.gl0l2)=="train_loss")] <- "loss"
    names(res.gl0l2)[which(names(res.gl0l2)=="train_loss_all")] <- "loss.all"
    names(res.gl0l2)[which(names(res.gl0l2)=='beta_all')] <- 'beta.all'
    names(res.gl0l2)[which(names(res.gl0l2)=="coef0_all")] <- 'coef0.all'
    res.gl0l2$nsample <- nrow(x)
    res.gl0l2$algorithm_type <- "GL0L2"
    res.gl0l2$method <- method
    res.gl0l2$type <- type
    res.gl0l2$line.search <- ifelse(line.search == 1, "gsection", "sequential")
    res.gl0l2$ic.type <- ifelse(is_cv == TRUE, "cv", c("AIC", "BIC", "GIC", "EBIC")[ic_type])
    res.gl0l2$lambda.max <- lambda.max
    res.gl0l2$lambda.min <- lambda.min
    if(!is.null(res.gl0l2$lambda_all)){
      names(res.gl0l2)[which(names(res.gl0l2)== "lambda_all")] <- 'lambda.all'
    }
    res.gl0l2$s.max <- s.max
    res.gl0l2$s.min <- s.min
    res.gl0l2$nlambda <- nlambda
    res.gl0l2$group.index <- group.index
    res.gl0l2$g_index <- g_index
    res.gl0l2$g_df <- g_df
    if(screening) res.gl0l2$screening_A <-  res.gl0l2$screening_A + 1;
    res.gl0l2$call <- match.call()
    class(res.gl0l2) <- "bess"
    #res.gl0l2$beta_all <- res.gl0l2$beta.all
    res.gl0l2$beta.all <- recover(res.gl0l2, F)
    res.gl0l2$bestmodel <- NULL

    set.seed(NULL)
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
                        powell_path = line.search, g_index = (1:ncol(x) -1), always_select=always.include, tao=1.1)
    beta.l0l2 <- res.l0l2$beta
    # length(which.max(beta.l0l2!=0))
    names(beta.l0l2) <- vn
    res.l0l2$beta <- beta.l0l2
    if(is_cv == TRUE) {
      names(res.l0l2)[which(names(res.l0l2) == "ic")] <- "cvm"
      names(res.l0l2)[which(names(res.l0l2) == "ic_all")] <- "cvm.all"
    }else{
      names(res.l0l2)[which(names(res.l0l2)=='ic_all')] <- 'ic.all'
    }
    res.l0l2$x <- x
    res.l0l2$y <- y
    res.l0l2$family <- family
    res.l0l2$s.list <- s.list
    names(res.l0l2)[which(names(res.l0l2)=="train_loss")] <- "loss"
    names(res.l0l2)[which(names(res.l0l2)=="train_loss_all")] <- "loss.all"
    names(res.l0l2)[which(names(res.l0l2)=='beta_all')] <- 'beta.all'
    names(res.l0l2)[which(names(res.l0l2)=="coef0_all")] <- 'coef0.all'
    res.l0l2$lambda.list <- lambda.list
    res.l0l2$nsample <- nrow(x)
    res.l0l2$algorithm_type <- "L0L2"
    res.l0l2$method <- method
    res.l0l2$type <- type
    res.l0l2$line.search <- ifelse(line.search == 1, "gsection", "sequential")
    res.l0l2$ic.type <- ifelse(is_cv == TRUE, "cv", c("AIC", "BIC", "GIC", "EBIC")[ic_type])
    res.l0l2$lambda.max <- lambda.max
    res.l0l2$lambda.min <- lambda.min
    if(!is.null(res.l0l2$lambda_all)){
      names(res.l0l2)[which(names(res.l0l2) == "lambda_all")] <- 'lambda.all'
    }
    res.l0l2$s.max <- s.max
    res.l0l2$s.min <- s.min
    res.l0l2$nlambda <- nlambda
    if(screening) res.l0l2$screening_A <-  res.l0l2$screening_A + 1;
    res.l0l2$call <- match.call()
    class(res.l0l2) <- "bess"
    #res.l0l2$beta_all <- res.l0l2$beta.all
    res.l0l2$beta.all <- recover(res.l0l2, F)
    res.l0l2$bestmodel <- NULL
    return(res.l0l2)

    set.seed(NULL)
  }
}


