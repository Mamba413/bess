#' Generate simulated data
#'
#' Generate data for simulations under the generalized linear model and Cox
#' model.
#'
#' We generate an \eqn{n \times p} random Gaussian matrix
#' \eqn{X} with mean 0 and a covariance matrix with an exponential structure
#' or a constant structure. For the exponential structure, the covariance matrix
#' has \eqn{(i,j)} entry equals \eqn{rho^{|i-j|}}. For the constant structure,
#' the \eqn{(i,j)} entry of the covariance matrix is \eqn{rho} for every \eqn{i
#' \neq j} and 1 elsewhere.
#'
#' For \code{family = "gaussian"} , the data model is \deqn{Y = X \beta +
#' \epsilon.}
#'
#' For \code{family= "binomial"}, the data model is \deqn{Prob(Y = 1) = \exp(X
#' \beta + \epsilon)/(1 + \exp(X \beta + \epsilon)).}
#'
#' For \code{family = "poisson"} , the data is modeled to have an exponential distribution: \deqn{Y = Exp(\exp(X \beta +
#' \epsilon)).}
#'
#'  For \code{family = "cox"}, the data model is
#'\deqn{T = (-\log(S(t))/\exp(X \beta))^{1/scal}.}
#'The centering time is generated from uniform distribution \eqn{[0, c]},
#'then we define the censor status as \eqn{\delta = I\{T \leq C\}, R = min\{T, C\}}.
#'
#' In the above models, \eqn{\epsilon \sim N(0,
#' \sigma^2 ),} where \eqn{\sigma^2} is determined by the \code{snr}. By default, the underlying
#' regression coefficient \eqn{\beta} has equi-spaced nonzero entries, each set to 1.
#'
#' @param n The number of observations.
#' @param p The number of predictors of interest.
#' @param k The number of nonzero coefficients in the underlying regression
#' model. Can be omitted if \code{beta} is supplied.
#' @param rho A parameter used to characterize the pairwise correlation in
#' predictors. Default is \code{0}.
#' @param family The distribution of the simulated data. \code{"gaussian"} for
#' gaussian data.\code{"binomial"} for binary data. \code{"poisson"} for count data. \code{"cox"}
#' for survival data.
#' @param beta The coefficient values in the underlying regression model.
#' @param cortype The correlation structure. \code{cortype = 1} denotes exponential structure,
#' where the covariance matrix has \eqn{(i,j)} entry equals \eqn{rho^{|i-j|}}.
#' code{cortype = 2} denotes constant structure, where the \eqn{(i,j)} entry of covariance
#' matrix is \eqn{rho} for every \eqn{i \neq j} and 1 elsewhere.
#' @param snr A numerical value controlling the signal-to-noise ratio (SNR). The SNR is defined as
#' as the variance of \eqn{x\beta} divided
#' by the variance of a gaussian noise: \eqn{\frac{Var(x\beta)}{\sigma^2}}.
#' The gaussian noise \eqn{\epsilon} is set with mean 0 and variance.
#' The noise is added to the linear predictor \eqn{\eta} = \eqn{x\beta}. Default is \code{snr = 10}.
#' @param censoring Whether data is censored or not. Valid only for \code{family = "cox"}. Default is \code{TRUE}.
#' @param c The censoring rate. Default is \code{1}.
#' @param scal A parameter in generating survival time based on the Weibull distribution. Only used for the "\code{cox}" family.
#' @param seed seed to be used in generating the random numbers.
#' @return %% ~Describe the value returned %% If it is a LIST, use
#' \item{x}{Design matrix of predictors.} \item{y}{Response variable.}
#' \item{Tbeta}{The coefficients used in the underlying regression model.} %%
#' @author Canhong Wen, Aijun Zhang, Shijie Quan, Liyuan Hu, Kangkang Jiang, Yanhang Zhang, Jin Zhu and Xueqin Wang.
#' @seealso \code{\link{bess}}, \code{\link{predict.bess}}.
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
#' Data <- gen.data(n, p, k, rho, family = "gaussian", cortype = cortype, snr = SNR, seed = seed)
#' x <- Data$x[1:140, ]
#' y <- Data$y[1:140]
#' x_new <- Data$x[141:200, ]
#' y_new <- Data$y[141:200]
#' lm.bss <- bess(x, y, method = "sequential")
#' lambda.list <- exp(seq(log(5), log(0.1), length.out = 10))
#' lm.bsrr <- bess(x, y, type = "bsrr", method = "pgsection")
#'
#'
gen.data <- function(n, p, k = NULL, rho = 0, family = c("gaussian", "binomial", "poisson", "cox"), beta = NULL, cortype = 1, snr = 10, censoring=TRUE, c=1, scal, seed = 1){
  family <- match.arg(family)
  set.seed(seed)
  if(is.null(beta)){
    Tbeta <- rep(0, p)
    Tbeta[1:k*floor(p/k):floor(p/k)] <- rep(1, k)
  } else{
    Tbeta <- beta
  }

  if(cortype == 1){
    Sigma <- matrix(0, p, p)
    Sigma <- rho^(abs(row(Sigma) - col(Sigma)))
  }else{
    Sigma <- matrix(rho, p, p) + diag(1-rho, p, p)
  }
  x <- mvrnorm(n, rep(0, p), Sigma)
  sigma <- sqrt((t(Tbeta)%*%Sigma%*%Tbeta)/snr)
  if(family == "gaussian"){
    y<-x%*%Tbeta+rnorm(n,0,sigma)
  } else if(family == "binomial"){
    eta<-x%*%Tbeta+rnorm(n,0,sigma)
    PB<-apply(eta, 1, generatedata2)
    y<-rbinom(n,1,PB)
  } else if(family == "cox"){
      time = (-log(runif(n))/drop(exp(x%*%Tbeta)))^(1/scal)
      if (censoring) {
        ctime = c*runif(n)
        status = (time < ctime) * 1
        censoringrate = 1 - sum(status)/n
        cat("censoring rate:", censoringrate, "\n")
        time = pmin(time, ctime)
      }else {
        status = rep(1, times = n)
        cat("no censoring", "\n")
      }
    y <- cbind(time = time, status = status)
  } else{
    eta <- x%*%Tbeta+rnorm(n,0,sigma)
    eta <- ifelse(eta>30, 30, eta)
    eta <- ifelse(eta<-30, -30, eta)
    eta <- exp(eta)
    y <- rpois(n, eta)
  }
  return(list(x = x, y = y, Tbeta = Tbeta))
}

generatedata2<-function(eta){
    a<-exp(eta)/(1+exp(eta))
    if(is.infinite(exp(eta))){
      a<-1
    }
    return(a)
  }
