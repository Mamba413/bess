
## not used
aic <- function(object, best.model, ...){
  if(object$algorithm_type == "PDAS" | object$algorithm_type == "L0L2"){
    n <- object$nsample
    if(best.model){
      # loglik <- Loglik(object)
      p <- sum(abs(object$beta)>1e-6)
      # p <- p + ifelse(abs(object$coef0)>1e-6, 1, 0)
      # AIC <- -2 * loglik + 2 * p
      # AIC <- n*object$train_loss + 2*p
      if(object$family == "gaussian"){
        AIC <- n*log(object$loss) + 2*p
      } else{
        AIC <- object$loss + 2*p
      }
    }else{
      loglik <- loglik(object, best.model=FALSE)
      if(object$method!="sequential"){
        df <- apply(object$beta_all, 2, function(x){sum(ifelse(abs(x) < 1e-6, 0, 1))})# + ifelse(abs(object$coef0_all)<1e-6, 0, 1)
        # AIC <- -2*loglik + 2*df
        if(object$family == "gaussian"){
          AIC <- n*log(unlist(object$loss.all)) + 2*df
        }else{
          AIC <- n*unlist(object$loss.all) +2*df
        }
      }else{
        df <- object$s.list #+ ifelse(abs(unlist(object$coef0_all)) < 1e-6, 0, 1)
        s.mat <- matrix(rep(df, each = length(object$beta_all)), length(df), byrow = TRUE)
        train_loss_all <- matrix(unlist(object$loss.all), nrow = length(object$s.list), byrow = F)
        # AIC <- -2*loglik + 2*s_mat
        if(object$family == "gaussian"){
          AIC <- n*log(train_loss_all) + 2*s.mat
        } else{
          AIC <- n*train_loss_all + 2*s.mat
        }
      }
    }
    return(AIC)
    ## for group
  } else{
    n = object$nsample
    if(best.model){
      p <- ifelse(abs(object$beta)>1e-6, 1, 0)
      group_p <- vector(0, length = length(object$beta))
      group_p[object$g_index + 1] <- 1
      p <- sum(p*group_p)
      if(object$family == "gaussian"){
        AIC <- n*log(object$loss) + 2*p
      } else{
        AIC <- object$loss + 2*p
      }
    }else{
      loglik <- loglik(object, best.model=FALSE)
      if(object$method!="sequential"){
        df <- apply(object$beta_all, 2, function(x){sum(ifelse(abs(x) < 1e-6, 0, 1))})# + ifelse(abs(object$coef0_all)<1e-6, 0, 1)
        # AIC <- -2*loglik + 2*df
        if(object$family == "gaussian"){
          AIC <- n*log(unlist(object$loss.all)) + 2*df
        }else{
          AIC <- n*unlist(object$loss.all) +2*df
        }
      }else{
        df <- object$s.list #+ ifelse(abs(unlist(object$coef0_all)) < 1e-6, 0, 1)
        s.mat <- matrix(rep(df, each = length(object$beta_all)), length(df), byrow = TRUE)
        train_loss_all <- matrix(unlist(object$loss.all), nrow = length(object$s.list), byrow = F)
        # AIC <- -2*loglik + 2*s_mat
        if(object$family == "gaussian"){
          AIC <- n*log(train_loss_all) + 2*s.mat
        } else{
          AIC <- n*train_loss_all + 2*s.mat
        }
      }
    }
    return(AIC)
  }

}

