
recover <- function(object, sparse = TRUE){
  if(!is.null(object$screening_A)){
    if(object$method == "sequential"){
      beta.all <- lapply(object$beta.all, list.beta, object, sparse)
    } else{
      beta.all = matrix(0, length(object$beta), ncol = ncol(object$beta.all))
      if(object$algorithm_type == "GL0L2" | object$algorithm_type == "GPDAS"){
        beta.all[which(object$group.index %in% object$screening_A), ] = object$beta.all
      } else{
        beta.all[object$screening_A, ] = object$beta.all
      }
      if(sparse){
        beta.all <- Matrix(beta.all)
      }
    }
  }else{
    beta.all <- object$beta.all
  }
  return(beta.all)
}

list.beta <- function(beta.mat, object, sparse){
  beta.all <- matrix(0, nrow=length(object$beta), ncol =  ncol(beta.mat))
  beta.all[object$screening_A, ] = beta.mat[[1]]
  if(sparse){
    beta.all <- Matrix(beta.all)
  }
  return(beta.all)
}



