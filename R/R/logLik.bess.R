logLik.bess =function(object,...){
  n=object$nsample
  if(object$family!="gaussian"){
    Loglik=-object$train_loss/2
  }else{
    Loglik=-n*log(object$train_loss)/2
  }
  names(Loglik)='Loglik'
  return(Loglik)

}
