logLik.bess =function(object,...){
  n=object$nsample
  if(object$family!="gaussian"){
    Loglik=-object$loss/2
  }else{
    Loglik=-n*log(object$loss)/2
  }
  names(Loglik)='Loglik'
  return(Loglik)

}
