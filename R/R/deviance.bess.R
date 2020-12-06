deviance.bess =function(object,...)
{
  n=object$nsample
  if(object$family!="gaussian"){
    deviance=object$loss
  }else{
    deviance=n*log(object$loss)
  }

  names(deviance)='deviance'
  return(deviance)

}


