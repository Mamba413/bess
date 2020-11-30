deviance.bess =function(object,...)
{
  n=object$nsample
  if(object$family!="gaussian"){
    deviance=object$train_loss
  }else{
    deviance=n*log(object$train_loss)
  }

  names(deviance)='deviance'
  return(deviance)

}


