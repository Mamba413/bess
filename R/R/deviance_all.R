deviance_all = function(object, training_error){
  n=object$nsample
  if(object$family!="gaussian"){
    deviance=training_error
  }else{
    deviance=n*log(unlist(training_error))
  }

  return(deviance)
}
