#' Produces a coefficient profile plot of the coefficient or loss function
#' paths
#'
#' Produces a coefficient profile plot of the coefficient or loss function
#' paths
#'
#'
#' @param x a \code{"bess"} object
#' @param type One of \code{"loss"}, \code{"tune"}, \code{"coefficients"}, \code{"both"}. This option is only valid for \code{"bess"} object obtained from \code{"bss"}.
#' If \code{type = "loss"} (\code{type = "tune"}), a path of loss function (corresponding information criterion or cross-validation loss) is provided.
#' If \code{type = "coefficients"}, it provides a coefficient profile plot of the coefficient.
#' If \code{type = "both"}, it combines the path of corresponding information criterion or cross-validation loss with the coefficient profile plot.
#' @param breaks If \code{TRUE}, a vertical line is drawn at a specified break point in
#' the coefficient paths.
#' @param K Which break point should the vertical line be drawn at. Default is the optimal model size.
#' @param sign.lambda A logical value indicating whether to show lambda on log scale. Default is 0. Valid for \code{"bess"} object obtained from \code{"bsrr"}.
#' @param \dots Other graphical parameters to plot
#' @author Canhong Wen, Aijun Zhang, Shijie Quan, Liyuan Hu, Kangkang Jiang, Yanhang Zhang, Jin Zhu and Xueqin Wang.
#' @seealso \code{\link{bess}}.
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
#' Data <- gen.data(n, p, k, rho, family = "gaussian", cortype = cortype, SNR = SNR, seed = seed)
#' x <- Data$x[1:140, ]
#' y <- Data$y[1:140]
#' x_new <- Data$x[141:200, ]
#' y_new <- Data$y[141:200]
#' lm.bss <- bess(x, y, method = "sequential")
#' lambda.list <- exp(seq(log(5), log(0.1), length.out = 10))
#' lm.bsrr <- bess(x, y, type = "bsrr", method = "pgsection")
#'
#' # generate plots
#' plot(lm.bss, type = "both", breaks = TRUE)
#' plot(lm.bsrr)
#'
#'
#'@method plot bess
#'@export
#'@export plot.bess
plot.bess<-function(x, type = c("loss", "tune", "coefficients","both"), breaks = TRUE, K = NULL, sign.lambda = 0, ...)
{
  if(x$algorithm_type == "GPDAS" | x$algorithm_type == "GL0L2") stop("plots for group selection not available now")
  if(x$algorithm_type == "PDAS"){
    type <- match.arg(type)
    # s.list=x$s.list
    df_list <- apply(matrix(unlist(x$beta_all), nrow = length(x$beta), byrow = F), 2, function(x){sum(ifelse(abs(x) < 1e-8, 0, 1))})

    if(is.null(K))  K <- length(which(x$beta!=0))#K<-df_list[length(df_list)]
    # if(x$family=="gaussian") dev=x$mse else dev=x$deviance

    if(type == "tune"){
      if(x$ic_type == "cv"){
        dev <- unlist(x$cvm_all)
      } else{
        dev <- unlist(x$ic_all)
      }
    }
    if(type == "loss"){
      dev = x$train_loss_all
    }
    if(type == "both"){
      if(x$ic_type == "cv"){
        dev <- unlist(x$cvm_all)
      } else{
        dev <- unlist(x$ic_all)
      }
    }

    beta_all <- matrix(unlist(x$beta_all), nrow = length(x$beta), byrow = F)
    df_order <- order(df_list)
    df_list <- df_list[df_order]
    dev <- dev[df_order]
    beta_all <- beta_all[,df_order]
    beta_all <- cbind(rep(0,length(x$beta)), beta_all)

    if(type=="loss" | type == "tune")
    {
      plot_loss(dev,df_list,K,breaks, mar = c(3,4,3,4), ic_type=x$ic_type)
    }
    if(type=="coefficients")
    {
      plot_solution(beta_all, c(0, df_list), K, breaks, mar = c(3,4,3,4))
    }
    if(type=="both")
    {
      layout(matrix(c(1,2),2,1,byrow=TRUE),heights=c(0.45,0.55), widths=1)
      oldpar <- par(las=1, mar=c(2,4,2,4), oma=c(2.5,0.5,1.5,0.5))
      plot_loss(dev,df_list,K,breaks,show_x = FALSE, ic_type = x$ic_type)
      plot_solution(beta_all, c(0, df_list), K,breaks)
      par(oldpar)
      par(mfrow=c(1,1))
    }
  } else{
    # plot_l0l2(x, sign.lambda, threeD)
    plot_heatmap(x,sign.lambda)
  }
}


plot_loss <- function(loss,df,K,breaks=TRUE,show_x=TRUE, mar = c(0,4,2,4), ic_type){

  plot.new()                            # empty plot
  plot.window(range(df), range(loss), xaxs="i")
  oldpar <- par(mar = mar,              # no bottom spacing
                lend="square")          # square line ends
  par(new=TRUE)                         # add to the plot
  if(show_x)
  {
    plot(df, loss, type = "b", ylab=ifelse(ic_type=="cv", "cross validation error", ic_type),
         xlim=c(0,max(df)))
  }else
  {
    plot(df, loss, type = "b", ylab=ifelse(ic_type=="cv", "cross validation error", ic_type),
         xlim=c(0,max(df)), xaxt='n')
  }
  title(xlab='Model size', line = 2)
  if(breaks)abline(v=K, col="orange", lwd=1.5, lty=2) ## add a vertical line
  grid()
  axis(2)
  #axis(4, pos=par("usr")[1], line=0.5)  # this would plot them 'inside'
  # box()                                 # outer box

  par(oldpar)
}

plot_solution <- function(beta, df, K, breaks = TRUE, mar = c(3,4,0,4)){
  p <- nrow(beta)
  plot.new()                            # empty plot
  plot.window(range(df), range(beta), xaxs="i")

  oldpar <- par(mar=mar,         # no top spacing
                lend="square")          # square line ends
  par(new=TRUE)                         # add to the plot

  plot(df, beta[1,], type="l",col=1, xlim=c(0,max(df)),xlab="",
       ylim=range(beta),ylab=expression(beta))
  title(xlab='Model size', line = 2)
  for(i in 2:p){
    lines(df, beta[i,], col=i,xlim=c(0,p+1))
  }
  if(breaks) abline(v=K, col="orange", lwd=1.5, lty=2) ## add a vertical line
  #matplot(df, t(beta), lty = 1, ylab="",  xaxs="i",type = "l",xlim=c(0,p+1))

  nnz <- p
  xpos <- max(df)-0.8
  pos <- 4
  xpos <- rep(xpos, nnz)
  ypos <- beta[, ncol(beta)]
  text(xpos, ypos, 1:p, cex = 0.8, pos = pos)

  grid()
  axis(2)
  box()                                 # outer box

  par(oldpar)
}

# plot_l0l2 <- function(x, sign.lambda, threeD){
#   # 3D
#   if(threeD == TRUE){
#     # powell
#     if(x$method == "powell"){
#       if(sign.lambda == 1){
#         lam_list <- log(x$lambda_all)
#       } else{
#         lam_list <- x$lambda_all
#       }
#       df_list <- apply(x$beta_all, 2, function(x){sum(ifelse(abs(x) < 1e-8, 0, 1))}) # x
#       if(x$ic_type == "cv"){
#         z <- x$cvm_all
#         #open3d()
#         plot3d(df_list, lam_list, z, type = "l", col = "blue", xlab = "DF", ylab =ifelse(sign.lambda == 1, expression(Log(lambda)), expression(lambda)), zlab = "mean cross validation error")
#         text3d(df_list,lam_list,z, texts= 1:length(x$lambda_all),adj = 0.1
#                , font=5, cex=1)
#         # z = x$cvm_all
#         # x0 = df_list
#         # y0 = lam_list
#         # z0 = z
#         # x1 = c(df_list[-1], df_list[length(df_list)])
#         # y1 = c(lam_list[-1], lam_list[length(lam_list)])
#         # z1 = c(z0[-1], z0[length(z0)])
#         # xyz = cbind(c(df_list,  df_list[length(df_list)]), c(lam_list, lam_list[length(lam_list)]), c(z, z[length(z)]))
#         # plot3d(df_list, lam_list, z, type = "l", col = "blue", xlab = "DF", ylab =ifelse(sign.lambda == 1, expression(Log(lambda)), expression(lambda)), zlab = "mean cross validation error")
#         # text3d(df_list,lam_list,z, text= 1:length(x$lambda_all),adj = 0.1
#         #        , font=5, cex=1)
#         # for(i in 1:length(z)){
#         #   arrow3d(xyz[i, ], xyz[i+1, ], col = "blue")
#         # }
#
#       } else{
#         z <- x$ic_all
#         zlab <- x$ic_type
#         #open3d()
#         plot3d(df_list, lam_list, z, type="l", col = "blue", xlab = "DF", ylab = ifelse(sign.lambda == 1, expression(Log(lambda)), expression(lambda)), zlab = zlab)
#         text3d(df_list,lam_list,z, texts= 1:length(x$lambda_all),adj = 0.1,
#                font=5, cex=1)
#       }
#       # sequential
#     } else{
#       if(sign.lambda == 1){
#         lam_list <- log(x$lambda.list)
#       } else{
#         lam_list <- x$lambda.list
#       }
#       lam_order <- order(lam_list)
#       lam_list <- lam_list[lam_order]
#       df_list <- x$s.list
#       if(x$ic_type == "cv"){
#         z <- x$cvm_all
#         nrz <- nrow(z)
#         ncz <- ncol(z)
#         nbcol <- 100
#         color <- jet.colors(nbcol)
#         zfacet <- z[-1, -1] + z[-1, -ncz] + z[-nrz, -1] + z[-nrz, -ncz]
#         facetcol <- cut(zfacet, nbcol)
#         persp3d(df_list, lam_list, z, col = "blue", phi = 30, theta = -30, xlab = "DF", ylab =ifelse(sign.lambda == 1, expression(Log(lambda)), expression(lambda)), zlab = "mean cross validation error")
#       } else{
#         z <- x$ic_all
#         nrz <- nrow(z)
#         ncz <- ncol(z)
#         nbcol <- 100
#         color <- jet.colors(nbcol)
#         zfacet <- z[-1, -1] + z[-1, -ncz] + z[-nrz, -1] + z[-nrz, -ncz]
#         facetcol <- cut(zfacet, nbcol)
#         zlab <- x$ic_type
#         z <- z[, lam_order]
#         persp3d(df_list, lam_list, z, col = "blue", phi = 30, theta = -30, xlab = "DF", ylab =ifelse(sign.lambda == 1, expression(Log(lambda)), expression(lambda)), zlab = zlab)
#       }
#     }
#     # 2D
#   } else{
#     # powell
#     if(x$method == "powell"){
#       if(sign.lambda == 1){
#         lam_list <- log(x$lambda_all)
#       } else{
#         lam_list <- x$lambda_all
#       }
#       df_list <- apply(x$beta_all, 2, function(x){sum(ifelse(abs(x) < 1e-8, 0, 1))}) # x
#       if(x$ic_type == "cv"){
#         z <- x$cvm_all
#         x0 <- df_list
#         y0 <- lam_list
#         z0 <- z
#         x1 <- c(df_list[-1], df_list[length(df_list)])
#         y1 <- c(lam_list[-1], lam_list[length(lam_list)])
#         z1 <- c(z0[-1], z0[length(z0)])
#         arrows3D(x0, y0, z0, x1, y1, z1,ticktype="detailed", xlab = "DF", ylab =ifelse(sign.lambda == 1, expression(Log(lambda)), expression(lambda)), zlab = "mean cross validation error")
#         # text3D(df_list,lam_list,z, labels = 1:length(x$lambda_all),add=TRUE)
#
#       } else{
#         z <- x$ic_all
#         zlab <- x$ic_type
#         x0 <- df_list
#         y0 <- lam_list
#         z0 <- z
#         x1 <- c(df_list[-1], df_list[length(df_list)])
#         y1 <- c(lam_list[-1], lam_list[length(lam_list)])
#         z1 <- c(z[-1], z[length(z)])
#         arrows3D(x0, y0, z0, x1, y1, z1,ticktype="detailed", xlab = "DF", ylab =ifelse(sign.lambda == 1, expression(Log(lambda)), expression(lambda)), zlab = zlab)
#         # text3D(df_list,lam_list,z, labels = 1:length(x$lambda_all),add=TRUE)
#       }
#       # sequential
#     } else{
#       if(sign.lambda == 1){
#         lam_list <- log(x$lambda.list)
#       } else{
#         lam_list <- x$lambda.list
#       }
#       lam_order <- order(lam_list)
#       lam_list <- lam_list[lam_order]
#       df_list <- x$s.list
#       if(x$ic_type == "cv"){
#         z <- x$cvm_all
#         nrz <- nrow(z)
#         ncz <- ncol(z)
#         jet.colors <- colorRampPalette(c("blue", "orangeRed"))
#         nbcol <- 100
#         color <- jet.colors(nbcol)
#         zfacet <- z[-1, -1] + z[-1, -ncz] + z[-nrz, -1] + z[-nrz, -ncz]
#         facetcol <- cut(zfacet, nbcol)
#         persp(df_list, lam_list, z, col = color[facetcol], phi = 30, theta = -30, ticktype = "detailed", xlab = "DF", ylab =ifelse(sign.lambda == 1, expression(Log(lambda)), expression(lambda)), zlab = "mean cross validation error")
#       } else{
#         z <- x$ic_all
#         nrz <- nrow(z)
#         ncz <- ncol(z)
#         jet.colors <- colorRampPalette(c("blue", "orangeRed"))
#         nbcol <- 100
#         color <- jet.colors(nbcol)
#         zfacet <- z[-1, -1] + z[-1, -ncz] + z[-nrz, -1] + z[-nrz, -ncz]
#         facetcol <- cut(zfacet, nbcol)
#         zlab <- x$ic_type
#         z <- z[, lam_order]
#         persp(df_list, lam_list, z, col = color[facetcol], phi = 30, theta = -30, ticktype = "detailed", xlab = "DF", ylab =ifelse(sign.lambda == 1, expression(Log(lambda)), expression(lambda)), zlab = zlab)
#
#        }
#     }
#   }
# }

plot_heatmap <- function(x, sign.lambda){
  # sequential path
  if(x$method == "sequential"){
    #val = ifelse(is.null(x$ic_all), x$cvm_all, x$ic_all)
    if(x$ic_type == "cv"){
      val = x$cvm_all
    } else{
      val = x$ic_all
    }
    if(length(x$lambda.list>15)){
      lambda_col =x$lambda.list
      lambda_col[5*(1:ceiling(length(lambda_col)/5))] = round(lambda_col[5*(1:ceiling(length(lambda_col)/5))], 3)
      lambda_col[-5*(1:ceiling(length(lambda_col)/5))] = ""
      if(sign.lambda)
        colnames(val) = exp(lambda_col)
      else
        colnames(val) = lambda_col
    } else{
      colnames(val) = round(x$lambda.list, 3)
      if(sign.lambda) colnames(val) = exp(colnames(val))
    }
    s_row = s.list
    if(length(s_row)>15){
      s_row[-5*(1:ceiling(length(s_row)/5))] = ""
    }
    rownames(val) = s_row
   # if(is.null(col)) col = heat.colors(nrow(x$ic_all) * ncol(x$ic_all))
    if(x$ic_type == "cv"){
      pheatmap(val, cluster_cols = FALSE, cluster_rows = FALSE, scale="none",  xlab = "lambda", ylab = "model size", main = "Cross-validation error")
    } else{
      main = x$ic_type
      pheatmap(val, cluster_cols = FALSE, cluster_rows = FALSE, scale="none", xlab = "lambda", ylab = "model size", main = main)
    }
    # powell path
  } else{
    df_list <- apply(x$beta_all, 2, function(x){sum(ifelse(abs(x) < 1e-8, 0, 1))})
    ## line search = sequential
    if(x$line.search == "sequential"){
      lambda.list = exp(seq(log(x$lambda.min), log(x$lambda.max),length.out = x$nlambda))
      s.list = x$s.min : x$s.max
      val = x$ic_mat
      # for(i in 1:length(x$lambda_all)){
      #   print(i)
      #   row_ind = which(s.list == df_list[i])
      #   col_ind = which(Isequal(x$lambda_all[i], lambda.list))
      #   # print(paste("col",col_ind))
      #   if(is.na(val[row_ind, col_ind])){
      #     val[row_ind, col_ind] = x$ic_all[i]
      #   } else{
      #     val[row_ind, col_ind] = pmin(val[row_ind, col_ind], x$ic_all[i])
      #   }
      # }

      if(length(lambda.list>15)){
        lambda_col = lambda.list
        lambda_col[5*(1:ceiling(length(lambda_col)/5))] = round(lambda.list[5*(1:ceiling(length(lambda.list)/5))], 3)
        lambda_col[-5*(1:ceiling(length(lambda_col)/5))] = ""
        if(sign.lambda) colnames(val) = exp(lambda_col)
        else colnames(val) = lambda_col
      } else{
        colnames(val) = round(lambda.list, 3)
        if(sign.lambda) colnames(val) = exp(colnames(val))
      }
      s_row = s.list
      if(length(s_row)>15){
        s_row[-5*(1:ceiling(length(s_row)/5))] = ""
      }
      rownames(val) = s_row
      #if(is.null(col)) col =heat.colors(length(x$lambda_all))
      if(x$ic_type == "cv"){
        # heatmap(val, Colv=NA, Rowv = NA, scale="none", col = col, xlab = "lambda", ylab = "model size", main = "Cross-validation error")
        pheatmap(val, cluster_cols = FALSE, cluster_rows = FALSE, scale="none", xlab = "lambda",
                 ylab = "model size", main = "Cross-validation error",na_col = "gray")
        # setHook("grid.newpage", NULL, "replace")
        # grid.text("xlabel example", y=-0.07, gp=gpar(fontsize=16))
        # grid.text("ylabel example", x=-0.07, rot=90, gp=gpar(fontsize=16))
      } else{
        main = x$ic_type
        # heatmap(val, Colv=NA, Rowv = NA ,scale="none", col = col, xlab = "lambda", ylab = "model size", main = main)
        pheatmap(val, cluster_cols = FALSE, cluster_rows = FALSE, scale="none", xlab = "lambda",
                 ylab = "model size", main = main,na_col = "gray")
      }
      ## line search = gsection
    }else{
      lambda.list = exp(seq(log(x$lambda.min), log(x$lambda.max),length.out = 100))
      s.list = x$s.min : x$s.max
      # val = matrix(NA, nrow = length(s.list), ncol = 100)
      # for(i in 1:length(x$lambda_all)){
      #   lower_ind = which(lambda.list <= x$lambda_all[i])[1]
      #   upper_ind = ifelse(lower_ind + 1 > 100, 100, lower_ind + 1)
      #   col_ind = ifelse(abs(lambda.list[lower_ind] - x$lambda_all[i]) < abs(lambda.list[lower_ind] - x$lambda_all[i]), lower_ind, upper_ind)
      #   row_ind = which(s.list == df_list[i])
      #   if(is.na(val[row_ind, col_ind])){
      #     val[row_ind, col_ind] = x$ic_all[i]
      #   } else{
      #     val[row_ind, col_ind] = pmin(val[row_ind, col_ind], x$ic_all[i])
      #   }
      # }
      val = x$ic_mat
      lambda_breaks = matrix(lambda.list, nrow = length(lambda.list))
      lambda_breaks[5*(1:20)] <-  round(lambda_breaks[5*(1:20)], 3)
      lambda_breaks[-(5*(1:20))] <-  ""
      colnames(val) <- lambda_breaks
      if(sign.lambda) colnames(val) = exp(colnames(val))
      s_row = s.list
      if(length(s_row)>15){
        s_row[-5*(1:ceiling(length(s_row)/5))] = ""
      }
      rownames(val) = s_row
      #val[which(is.na(val))] = min(x$ic_all - 10)
      #if(is.null(col)) col = heat.colors(length(x$lambda_all))
      if(x$ic_type == "cv"){
        pheatmap(val, cluster_cols = FALSE, cluster_rows = FALSE, scale="none", #color = col,
                 xlab = "lambda", ylab = "model size", main = "Cross-validation error", na_col = "gray"
                 )
      } else{
        main = x$ic_type
        pheatmap(val, cluster_cols = FALSE, cluster_rows = FALSE, scale="none", #color = col,
                 xlab = "lambda", ylab = "model size", main = main, na_col = "gray"
                 )
      }
    }
  }
}

# plot_heatmap <- function(x, col){
#   # sequential path
#   if(x$method == "sequential"){
#     val = x$ic_all
#     if(length(x$lambda.list>15)){
#       lambda_col = x$lambda.list
#       lambda_col[5*(1:ceiling(length(lambda_col)/5))] = round(x$lambda.list[5*(1:ceiling(length(x$lambda.list)/5))], 3)
#       lambda_col[-5*(1:ceiling(length(lambda_col)/5))] = ""
#       colnames(val) = lambda_col
#     } else{
#       colnames(val) = round(x$lambda.list, 3)
#     }
#     rownames(val) = x$s.list
#     if(is.null(col)) col = heat.colors(nrow(x$ic_all) * ncol(x$ic_all))
#     if(x$ic_type == "cv"){
#       pheatmap(val, cluster_cols = FALSE, cluster_rows = FALSE, scale="none", color = col, xlab = "lambda", ylab = "model size", main = "Cross-validation error")
#     } else{
#       main = x$ic_type
#       pheatmap(val, cluster_cols = FALSE, cluster_rows = FALSE, scale="none", color = col, xlab = "lambda", ylab = "model size", main = main)
#     }
#   # powell path
#   } else{
#     df_list <- apply(x$beta_all, 2, function(x){sum(ifelse(abs(x) < 1e-8, 0, 1))})
#     ## line search = sequential
#     if(x$line.search == "sequential"){
#       lambda.list = exp(seq(log(x$lambda.max), log(x$lambda.min),length.out = x$nlambda))
#       s.list = x$s.min : x$s.max
#       val = matrix(NA, nrow = length(s.list), ncol = x$nlambda)
#       for(i in 1:length(x$lambda_all)){
#         print(i)
#         row_ind = which(s.list == df_list[i])
#         col_ind = which(Isequal(x$lambda_all[i], lambda.list))
#         # print(paste("col",col_ind))
#         if(is.na(val[row_ind, col_ind])){
#           val[row_ind, col_ind] = x$ic_all[i]
#         } else{
#           val[row_ind, col_ind] = pmin(val[row_ind, col_ind], x$ic_all[i])
#         }
#       }
#
#       if(length(lambda.list>15)){
#         lambda_col = lambda.list
#         lambda_col[5*(1:ceiling(length(lambda_col)/5))] = round(lambda.list[5*(1:ceiling(length(lambda.list)/5))], 3)
#         lambda_col[-5*(1:ceiling(length(lambda_col)/5))] = ""
#         colnames(val) = lambda_col
#       } else{
#         colnames(val) = round(lambda.list, 3)
#       }
#       rownames(val) = s.list
#       if(is.null(col)) col =heat.colors(length(x$lambda_all))
#       if(x$ic_type == "cv"){
#         # heatmap(val, Colv=NA, Rowv = NA, scale="none", col = col, xlab = "lambda", ylab = "model size", main = "Cross-validation error")
#         pheatmap(val, cluster_cols = FALSE, cluster_rows = FALSE, scale="none", color = col, xlab = "lambda",
#                  ylab = "model size", main = "Cross-validation error", na_col = "white")
#       } else{
#         main = x$ic_type
#         # heatmap(val, Colv=NA, Rowv = NA ,scale="none", col = col, xlab = "lambda", ylab = "model size", main = main)
#         pheatmap(val, cluster_cols = FALSE, cluster_rows = FALSE, scale="none", color = col, xlab = "lambda",
#                  ylab = "model size", main = main, na_col = "white")
#       }
#     ## line search = gsection
#     }else{
#       lambda.list = exp(seq(log(x$lambda.max), log(x$lambda.min),length.out = 100))
#       s.list = x$s.min : x$s.max
#       val = matrix(NA, nrow = length(s.list), ncol = 100)
#       for(i in 1:length(x$lambda_all)){
#         lower_ind = which(lambda.list <= x$lambda_all[i])[1]
#         upper_ind = ifelse(lower_ind + 1 > 100, 100, lower_ind + 1)
#         col_ind = ifelse(abs(lambda.list[lower_ind] - x$lambda_all[i]) < abs(lambda.list[lower_ind] - x$lambda_all[i]), lower_ind, upper_ind)
#         row_ind = which(s.list == df_list[i])
#         if(is.na(val[row_ind, col_ind])){
#           val[row_ind, col_ind] = x$ic_all[i]
#         } else{
#           val[row_ind, col_ind] = pmin(val[row_ind, col_ind], x$ic_all[i])
#         }
#       }
#       lambda_breaks = matrix(lambda.list, nrow = length(lambda.list))
#       lambda_breaks[5*(1:20)] <-  round(lambda_breaks[5*(1:20)], 3)
#       lambda_breaks[-(5*(1:20))] <-  ""
#       colnames(val) <- lambda_breaks
#       rownames(val) = s.list
#       #val[which(is.na(val))] = min(x$ic_all - 10)
#       if(is.null(col)) col = heat.colors(length(x$lambda_all))
#       if(x$ic_type == "cv"){
#         pheatmap(val, cluster_cols = FALSE, cluster_rows = FALSE, scale="none", color = col,
#                  xlab = "lambda", ylab = "model size", main = "Cross-validation error", na_col = "white")
#       } else{
#         main = x$ic_type
#         pheatmap(val, cluster_cols = FALSE, cluster_rows = FALSE, scale="none", color = col,
#                  xlab = "lambda", ylab = "model size", main = main, na_col = "white")
#       }
#     }
#   }
# }

# Isequal <- function(x, y){
#   return(ifelse(abs(x-y)<1e-5, TRUE, FALSE))
# }

