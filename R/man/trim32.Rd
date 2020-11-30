\name{trim32}
\alias{trim32}
\docType{data}
\title{
trim32 data set
}
\description{
the data set reported in Scheetz Kim et al, which contains gene expression levels of 18976 probes obtained from 120 rats. The gene trim32 (1389163_at) had been known to cause Bardet-Biedl syndrome, which is a genetic disease of multiple organ systems including the retina
}
\usage{data("trim32")}
\format{
  A data frame with 120 observations on the following 2 variables.
  \describe{
    \item{\code{x}}{expression of 18975 probes predictors}
    \item{\code{y}}{the expression level of TRIM32}
  }
}
\references{
Scheetz T E, Kim K Y A, Swiderski R E, et al. Regulation of gene expression in the mammalian eye and its relevance to eye disease[J]. Proceedings of the National Academy of Sciences, 2006, 103(39): 14429-14434.
}
\examples{
data(trim32)
}
\keyword{datasets}
