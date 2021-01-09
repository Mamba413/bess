#' @title The Bardet-Biedl syndrome Gene expression data
#' @name trim32
#' @docType data
#' @description Gene expression data (500 gene probes for 120 samples) from the microarray experiments of mammalianeye tissue samples of Scheetz et al. (2006).
#'
#' @details In this study, laboratory rats (Rattus norvegicus) were studied to learn about gene expression and regulation in the mammalian eye.
#' Inbred rat strains were crossed and tissue extracted from the eyes of 120 animals from the F2 generation. Microarrays were used to measure levels of RNA expression in the isolated eye tissues of each subject.
#' Of the 31,000 different probes, 18,976 were detected at a sufficient level to be considered expressed in the mammalian eye.
#' For the purposes of this analysis, we treat one of those genes, Trim32, as the outcome.
#' Trim32 is known to be linked with a genetic disorder called Bardet-Biedl Syndrome (BBS): the mutation (P130S) in Trim32 gives rise to BBS.
#'
#' @note This data set contains 120 samples with 500 predictors. The 500 predictors are features with maximum marginal correlation to Trim32 gene.
#'
#' @format A data frame with 120 rows and 501 variables, where the first variable is the expression level of TRIM32 gene,
#' and the remaining 500 variables are 500 gene probes.
#'
#' @references T. Scheetz, k. Kim, R. Swiderski, A. Philp, T. Braun, K. Knudtson, A. Dorrance, G. DiBona, J. Huang, T. Casavant, V. Sheffield, E. Stone. Regulation of gene expression in the mammalian eye and its relevance to eye disease. Proceedings of the National Academy of Sciences of the United States of America, 2006.
#'
#' @source \url{https://www.ncbi.nlm.nih.gov/geo} (accession no. GSE5680)
NULL



#' @title Duke breast cancer data
#' @name duke
#' @docType data
#' @description This data set details microarray experiment for breast cancer patients.
#'
#' @details The binary variable Status is used to classify the patients into
#' estrogen receptor-positive (y = 0) and estrogen receptor-negative (y = 1).
#' The other variables contain the expression level of the considered genes.
#'
#' @format A data frame with 86 rows and 501 variables,
#' where the first variable is the label of estrogen receptor-positive/negative,
#' and the remaining 500 variables are 500 gene.
#'
#' @references M. West, C. Blanchette, H. Dressman, E. Huang, S. Ishida, R. Spang, H. Zuzan, J.A. Olson, Jr., J.R. Marks and Joseph R. Nevins (2001) <doi:10.1073/pnas.201162998> Predicting the clinical status of human breast cancer by using gene expression profiles, Proceedings of the National Accademy of Sciences of the USA, Vol 98(20), 11462-11467.
#'
#' @source \url{www.kaggle.com/andreicosma/duke-breast-cancer-dataset}
NULL
