library(readr)
library(SIS)

################## Duke Breast Cancer ##################
duke_breast_cancer <- read_table2("F:/pdas_l0l2/bess/data/duke-breast-cancer.txt/duke-breast-cancer.txt",
                                  col_names = FALSE)
duke_breast_cancer <- as.data.frame(duke_breast_cancer)

sis_fit <- SIS(as.matrix(duke_breast_cancer[, -1]), as.vector(duke_breast_cancer[, 1]),
               family = "binomial", iter = FALSE, nsis = 500)

duke <- duke_breast_cancer[, sis_fit[["sis.ix0"]]]
duke <- cbind.data.frame("y" = duke_breast_cancer[, 1], duke)
usethis::use_data(duke, duke, overwrite = TRUE)


################## Trim 32 dataset ##################
trim32 <- readRDS("F:\\pdas_l0l2\\bess\\data\\trim32.rds")

sis_fit <- SIS(as.matrix(trim32[["x"]]), as.vector(trim32[["y"]]),
               family = "gaussian", iter = FALSE, nsis = 500)

trim32[["x"]] <- trim32[["x"]]
trim32_x <- trim32[["x"]][, sis_fit[["sis.ix0"]]]

trim32 <- cbind.data.frame("y" = trim32[["y"]], trim32_x)
usethis::use_data(trim32, trim32, overwrite = TRUE)
