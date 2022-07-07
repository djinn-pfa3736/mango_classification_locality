Exp20200609 <- function() {

  source("diff_each.R")

  lab.a <- read.csv("lab_a.csv")
  lab.b <- read.csv("lab_b.csv")
  lab.c <- read.csv("lab_c.csv")

  # lab.data <- rbind(diff_each(lab.a[,2:4]), diff_each(lab.b[,2:4]), diff_each(lab.c[,2:4]))
  # lab.pca.res <- prcomp(lab.data)  

  lab.data <- rbind(lab.a, lab.b, lab.c)
  lab.pca.res <- prcomp(lab.data[, 2:4])

  plot(lab.pca.res$x[,1:2], col=c(rep("red", 50), rep("green", 55), rep("blue", 9)), pch=c(rep(1, 50), rep(2, 55), rep(3, 9)))

  return(lab.pca.res)
}
