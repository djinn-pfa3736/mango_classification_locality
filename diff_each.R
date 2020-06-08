diff_each <- function(data.x) {

  sub1 <- data.x[seq(2, nrow(data.x), 2),]
  sub2 <- data.x[seq(1, nrow(data.x), 2),]

  diff <- (sub1 + sub2)/2

  # browser()
  return(diff)

}
