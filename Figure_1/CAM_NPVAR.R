rm(list = ls())
source('mydir/NPVAR/NPVAR.R')
source('mydir/NPVAR/utils.R')


# repeat the following for n in N = seq(from = 100, to = 2000, length.out = 20)
# repeat each n 100 times

# example of s0 = 100, m = 1 (m is the index for repetition), and h(x) = exp(x)

n <- 100
m <- 1


# ======================================================================


edges_star = matrix(c(0,0,0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0),5)

FUNS = list(
  function(x) {
    0.5 * x^3
  },
  function(x) {
    sin(x)
  },
  function(x) {
    atan(x^2)
  },
  function(x) {
    abs(x)
  },
  function(x) {
    exp(x)
  }
)

func_name = "exp(x)"
h = FUNS[[7]]


N = seq(from = 100, to = 2000, length.out = 20)
M = 100

RS = matrix(0, nrow = 0, ncol = 5)

# for equal
x1 = rnorm(n, 0, 0.5)
x2 = x1^2 - mean(x1^2) + rnorm(n, 0, 0.5)
x3 = 2 * x1^2 + h(x2) - mean(2 * x1^2) - mean(h(x2))  + rnorm(n, 0, 0.5)
x4 = rnorm(n, 0, 0.5)
x5 = rnorm(n, 0, 0.5)
X = cbind(x1, x2, x3, x4, x5)
name = paste0("./data/equal_", n, "_", m, ".csv")
write.table(X, name, col.names = FALSE, row.names = FALSE)
cam_obj = CAM::CAM(X, pruning = TRUE)
rec_cam = all(cam_obj$Adj == edges_star)
shd_cam = sum(abs(edges_star - cam_obj$Adj))
RS = rbind(RS, c(n, m, "equal", "CAM", rec_cam, shd_cam))

npvar_obj = NPVAR(x = X)
rec_npvar = all(prune(X, npvar_obj) == edges_star)
shd_npvar = sum(abs(edges_star - prune(X, npvar_obj)))
RS = rbind(RS, c(n, m, "equal", "NPVAR", rec_npvar, shd_npvar))


# for unequal
x1 = rnorm(n, 0, 0.5)
x2 = x1^2 - mean(x1^2) + rnorm(n, 0, 0.1)
x3 = 2 * x1^2 + h(x2) - mean(2 * x1^2) - mean(h(x2))  + rnorm(n, 0, 0.3)
x4 = rnorm(n, 0, 0.5)
x5 = rnorm(n, 0, 0.5)
X = cbind(x1, x2, x3, x4, x5)
name = paste0("./data/unequal_", n, "_", m, ".csv")
write.table(X, name, col.names = FALSE, row.names = FALSE)

cam_obj = CAM::CAM(X)
rec_cam = all(cam_obj$Adj == edges_star)
shd_cam = sum(abs(edges_star - cam_obj$Adj))
RS = rbind(RS, c(n, m, "unequal", "CAM", rec_cam, shd_cam))

npvar_obj = NPVAR(x = X)
rec_npvar = all(prune(X, npvar_obj) == edges_star)
shd_npvar = sum(abs(edges_star - prune(X, npvar_obj)))
RS = rbind(RS, c(n, m, "unequal", "NPVAR", rec_npvar, shd_npvar))



RS = cbind(RS, func_name)


name = paste0("./results/", n, "_", m, ".RDS")
saveRDS(RS, name)



# collect results ========
# collect all results (including all n and 100 trials) into a data frame called "exp(x).csv"
# The column names are:
# [1] "n"      "m"      "var"    "method" "rec"    "SHD"    "func" 
