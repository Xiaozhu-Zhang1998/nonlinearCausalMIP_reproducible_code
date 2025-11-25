source('mydir/NPVAR/NPVAR.R')
source('mydir/NPVAR/utils.R')
source('mydir/EqVarDAG/source.R')
source('mydir/RESIT/source.R')
source('functions.R')

# specify the data_name and variance scheme
# repeat the following for m in M = 0:29
# example of m = 1, structure `dsep` and equal-variance

m = 1
data_name = "dsep"
equalvar = TRUE


# ======================================================================


# put the true edge set as "data_name.txt" under the ./dataset folder
edges_file <- read.csv(paste0("./dataset/", data_name, ".txt", collapse = ""), header = FALSE)
vec_list <- apply(edges_file, 1, function(row) c(row[1], row[2]))
edges_star <- lapply(1:nrow(edges_file), function(i) c(edges_file[i, 1], edges_file[i, 2]))


if (equalvar){
    name = paste0("./dataset/", data_name, "_equalvar/W_", m, ".csv")
} else {
    name = paste0("./dataset/", data_name, "_diffvar/W_", m, ".csv")
}
W = as.matrix(read.csv(name, header = FALSE))


W_train = W[1:500,]
p = ncol(W)

RS = matrix(0, nrow = 0, ncol = 5)
cat("finished all setup!\n")


# ======================================================================


if (equalvar) {

    cat("started bootstrapping NPVAR...\n")
    t1_bootstrap = Sys.time()

    Adj = matrix(0, p, p)
    superset = list()
    B = 20
    for (b in 1:B) {
        idx = sample(1:nrow(W_train), size = nrow(W_train), replace = TRUE)
        W_idx = W_train[idx,]
        eta = 0.00001
        while(TRUE) {
            result1 = NPVAR(x = W_idx, layer.select = TRUE, eta = eta)
            if (length(result1$layers) == p) {
                break
            } else {
                eta = eta / 2
            }
        }
        Cutoff = exp(seq(from = log(0.1), to = log(0.00001), length.out = 20))
        Loss_val = c()
        for (cutoff in Cutoff) {
        est = prune(W_idx, result1$ancestors, cutoff = cutoff)
        loss = loss_from_adj(est, W_train, W_train)
        Loss_val = c(Loss_val, loss + sum(est) * log(length(idx)) / length(idx) )
        }

        idx = which.min(Loss_val)
        cutoff = Cutoff[idx]
        est = prune(W_idx, result1$ancestors, cutoff = cutoff)
        Adj = Adj + est

        cat("finished b=", b, "out of", B, "\n")
    }
    
    prob_adj = Adj / B
    t2_bootstrap = Sys.time()
    cat("finished bootstrapping NPVAR...\n\n")


    # record adj matrix
    if (equalvar) {
        name = paste0("./dataset/", data_name, "_equalvar/prob_adj_NPVAR_", m, ".csv")
    } else {
        name = paste0("./dataset/", data_name, "_diffvar/prob_adj_NPVAR_", m, ".csv")
    }
    write.table(
        prob_adj, name,
        sep = ",", col.names = FALSE, row.names = FALSE
    )


    # record bootstrapping time
    rs = c(
        m, p, "Bootstrapping",
        NA,
        difftime(t2_bootstrap, t1_bootstrap, units = "secs")
    )
    RS = rbind(RS, rs)

}




# ======================================================================




## 1. NPVAR ====
cat("Started NPVAR....\n")
eta = 0.00001
while(TRUE) {
    result1 = NPVAR(x = W_train, layer.select = TRUE, eta = eta)
    if (length(result1$layers) == p) {
      break
    } else {
      eta = eta / 2
    }
}


Cutoff = exp(seq(from = log(0.1), to = log(0.00001), length.out = 10))
Loss_val = c()
for (cutoff in Cutoff) {
    est = prune(W_train, result1$ancestors, cutoff = cutoff)
    Loss_val = c(Loss_val, loss_from_adj(est, W_train, W_train) + sum(est) * log(nrow(W_train)) / nrow(W_train) )
}
idx = which.min(Loss_val)
cutoff = Cutoff[idx]

t1 = Sys.time()
result1 = NPVAR(x = W_train, layer.select = TRUE, eta = eta)
est = prune(W_train, result1$ancestors, cutoff = cutoff)
t2 = Sys.time()


# record
rs = c(
    m, p, "NPVAR",
    edge_diff_npvar(edges_star, est, p),
    difftime(t2, t1, units = "secs")
)
RS = rbind(RS, rs)


loss = loss_from_adj(est, W_train, W_train)
Loss_NPVAR = loss + sum(est) * log(nrow(W_train)) / nrow(W_train)



if (equalvar) {
    name = paste0("./dataset/", data_name, "_equalvar/Loss_NPVAR_", m, ".txt")
} else {
    name = paste0("./dataset/", data_name, "_diffvar/Loss_NPVAR_", m, ".txt")
}
readr::write_file(
    as.character(Loss_NPVAR),
    name
)


if (equalvar) {
    name = paste0("./dataset/", data_name, "_equalvar/s0_NPVAR_", m, ".txt")
    readr::write_file(
        as.character(sum(est)),
        name
    )
}

cat("Finished NPVAR....\n\n")




## 2. eqvarDAG_TD ====
cat("Started eqvarDAG TD....\n")
## eqvarDAG_TD ====
Loss_val = c()
for (cutoff in Cutoff) {
    t1 = Sys.time()
    EqVarDAG_TD_obj = EqVarDAG_TD(W_train, mtd = "cvlasso", alpha = cutoff)
    t2 = Sys.time()
    Loss_val = c(Loss_val, loss_from_adj(EqVarDAG_TD_obj$adj, W_train, W_train) + sum(EqVarDAG_TD_obj$adj) * log(nrow(W_train)) / nrow(W_train) )
    cat("cutoff:", cutoff, difftime(t2, t1, units = "secs"), "\n")
}
idx = which.min(Loss_val)
cutoff = Cutoff[idx]
  
t1 = Sys.time()
EqVarDAG_TD_obj = EqVarDAG_TD(W_train, mtd = "cvlasso", alpha = cutoff)
t2 = Sys.time()

# record
rs = c(
    m, p, "eqvar_TD",
    edge_diff_npvar(edges_star, EqVarDAG_TD_obj$adj, p),
    difftime(t2, t1, units = "secs")
)
RS = rbind(RS, rs)
cat("Finished NPVAR....\n\n")




## 3. eqvarDAG_BU ====
cat("Started eqvarDAG BU....\n")
Loss_val = c()
for (cutoff in Cutoff) {
    t1 = Sys.time()
    EqVarDAG_BU_obj = EqVarDAG_BU(W_train, mtd = "cvlasso", alpha = cutoff)
    t2 = Sys.time()
    Loss_val = c(Loss_val, loss_from_adj(EqVarDAG_BU_obj$adj, W_train, W_train) + sum(EqVarDAG_BU_obj$adj) * log(nrow(W_train)) / nrow(W_train) )
    cat("cutoff:", cutoff, difftime(t2, t1, units = "secs"), "\n")
}
idx = which.min(Loss_val)
cutoff = Cutoff[idx]
  
t1 = Sys.time()
EqVarDAG_BU_obj = EqVarDAG_BU(W_train, mtd = "cvlasso", alpha = cutoff)
t2 = Sys.time()
  
# record
rs = c(
    m, p, "eqvar_BU",
    edge_diff_npvar(edges_star, EqVarDAG_BU_obj$adj, p),
    difftime(t2, t1, units = "secs")
)
RS = rbind(RS, rs)
cat("Finished eqvarDAG BU....\n\n")




if (equalvar) {
    name = paste0("./dataset/", data_name, "_equalvar/R_baselines_", m, ".rds")
} else {
    name = paste0("./dataset/", data_name, "_diffvar/R_baselines_", m, ".rds")
}
saveRDS(RS, file = name)



## 4. RESIT ====
cat("Started RESIT....\n")
Loss_val = c()
for (cutoff in Cutoff) {
    t1 = Sys.time()
    RESIT_obj = ICML(W_train, alpha = cutoff, model = train_gam)
    t2 = Sys.time()
    Loss_val = c(Loss_val, loss_from_adj(RESIT_obj, W_train, W_train) + sum(RESIT_obj) * log(nrow(W_train)) / nrow(W_train)  )
    cat("cutoff:", cutoff, difftime(t2, t1, units = "secs"), "\n")
}
idx = which.min(Loss_val)
cutoff = Cutoff[idx]
  
t1 = Sys.time()
RESIT_obj = ICML(W_train, alpha = cutoff, model = train_gam)
t2 = Sys.time()
  
# record
rs = c(
    m, p, "RESIT",
    edge_diff_npvar(edges_star, RESIT_obj, p),
    difftime(t2, t1, units = "secs")
)
RS = rbind(RS, rs)
cat("Finished RESIT....\n\n")




cat("finished all R baselines!\n\n")

if (equalvar) {
    name = paste0("./dataset/", data_name, "_equalvar/R_baselines_", m, ".rds")
} else {
    name = paste0("./dataset/", data_name, "_diffvar/R_baselines_", m, ".rds")
}
saveRDS(RS, file = name)



# ======================================================================

cat("\n\n")
cat("finished...")
