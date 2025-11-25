# specify the data_name and variance scheme
# example of structure `dsep` and equal-variance


data_name = "dsep"
equalvar = TRUE


# ======================================================================

M = 30
RS = matrix(0, nrow = 0, ncol = 5)
for(m in (1:M-1)) {
    if (equalvar) {
        name = paste0("./dataset/", data_name, "_equalvar/R_baselines_", m, ".rds")
    } else {
        name = paste0("./dataset/", data_name, "_diffvar/R_baselines_", m, ".rds")
    }
    rs = readRDS(name)
    RS = rbind(RS, rs)
}


if (equalvar) {
    name = paste0("./dataset/", data_name, "_equalvar/R_outputs", ".csv")
} else {
    name = paste0("./dataset/", data_name, "_diffvar/R_outputs", ".csv")
}

RS_df = data.frame(RS, row.names = NULL)
colnames(RS_df) = c("m", "p", "method", "diff", "time")
write.csv(RS_df, file = name, row.names = FALSE, col.names = TRUE)


cat("finished...")
