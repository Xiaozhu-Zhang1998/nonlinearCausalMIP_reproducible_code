library(tidyverse)

# navigate to the directory where "Python_outputs.csv" and "R_outputs.csv" are located


rbind(
  read.csv("Python_outputs.csv"),
  read.csv("R_outputs.csv")
) %>%
  filter(method != "Bootstrapping") %>%
  mutate(method = factor(method, levels = c("NPVAR", "eqvar_TD", "eqvar_BU", "NoTears", "RESIT", "CCDr", "CAM", "MIP (super)", "MIP (moral)")),
         method = fct_recode(method, "EqVar (TD)" = "eqvar_TD", "EqVar (BU)" = "eqvar_BU")) 
