library("tidyverse")


# navigate to the directory where the folders "Equal" and "Unequal2" are located


Name = c("dsep", "asia", "bowling", "inssmall", "rain")


show_results = function(path = "./Equal/") {
  RS = list()
  for (data_name in Name){
    data = rbind(
      read.csv(paste0(path, data_name, "/Python_outputs.csv")),
      read.csv(paste0(path, data_name, "/R_outputs.csv"))
    ) %>%
      mutate(method = factor(method, levels = c("NPVAR", "eqvar_TD", "eqvar_BU", "NoTears", "RESIT", "CCDr", "CAM", "MIP (super)", "MIP (moral)", "Bootstrapping")),
             method = fct_recode(method, "eqvar (TD)" = "eqvar_TD", "eqvar (BU)" = "eqvar_BU")) %>%
      group_by(method) %>%
      summarise(
        sd_diff = round(sd(diff), 1),
        sd_time = round(sd(time), 1),
        diff = round(mean(diff), 1),
        time = round(mean(time), 1),
        .groups = "keep") %>%
      dplyr::select(diff, sd_diff, time, sd_time) %>% t()
    
    RS[data_name] = list(data)
  }
  return(RS)
}

# b0 = 1: "./Unequal1/"
# b0 = 2: "./Unequal2/"
# b0 = 3: "./Unequal3/"
# equal:  "./Equal/"

tab_rs = show_results("./Equal/")