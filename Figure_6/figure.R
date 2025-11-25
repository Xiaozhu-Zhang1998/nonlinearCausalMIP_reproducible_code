library(tidyverse)
library(patchwork)
library(latex2exp)

# navigate to the directory where the folders "Equal", "Unequal3", "Unequal2" and "Unequal1" are located


df = matrix(0, nrow = 0, ncol = 7) %>% data.frame()
for(data_name in c("bowling", "rain", "insurance")) {
  equalvar_data = 
    rbind(
      read.csv(paste0("./Equal/", data_name, "/Python_outputs.csv")),
      read.csv(paste0("./Equal/", data_name, "/R_outputs.csv"))
    ) %>%
    mutate(var = "equal", data_name = data_name)
  
  diffvar3_data = 
    rbind(
      read.csv(paste0("./Unequal3/", data_name, "/Python_outputs.csv")),
      read.csv(paste0("./Unequal3/", data_name, "/R_outputs.csv"))
    ) %>%
    mutate(var = "diff3", data_name = data_name)
  
  diffvar2_data = 
    rbind(
      read.csv(paste0("./Unequal2/", data_name, "/Python_outputs.csv")),
      read.csv(paste0("./Unequal2/", data_name, "/R_outputs.csv"))
    ) %>%
    mutate(var = "diff2", data_name = data_name)
  
  
  diffvar1_data = 
    rbind(
      read.csv(paste0("./Unequal1/", data_name, "/Python_outputs.csv")),
      read.csv(paste0("./Unequal1/", data_name, "/R_outputs.csv"))
    ) %>%
    mutate(var = "diff1", data_name = data_name)
  
  df = rbind(
    df,
    diffvar1_data,
    diffvar2_data,
    diffvar3_data,
    equalvar_data
  )
}


df = df %>%
  filter(method != "Bootstrapping") %>%
  mutate(method = factor(method, levels = c("NPVAR", "eqvar_TD", "eqvar_BU", "NoTears", "RESIT", "CCDr", "CAM", "MIP (super)", "MIP (moral)")),
         method = fct_recode(method, "EqVar (TD)" = "eqvar_TD", "EqVar (BU)" = "eqvar_BU"),
         data_name = factor(data_name, levels = c("bowling", "rain", "insurance")),
         data_name = fct_recode(data_name, "Bowling. 9.11" = "bowling", "Rain. 14.18" = "rain", "Insurance. 27.52" = "insurance")
  )


ggplot(df, aes(x = method, y = diff, col = var)) +
  geom_boxplot(width = 0.7) +
  facet_grid(data_name ~ var, scales = "free_y") +
  scale_color_discrete(
    name = "",  # legend title
    labels = c(
      "diff1" = TeX("Unequal variances $\\mu_0 = 1$"),
      "diff2" = TeX("Unequal variances $\\mu_0 = 2$"),
      "diff3" = TeX("Unequal variances $\\mu_0 = 3$"),
      "equal" = TeX("Equal variances")
    )) +
  labs(x = "", y = TeX("SHD $d$")) + 
  theme_bw() +
  theme(
    strip.text.x = element_blank(),     # remove text in column strips
    strip.background.x = element_blank(),
    legend.position = "bottom",
    axis.text.x = element_text(angle = 45, hjust = 1),
    axis.title.x = element_text(size = 10),    # x-axis label
    axis.title.y = element_text(size = 10),    # y-axis label
    legend.title  = element_text(size = 10),   # legend title
    legend.text   = element_text(size = 10),
    plot.title    = element_text(size = 10)
  )

