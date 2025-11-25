library(tidyverse)
library(patchwork)
library(latex2exp)

# navigate to the directory where the "performThres_outputs_fixed_stab.csv" and "performThres_outputs_varied_stab.csv" are located

tab <- read_csv("./performThres_outputs_fixed_stab.csv")

p1 = tab %>%
  mutate(thres_par = factor(thres_par)) %>%
  ggplot(aes(x = thres_par, y = diff)) +
  geom_boxplot() +
  labs(x = TeX("Partial set threshold $\\tau^p$"), y = "SHD d", 
       title = TeX("Stable set threshold $\\tau^s = 1$")) +
  theme_bw()  +
  theme(
    plot.title = element_text(size = 10),  # Title size
    axis.title.x = element_text(size = 10),  # x-axis label size
    axis.title.y = element_text(size = 10)   # y-axis label size
  )


p2= tab %>%
  mutate(thres_par = factor(thres_par)) %>%
  ggplot(aes(x = thres_par, y = time)) +
  geom_boxplot() +
  labs(x = TeX("Partial set threshold $\\tau^p$"), y = "Running time", 
       title = TeX("Stable set threshold $\\tau^s = 1$")) +
  theme_bw()  +
  theme(
    plot.title = element_text(size = 10),  # Title size
    axis.title.x = element_text(size = 10),  # x-axis label size
    axis.title.y = element_text(size = 10)   # y-axis label size
  )


tab <- read_csv("./performThres_outputs_varied_stab.csv")

p3 = tab %>%
  mutate(thres_par = factor(thres_par)) %>%
  ggplot(aes(x = thres_par, y = diff)) +
  geom_boxplot() +
  labs(x = TeX("Partial set threshold $\\tau^p$"), y = "SHD d", 
       title = TeX("Stable set threshold $\\tau^s$ = partial set threshold $\\tau^p$")) +
  theme_bw()  +
  theme(
    plot.title = element_text(size = 10),  # Title size
    axis.title.x = element_text(size = 10),  # x-axis label size
    axis.title.y = element_text(size = 10)   # y-axis label size
  )


p4 = tab %>%
  mutate(thres_par = factor(thres_par)) %>%
  ggplot(aes(x = thres_par, y = time)) +
  geom_boxplot() +
  labs(x = TeX("Partial set threshold $\\tau^p$"), y = "Running time", 
       title = TeX("Stable set threshold $\\tau^s$ = partial set threshold $\\tau^p$")) +
  theme_bw()  +
  theme(
    plot.title = element_text(size = 10),  # Title size
    axis.title.x = element_text(size = 10),  # x-axis label size
    axis.title.y = element_text(size = 10)   # y-axis label size
  )


p1 + p2 + p3 + p4
