library(tidyverse)
library(patchwork)

colnames = c("n", "m", "var", "method", "rec", "SHD", "func")
tab = rbind(
  read.csv("./results_shd/0.5x^3.csv") %>% `colnames<-`(colnames),
  read.csv("./results_shd/0.5x^3_python.csv") %>% `colnames<-`(colnames),
  read.csv("./results_shd/sin(x).csv") %>% `colnames<-`(colnames),
  read.csv("./results_shd/sin(x)_python.csv") %>% `colnames<-`(colnames),
  read.csv("./results_shd/atan(x^2).csv") %>% `colnames<-`(colnames),
  read.csv("./results_shd/atan(x^2)_python.csv") %>% `colnames<-`(colnames),
  read.csv("./results_shd/abs(x).csv") %>% `colnames<-`(colnames),
  read.csv("./results_shd/abs(x)_python.csv") %>% `colnames<-`(colnames),
  read.csv("./results_shd/exp(x).csv") %>% `colnames<-`(colnames),
  read.csv("./results_shd/exp(x)_python.csv") %>% `colnames<-`(colnames)
) 


p1 = tab %>%
  filter(!(method %in% c("Mips_equal", "Mips_unequal"))) %>%
  mutate(rec = case_when(
    rec %in% c("0", "FALSE", "False") ~ 0,
    rec %in% c("1", "TRUE", "True") ~ 1
  )) %>%
  group_by(n, var, method, func) %>%
  summarise(rec = mean(rec)) %>%
  mutate(var = recode(var,
                      "equal" = "Equal variances",
                      "unequal" = "Unequal variances"),
         method = factor(method,
                         levels = c("CAM", "NPVAR", "linear", "Mips_unequal", "Mips_equal", "Mips"),
                         labels = c("CAM-IncEdge", "NPVAR", "MIP-linear", 
                                    "MIP-nonlinear (our approach)", "MIP-nonlinear (our approach)", "MIP-nonlinear (our approach)")),
         func = factor(func, 
                       levels = c("sin(x)", "0.5x^3", "log(x^2)", "atan(x^2)", "abs(x)", "2cos(x)", "0.8x^2", "exp(x)"),
                       labels = c("h(x)==sin(x)", "h(x)==0.5*x^3", "h(x)==log(x^2)", 
                                  "h(x)==arctan(x^2)", "h(x)==abs(x)", "h(x)==2*cos(x)", "h(x)==0.8*x^2", "h(x)==exp(x)"))
  ) %>%
  ggplot(aes(x = n, y = rec, col = method, shape = method)) +
  geom_point(size = 1.2) +
  # geom_line() +
  facet_grid(rows = vars(var), cols = vars(func), labeller = labeller(func = label_parsed)) +
  scale_shape_manual(
    values = c("CAM-IncEdge" = 7, "NPVAR" = 9, "MIP-linear" = 8, "MIP-nonlinear (our approach)" = 19),
  ) +
  labs(x = "Sample size n", y = "P (correct recovery)", col = "Method", shape = "Method") +
  theme_bw() +
  theme(legend.position = "bottom",
        legend.text = element_text(size = 10), 
        plot.title = element_text(size = 10),  # Title size
        axis.title.x = element_text(size = 10),  # x-axis label size
        axis.title.y = element_text(size = 10)   # y-axis label size
  )


p2 = tab %>%
  filter(!(method %in% c("Mips_equal", "Mips_unequal"))) %>%
  group_by(n, var, method, func) %>%
  summarise(SHD = mean(SHD)) %>%
  mutate(var = recode(var,
                      "equal" = "Equal variances",
                      "unequal" = "Unequal variances"),
         method = factor(method,
                         levels = c("CAM", "NPVAR", "linear", "Mips_unequal", "Mips_equal", "Mips"),
                         labels = c("CAM-IncEdge", "NPVAR", "MIP-linear", 
                                    "MIP-nonlinear (our approach)", "MIP-nonlinear (our approach)", "MIP-nonlinear (our approach)")),
         func = factor(func, 
                       levels = c("sin(x)", "0.5x^3", "log(x^2)", "atan(x^2)", "abs(x)", "2cos(x)", "0.8x^2", "exp(x)"),
                       labels = c("h(x)==sin(x)", "h(x)==0.5*x^3", "h(x)==log(x^2)", 
                                  "h(x)==arctan(x^2)", "h(x)==abs(x)", "h(x)==2*cos(x)", "h(x)==0.8*x^2", "h(x)==exp(x)"))
  ) %>%
  ggplot(aes(x = n, y = SHD, col = method, shape = method)) +
  geom_point(size = 1.2) +
  # geom_line() +
  facet_grid(rows = vars(var), cols = vars(func), labeller = labeller(func = label_parsed)) +
  scale_shape_manual(
    values = c("CAM-IncEdge" = 7, "NPVAR" = 9, "MIP-linear" = 8, "MIP-nonlinear (our approach)" = 19),
  ) +
  labs(x = "Sample size n", y = "SHD d", col = "Method", shape = "Method") +
  theme_bw() +
  theme(legend.position = "bottom",
        legend.text = element_text(size = 10), 
        plot.title = element_text(size = 10),  # Title size
        axis.title.x = element_text(size = 10),  # x-axis label size
        axis.title.y = element_text(size = 10)   # y-axis label size
  )

(p1 / p2) +
  plot_layout(guides = "collect") &
  theme(legend.position = "bottom")
  
