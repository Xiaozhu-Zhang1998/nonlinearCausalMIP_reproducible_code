library(tidyverse)
library(patchwork)
library(latex2exp)

# navigate to the directory where the "early_stopping.csv" is located


tab = read.csv("early_stopping.csv")

p1 = tab %>%
  mutate(tau = factor(tau)) %>%
  ggplot(aes(x = tau, y = diff)) +
  geom_boxplot() +
  labs(x = TeX("Early stopping threshold $\\tau^{early}$"), y = "SHD d") +
  theme_bw()  +
  theme(
    plot.title = element_text(size = 10),  # Title size
    axis.title.x = element_text(size = 10),  # x-axis label size
    axis.title.y = element_text(size = 10)   # y-axis label size
  )


p2 = tab %>%
  mutate(tau = factor(tau)) %>%
  ggplot(aes(x = tau, y = time)) +
  geom_boxplot() +
  labs(x = TeX("Early stopping threshold $\\tau^{early}$"), y = "Running time") +
  scale_y_continuous(breaks = seq(0, max(tab$time), by = 125)) +
  theme_bw()  +
  theme(
    plot.title = element_text(size = 10),  # Title size
    axis.title.x = element_text(size = 10),  # x-axis label size
    axis.title.y = element_text(size = 10)   # y-axis label size
  )

p1 + p2
