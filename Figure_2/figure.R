library(tidyverse)
library(latex2exp)

# navigate to the directory where l0l1.csv is located

df = read.csv("l0l1.csv") %>%
  mutate(lambda = round(exp(lambda), 4))

df %>% 
  mutate(lambda = factor(lambda)) %>%
  ggplot(aes(x = lambda, y = diff / 25, col = type)) +
  geom_boxplot() +
  labs(x = TeX("\\lambda"), y = TeX("$d/s^*$")) +
  scale_color_discrete(
    name = "Penalty",  
    labels = c(
      "l0" = TeX("Group $L_0$"),
      "l1" = TeX("Group $L_1$")
    )
  ) +
  theme_bw() +
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = 10), 
    plot.title = element_text(size = 10),  # Title size
    axis.text.x = element_text(angle = 20, hjust = 1),
    axis.title.x = element_text(size = 10),  # x-axis label size
    axis.title.y = element_text(size = 10),   # y-axis label size
  )

common_scale = scale_color_discrete(
  name = "Formultaion",  # legend title
  labels = c(
    "default" = "Default",
    "suff_stat" = "Sufficient stat"
  )
)