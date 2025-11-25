library(tidyverse)
library(patchwork)

# navigate to the directory where the suff_stat_output files are located

df = mutate(read.csv("suff_stat_output_n300.csv"), n = 300)
N = c((4:16) * 100, 2000, 2500)
for(n in N) {
  name = paste0("suff_stat_output_n", n, ".csv")
  df = rbind(
    df,
    mutate(read.csv(name), n = n)
  )
}

df = df %>%
  pivot_longer(-c("n", "m", "method"), names_to = "time_type", values_to = "time") %>%
  mutate(n = factor(n))

common_scale = scale_color_discrete(
  name = "Formultaion",  # legend title
  labels = c(
    "default" = "Default",
    "suff_stat" = "Sufficient stat"
  )
)

p1 = df %>% 
  filter(time_type == "time_obj") %>%
  ggplot(aes(x = n, y = time, col = method)) +
  geom_boxplot(outlier.size = 1) +
  labs(x = "Sample size n", y = "Time", title = "Specifying obejective functions") +
  common_scale +
  scale_y_log10() +
  theme_bw() +
  theme(
    plot.title = element_text(size = 10),  # Title size
    axis.title.x = element_text(size = 10),  # x-axis label size
    axis.title.y = element_text(size = 10),   # y-axis label size
    axis.text.x = element_text(angle = 45, hjust = 1)
  )


p2 = df %>% 
  filter(time_type == "time_bigM") %>%
  ggplot(aes(x = n, y = time, col = method)) +
  geom_boxplot(outlier.size = 1) +
  labs(x = "Sample size n", y = "Time", title = "The pilot MIP that finds big M") +
  common_scale +
  scale_y_log10() +
  theme_bw() +
  theme(
    plot.title = element_text(size = 10),  # Title size
    axis.title.x = element_text(size = 10),  # x-axis label size
    axis.title.y = element_text(size = 10),   # y-axis label size
    axis.text.x = element_text(angle = 45, hjust = 1)
  )


p3 = df %>% 
  filter(time_type == "time_opt") %>%
  ggplot(aes(x = n, y = time, col = method)) +
  geom_boxplot(outlier.size = 1) +
  labs(x = "Sample size n", y = "Time", title = "The core MIP optimization") +
  common_scale +
  scale_y_log10() +
  theme_bw() +
  theme(
    plot.title = element_text(size = 10),  # Title size
    axis.title.x = element_text(size = 10),  # x-axis label size
    axis.title.y = element_text(size = 10),   # y-axis label size
    axis.text.x = element_text(angle = 45, hjust = 1)
  )


p4 = df %>% 
  filter(time_type == "time_total") %>%
  ggplot(aes(x = n, y = time, col = method)) +
  geom_boxplot(outlier.size = 1) +
  labs(x = "Sample size n", y = "Time", title = "Total running time") +
  common_scale +
  scale_y_log10() +
  theme_bw() +
  theme(
    plot.title = element_text(size = 10),  # Title size
    axis.title.x = element_text(size = 10),  # x-axis label size
    axis.title.y = element_text(size = 10),   # y-axis label size
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

p1 + p2 + p3 + p4 + 
  plot_layout(guides = "collect") & 
  theme(legend.position = "bottom", 
        legend.title = element_text(size = 10))

