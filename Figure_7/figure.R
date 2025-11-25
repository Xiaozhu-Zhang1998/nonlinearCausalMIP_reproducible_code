library(tidyverse)
library(latex2exp)
library(patchwork)


p = 15
Kappa = ((2:16) + 1) / 30

# navigate to the directory where the "non_parametric.csv" and "naive_MIP.csv" are located


tab1 = read.csv("non_parametric.csv") %>%
  mutate(
    kappa = round(Kappa[l+1], 2),
    kappa = factor(kappa),
    l2 = l2 / p,
    Rn = Rn + 3
  ) %>%
  filter(l != 14)


tab2 = read.csv("naive_MIP.csv") %>%
  mutate(
    Rn = 1,
    l2 = l2 / p,
    kappa = "Linear"
  ) 

df = rbind(tab2, tab1) %>%
  mutate(kappa = factor(kappa),
         kappa = fct_inorder(kappa))


# Define transformation constants
l2_min <- min(df$l2)
l2_max <- max(df$l2)
Rn_min <- 1
Rn_max <- max(df$Rn)

# Function to scale Rn into perform-range
rescale_rn_l2 <- function(x) {
  (x - Rn_min) / (Rn_max - Rn_min) * (l2_max - l2_min) + l2_min
}

# Inverse function for labeling the secondary axis
inv_rescale_rn_l2 <- function(y) {
  (y - l2_min) / (l2_max - l2_min) * (Rn_max - Rn_min) + Rn_min
}


# Prepare Rn data
df_rn <- df %>%
  group_by(kappa) %>%
  summarise(Rn = first(Rn)) %>%
  mutate(Rn_rescaled_l2 = rescale_rn_l2(Rn))

# Main plot
p1 = ggplot(df, aes(x = factor(kappa))) +
  geom_boxplot(aes(y = l2), fill = "lightblue", outlier.shape = NA) +
  geom_point(data = df_rn, aes(y = Rn_rescaled_l2), 
             color = "red", size = 3, shape = 18) +
  scale_y_continuous(
    name = TeX("Estimation error for $\\sigma_j^{*2}$'s"),
    breaks = seq(0, 0.6, by = 0.1),
    sec.axis = sec_axis(
      ~inv_rescale_rn_l2(.), 
      name = TeX("Number of basis $R_n$"),
      breaks = seq(5, 35, by = 5)
    )
  ) +
  labs(x = TeX("Factor \\zeta")) +
  theme_bw() +
  theme(
    axis.title.x = element_text(size = 10),    # x-axis label
    axis.title.y = element_text(size = 10),    # y-axis label
    plot.title    = element_text(size = 10),
    axis.text.x = element_text(angle = 45, hjust = 1)
  ) 




# Define transformation constants
diff_min <- min(df$diff)
diff_max <- max(df$diff)
Rn_min <- 3
Rn_max <- max(df$Rn)

# Function to scale Rn into perform-range
rescale_rn_diff <- function(x) {
  (x - Rn_min) / (Rn_max - Rn_min) * (diff_max - diff_min) + diff_min
}

# Inverse function for labeling the secondary axis
inv_rescale_rn_diff <- function(y) {
  (y - diff_min) / (diff_max - diff_min) * (Rn_max - Rn_min) + Rn_min
}


# Prepare Rn data
df_rn_diff <- df %>%
  group_by(kappa) %>%
  summarise(Rn = first(Rn)) %>%
  mutate(Rn_rescaled_diff = rescale_rn_diff(Rn))

# Main plot
p2 = ggplot(df, aes(x = factor(kappa))) +
  geom_boxplot(aes(y = diff), fill = "lightblue", outlier.shape = NA) +
  geom_point(data = df_rn_diff, aes(y = Rn_rescaled_diff), 
             color = "red", size = 3, shape = 18) +
  scale_y_continuous(
    name = TeX("SHD d"),
    sec.axis = sec_axis(
      ~inv_rescale_rn_diff(.), 
      name = TeX("Number of basis $R_n$"),
      breaks = seq(5, 35, by = 5)
    )
  ) +
  labs(x = TeX("Factor \\zeta")) +
  theme_bw() +
  theme(
    axis.title.x = element_text(size = 10),    # x-axis label
    axis.title.y = element_text(size = 10),    # y-axis label
    plot.title    = element_text(size = 10),
    axis.text.x = element_text(angle = 45, hjust = 1)
  ) 


p2 + p1
